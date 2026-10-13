"""Dataset building — every image this project trains on comes from here.

One module, three datasets, all pulled from iNaturalist under open licences and
recorded in a manifest for provenance:

  ``negatives``  background images for the DAMAGE detector — insects that are
                 not Helopeltis, and insect-free tea leaves. They exist because
                 the first detector, trained on an all-positive set, fired on
                 essentially any image. Labels are empty files: in YOLO that
                 means "background".
  ``groups``     ten insect groups for the general WHO classifier.
  ``helopeltis`` the genus plus its own family as hard negatives — the data for
                 "is this Helopeltis or not".

Two traps this module is built to avoid, both of which produced fake results
before they were fixed:

  * **Miridae in the detector negatives.** Helopeltis IS a mirid. An image
    labelled "background" that contains one teaches the detector to ignore its
    own target. ``TAXA`` therefore excludes the family.
  * **Test images that were trained on.** Every download excludes observation
    ids already recorded in the relevant manifest, so held-out splits stay
    held out.

Commands (from the repository root):

    python -m src.data download               # START HERE: datasets + trained models
    python -m src.data negatives              # detector backgrounds, train+valid
    python -m src.data negatives --test       # held-out detector backgrounds
    python -m src.data groups                 # ten insect groups
    python -m src.data helopeltis             # Helopeltis + other mirids
    python -m src.data regional               # same-region mirids (geo balance)
    python -m src.data rebuild                # exact rebuild from manifest.csv
    python -m src.data flush <Raw Data dir>   # leaf-stage images (Mendeley)

No images are versioned in git. ``download`` fetches the exact files the
published results were measured on (with SHA-256 checks); the iNaturalist
commands rebuild or extend the sets from source.
"""
import argparse
import csv
import hashlib
import io
import os
import zipfile
import json
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "TeaGuard-dataset-builder/1.0 (academic research)"}

HELO_DIR = HERE.parent / "data" / "raw" / "helopeltis"
FLUSH_DIR = HERE.parent / "data" / "raw" / "flush_stages"
SPLIT_DIR = HELO_DIR / "splits"
MODELS_DIR = HERE.parent / "models"


def api(path, **params):
    url = f"https://api.inaturalist.org/v1/{path}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def resolve(name):
    """Taxon name -> (id, canonical name). Names, not hard-coded fragile ids."""
    res = api("taxa", q=name, per_page=1)["results"]
    if not res:
        return None, None
    return res[0]["id"], res[0]["name"]


# ============================================================================
# 1. Detector backgrounds  (data/raw/helopeltis_negatives/)
# ============================================================================
NEG = HERE.parent / "data" / "raw" / "helopeltis_negatives"

# (taxon name, images to fetch). Names resolved via the iNat taxa endpoint so we
# don't hard-code fragile IDs. NO Miridae here (would risk real Helopeltis).
TAXA = [
    ("Apis mellifera", 18),      # honeybee (easy)
    ("Coccinellidae", 18),       # ladybirds (easy)
    ("Syrphidae", 18),           # hoverflies (easy)
    ("Formicidae", 16),          # ants
    ("Papilionoidea", 16),       # butterflies
    ("Araneae", 14),             # spiders
    ("Cicadellidae", 18),        # leafhoppers (hard: small, bug-like, not Miridae)
    ("Coreidae", 18),            # leaf-footed bugs (hard: true bugs, not Miridae)
    ("Camellia sinensis", 22),   # TEA LEAVES, no insect (pure background)
    ("Plantae", 20),             # generic foliage (pure background)
]
VALID_EVERY = 5  # every 5th image of each taxon goes to the valid split

# Held-out TEST negatives (--test). Fewer per taxon than training, but weighted
# towards the pure-background classes: "insect-free tea leaf reported as
# Helopeltis" is the failure mode that matters most in the field.
TEST_TAXA = [
    ("Apis mellifera", 6),
    ("Coccinellidae", 6),
    ("Syrphidae", 6),
    ("Formicidae", 6),
    ("Papilionoidea", 6),
    ("Araneae", 6),
    ("Cicadellidae", 8),
    ("Coreidae", 8),
    ("Camellia sinensis", 12),
    ("Plantae", 10),
]


def observations(taxon_id, n, exclude=frozenset()):
    """Fetch up to n observations, skipping any obs_id in `exclude`.

    `exclude` is how the test split stays disjoint from train/valid: the same
    query is issued, already-downloaded observations are dropped, so what comes
    back is further down the same ranking.
    """
    out, page = [], 1
    while len(out) < n and page <= 8:
        data = api("observations", taxon_id=taxon_id, quality_grade="research",
                   photos="true", photo_license="cc0,cc-by,cc-by-nc",
                   per_page=30, page=page, order_by="votes")
        results = data.get("results", [])
        if not results:
            break
        for obs in results:
            ph = obs.get("photos") or []
            if not ph or obs["id"] in exclude:
                continue
            out.append({
                "obs_id": obs["id"],
                "url": ph[0]["url"].replace("square", "medium"),
                "license": ph[0].get("license_code"),
                "attribution": (ph[0].get("attribution") or "")[:140],
            })
            if len(out) >= n:
                break
        page += 1
    return out


def known_obs_ids():
    """Observation ids already downloaded, read back from manifest.csv."""
    path = NEG / "manifest.csv"
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {int(row["obs_id"]) for row in csv.DictReader(f) if row.get("obs_id")}


def write_manifest(rows, append):
    """Write (or append to) manifest.csv, keeping provenance for every image."""
    path = NEG / "manifest.csv"
    mode = "a" if (append and path.exists()) else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if mode == "w":
            w.writeheader()
        w.writerows(rows)


def build_detector_negatives(test_mode=False):
    taxa = TEST_TAXA if test_mode else TAXA
    splits = ("test",) if test_mode else ("train", "valid")
    # Never re-download an image that is already in another split: a background
    # the model trained on would make the test-set precision meaningless.
    exclude = known_obs_ids() if test_mode else set()

    for split in splits:
        (NEG / split / "images").mkdir(parents=True, exist_ok=True)
        (NEG / split / "labels").mkdir(parents=True, exist_ok=True)

    manifest = []
    totals = {s: 0 for s in splits}
    for name, n in taxa:
        tid, resolved = resolve(name)
        if tid is None:
            print(f"[warn] could not resolve taxon '{name}'")
            continue
        obs = observations(tid, n, exclude=exclude)
        got = 0
        for i, rec in enumerate(obs):
            split = "test" if test_mode else ("valid" if (i % VALID_EVERY == 0) else "train")
            stem = f"neg_{tid}_{rec['obs_id']}"
            img_path = NEG / split / "images" / f"{stem}.jpg"
            lbl_path = NEG / split / "labels" / f"{stem}.txt"
            try:
                data = urllib.request.urlopen(
                    urllib.request.Request(rec["url"], headers=UA), timeout=60).read()
                img_path.write_bytes(data)
                lbl_path.write_text("")  # empty label => YOLO background image
                got += 1
                totals[split] += 1
                manifest.append({
                    "file": f"{split}/images/{stem}.jpg", "taxon": resolved,
                    "obs_id": rec["obs_id"], "license": rec["license"],
                    "attribution": rec["attribution"],
                    "obs_url": f"https://www.inaturalist.org/observations/{rec['obs_id']}",
                })
            except Exception as e:
                print(f"  [skip] {rec['obs_id']}: {e}")
        print(f"{name:<20} ({resolved}, id {tid}): {got} images")

    if not manifest:
        print("[warn] nothing downloaded — manifest unchanged")
        return

    write_manifest(manifest, append=test_mode)

    summary = " ".join(f"{s}={totals[s]}" for s in splits)
    print(f"\nTotal negatives: {summary} (={sum(totals.values())})")
    print(f"Manifest: {NEG / 'manifest.csv'}")



# ============================================================================
# 2 & 3. Insect groups and the Helopeltis task  (data/raw/insect_families/)
# ============================================================================
FAM = HERE.parent / "data" / "raw" / "insect_families"

# (iNaturalist query, folder/class name, ecological role).
# Miridae FIRST because it is the one that matters: Helopeltis is a mirid, so
# this class is what makes the model able to say "this is the pest's family"
# rather than merely "this is not a bee".
GROUPS = [
    ("Miridae",         "miridae",       "pest"),
    ("Cicadellidae",    "cicadellidae",  "pest"),
    ("Coreidae",        "coreidae",      "pest"),
    ("Aphididae",       "aphididae",     "pest"),
    ("Coccinellidae",   "coccinellidae", "beneficial"),
    ("Araneae",         "araneae",       "beneficial"),
    ("Apidae",          "apidae",        "pollinator"),
    ("Syrphidae",       "syrphidae",     "pollinator"),
    ("Formicidae",      "formicidae",    "neutral"),
    ("Papilionoidea",   "papilionoidea", "pollinator"),
]

PER_CLASS = 150   # target images per class; iNat may return fewer for rare taxa

# --- The Helopeltis task (--helopeltis) --------------------------------------
# The question the project actually needs answered is "is this Helopeltis or
# not?", which the ten-group model cannot answer: it stops at the family, and
# Miridae holds thousands of species. These two taxa turn it into a real class.
#
# `other_miridae` is the class that matters. Telling Helopeltis from a bee is
# trivial; telling it from another mirid is the whole problem, so the negatives
# have to be its own family, with the genus itself excluded.
HELOPELTIS_TAXON = 474380   # genus Helopeltis — 795 openly licensed photos
MIRIDAE_TAXON = 53505       # family Miridae — 198k once Helopeltis is removed
HELO_GROUPS = [
    ("helopeltis",    "pest", {"taxon_id": HELOPELTIS_TAXON}),
    ("other_miridae", "pest", {"taxon_id": MIRIDAE_TAXON,
                               "without_taxon_id": HELOPELTIS_TAXON}),
]

# Where Helopeltis is actually photographed. The first `--helopeltis` pull left
# the classes split by continent — Helopeltis from tropical Asia, the negatives
# from the USA and Europe — and the model partly learned the region instead of
# the insect: it called 22.8% of same-region mirids Helopeltis, against 9.4% on
# the mixed-geography test split (see models/helopeltis_geo_control.json).
# `--regional-negatives` closes that gap by adding mirids from these places to
# the negative class, so the boundary the model has to learn is the animal.
HELOPELTIS_REGIONS = ["Singapore", "Hong Kong", "India", "Taiwan", "Malaysia",
                      "Indonesia", "Sri Lanka", "Thailand", "Vietnam", "Philippines"]


def fetch(taxon_id, n, **extra):
    """Fetch up to n research-grade, openly licensed observation photos.

    `extra` passes through iNat query params — `without_taxon_id` is what keeps
    the Helopeltis genus out of the other-mirids class.
    """
    out, page = [], 1
    while len(out) < n and page <= 20:
        data = api("observations", taxon_id=taxon_id, quality_grade="research",
                   photos="true", photo_license="cc0,cc-by,cc-by-nc",
                   per_page=100, page=page, order_by="votes", **extra)
        results = data.get("results", [])
        if not results:
            break
        for obs in results:
            ph = obs.get("photos") or []
            if not ph:
                continue
            out.append({
                "obs_id": obs["id"],
                "url": ph[0]["url"].replace("square", "medium"),
                "license": ph[0].get("license_code"),
                "attribution": (ph[0].get("attribution") or "")[:140],
            })
            if len(out) >= n:
                break
        page += 1
    return out


def resolve_place(name):
    res = api("places/autocomplete", q=name, per_page=1)["results"]
    return (res[0]["id"], res[0]["name"]) if res else (None, None)


def build_regional_negatives(per_place=60):
    """Add same-region mirids to the negative class.

    The geographic control set in `_geo_control/` is EXCLUDED by observation id.
    It is the only unbiased measurement available; training on it would destroy
    the very number that tells us whether this fix worked.
    """
    control = {int(p.stem.split("_")[-1]) for p in (FAM / "_geo_control").glob("*.jpg")}
    print(f"Excluding {len(control)} control observations from the download\n")

    dest = FAM / "other_miridae"
    dest.mkdir(parents=True, exist_ok=True)
    manifest, before = [], len(list(dest.glob("*.jpg")))

    for place in HELOPELTIS_REGIONS:
        pid, resolved = resolve_place(place)
        if pid is None:
            print(f"[warn] could not resolve place '{place}'")
            continue
        got = 0
        for rec in fetch(MIRIDAE_TAXON, per_place + len(control),
                         without_taxon_id=HELOPELTIS_TAXON, place_id=pid):
            if rec["obs_id"] in control or got >= per_place:
                continue
            path = dest / f"other_miridae_{rec['obs_id']}.jpg"
            if path.exists():
                continue
            try:
                path.write_bytes(urllib.request.urlopen(
                    urllib.request.Request(rec["url"], headers=UA), timeout=60).read())
                got += 1
                manifest.append({
                    "file": f"other_miridae/{path.name}", "class": "other_miridae",
                    "role": "pest", "taxon": f"Miridae ({resolved})",
                    "obs_id": rec["obs_id"], "license": rec["license"],
                    "attribution": rec["attribution"],
                    "obs_url": f"https://www.inaturalist.org/observations/{rec['obs_id']}",
                })
            except Exception as e:
                print(f"  [skip] {rec['obs_id']}: {e}")
        print(f"{resolved:24s} +{got:3d}")

    if manifest:
        write_manifest_rows(manifest)
    after = len(list(dest.glob("*.jpg")))
    print(f"\nother_miridae: {before} -> {after} (+{after - before} from Helopeltis regions)")
    print("\nNext: python -m src.train insects --task helopeltis")
    print("Then : python -m src.evaluate geo")


def build_helopeltis(per_class=600):
    """Download the two taxa the Helopeltis-vs-not question needs.

    `taxon_id` is recorded per image so contamination stays auditable: the
    other-mirids class must never contain a Helopeltis, or the model would be
    trained to call its own target a negative.
    """
    manifest = []
    for cls, role, params in HELO_GROUPS:
        (FAM / cls).mkdir(parents=True, exist_ok=True)
        got = 0
        for rec in fetch(n=per_class, **params):
            path = FAM / cls / f"{cls}_{rec['obs_id']}.jpg"
            if path.exists():
                continue
            try:
                data = urllib.request.urlopen(
                    urllib.request.Request(rec["url"], headers=UA), timeout=60).read()
                path.write_bytes(data)
                got += 1
                manifest.append({
                    "file": f"{cls}/{path.name}", "class": cls, "role": role,
                    "taxon": cls, "obs_id": rec["obs_id"],
                    "license": rec["license"], "attribution": rec["attribution"],
                    "obs_url": f"https://www.inaturalist.org/observations/{rec['obs_id']}",
                })
            except Exception as e:
                print(f"  [skip] {rec['obs_id']}: {e}")
        total = len(list((FAM / cls).glob("*.jpg")))
        print(f"{cls:16s} +{got:4d} downloaded, {total:4d} on disk")

    if manifest:
        write_manifest_rows(manifest)
    print(f"\nNew images: {len(manifest)}")
    print("\nNext: python -m src.train insects --task helopeltis")


def write_manifest_rows(rows):
    path = FAM / "manifest.csv"
    mode = "a" if path.exists() else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if mode == "w":
            w.writeheader()
        w.writerows(rows)


def rebuild_from_manifest():
    """Re-download exactly the images listed in manifest.csv.

    Photo URLs are not stored (they can change), so observations are re-fetched
    by id — the API takes up to 200 ids per call, so a full rebuild is a handful
    of requests, not one per image.
    """
    path = FAM / "manifest.csv"
    if not path.exists():
        raise SystemExit(f"No manifest at {path} — nothing to rebuild from.")

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    wanted = {int(r["obs_id"]): r for r in rows}
    missing = {oid: r for oid, r in wanted.items() if not (FAM / r["file"]).exists()}
    print(f"{len(wanted)} images in manifest, {len(missing)} missing locally")
    if not missing:
        return

    ids = list(missing)
    got = 0
    for i in range(0, len(ids), 200):
        batch = ids[i:i + 200]
        data = api("observations", id=",".join(str(x) for x in batch), per_page=200)
        for obs in data.get("results", []):
            row = missing.get(obs["id"])
            ph = obs.get("photos") or []
            if not row or not ph:
                continue
            dest = FAM / row["file"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                url = ph[0]["url"].replace("square", "medium")
                dest.write_bytes(urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=60).read())
                got += 1
            except Exception as e:
                print(f"  [skip] {obs['id']}: {e}")
        print(f"  ...{got}/{len(missing)}")

    gone = len(missing) - got
    print(f"\nRestored {got} images." +
          (f" {gone} no longer available on iNaturalist." if gone else ""))


def build_insect_groups(per_class=PER_CLASS):
    manifest = []
    print(f"Target: {per_class} images per class, {len(GROUPS)} classes\n")

    for query, cls, role in GROUPS:
        (FAM / cls).mkdir(parents=True, exist_ok=True)
        tid, resolved = resolve(query)
        if tid is None:
            print(f"[warn] could not resolve '{query}' — class skipped")
            continue

        got = 0
        for rec in fetch(tid, per_class):
            path = FAM / cls / f"{cls}_{rec['obs_id']}.jpg"
            if path.exists():        # re-runs top up rather than re-download
                continue
            try:
                data = urllib.request.urlopen(
                    urllib.request.Request(rec["url"], headers=UA), timeout=60).read()
                path.write_bytes(data)
                got += 1
                manifest.append({
                    "file": f"{cls}/{path.name}", "class": cls, "role": role,
                    "taxon": resolved, "obs_id": rec["obs_id"],
                    "license": rec["license"], "attribution": rec["attribution"],
                    "obs_url": f"https://www.inaturalist.org/observations/{rec['obs_id']}",
                })
            except Exception as e:
                print(f"  [skip] {rec['obs_id']}: {e}")

        total = len(list((FAM / cls).glob("*.jpg")))
        flag = "" if total >= per_class * 0.6 else "   <-- THIN, needs another source"
        print(f"{cls:16s} {role:11s} +{got:4d} downloaded, {total:4d} on disk{flag}")

    if not manifest:
        print("\nNothing new downloaded.")
        return

    path = FAM / "manifest.csv"
    mode = "a" if path.exists() else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        if mode == "w":
            w.writeheader()
        w.writerows(manifest)

    print(f"\nNew images: {len(manifest)}")
    print(f"Manifest  : {path}")
    print("\nNext: python -m src.train insects --task groups")


# ============================================================================
# Capture-aware splitting
# ============================================================================
# Both public datasets are burst photography: someone walking a plantation,
# shooting continuously. Consecutive frames are the same leaf, seconds apart,
# same light and same background. Splitting those at random puts near-duplicates
# on both sides of the train/test line, and the test set stops measuring
# generalisation — it measures recall of something almost seen. Measured on the
# Roboflow split as it shipped: 45 of its 49 test images sat immediately beside
# a training image in the original capture order, one of them 5 seconds away.
#
# The fix is to split by *capture session* instead of by file: cluster the
# photos back into the runs they were taken in, and send each run entirely to
# one side. Filenames carry the ordering — a clock timestamp, or a sequence
# counter — which is enough to reconstruct it.
#
# The two gap thresholds below are not round numbers picked for looks. In the
# detector set, 224 of 244 consecutive clock gaps are under two minutes and the
# tail begins there; the sequence counters run in steps of 1 to 10 with a clear
# break above 20. Anything closer than these is treated as one session.
SESSION_GAP_SECONDS = 120
SESSION_GAP_INDEX = 10

_WA_NAME = re.compile(r"IMG-(\d{8})-WA(\d+)")          # B_IMG-20231224-WA0105
_CLOCK_NAME = re.compile(r"(\d{8})[_-](\d{6})")        # A_20231224_160737
_SEQ_NAME = re.compile(r"^image_0_(\d+)$")             # image_0_1016


def _capture_key(stem):
    """Locate a photo in its capture sequence: (namespace, ordinal).

    Ordinals are only comparable inside a namespace — a WhatsApp counter and a
    wall clock do not share a timeline. A name that carries neither becomes its
    own namespace, i.e. a session of one, which is the safe default.
    """
    m = _WA_NAME.search(stem)
    if m:
        return f"wa:{m.group(1)}", float(int(m.group(2)))
    m = _CLOCK_NAME.search(stem)
    if m:
        return "clock", datetime.strptime(m.group(1) + m.group(2),
                                          "%Y%m%d%H%M%S").timestamp()
    m = _SEQ_NAME.match(stem)
    if m:
        return "seq", float(m.group(1))
    return f"lone:{stem}", 0.0


def capture_groups(stems):
    """Map each filename stem to the id of the capture session it belongs to.

    A session is a maximal run of photos adjacent in the original order. Two
    photos in different sessions are guaranteed to be at least one gap
    threshold apart, so a split that keeps sessions whole cannot put two
    near-duplicates on opposite sides of it.
    """
    by_namespace = defaultdict(list)
    for stem in stems:
        namespace, ordinal = _capture_key(stem)
        by_namespace[namespace].append((ordinal, stem))

    groups = {}
    for namespace, items in sorted(by_namespace.items()):
        items.sort()
        gap = SESSION_GAP_SECONDS if namespace == "clock" else SESSION_GAP_INDEX
        session = 0
        for i, (ordinal, stem) in enumerate(items):
            if i and ordinal - items[i - 1][0] > gap:
                session += 1
            groups[stem] = f"{namespace}#{session}"
    return groups


def assign_sessions(groups, ratios=(0.8, 0.1, 0.1), names=("train", "valid", "test"),
                    labels=None):
    """Send whole sessions to splits, largest first, filling the emptiest quota.

    Deterministic: no RNG, so the split is a function of the filenames and
    reproduces anywhere without a seed to remember.

    `labels` (stem -> class) turns the quota per-class. Without it, a size-only
    greedy piles one class into one split whenever sessions are class-pure —
    which is exactly what happened to the flush classifier: 20 large sessions,
    most of them a single stage, and every mature_leaf session landed in the
    test split, leaving it with zero new_flush. Tracking a quota per class and
    scoring a session by how much each candidate split still needs *its* classes
    keeps every class present in every split, sessions still whole.
    """
    members = defaultdict(list)          # session -> [stem, ...]
    for stem, session in groups.items():
        members[session].append(stem)

    def profile(session):
        """Class counts for a session — {None: size} when labels are absent."""
        if labels is None:
            return {None: len(members[session])}
        c = Counter(labels[s] for s in members[session])
        return dict(c)

    classes = ({None} if labels is None else set(labels.values()))
    totals = Counter()
    for session in members:
        for cls, n in profile(session).items():
            totals[cls] += n
    quota = {(n, c): r * totals[c] for n in names for r, c in
             [(ratios[names.index(n)], c) for c in classes]}
    filled = {(n, c): 0 for n in names for c in classes}

    placement = {}
    for session in sorted(members, key=lambda s: (-len(members[s]), s)):
        prof = profile(session)
        # Prefer the split whose remaining need for *this session's* classes is
        # greatest, weighted by how many of each the session carries.
        def score(n):
            return sum(cnt * (quota[(n, c)] - filled[(n, c)]) / max(quota[(n, c)], 1e-9)
                       for c, cnt in prof.items())
        pick = max(names, key=score)
        placement[session] = pick
        for c, cnt in prof.items():
            filled[(pick, c)] += cnt
    return placement, filled


def verify_split(stems_by_split):
    """Report the closest any two photos from different splits come to each other.

    This is the number the whole exercise exists to move. On the Roboflow split
    it was 5 seconds; a correct capture-aware split cannot come in below the
    gap thresholds.
    """
    keyed = {split: [_capture_key(s) for s in stems]
             for split, stems in stems_by_split.items()}
    worst = {}
    for split, keys in keyed.items():
        if split == "train":
            continue
        train_keys = defaultdict(list)
        for ns, o in keyed["train"]:
            train_keys[ns].append(o)
        closest = None
        for ns, o in keys:
            if ns not in train_keys:
                continue
            d = min(abs(o - t) for t in train_keys[ns])
            unit = "s" if ns == "clock" else " positions"
            if closest is None or d < closest[0]:
                closest = (d, unit)
        worst[split] = closest
    return worst


def split_detector():
    """Re-split the positive detector set by capture session.

    Leaves the Roboflow folders untouched — they are the upstream data. Writes
    the session assignment (versioned, machine-independent) plus the absolute
    path lists Ultralytics reads (generated, gitignored).
    """
    pos = HELO_DIR
    images = {}
    for split in ("train", "valid", "test"):
        for p in sorted((pos / split / "images").glob("*.jpg")):
            images[p.stem.split("_jpg.rf.")[0]] = p

    if not images:
        raise SystemExit(f"No images under {pos} — nothing to split.")

    groups = capture_groups(images)
    placement, filled = assign_sessions(groups)

    by_split = defaultdict(list)
    for stem, session in groups.items():
        by_split[placement[session]].append(stem)

    print(f"{len(images)} images in {len(set(groups.values()))} capture sessions\n")
    for name in ("train", "valid", "test"):
        stems = by_split[name]
        sessions = len({groups[s] for s in stems})
        print(f"  {name:5s} {len(stems):4d} images  ({len(stems)/len(images):.0%})"
              f"  from {sessions:3d} sessions")

    print("\nClosest approach to a training photo (was 5s on the Roboflow split):")
    for split, closest in verify_split({k: v for k, v in by_split.items()}).items():
        if closest is None:
            print(f"  {split:5s} shares no capture namespace with train")
        else:
            print(f"  {split:5s} {closest[0]:.0f}{closest[1]}")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_DIR / "sessions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "session", "split"])
        for stem in sorted(groups):
            w.writerow([images[stem].name, groups[stem], placement[groups[stem]]])

    MODELS_DIR.mkdir(exist_ok=True)
    for name in ("train", "valid", "test"):
        listing = MODELS_DIR / f"split_{name}.txt"
        listing.write_text("\n".join(
            str(images[s].resolve()).replace("\\", "/") for s in sorted(by_split[name])
        ) + "\n", encoding="utf-8")

    print(f"\nSession assignment -> {SPLIT_DIR / 'sessions.csv'} (versioned)")
    print(f"Path lists         -> {MODELS_DIR}/split_*.txt (generated)")
    print("\nNext: python -m src.evaluate detector   (or retrain: python -m src.train detector)")


# ============================================================================
# Flush-stage images — TeaLeafAgeQuality (Mendeley Data), organised by class
# ============================================================================

FLUSH_CATEGORY_TO_CLASS = {"A": "new_flush", "B": "new_flush",
                           "C": "mature_leaf", "D": "mature_leaf"}
# The dataset spells it both "Category" and "Catagory".
_CATEGORY_DIR = re.compile(r"cat[ae]gory[\s_-]*([A-D])\b", re.IGNORECASE)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def prepare_flush(raw_dir, out_dir=FLUSH_DIR):
    """Copy the four age categories into the two flush-stage class folders."""
    import shutil
    folders = {}
    for folder in sorted(p for p in Path(raw_dir).rglob("*") if p.is_dir()):
        m = _CATEGORY_DIR.search(folder.name)
        if m and m.group(1).upper() not in folders:
            folders[m.group(1).upper()] = folder
    missing = sorted(set(FLUSH_CATEGORY_TO_CLASS) - set(folders))
    if missing:
        raise SystemExit(f"Category folder(s) {missing} not found under {raw_dir}")

    counts = Counter()
    for letter, folder in sorted(folders.items()):
        cls = FLUSH_CATEGORY_TO_CLASS[letter]
        (Path(out_dir) / cls).mkdir(parents=True, exist_ok=True)
        for img in sorted(folder.rglob("*")):
            if img.suffix.lower() in _IMAGE_SUFFIXES:
                shutil.copy2(img, Path(out_dir) / cls / f"{letter}_{img.name}")
                counts[cls] += 1
    for cls in ("new_flush", "mature_leaf"):
        print(f"  {cls:12s} {counts[cls]} images")
    print("\nNext: python -m src.train flush")
    return dict(counts)


# ============================================================================
# Download — the exact files the published numbers were measured on
# ============================================================================

# Release assets on GitHub: zips are kept out of the git history so cloning
# stays light, but they are pinned by checksum so every download is the same
# set of bytes. Override the location with TEAGUARD_DATA_URL (a mirror, or a
# local folder for offline work).
DATA_RELEASE_URL = "https://github.com/angelbohorquez05/Tea-Guard/releases/download/data-v1"
RELEASE_ASSETS = {
    # asset name -> (sha256, extract into, folder that proves it is installed)
    "helopeltis.zip": (
        "9f11429948bfa2be281508cb130a8c127dfa4f508e704b9c15095be00e918c07",
        HERE.parent / "data" / "raw", HELO_DIR / "test" / "images"),
    "helopeltis_negatives.zip": (
        "0749e5b79e36e365508af29455844670988bae6bfb0a1b9bd771ddaecf4a5863",
        HERE.parent / "data" / "raw",
        HERE.parent / "data" / "raw" / "helopeltis_negatives" / "test" / "images"),
    "models.zip": (
        "5c85a9b260069dffc97702b66329c99c3d39e50b5e899145f5696c40dc0c1e28",
        MODELS_DIR, MODELS_DIR / "helopeltis_damage.onnx"),
}


def _read_asset(base, name):
    """Bytes of one release asset, from a URL or a local folder."""
    if base.startswith(("http://", "https://")):
        req = urllib.request.Request(f"{base}/{name}", headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()
    return (Path(base) / name).read_bytes()


def download(force=False, base=None):
    """Fetch datasets and trained models, verify them, extract them, then split."""
    base = (base or os.environ.get("TEAGUARD_DATA_URL") or DATA_RELEASE_URL).rstrip("/")
    for name, (sha256, dest, marker) in RELEASE_ASSETS.items():
        present = marker.is_dir() and any(marker.iterdir()) if marker.suffix == "" \
            else marker.exists()
        if present and not force:
            print(f"  {name:26s} already installed, skipping (--force to refresh)")
            continue
        print(f"  {name:26s} downloading...", end="", flush=True)
        blob = _read_asset(base, name)
        digest = hashlib.sha256(blob).hexdigest()
        if digest != sha256:
            raise SystemExit(f"\n{name}: checksum mismatch (got {digest[:12]}..., "
                             f"expected {sha256[:12]}...). Refusing to install it.")
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            z.extractall(dest)
            n = len(z.namelist())
        print(f" {len(blob) / 1e6:.1f} MB, {n} files, checksum OK")

    print()
    split_detector()


# ============================================================================
# Command line
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="datasets + trained models (start here)")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.add_argument("--from", dest="base", help="release URL or local folder")

    p = sub.add_parser("negatives", help="detector background images")
    p.add_argument("--test", action="store_true", help="held-out test split")

    p = sub.add_parser("groups", help="ten insect groups")
    p.add_argument("--per", type=int, default=PER_CLASS, help="images per class")

    p = sub.add_parser("helopeltis", help="Helopeltis + other mirids")
    p.add_argument("--per", type=int, default=600, help="images per class")

    p = sub.add_parser("regional", help="same-region mirids, to balance geography")
    p.add_argument("--per", type=int, default=60, help="images per region")

    sub.add_parser("rebuild", help="exact rebuild from manifest.csv")

    p = sub.add_parser("flush", help="organise TeaLeafAgeQuality into 2 classes")
    p.add_argument("raw_dir", help="extracted 'Raw Data' folder of the dataset")

    sub.add_parser("split", help="re-split the detector set by capture session")

    args = ap.parse_args()
    if args.command == "download":
        download(force=args.force, base=args.base)
    elif args.command == "negatives":
        build_detector_negatives(test_mode=args.test)
    elif args.command == "groups":
        build_insect_groups(per_class=args.per)
    elif args.command == "helopeltis":
        build_helopeltis(per_class=args.per)
    elif args.command == "regional":
        build_regional_negatives(per_place=args.per)
    elif args.command == "split":
        split_detector()
    elif args.command == "rebuild":
        rebuild_from_manifest()
    elif args.command == "flush":
        prepare_flush(args.raw_dir)


if __name__ == "__main__":
    main()
