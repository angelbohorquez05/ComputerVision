# insect_families — Insect groups for a general WHO

Images for a classifier that answers a **different question** from the damage
detector:

| Model | Input | Question it answers |
|---|---|---|
| `helopeltis_damage.onnx` | photo of a tea leaf | is there Helopeltis feeding damage? |
| `insect_families.onnx` | photo of an insect | which broad group, and is it a problem? |

These are **two entry points, not a pipeline**. The damage detector boxes
lesions on a leaf; a lesion crop contains no insect, so its output cannot be fed
to this classifier.

## Why this exists

The damage detector has a single class. It cannot name anything — the
`"Helopeltis spp."` label in `src/pipeline.py` is a hard-coded string stamped on
every detection, not a prediction. This dataset is the raw material for a model
that can actually say *what* is in a photo, at the group level.

## Classes

One folder per class; the group maps to an ecological role, which is what
drives a decision in the field.

| Class | Role | Note |
|---|---|---|
| `miridae` | pest | **Helopeltis's own family** — the class that matters |
| `cicadellidae` | pest | leafhoppers |
| `coreidae` | pest | leaf-footed bugs |
| `aphididae` | pest | aphids |
| `coccinellidae` | beneficial | ladybirds |
| `araneae` | beneficial | spiders (predators) |
| `apidae` | pollinator | bees |
| `syrphidae` | pollinator | hoverflies |
| `papilionoidea` | pollinator | butterflies |
| `formicidae` | neutral | ants |

**On the word "families":** `Araneae` is an order and `Papilionoidea` a
superfamily, so these are not all families in the taxonomic sense. Mixing ranks
is deliberate — the target is the ecological group a grower acts on, not a
determination key.

## Do NOT merge this with `helopeltis_negatives/`

The negatives deliberately **exclude Miridae**, because an image labelled
"background" that contained a real Helopeltis would teach the detector to ignore
its own target. Here Miridae is a required class. Same source (iNaturalist),
opposite requirement. Keep them apart.

## What this dataset cannot do

- **It is not a field detector.** These are iNaturalist photos: one insect,
  large and centred. A real scouting photo — a small bug somewhere in a wide
  frame of foliage — is a different problem, and needs detection with boxes plus
  tiled high-resolution inference.
- **It cannot separate Helopeltis from other mirids.** The class is the whole
  family. Doing better needs boxed, species-level annotations.
- **Group accuracy is not the honest headline.** Confusing Coreidae with
  Cicadellidae costs a grower nothing (both pests, same action). Confusing a
  pest with a pollinator means spraying something that was helping. That is why
  `train.py` reports **role accuracy** alongside group accuracy.

## The images are NOT versioned

Unlike `helopeltis_negatives/` (~38 MB, versioned), this set is ~390 MB — too heavy for the repository. `manifest.csv` **is**
versioned, and it holds the
observation id, licence and attribution of every image, so the set rebuilds
exactly:

```bash
python -m src.data rebuild   # exact rebuild from manifest.csv
python -m src.data groups --per 150         # fresh pull (a DIFFERENT sample)
python -m src.data groups --per 5           # smoke test
```

The distinction matters: a fresh pull does not reproduce the same set, because
iNaturalist's sort-by-votes ordering shifts over time. Only `rebuild`
reproduces it.

## Licences and attribution

Every image comes from iNaturalist under `cc0`, `cc-by` or `cc-by-nc`. Author
and observation link are in `manifest.csv`. **Academic / non-commercial** use,
consistent with `cc-by-nc`.
