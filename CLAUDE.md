# CLAUDE.md — TINAT

> Contesto del progetto per Claude Code. Leggilo all'inizio di ogni sessione.

## Cos'è TINAT

**TINAT** è il lavoro del **tirocinio** di Angel (lavoro individuale): *object detection* di
Helopeltis (famiglia Miridae) su colture di tè con YOLOv8, più una metrica ecologica
WHO×WHERE×WHEN e una capa di fusione.

Questo repository è la versione **anonimizzata e pubblicabile** del progetto reale di tirocinio:
usa esclusivamente dati pubblici e codice scritto da Angel. Il nome reale del progetto, l'azienda,
i colleghi, i luoghi e le infrastrutture NON compaiono mai qui — solo "TINAT".

## Riservatezza e anonimizzazione (IMPORTANTE)

- NON menzionare mai il nome reale del progetto né l'azienda in codice, docs, commit messages,
  nomi di file o commenti. Il progetto si chiama solo **TINAT**.
- NON copiare codice, dati, pesi di modelli o documenti proprietari dell'azienda.
- **Migrazione selettiva:** entra nel repo solo ciò che è di Angel e/o pubblico (dataset da
  internet, codice scritto da Angel, struttura ridefinita da zero). Ogni pezzo migrato = un commit
  dedicato con messaggio descrittivo.
- Le decisioni possono essere descritte in modo generico ("il supervisore del tirocinio ha
  indicato…" senza nomi né dettagli riservati).
- Prima di ogni commit, `grep` di nomi reali sui file trackeati (escluso `_private/`); se qualcosa
  è già stato pushato, non basta correggere il file: va purgato anche lo storico.

## Due livelli: repo anonimizzato + `_private/` locale

- **Livello pubblico/condivisibile** = questo repo TINAT anonimizzato (ciò che si committa e si
  pubblica).
- **Livello privato** = cartella locale **`_private/`**, in `.gitignore`, **mai versionata né
  pubblicata**: materiale reale del tirocinio (documenti aziendali, note, conversazioni con il
  supervisore, versioni con nomi reali). Prima di spostare qualsiasi cosa da `_private/` al repo,
  va anonimizzata.

## Lingue

- **Documentazione** (README, decision-log, docs/): **italiano**.
- **Codice, commenti, nomi di variabili, commit messages:** **inglese**.
- **Conversazione con Angel:** spagnolo (informale).

## Decision log

Le decisioni tecniche e organizzative rilevanti vanno registrate in `docs/decision-log.md` con:
(1) la decisione, (2) le alternative considerate, (3) perché è stata scelta questa. Formato di ogni
voce:

```markdown
## [YYYY-MM-DD] Titolo breve della decisione

**Contesto:** qual era il problema/bivio.
**Alternative considerate:** A, B, C — e perché sono state scartate.
**Decisione:** cosa è stato scelto.
**Motivazione:** perché questa e non le altre.
**Conseguenze/rischi:** cosa comporta questa scelta.
```

## Struttura del repository

```
TINAT/
├── README.md                  # descrizione del progetto
├── CLAUDE.md                  # questo file (contesto per l'assistente)
├── src/                       # codice (preparazione dati, training, inferenza, fusione)
├── tests/                     # pytest — test di base su src/
├── notebooks/                 # fine-tuning detector (01) e classificatore di stadio (02)
├── data/                      # dataset pubblico versionato (~5 MB)
├── models/                    # artefatti di risultato del training (pesi esclusi)
├── docs/                      # README di dominio, model card, decision-log
├── .github/                   # CI (GitHub Actions) + template di PR
├── requirements.txt
└── pytest.ini
```

Struttura minima e volutamente essenziale: si aggiungono cartelle solo quando contengono materiale
vero, non scheletri vuoti.

## Workflow Git

- **Branching:** GitHub Flow leggero — `main` sempre verde; ogni modifica su un branch a vita breve
  con prefisso di tipo (`feature/`, `fix/`, `docs/`, `ci/`, `build/`); merge su `main` via Pull
  Request con CI verde.
- **Commit messages:** in inglese, formato convenzionale (`feat:`, `fix:`, `docs:`, `test:`,
  `chore:`). Commit piccoli e frequenti.
- **`.gitignore`:** esclude dati pesanti, pesi dei modelli, `__pycache__/`, ambienti virtuali,
  `_private/`.
- **Angel fa i commit e i push da solo:** preparare i cambiamenti e dargli il comando; non eseguire
  `git commit`/`git push` senza richiesta esplicita.

## Come deve lavorare Claude Code in questo repo

1. **Prima di ogni modifica rilevante:** proporre la decisione ad Angel e, se approvata, aggiungere
   la voce al decision-log.
2. **Non inventare storia di progetto:** se manca un'informazione (date, motivazioni reali, feedback
   del supervisore), **chiedere ad Angel** invece di riempire con dettagli plausibili ma falsi.
3. **Output onesti:** se qualcosa non funziona end-to-end, documentarlo nella sezione "Rischi,
   assunzioni e limiti" del README, non nasconderlo.
4. **Mantenere le lingue:** docs in italiano, codice in inglese.
