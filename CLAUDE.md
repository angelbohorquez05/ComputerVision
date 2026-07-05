# CLAUDE.md — Progetto d'esame UFS14 (Processo e Sviluppo del Software)

> Questo file dà a Claude Code il contesto del progetto. Leggilo all'inizio di ogni sessione.

## Contesto

- **Corso:** UFS14 — Processo e Sviluppo del Software (DevOps-Agile), ITS Angelo Rizzoli, Milano.
- **Studente:** Angel (lavoro **individuale**).
- **Progetto:** **TINAT** — rilevamento di insetti (Helopeltis/Miridae) con YOLOv8. Versione "specchio" anonimizzata di un progetto di tirocinio (il nome reale del progetto e l'azienda NON vanno mai menzionati nel repo).
- **Consegna:** repo GitHub **personale e indipendente** (sul profilo di Angel), non il repo condiviso di classe. Scelta motivata: controllo totale sullo storico dei commit (che è l'artefatto da mostrare), nessuna dipendenza da PR/issue accettati da altri. Presentazione in classe dal computer di Angel; eventualmente la docente viene invitata come collaboratrice per rivedere lo storico e commentare come migliorarlo.
- **Priorità del corso:** alla docente interessa il **processo** (organizzazione, decisioni, tracciabilità, ciclo di vita ML, MLOps), NON la perfezione del codice o del modello. Il progetto può essere incompleto: l'importante è documentare onestamente dove si trova e perché.

## Regola d'oro del corso

> "Se una decisione è importante, deve essere rintracciabile in un documento o in un artefatto nel repository."

Quindi: **ogni decisione tecnica o organizzativa rilevante va registrata in `exam/decision-log.md`** con: (1) la decisione, (2) le alternative considerate, (3) perché è stata scelta questa e non un'altra.

## Lingue

- **Documentazione** (README, decision-log, docs/): **italiano**.
- **Codice, commenti, nomi di variabili, commit messages:** **inglese**.
- **Conversazione con Angel:** spagnolo (informale, va bene).

## Riservatezza e anonimizzazione (IMPORTANTE)

Questo è un **repo "specchio" nuovo, personale, per l'esame**, NON il repo del tirocinio:
- NON menzionare mai il nome reale del progetto né l'azienda in codice, docs, commit messages, nomi di file o commenti. Il progetto si chiama solo **TINAT**.
- NON copiare codice, dati, pesi di modelli o documenti proprietari dell'azienda.
- **Migrazione selettiva:** dal progetto locale entra solo ciò che è di Angel e/o pubblico (dataset da internet come IP102, codice scritto da Angel, struttura ridefinita da zero). Ogni "pezzo" migrato = un commit dedicato con messaggio descrittivo.
- Le decisioni possono essere descritte in modo generico ("il supervisore del tirocinio ha indicato…" senza nomi né dettagli riservati).
- Prima di ogni commit, controllare che non ci siano riferimenti al nome reale, all'azienda, a URL interni o a persone (fare un `grep` se serve).

## Struttura del repository

```
TINAT/
├── CLAUDE.md                  # questo file
├── README.md                  # segue il template della docente (vedi sotto)
├── exam/
│   └── decision-log.md        # log delle decisioni (formato sotto)
├── data/
│   ├── raw/                   # dati grezzi (NON versionare file pesanti; usare .gitignore)
│   └── processed/             # dati puliti/annotati
├── notebooks/                 # esplorazione (EDA, prove) — codice "sporco" ammesso
├── src/                       # codice "pulito" riutilizzabile (scripts, moduli)
├── tests/                     # pytest — almeno test di base su src/
├── models/                    # pesi/output (in .gitignore se pesanti)
├── docs/                      # documentazione extra (diagrammi, note di processo)
└── requirements.txt           # dipendenze con versioni pinnate
```

**Separazione notebooks/ vs src/:** i notebook sono il laboratorio, src/ è la fabbrica. Il codice che si stabilizza migra da notebooks/ a src/ (e questa migrazione è essa stessa una decisione da loggare).

## Template del README.md (obbligatorio, dalla docente)

Il README deve contenere queste sezioni, in italiano:

1. **Project name/idea**
2. **Setup / How to run this project** — comandi per avviare, requisiti di sistema (versione Python, librerie esterne). *Niente è scontato:* scrivere come per chi non ha mai visto il progetto.
3. **Spiegazione del progetto** — obiettivo, problema che risolve.
4. **Dati** — origine del dataset, perché quello, caratteristiche, gestione di dati mancanti/malformattati.
5. **Ciclo di vita ML** — come si applicano al progetto: raccolta dati, training, validazione, deploy, monitoring.
6. **MLOps** — cosa monitorare; criteri per il re-training (es. drift stagionale delle immagini, nuovi falsi negativi, soglia di mAP).
7. **Rischi, assunzioni e limiti** — limiti identificati, assunzioni fatte, stato onesto del progetto (funzionante end-to-end o no, difficoltà incontrate, come ampliarlo).
8. **Ulteriori informazioni**

## Formato del decision-log.md

Ogni voce così:

```markdown
## [YYYY-MM-DD] Titolo breve della decisione

**Contesto:** qual era il problema/bivio.
**Alternative considerate:** A, B, C — e perché sono state scartate.
**Decisione:** cosa è stato scelto.
**Motivazione:** perché questa e non le altre (costi, tempo, vincoli, dati disponibili).
**Conseguenze/rischi:** cosa comporta questa scelta.
```

Decisioni già note da registrare (chiedere dettagli ad Angel se mancano):
- Repo nuovo "specchio" anonimizzato, **personale e indipendente**, con nome generico "TINAT" (alternative scartate: usare il repo aziendale originale; contribuire al repo condiviso di classe). Motivo: riservatezza del tirocinio + controllo totale sullo storico dei commit; la demo avviene in classe dal computer di Angel, con eventuale invito alla docente come collaboratrice.
- Migrazione selettiva dal progetto locale (pezzo per pezzo, un commit per pezzo) invece di un clone completo. Motivo: controllo su cosa entra nel repo + storico di commit granulare come evidenza di processo.
- Scelta di YOLOv8 (vs altre architetture di detection).
- Partire da IP102 / classe Miridae come base per il fine-tuning verso Helopeltis.
- Separazione notebooks/ vs src/.
- Strategia di branching Git (vedi sotto).
- Lingue: docs IT / codice EN.

## Workflow Git (DevOps-Agile)

- **Branching:** `main` protetto; una branch per feature/task (`feature/...`, `docs/...`, `fix/...`); merge via commit descrittivi (o PR se Angel vuole simulare il flusso completo).
- **Commit messages:** in inglese, formato convenzionale: `feat: ...`, `fix: ...`, `docs: ...`, `test: ...`, `chore: ...`.
- **Commit piccoli e frequenti** — la storia dei commit è un artefatto del processo che la docente può guardare.
- **`.gitignore`:** escludere dati pesanti, pesi dei modelli, `__pycache__/`, ambienti virtuali.

## Testing

- Usare **pytest** (visto nel corso). Minimo: test sulle funzioni di `src/` (es. preprocessing, parsing delle annotazioni).
- I test girano con `pytest tests/` dalla root.
- Anche "abbiamo pochi test perché X" è una decisione da loggare — onestà > copertura finta.

## Come deve lavorare Claude Code in questo repo

1. **Prima di ogni modifica rilevante:** proporre la decisione ad Angel e, se approvata, aggiungere la voce al decision-log.
2. **Non inventare storia di progetto:** se manca un'informazione (date, motivazioni reali, feedback dei supervisori), **chiedere ad Angel** invece di riempire con dettagli plausibili ma falsi.
3. **Priorità dei task:** documentazione e struttura > codice funzionante > performance del modello.
4. **Output onesti:** se qualcosa non funziona end-to-end, documentarlo nella sezione "Rischi, assunzioni e limiti", non nasconderlo.
5. **Mantenere le lingue:** docs in italiano, codice in inglese.

## Backlog iniziale (ordine suggerito)

- [x] Inizializzare repo personale "TINAT" con struttura di cartelle + .gitignore + requirements.txt
- [ ] README.md con il template (sezioni vuote o bozza)
- [ ] exam/decision-log.md con le prime 3–4 voci (decisioni già prese: repo specchio personale, migrazione selettiva, YOLOv8, IP102/Miridae)
- [ ] Migrazione selettiva: portare dal progetto locale i primi file (solo materiale di Angel/pubblico), un commit per pezzo, verificando l'anonimizzazione
- [ ] Script minimo in src/ (es. download/preparazione dati IP102 classe Miridae)
- [ ] Notebook EDA di base sul dataset
- [ ] 2–3 test pytest sulle funzioni di src/
- [ ] Sezioni Ciclo di vita ML + MLOps del README (testo, non codice)
- [ ] Sezione Rischi/assunzioni/limiti — stato onesto del progetto
