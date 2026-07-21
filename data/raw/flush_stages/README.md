# flush_stages — Dati per il Workstream B (classificatore flush-stage)

Immagini di foglie di tè per stadio, usate per addestrare il classificatore
descritto nell'outline §5. **Le immagini NON sono versionate** (dataset pesante,
~2200 file): git tiene solo questa struttura e il README. Vanno ri-estratte in
locale seguendo le istruzioni sotto.

## Struttura (2 classi)

```
flush_stages/
├── new_flush/     # germogli teneri (foglia giovane)      → 1177 immagini
└── mature_leaf/   # foglia matura / non più adatta al tè  → 1031 immagini
```

Scelta a **2 classi** (step "coarse: active flush vs not" dell'outline §5.2),
ed è lo scope definitivo del pilot. Una terza classe `dormant` (nessun germoglio)
è **fuori scope**: è la fase con meno attività di parassiti e non esiste un
dataset pubblico adatto (vedi decision-log, 2026-07-21).

## Fonte dei dati

**TeaLeafAgeQuality — Age-Stratified Tea Leaf Quality Classification Dataset**
(Mendeley Data, licenza **CC BY 4.0**). Parte usata: `TeaLeafAgeQuality(Raw Data)`,
2208 immagini organizzate per età della foglia in 4 categorie.

## Mappatura categoria → classe

| Categoria originale (età) | Classe | N. |
|---|---|---|
| Catagory A (1-2 giorni) | `new_flush`   | 562  |
| Category B (2-4 giorni) | `new_flush`   | 615  |
| Category C (4-7 giorni) | `mature_leaf` | 508  |
| Category D (7+ giorni, "Not for Tea") | `mature_leaf` | 523 |

> Nota: il dataset è per **età/qualità** della foglia, un *proxy* dello stadio
> di flush. La mappatura giovane→`new_flush`, vecchia→`mature_leaf` è coerente
> con il primo passo dell'outline.

## Come ri-estrarre le immagini (chi clona il repo)

1. Scaricare lo zip `TeaLeafAgeQuality...Dataset.zip` da Mendeley Data.
2. Estrarre **solo** la parte `Raw Data`, applicando la mappatura sopra:
   A+B → `new_flush/`, C+D → `mature_leaf/`. Prefissare i file con la lettera
   della categoria (`A_`, `B_`, …) per evitare collisioni di nome.
3. Lo zip non va tenuto nel repo (è in `.gitignore`).

## Fonte alternativa (produzione)

Cattura diretta in campo — preferibile per il modello di produzione; non usata
in questo pilot per vincoli di tempo/accesso.
