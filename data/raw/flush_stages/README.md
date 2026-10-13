# flush_stages — Data for the flush-stage classifier

Tea-leaf images by stage, used to train the growth-stage (WHEN) classifier.
**The images are NOT versioned** (~4 GB, ~2200 files): git keeps only this
README. Rebuild them locally with the command below.

## Structure (2 classes)

```
flush_stages/
├── new_flush/     # tender shoots (young leaf)          → 1177 images
└── mature_leaf/   # mature leaf / no longer tea-worthy  → 1031 images
```

A **2-class** choice: active flush vs not. A third `dormant` class (no shoots)
is **out of scope**: it is the lowest-pest-activity stage and no suitable public
dataset exists.

## Data source

**TeaLeafAgeQuality — Age-Stratified Tea Leaf Quality Classification Dataset**
(Mendeley Data, **CC BY 4.0** licence). Part used: `TeaLeafAgeQuality(Raw Data)`,
2208 images organised by leaf age into 4 categories.

## Category → class mapping

| Original category (age) | Class | N. |
|---|---|---|
| Category A (1-2 days) | `new_flush`   | 562  |
| Category B (2-4 days) | `new_flush`   | 615  |
| Category C (4-7 days) | `mature_leaf` | 508  |
| Category D (7+ days, "Not for Tea") | `mature_leaf` | 523 |

> Note: the dataset is about leaf **age/quality**, a *proxy* of the flush stage.
> The mapping young→`new_flush`, old→`mature_leaf` is the coarsest useful cut.

## How to rebuild the images

1. Download `TeaLeafAgeQuality...Dataset.zip` from Mendeley Data and extract it
   anywhere outside the repository.
2. Run, pointing at the extracted `Raw Data` folder:

   ```bash
   python -m src.data flush "path/to/TeaLeafAgeQuality/Raw Data"
   ```

   It applies the mapping above (A+B → `new_flush/`, C+D → `mature_leaf/`) and
   prefixes each file with its category letter (`A_`, `B_`, ...) to avoid name
   collisions.
3. Train with `python -m src.train flush`.

The trained model itself comes with `python -m src.data download`, so this is
only needed to retrain or re-measure the classifier.
