# flush_stages — Training Data for Workstream B

Place tea plant images here, one subfolder per class:

```
flush_stages/
├── dormant/        # No visible new shoots (Months 0-2)
├── new_flush/      # Bright green, 2-3 tender leaves + bud (Months 3-4) ⚠️ HIGH RISK
└── mature_leaf/    # Dark, fully expanded leaves (Months 5+)
```

## Target counts
- ≥150 images per class before training (see NEXT_STEPS_WORKSTREAM_B.md)

## Data sources (fastest to slowest)
1. **Mendeley** — TeaLeafAgeQuality dataset (2,208 images, CC BY 4.0)
   Map: T1+T2 → new_flush/, T3+T4 → mature_leaf/
   Add dormant from iNaturalist searches.
2. **Partner farm (Peru)** — preferred for production; contact the farm contact for access.
