# PRD 3.1.1 Garment Instance Segmentation

The current short-term goal is to complete the data and baseline path for PRD
3.1.1: garment instance segmentation.

## Requirement

Input:

```text
RGB product image
```

Output:

```text
instance mask, bounding box, category label
```

Target PRD categories:

```text
top, pants, skirt, outerwear, dress, shoes, bag, accessory
```

DeepFashion2 currently covers garment categories that map to:

```text
top, pants, skirt, outerwear, dress
```

Shoes, bags, and accessories require additional data sources before they can be
trained with the same quality.

## Convert DeepFashion2 To COCO

Smoke test:

```bash
python scripts/convert_deepfashion2_to_coco.py --split train --limit 10
```

Full conversion:

```bash
python scripts/convert_deepfashion2_to_coco.py --split train
python scripts/convert_deepfashion2_to_coco.py --split validation
```

Generated files are stored under:

```text
data/processed/autodl/segmentation
```

The COCO files use project-relative image paths so they can be reused across
local and AutoDL environments.
