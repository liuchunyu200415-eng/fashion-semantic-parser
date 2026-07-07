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

## Train Mask R-CNN Baseline

The first baseline for PRD 3.1.1 uses Detectron2 Mask R-CNN with a ResNet-50
FPN backbone. Detectron2 is intentionally treated as an optional cloud-GPU
dependency so local data tooling can still run without it.

Install PyTorch and Detectron2 in the AutoDL environment, then run:

```bash
python scripts/train_segmentation_baseline.py
```

The default training config is:

```text
configs/segmentation_mask_rcnn.yaml
```

Useful smoke-test override:

```bash
python scripts/train_segmentation_baseline.py --max-iter 20
```

The baseline registers the converted COCO files and trains against all eight
PRD categories. DeepFashion2 currently only provides high-quality examples for
top, pants, skirt, outerwear, and dress, so shoes, bags, and accessories should
be treated as data gaps until additional datasets are added.

## Predict One Image

After training, run single-image inference with the trained weights:

```bash
python scripts/predict_segmentation.py \
  data/raw/example.jpg \
  --weights outputs/segmentation/mask_rcnn_r50_fpn/model_final.pth \
  --output outputs/segmentation/example_prediction.json
```

Prediction JSON contains:

```text
image_path
instances[].category_id
instances[].category_label
instances[].confidence
instances[].box
instances[].mask
```

This matches the PRD 3.1.1 output contract: instance mask, bounding box, and
category label.
