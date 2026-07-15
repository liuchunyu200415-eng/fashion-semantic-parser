# PRD 3.1.1 Garment Instance Segmentation

The current short-term goal is to complete the data, baseline, and PRD-aligned
model path for PRD 3.1.1: garment instance segmentation.

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

## Check AutoDL Environment

After pulling the latest code on AutoDL, check whether the segmentation training
environment is ready:

```bash
python scripts/check_segmentation_env.py
```

The report checks:

```text
PyTorch and CUDA availability
OpenCV
Detectron2
Mask2Former
converted COCO files
project config files
```

If Detectron2 or Mask2Former is missing, install those dependencies before
running training.

## Train Mask R-CNN Baseline

The first engineering baseline uses Detectron2 Mask R-CNN with a ResNet-50 FPN
backbone. This baseline is kept because it is stable, COCO-compatible, and good
for validating data conversion, class mapping, and the train/inference loop.

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

## Train Mask2Former Target Model

Mask2Former is the PRD-aligned target model for 3.1.1 because the PRD technical
stack explicitly includes Mask2Former and the task requires instance masks plus
category labels.

Recommended AutoDL setup:

```bash
cd /root/fashion-semantic-parser
mkdir -p external
git clone https://github.com/facebookresearch/Mask2Former.git external/Mask2Former
export PYTHONPATH=$PWD/external/Mask2Former:$PYTHONPATH
```

After Detectron2 and Mask2Former dependencies are available, run a smoke test:

```bash
python scripts/train_segmentation_baseline.py \
  --config configs/segmentation_mask2former.yaml \
  --max-iter 20
```

The project Mask2Former config loads the official COCO instance segmentation
R50 50ep pretrained checkpoint by default:

```text
https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl
```

This checkpoint is required for useful short fine-tuning runs. If the training
log says:

```text
[DetectionCheckpointer] Loading from  ...
```

then no pretrained weights were loaded and a short run is expected to produce
near-zero AP.

The Mask2Former config file is:

```text
configs/segmentation_mask2former.yaml
```

Mask2Former should become the main 3.1.1 model after the baseline train loop is
verified. Mask R-CNN remains a debugging baseline.

Mask2Former is a mask-first architecture, so the project derives invalid or
empty predicted boxes from the predicted masks before returning the PRD output
schema and before COCO validation serializes predictions. This keeps the
required `mask + bounding box + category label` output consistent even when the
upstream model head does not emit meaningful boxes directly.

The training path registers the converted COCO files and trains against all
eight PRD categories. DeepFashion2 currently only provides high-quality
examples for top, pants, skirt, outerwear, and dress, so shoes, bags, and
accessories should be treated as data gaps until additional datasets are added.

For a fast experiment, keep the full training file but limit validation:

```bash
python scripts/convert_deepfashion2_to_coco.py --split validation --limit 500
python scripts/train_segmentation_baseline.py \
  --config configs/segmentation_mask2former.yaml \
  --max-iter 500

python scripts/evaluate_segmentation_baseline.py \
  --config configs/segmentation_mask2former.yaml \
  --weights outputs/segmentation/mask2former_r50/model_final.pth \
  --score-threshold 0.0
```

Full validation currently contains tens of thousands of images and can take
more than an hour, so use it only for formal reporting.

## Staged Mask2Former Training

Before a formal run, regenerate the full training COCO file. The smoke-test
command with `--limit 10` writes to the same default path and must not remain as
the trainer input:

```bash
python scripts/convert_deepfashion2_to_coco.py --split train
python scripts/convert_deepfashion2_to_coco.py --split validation --limit 500
python scripts/check_segmentation_env.py
```

Check the `datasets.train.image_count` field before training. A full
DeepFashion2 training conversion should contain far more than 10 images. The
500-image validation subset keeps iteration experiments practical; regenerate
the full validation file only for formal final metrics.

The target config now defines a 20,000-iteration first training stage and saves
a checkpoint every 1,000 iterations. It skips the slow final validation so
training and evaluation can be run independently:

```bash
python scripts/train_segmentation_baseline.py \
  --config configs/segmentation_mask2former.yaml
```

To continue an interrupted run in the same output directory, preserve
`last_checkpoint` and pass `--resume`. `max_iter` is the final target iteration,
not the number of additional iterations:

```bash
python scripts/train_segmentation_baseline.py \
  --config configs/segmentation_mask2former.yaml \
  --resume \
  --max-iter 20000
```

Do not resume a checkpoint that was trained from a different or 10-image COCO
file. Start a clean output directory in that case. After the stage finishes,
run `evaluate_segmentation_baseline.py` against `model_final.pth`, then decide
whether to resume to 50,000 iterations from the metric trend and visual masks.

The Mask2Former trainer uses the optimizer path from the official project:
AdamW parameter groups, a `0.1` backbone learning-rate multiplier, special
weight decay for normalization and embedding parameters, DeepLab learning-rate
scheduling, and full-model gradient clipping. Detectron2's generic default
trainer uses SGD in version 0.6 and must not be used for this model family.

The single-GPU target config starts with batch size 4 and learning rate
`2.5e-5`. Before a long run on a new GPU, use a 1,000-iteration stability stage:

```bash
python scripts/train_segmentation_baseline.py \
  --config configs/segmentation_mask2former.yaml \
  --output-dir outputs/segmentation/mask2former_r50_official_stage2 \
  --max-iter 1000
```

If memory and mask losses are stable, resume the same directory with a larger
final `--max-iter`. Do not resume checkpoints produced by the former generic
Detectron2 optimizer because their optimizer and scheduler states are not
compatible with the corrected Mask2Former trainer.

## Predict One Image

After training, run single-image inference with the trained weights:

```bash
python scripts/predict_segmentation.py \
  data/raw/example.jpg \
  --config configs/segmentation_mask2former.yaml \
  --weights outputs/segmentation/mask2former_r50/model_final.pth \
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

## Visualize One Prediction

To inspect visible Mask2Former results for reporting or debugging, save a
mask-overlay image:

```bash
python scripts/visualize_segmentation_prediction.py \
  data/raw/example.jpg \
  --config configs/segmentation_mask2former.yaml \
  --weights outputs/segmentation/mask2former_r50/model_final.pth \
  --score-threshold 0.1 \
  --output outputs/segmentation/visualizations/example.png \
  --json-output outputs/segmentation/visualizations/example.json
```

The visualization draws translucent predicted masks, mask-derived boxes, class
labels, and confidence scores.

If a validation image contains a full scene, provide a subject/person ROI to
focus diagnosis on the model region:

```bash
python scripts/visualize_segmentation_prediction.py \
  data/raw/deepfashion2/validation/image/000001.jpg \
  --config configs/segmentation_mask2former.yaml \
  --weights outputs/segmentation/mask2former_r50/model_final.pth \
  --score-threshold 0.05 \
  --subject-roi 170,80,330,520 \
  --output outputs/segmentation/visualizations/val_000001_roi.png \
  --json-output outputs/segmentation/visualizations/val_000001_roi.json
```

The ROI format is `x_min,y_min,x_max,y_max` in image pixel coordinates. This is
a manual stand-in for the planned person/subject detector stage.

## Evaluate Existing Weights

To compare mask quality without retraining, evaluate a saved checkpoint at
different score thresholds:

```bash
python scripts/evaluate_segmentation_baseline.py \
  --config configs/segmentation_mask2former.yaml \
  --weights outputs/segmentation/mask2former_r50/model_final.pth \
  --score-threshold 0.0
```

Use a threshold of `0.0` for formal COCO AP because AP itself ranks predictions
by confidence. Higher thresholds are useful for visualization and deployment,
but can hide true positives and understate model quality during evaluation. For
PRD 3.1.1 diagnosis, prioritize `segm` metrics and visible mask overlays.
Bounding boxes are derived from the predicted mask region, so they are a
secondary output once mask quality is acceptable.

The evaluator also reports `AP85` and `AP90`, including per-category values.
For `segm`, these metrics use predicted-mask IoU and directly show precision at
the PRD-relevant IoU 0.85 boundary. They are stricter than standard `AP75` and
should be used alongside the overall COCO AP and visual mask inspection.
