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

## Audit And Convert Fashionpedia

Fashionpedia supplements the five DeepFashion2-backed categories with instance
masks for shoes, bags, and accessories. Keep the official layout under the
configured `data/raw/fashionpedia` root:

```text
data/raw/fashionpedia/
├── annotations/
│   ├── instances_attributes_train2020.json
│   └── instances_attributes_val2020.json
├── train/
└── test/
```

The official training archive extracts to `train/`. The combined validation and
test archive extracts all 3,200 images to `test/`; the validation annotation
file selects its labelled 1,158-image subset by filename. The converter also
accepts `train2020/`, `val/`, and `val2020/` directory names for previously
organized copies.

The official JSON contains 27 main-apparel classes and 19 overlapping garment
part classes. The project maps 26 unambiguous main classes into the eight PRD
categories, excludes garment parts, and drops every image containing a
`jumpsuit`. Dropping the complete image prevents an excluded whole-body garment
from becoming unlabeled background. The initial mapping treats cardigans,
jackets, coats, and capes as outerwear; shoes and bags remain dedicated classes;
and wearable items such as hats, glasses, belts, watches, and scarves map to
accessory.

Audit annotation mapping before downloading or extracting images:

```bash
python scripts/convert_fashionpedia_to_coco.py \
  --split validation \
  --audit-only
```

The audit reports source and output image/annotation counts, every source-class
count, mapped PRD counts, missing images, excluded parts, ambiguous images, and
invalid records. It writes no training file and does not require image files.

After the official image archive is extracted, run a deterministic smoke test:

```bash
python scripts/convert_fashionpedia_to_coco.py \
  --split validation \
  --limit 10
```

Then convert both complete labelled splits:

```bash
python scripts/convert_fashionpedia_to_coco.py --split train
python scripts/convert_fashionpedia_to_coco.py --split validation
```

The converter preserves official polygon or RLE masks and mask area, remaps
category IDs to the existing PRD `1..8` contract, and writes project-relative
image paths. Conversion fails when selected image files are missing; use
`--audit-only` for annotation-only inspection.

These Fashionpedia outputs are intentionally independent from the current
DeepFashion2 training JSON. Do not point the production config at them yet. The
next data stage must build a mixed-data training policy with source balancing
and keep both source validation sets separate. Fashionpedia annotations use CC
BY 4.0, while each image remains subject to its original source license; retain
attribution and review image rights before non-research deployment.

## Establish The Fashionpedia Transfer Baseline

Before fine-tuning, evaluate the selected DeepFashion2-backed checkpoint on the
complete Fashionpedia validation set. This records the cross-dataset baseline,
including the previously untrained shoes, bag, and accessory classes. Formal
COCO evaluation keeps the score threshold at zero. `--metrics-output` writes a
clean JSON file even when Detectron2 emits verbose logs:

```bash
python scripts/evaluate_segmentation_baseline.py \
  --config configs/segmentation_mask2former_fashionpedia.yaml \
  --weights outputs/segmentation/mask2former_r50_official_stage2/model_official_0004999.pth \
  --output-dir outputs/segmentation/fashionpedia/zero_shot \
  --metrics-output outputs/segmentation/fashionpedia/zero_shot/metrics.json
```

The experimental Fashionpedia config starts a new optimizer schedule from the
validated checkpoint. It uses all eight PRD classes, batch size four, a
conservative `1e-5` transfer learning rate, and a 10,000-iteration target. Run a
1,000-iteration stability stage first:

```bash
python scripts/train_segmentation_baseline.py \
  --config configs/segmentation_mask2former_fashionpedia.yaml \
  --max-iter 1000
```

Evaluate that checkpoint on Fashionpedia and DeepFashion2 separately before
resuming. If mask losses and both validation suites remain acceptable, resume
the same output directory to the next planned checkpoint rather than starting a
new optimizer:

```bash
python scripts/train_segmentation_baseline.py \
  --config configs/segmentation_mask2former_fashionpedia.yaml \
  --resume \
  --max-iter 5000
```

This Fashionpedia-only stage is not the final production model. Its purpose is
to activate and measure all eight classes; a later mixed-data consolidation
stage must recover or preserve DeepFashion2 performance before deployment.

## Check AutoDL Environment

After pulling the latest code on AutoDL, check whether the segmentation training
environment is ready:

```bash
python scripts/check_segmentation_env.py
```

The report checks:

```text
PyTorch and CUDA availability
active GPU compute capability and memory
OpenCV
Detectron2 CUDA architecture compatibility
Mask2Former
DeepFashion2 and Fashionpedia converted COCO files
Fashionpedia image directories and broken data-volume symlinks
project config files
```

When moving between GPU generations, update `TORCH_CUDA_ARCH_LIST` and rerun
this check before training. A cloned system disk may retain CUDA extensions for
the previous GPU while project symlinks point to a data disk that was not
migrated.

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

`AP85` is not the mean mask IoU. The `segm` result now also reports direct mask
IoU statistics. Predictions and ground truth are matched one-to-one within the
same image and category, in descending IoU order. A pair must have IoU at least
`0.50` to count as a valid match. All values below are percentages on a `0-100`
scale:

```text
MatchedMeanIoU       mean IoU of valid matched pairs
MatchedMedianIoU     median IoU of valid matched pairs
AllGTMeanIoU         mean IoU with every unmatched ground-truth item counted as 0
Precision50          valid matches / retained predictions
Recall50             valid matches / ground-truth instances
MatchedIoU85Rate     fraction of matched pairs whose IoU is at least 0.85
AllGTIoU85Rate       fraction of all ground-truth items matched at IoU at least 0.85
```

Per-category forms such as `MatchedMeanIoU-top` and `AllGTIoU85Rate-dress` are
included. Aggregate values only include categories that have ground-truth
coverage, so the current unlabelled shoes, bag, and accessory classes do not
become false positives in the five-class DeepFashion2 report. Use
`MatchedMeanIoU` to describe mask-boundary quality, and always report `Recall50`
or `AllGTIoU85Rate` beside it so missed garments are visible. Run COCO AP at
score threshold `0.0`; run a second direct-IoU report at the selected deployment
threshold. The validated five-class deployment profile uses `0.8`:

```bash
python scripts/evaluate_segmentation_baseline.py \
  --config configs/segmentation_mask2former.yaml \
  --weights outputs/segmentation/mask2former_r50_official_stage2/model_official_0004999.pth \
  --score-threshold 0.8
```

If full inference has already completed at score threshold `0.0`, reuse the COCO
prediction file instead of running the GPU model again. First locate the file:

```bash
find outputs/segmentation \
  -path "*/inference/coco_instances_results.json" \
  -print
```

Then calculate direct IoU at the deployment threshold entirely offline:

```bash
python scripts/evaluate_segmentation_predictions.py \
  --val-json data/processed/autodl/segmentation/deepfashion2_validation.json \
  --predictions outputs/segmentation/mask2former_r50_official_stage2/inference/coco_instances_results.json \
  --score-threshold 0.1 \
  --output outputs/segmentation/mask2former_r50_official_stage2/direct_mask_iou.json
```

This path loads neither Mask2Former weights nor images and does not require a
GPU. It filters the saved predictions by confidence and rebuilds only the COCO
mask-IoU matrix and one-to-one match statistics.

## Benchmark Single-Image Latency

Measure the PRD latency target with a loaded model and a deterministic sample of
validation images:

```bash
python scripts/benchmark_segmentation_latency.py \
  --config configs/segmentation_mask2former.yaml \
  --weights outputs/segmentation/mask2former_r50_official_stage2/model_official_0004999.pth \
  --val-json data/processed/autodl/segmentation/deepfashion2_validation.json \
  --image-limit 20 \
  --warmup-runs 10 \
  --runs 100 \
  --score-threshold 0.1 \
  --output outputs/segmentation/mask2former_r50_official_stage2/latency.json
```

The benchmark loads weights once, decodes source images before timing, warms up
CUDA, and synchronizes the GPU around each sample. `predictor_ms` includes
Detectron2 preprocessing, Mask2Former inference, and Detectron2 model
postprocessing. `pipeline_ms` additionally includes transfer to CPU, score
filtering, mask-to-polygon conversion, mask-derived boxes, and project response
construction. Report median and p95; compare `pipeline_ms.p95` with the PRD
single-image latency target for the strict end-to-end interpretation.

The output path applies the confidence threshold to Detectron2 instances before
copying masks to CPU. Retained masks stay as dense NumPy arrays during OpenCV
contour extraction; converting every pixel of all 100 Mask2Former queries into
nested Python lists is intentionally avoided.

Use `--precision fp16` to benchmark CUDA autocast on GPUs with Tensor Cores.
FP32 remains the default, and the output report records the selected precision
so latency artifacts remain comparable. Run the same image sample, warmup count,
measurement count, score threshold, and weights when comparing FP32 with FP16.

If precision alone does not meet the latency target, benchmark a lower inference
resolution without retraining the model. `--min-size-test` sets the resized short
edge and `--max-size-test` caps the long edge. The report records both effective
values under `input_size`; omitting them preserves the values inherited from the
Mask2Former config.

Start with `640/1067` and run a short smoke test before the formal benchmark:

```bash
python scripts/benchmark_segmentation_latency.py \
  --config configs/segmentation_mask2former.yaml \
  --weights outputs/segmentation/mask2former_r50_official_stage2/model_official_0004999.pth \
  --val-json data/processed/autodl/segmentation/deepfashion2_validation_full.json \
  --image-limit 5 \
  --warmup-runs 5 \
  --runs 20 \
  --precision fp16 \
  --min-size-test 640 \
  --max-size-test 1067 \
  --score-threshold 0.8 \
  --output outputs/segmentation/eval_official_05000_full/latency_640_fp16_smoke.json
```

Reducing resolution changes the model input and can reduce mask-boundary quality.
After selecting a latency candidate, run `evaluate_segmentation_baseline.py` with
the same `--min-size-test` and `--max-size-test` values on the full validation set
before treating it as the deployment configuration.

When latency is benchmarked with `--precision fp16`, pass the same precision to
evaluation, prediction, and visualization. This keeps reported mask quality on
the exact autocast inference path used by the deployment candidate instead of
silently evaluating FP32 outputs.

## Validated Deployment Profile

The current five-class DeepFashion2-backed deployment profile is stored in
`configs/segmentation_mask2former_deployment.yaml`. It fixes the selected
checkpoint and the exact measured inference settings:

```text
weights             model_official_0004999.pth
precision           fp16
minimum test size   384
maximum test size   640
score threshold     0.8
```

On an NVIDIA GeForce RTX 4080 SUPER, using 100 source images, 20 warmup runs,
and 200 measured runs, the loaded-model pipeline produced mean `49.82 ms`,
median `47.38 ms`, p95 `57.17 ms`, and `20.07 FPS`. Model/weight loading and
image decoding were excluded. The mean meets the 50 ms target, but p95 does not;
the result is hardware- and protocol-specific rather than a universal latency
guarantee.

Full-validation FP16 accuracy at this resolution was mask AP `57.09`, AP50
`75.36`, AP75 `65.92`, AP85 `47.95`, and AP90 `28.48`. At deployment threshold
`0.8`, direct matched mask IoU was `88.41%`, precision was `77.94%`, recall was
`76.62%`, and F1 was `77.28%`. Unmatched ground-truth masks count as zero in
AllGTMeanIoU, which was `67.74%`.

Threshold `0.75` retains more skirt and outerwear predictions with essentially
the same F1, but its measured full-pipeline mean was `51.20 ms`; keep it as a
higher-recall experiment rather than the latency-compliant default.

Run the recorded deployment profile without repeating its flags:

```bash
python scripts/predict_segmentation.py \
  data/raw/example.jpg \
  --config configs/segmentation_mask2former_deployment.yaml \
  --output outputs/segmentation/example_deployment_prediction.json
```

The validated accuracy currently covers top, pants, skirt, outerwear, and dress.
DeepFashion2 has no ground truth for shoes, bag, or accessory, so the full
eight-class PRD contract still requires additional labelled data. Outerwear is
also the weakest covered category and remains a priority for data and model
improvement.

### Experimental category-conflict filter

Some images receive one large `dress` prediction over a stronger `top` plus
`pants/skirt` pair. Do not raise the global score threshold to hide this case,
because the correct lower garment may have a similar score. The experimental
post-processor only suppresses a dress when its box closely matches the union
of a vertically ordered top and lower garment, both components are covered by
the dress box, and their average score is at least as high as the dress score.

Generate a filtered prediction file without changing API defaults:

```bash
python scripts/postprocess_segmentation_predictions.py \
  --predictions outputs/segmentation/eval_official_05000_384_fp16_full/inference/coco_instances_results.json \
  --score-threshold 0.8 \
  --min-union-iou 0.8 \
  --min-component-coverage 0.8 \
  --score-margin 0.0 \
  --output outputs/segmentation/eval_official_05000_384_fp16_full/predictions_score_080_conflict_filtered.json \
  --report outputs/segmentation/eval_official_05000_384_fp16_full/conflict_filter_score_080_report.json
```

Evaluate the generated file on the complete validation set before enabling the
policy in runtime inference. It remains experimental unless aggregate F1
improves without a material dress or lower-garment recall regression.

The complete validation-set sweep at the `0.8` deployment threshold rejected
this policy for runtime use:

| Score margin | Dresses removed | P50 | R50 | F1 | GT at IoU 0.85 | Dress R50 | Dress GT at IoU 0.85 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Disabled | 0 | 77.94 | 76.62 | 77.28 | 58.94 | 74.01 | 60.81 |
| 0.02 | 529 | 78.42 | 76.30 | 77.34 | 58.71 | 72.16 | 59.48 |
| 0.05 | 337 | 78.26 | 76.43 | 77.34 | 58.81 | 72.94 | 60.09 |
| 0.10 | 90 | 78.03 | 76.57 | 77.29 | 58.91 | 73.72 | 60.65 |

Margins `0.02` and `0.05` traded a negligible aggregate F1 increase for a
material dress-recall regression. Margin `0.10` reduced that regression, but
its `0.01` F1 gain was not meaningful enough to justify runtime complexity or
false suppression risk. Keep the API on the unfiltered predictions; retain
this command only as a reproducible diagnostic experiment.

## Visual Acceptance Review

Do not use early low-threshold debug overlays as final visual evidence. Build
the acceptance set from the exact saved predictions used by the full-validation
metrics. The command below creates `Original | Ground truth | Prediction`
comparisons, two deterministic samples per labelled category, one explicit
category-level miss per category when available, a contact sheet, and a JSON
manifest:

```bash
python scripts/visualize_segmentation_acceptance.py \
  --val-json data/processed/autodl/segmentation/deepfashion2_validation_full.json \
  --predictions outputs/segmentation/eval_official_05000_384_fp16_full/inference/coco_instances_results.json \
  --image-root . \
  --score-threshold 0.8 \
  --samples-per-category 2 \
  --misses-per-category 1 \
  --output-dir outputs/segmentation/eval_official_05000_384_fp16_full/acceptance_visuals
```

Review `acceptance_visuals/contact_sheet.jpg` first, then inspect the individual
PNG files at full resolution. Each comparison and the manifest record why the
image was selected, using labels such as `sample:top` and `miss:outerwear`, so
the review set is repeatable and does not hide known failure cases.

## FastAPI Inference Service

The application uses `configs/app.yaml` to select the validated deployment
profile. The Detectron2/Mask2Former predictor is loaded on the first valid
request and then reused, so model and weight loading are not repeated for every
image.

On AutoDL, start the service from the project root:

```bash
export OMP_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST="8.9"
export PYTHONPATH=$PWD/src:$PWD/external/Mask2Former:$PYTHONPATH

python -m uvicorn fashion_semantic_parser.api.app:app \
  --host 0.0.0.0 \
  --port 8000
```

Call the dedicated PRD 3.1.1 endpoint with a project-relative image path:

```bash
curl -X POST http://127.0.0.1:8000/v1/segment \
  -H 'Content-Type: application/json' \
  -d '{"image_path":"data/raw/example.jpg"}'
```

The response uses `SegmentationPrediction` and includes `category_id`,
`category_label`, `confidence`, an `xyxy` box, and polygon mask coordinates for
every retained instance. An optional manual subject ROI remains available:

```json
{
  "image_path": "data/raw/example.jpg",
  "subject_roi": {
    "x_min": 100,
    "y_min": 20,
    "x_max": 700,
    "y_max": 980
  }
}
```

`POST /v1/query` invokes the same segmentation runtime and returns both compact
`regions` and the complete `segmentation` object. Its answer text only reports
that PRD 3.1.1 segmentation completed; it does not claim that language-guided
grounding, attribute extraction, RAG, or multimodal answer generation is ready.

For security and reproducibility, API image paths must stay inside the project
checkout and must be relative. Invalid or missing paths return HTTP 400.
Missing model dependencies, invalid runtime configuration, or unavailable CUDA
return HTTP 503. The first request includes model-loading latency; use the
benchmark script, not first-request wall time, when comparing the validated
loaded-model latency profile.
