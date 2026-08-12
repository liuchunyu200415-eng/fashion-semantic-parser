# PRD 3.1.2 Language-Guided Region Localization

This module localizes the fashion region described by a natural-language query.
The PRD contract is:

- input: RGB product image plus a query such as `这件衣服的领口`
- output: target-region mask and bounding box
- example regions: collar, cuff, hem, pocket, shoulder, waist, pattern, and
  decoration
- acceptance targets: localization accuracy at least `92%` and localization
  latency at most `30 ms`

## Current Status

The first executable engineering slice is implemented:

- a separate Fashionpedia local-part COCO conversion path
- 19 directly annotated part categories with Chinese and English prompt terms
- explicit PRD coverage reporting instead of relabeling weak proxies
- typed request and response models for local-region masks and boxes
- Chinese and English query normalization for the Grounding DINO text encoder
- automatic person ROI cropping with the accepted `0.35` context margin
- Grounding DINO candidate boxes followed by batched SAM-HQ mask refinement
- output boxes derived from final masks instead of copied detector boxes
- a lazy, reusable `POST /v1/localize` runtime
- saved-result visualization against query-aligned Fashionpedia ground truth
- direct one-to-one mask IoU diagnostics for full-image and subject-ROI modes

The external models and checkpoints have now passed the AutoDL readiness check,
and the first real-image full-image request completed end to end. It returned
four collar candidates in `7.06 s` including cold model loading. Their
scores were only `0.265` to `0.296`, and large false-positive candidates were
present, so this proves runtime integration but not localization quality. The
PRD `92%` accuracy and `30 ms` latency targets remain unverified. Existing
`/v1/segment` and `/v1/query` behavior is unchanged.

## Annotation Coverage

Fashionpedia contains direct masks for the following 19 categories:

```text
hood, collar, lapel, epaulette, sleeve, pocket, neckline,
buckle, zipper, applique, bead, bow, flower, fringe, ribbon,
rivet, ruffle, sequin, tassel
```

Coverage against the example regions in PRD 3.1.2 is:

| PRD region | Coverage | Available supervision |
| --- | --- | --- |
| collar | exact | collar, lapel, neckline |
| cuff | missing | sleeve covers the full sleeve, not the cuff |
| hem | missing | no direct mask |
| pocket | exact | pocket |
| shoulder | partial | epaulette only |
| waist | missing | belt is an object, not a waist-region mask |
| pattern | missing | pattern attributes exist, but not general pattern masks |
| decoration | exact | closures and 10 decoration categories |

The converter preserves Fashionpedia `attribute_ids` for later PRD 3.1.3 work,
but it does not use attribute labels as region masks.

## Audit On AutoDL

Pull the implementation and audit the official annotations without requiring
image files:

```bash
cd /root/fashion-semantic-parser
git pull

export PYTHONPATH=$PWD/src:$PYTHONPATH

python scripts/convert_fashionpedia_parts_to_coco.py \
  --split validation \
  --audit-only

python scripts/convert_fashionpedia_parts_to_coco.py \
  --split train \
  --audit-only
```

The official validation audit reports `4,093` valid part masks across `1,150`
images. This is `77` more than the old PRD 3.1.1 exclusion count because the
garment converter dropped entire images containing ambiguous jumpsuits before
counting their parts. The localization converter keeps valid local-part masks
from those images because main-garment ambiguity does not invalidate the part
annotation.

The official training audit reports `170,341` selected part annotations across
`44,898` part-containing images. Of those annotations, `170,332` have valid
masks and boxes; the converter safely skips the remaining `9`, and all required
image files are present. The old `167,406` segmentation exclusion count was a
lower bound produced by the PRD 3.1.1 image-filtering order, not the complete
PRD 3.1.2 part count.

Run a 10-image conversion smoke test before creating full outputs:

```bash
python scripts/convert_fashionpedia_parts_to_coco.py \
  --split validation \
  --limit 10 \
  --output data/processed/autodl/localization/fashionpedia_parts_validation_smoke10.json

test -s \
  data/processed/autodl/localization/fashionpedia_parts_validation_smoke10.json \
  && echo "Localization smoke conversion: complete"
```

Then create the full train and validation COCO files:

```bash
python scripts/convert_fashionpedia_parts_to_coco.py --split train
python scripts/convert_fashionpedia_parts_to_coco.py --split validation

ls -lh data/processed/autodl/localization/fashionpedia_parts_*.json
```

The completed official conversion contains:

| Split | Images | Valid masks | Invalid masks | Missing images | File size |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 44,898 | 170,332 | 9 | 0 | 212.0 MB |
| validation | 1,150 | 4,093 | 0 | 0 | 5.2 MB |

These outputs are independent from
`data/processed/autodl/segmentation/fashionpedia_*.json`; converting local parts
does not change the accepted PRD 3.1.1 training data.

## Model Setup On AutoDL

The first accuracy baseline uses the official Grounding DINO Swin-T checkpoint
and SAM-HQ ViT-B. ViT-B is selected before ViT-H because it is a more practical
starting point on the 24 GB RTX 3090 while retaining the high-quality mask
decoder. `hq_token_only` remains `false`, matching the SAM-HQ recommendation
for quantitative evaluation and images that can contain multiple objects.

Install the official runtime and download the two checkpoints:

```bash
cd /root/fashion-semantic-parser

export OMP_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_HOME=/usr/local/cuda
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PWD/src:$PWD/external/Mask2Former:$PYTHONPATH

mkdir -p external models/checkpoints/localization

test -d external/GroundingDINO/.git || \
  git clone https://github.com/IDEA-Research/GroundingDINO.git \
  external/GroundingDINO

python -m pip install \
  "transformers==4.35.2" \
  segment-anything-hq \
  gdown

python -m pip install --no-build-isolation -e external/GroundingDINO

test -s models/checkpoints/localization/groundingdino_swint_ogc.pth || \
  curl -L --fail --retry 3 \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
    -o models/checkpoints/localization/groundingdino_swint_ogc.pth

SAM_HQ_WEIGHTS=models/checkpoints/localization/sam_hq_vit_b.pth
if [ ! -s "$SAM_HQ_WEIGHTS" ]; then
  rm -f "${SAM_HQ_WEIGHTS}.part"
  python -m gdown --fuzzy \
    "https://drive.google.com/file/d/11yExZLOve38kRZPfRx_MRxfIAKmfMY47/view?usp=sharing" \
    -O "${SAM_HQ_WEIGHTS}.part"
  echo \
    "14a9d662cd6f5a9c2dba6d40ab0058d88d287e4a18fd6fdc6ad5fb1a3fdeaa57  ${SAM_HQ_WEIGHTS}.part" \
    | sha256sum -c -
  mv "${SAM_HQ_WEIGHTS}.part" "$SAM_HQ_WEIGHTS"
fi

python - <<'PY'
from transformers import AutoTokenizer, BertModel

print("状态：正在缓存 Grounding DINO 的 bert-base-uncased 文本编码器...")
AutoTokenizer.from_pretrained("bert-base-uncased")
BertModel.from_pretrained("bert-base-uncased")
print("状态：bert-base-uncased 已缓存")
PY
```

Grounding DINO does not constrain its `transformers` dependency. Pinning
`4.35.2` prevents newer releases from disabling model support for the
project's PyTorch `2.1.2` environment. The mirror endpoint is also exported
before pre-caching `bert-base-uncased`, which Grounding DINO otherwise tries to
download during the first request. The SAM-HQ ViT-B file is downloaded from
the official Google Drive link and accepted only after its published SHA256
checksum passes.

Run the readiness check before starting inference:

```bash
python scripts/check_localization_env.py
```

A ready instance reports an empty `recommendations` list. The checker verifies
the converted train/validation COCO files, CUDA, a PyTorch-compatible
Transformers BERT implementation, both official imports, model configuration,
and both checkpoints.

## First Real-Image Smoke Test

Select one validation image whose ground truth contains a direct `collar`
annotation:

```bash
IMAGE_PATH="$(python - <<'PY'
import json

path = "data/processed/autodl/localization/fashionpedia_parts_validation.json"
data = json.load(open(path))
collar_id = next(
    category["id"]
    for category in data["categories"]
    if category["name"] == "collar"
)
image_id = next(
    annotation["image_id"]
    for annotation in data["annotations"]
    if annotation["category_id"] == collar_id
)
image = next(row for row in data["images"] if row["id"] == image_id)
print(image["file_name"])
PY
)"

python scripts/predict_localization.py \
  --image "$IMAGE_PATH" \
  --query "这件衣服的衣领" \
  --full-image \
  --output outputs/localization/grounded_sam_hq_smoke/collar.json
```

The command prints `状态：正在加载模型并执行语言引导区域定位...` while the
first request loads both checkpoints, then prints the result count and elapsed
time. The first call intentionally uses the full image to isolate Grounding
DINO and SAM-HQ startup from the independent person detector. After it passes,
repeat the command without `--full-image` to exercise automatic subject ROI.
From another terminal, progress can be checked with:

```bash
pgrep -af predict_localization.py \
  || echo "状态：定位进程已结束"

test -s outputs/localization/grounded_sam_hq_smoke/collar.json \
  && echo "状态：定位结果已生成"
```

The first completed smoke result is a functional check only. Confidence is not
mask accuracy, and multiple low-confidence candidates cannot be accepted by
inspection alone. Generate the automatic-person-ROI result for the exact same
image:

```bash
cd /root/fashion-semantic-parser

export OMP_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST="8.6"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PWD/src:$PWD/external/Mask2Former:$PYTHONPATH

RUN_DIR=outputs/localization/grounded_sam_hq_smoke
IMAGE_PATH="$(python - <<'PY'
import json

print(json.load(open(
    "outputs/localization/grounded_sam_hq_smoke/collar.json"
))["image_path"])
PY
)"

nohup python scripts/predict_localization.py \
  --image "$IMAGE_PATH" \
  --query "这件衣服的衣领" \
  --output "$RUN_DIR/collar_auto.json" \
  > "$RUN_DIR/collar_auto_stdout.json" \
  2> "$RUN_DIR/collar_auto.log" &

echo $! | tee "$RUN_DIR/collar_auto.pid"
```

Always check progress and completion before starting another process:

```bash
RUN_DIR=outputs/localization/grounded_sam_hq_smoke
PID="$(cat "$RUN_DIR/collar_auto.pid")"

if ps -p "$PID" > /dev/null; then
  echo "状态：自动 ROI 定位仍在运行"
elif [ -s "$RUN_DIR/collar_auto.json" ]; then
  echo "状态：自动 ROI 定位已完成"
else
  echo "状态：进程已结束，但没有结果，检查日志"
fi

tail -n 20 "$RUN_DIR/collar_auto.log"
```

After completion, create an `Original / Ground Truth / full / auto_roi`
comparison without loading either model again:

```bash
python scripts/visualize_localization_comparison.py \
  --val-json data/processed/autodl/localization/fashionpedia_parts_validation.json \
  --prediction full="$RUN_DIR/collar.json" \
  --prediction auto_roi="$RUN_DIR/collar_auto.json" \
  --output "$RUN_DIR/collar_full_vs_auto.png" \
  --metrics-output "$RUN_DIR/collar_full_vs_auto_metrics.json"
```

The query `这件衣服的衣领` resolves to the exact Fashionpedia `collar` category.
Exact category queries are not penalized for separate `lapel` or `neckline`
annotations; broader targets such as `decoration` use all categories in that
semantic group. Matching is one-to-one at mask IoU `>= 0.50`. Print only the
acceptance-relevant summary:

```bash
python - <<'PY'
import json

path = (
    "outputs/localization/grounded_sam_hq_smoke/"
    "collar_full_vs_auto_metrics.json"
)
data = json.load(open(path))
for label, metrics in data["predictions"].items():
    print(label, {
        key: None if metrics[key] is None else round(metrics[key], 2)
        for key in (
            "PredictionCount",
            "MatchedCount",
            "Precision50",
            "Recall50",
            "MatchedMeanIoU",
            "AllGTMeanIoU",
            "AllGTIoU85Rate",
        )
    })
print("visualization", data["visualization"])
PY
```

This is a one-image diagnostic, not an unbiased validation-set accuracy. The
PNG is the visual acceptance artifact; the JSON determines whether automatic
ROI improves mask matching and suppresses the large background candidates.

The completed collar example returned four candidates in both modes. Exactly
one candidate matched at IoU `58.76%`, giving `P50=25%` and `R50=100%`.
Full-image and automatic-ROI predictions were identical. The matching region
was also the highest-confidence candidate (`0.2956`), so Top-1 is promising on
this image, but the result is not sufficient evidence to change deployment
defaults.

## Candidate Ranking Validation

The dataset predictor loads Grounding DINO and SAM-HQ once, selects validation
images with exact-category ground truth, and saves up to five candidates per
image as flat COCO results. This benchmark is conditioned on images where the
target category is present; it does not measure false positives for queries
whose target is absent. Start with a 10-image full-image smoke run:

```bash
cd /root/fashion-semantic-parser

export OMP_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST="8.6"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PWD/src:$PWD/external/Mask2Former:$PYTHONPATH

RUN_DIR=outputs/localization/collar_candidate_benchmark
mkdir -p "$RUN_DIR"

nohup python scripts/predict_localization_dataset.py \
  --category collar \
  --query "这件衣服的衣领" \
  --roi-mode full \
  --image-limit 10 \
  --progress-every 1 \
  --max-regions 5 \
  --output "$RUN_DIR/collar_full_smoke10_raw.json" \
  > "$RUN_DIR/collar_full_smoke10.log" 2>&1 &

echo $! | tee "$RUN_DIR/collar_full_smoke10.pid"
```

Check progress before starting evaluation:

```bash
RUN_DIR=outputs/localization/collar_candidate_benchmark
PID="$(cat "$RUN_DIR/collar_full_smoke10.pid")"

if ps -p "$PID" > /dev/null; then
  echo "状态：10 图候选生成仍在运行"
elif [ -s "$RUN_DIR/collar_full_smoke10_raw.json" ] && \
     [ -s "$RUN_DIR/collar_full_smoke10_raw_summary.json" ]; then
  echo "状态：10 图候选生成已完成"
else
  echo "状态：进程结束，但结果不完整"
fi

tail -n 30 "$RUN_DIR/collar_full_smoke10.log"
```

One raw inference run supports an offline grid over per-image Top-K and output
score thresholds. These score filters apply after the model's configured
Grounding DINO threshold (`0.25`) and do not rerun the GPU models:

```bash
RUN_DIR=outputs/localization/collar_candidate_benchmark

for top_k in 1 3 5; do
  for threshold in 0.0 0.28 0.29; do
    python scripts/evaluate_localization_predictions.py \
      --predictions "$RUN_DIR/collar_full_smoke10_raw.json" \
      --category collar \
      --top-k "$top_k" \
      --score-threshold "$threshold" \
      --output \
        "$RUN_DIR/metrics_top${top_k}_score${threshold}.json" \
      > /dev/null
  done
done
```

Print a compact comparison:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/localization/collar_candidate_benchmark")
rows = [json.load(open(path)) for path in root.glob("metrics_top*_score*.json")]

for data in sorted(rows, key=lambda row: (row["top_k"], row["score_threshold"])):
    coco = data["segm_coco"]
    direct = data["segm_direct_iou"]
    print({
        "top_k": data["top_k"],
        "score": data["score_threshold"],
        "kept": data["candidate_count_after_filter"],
        "AP": None if coco["AP"] is None else round(coco["AP"], 2),
        "AP50": None if coco["AP50"] is None else round(coco["AP50"], 2),
        "P50": None if direct["Precision50"] is None else round(
            direct["Precision50"], 2
        ),
        "R50": None if direct["Recall50"] is None else round(
            direct["Recall50"], 2
        ),
        "F1": None if direct["F1_50"] is None else round(direct["F1_50"], 2),
        "MatchedIoU": None if direct["MatchedMeanIoU"] is None else round(
            direct["MatchedMeanIoU"], 2
        ),
        "AllGTIoU": None if direct["AllGTMeanIoU"] is None else round(
            direct["AllGTMeanIoU"], 2
        ),
    })
PY
```

The sidecar summary stores every evaluated image ID, including images where the
model returned no candidate. Offline filtering therefore preserves misses in
Recall and All-GT IoU. Run the complete collar subset only after the 10-image
smoke verifies the environment and estimated runtime.

### Collar Proposal-Recall Diagnosis

The first 10-image collar smoke produced 24 candidates but only two
ground-truth matches at IoU `0.50`: Top-1 reached `P50=20`, `R50=20`, and
`F1=20`. Top-3 and Top-5 did not improve recall. Matched mean IoU was `71.72`,
while All-GT mean IoU was only `14.34`. Five images had no candidate above
`7.63` IoU, and three more had a best candidate between `28.62` and `34.17`
IoU. This identifies proposal/mask recall as the current bottleneck rather
than post-inference Top-K filtering.

The next smoke separates two factors without changing the deployment YAML:

1. lower the Grounding DINO box threshold from `0.25` to `0.15`
2. compare the default `collar` prompt with a contextual prompt,
   `shirt collar . clothing collar`

`--grounding-prompt` changes only the English text sent to Grounding DINO. The
original query still determines the returned API label and evaluation
category. These overrides are experimental and are recorded in each run
summary.

### Supervised Part Mask2Former

The proposal-recall smoke established the limit of the zero-shot baseline.
Lowering Grounding DINO's box threshold from `0.25` to `0.15` increased
Top-10 collar recall from `20%` to `40%`, but Top-1 recall stayed at `20%` and
precision fell to `6.25%`. The contextual prompt produced no recall gain.
Grounded SAM-HQ therefore remains an open-vocabulary fallback rather than the
primary exact-category path.

`configs/localization_mask2former_parts.yaml` trains a separate 19-class
Mask2Former on the `44,898` Fashionpedia training images and `170,332` valid
part masks. It transfers the accepted fashion checkpoint's compatible visual
features, replaces the incompatible eight-class classifier, and uses
Detectron2's `RepeatFactorTrainingSampler` for categories appearing in fewer
than `1%` of training images. The eight-class deployment model is not modified.
This profile also replaces the four upstream TorchScript-wrapped Dice/BCE loss
functions with their original eager implementations. This leaves the formulas
and gradients unchanged while avoiding the PyTorch 2.1
`Global alloc not supported yet` fusion failure observed on a target shape that
was not selected by the 20-iteration smoke.

Run a 20-iteration infrastructure smoke before starting a longer stage:

```bash
cd /root/fashion-semantic-parser

export OMP_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST="8.6"
export PYTHONPATH=$PWD/src:$PWD/external/Mask2Former:$PYTHONPATH

RUN_DIR=outputs/localization/mask2former_parts_r50_smoke20
mkdir -p "$RUN_DIR"

nohup python scripts/train_segmentation_baseline.py \
  --config configs/localization_mask2former_parts.yaml \
  --output-dir "$RUN_DIR" \
  --max-iter 20 \
  --checkpoint-period 20 \
  --skip-final-eval \
  > "$RUN_DIR/train.log" 2>&1 &

echo $! | tee "$RUN_DIR/train.pid"
```

Check progress and artifacts before starting another run:

```bash
RUN_DIR=outputs/localization/mask2former_parts_r50_smoke20
PID="$(cat "$RUN_DIR/train.pid")"

if ps -p "$PID" > /dev/null; then
  echo "状态：19 类部位 Smoke 仍在训练"
elif [ -s "$RUN_DIR/model_final.pth" ]; then
  echo "状态：19 类部位 Smoke 已完成"
else
  echo "状态：进程结束，但没有最终 checkpoint"
fi

grep -E "iter:|max_mem|Traceback|out of memory" \
  "$RUN_DIR/train.log" | tail -10
ls -lh "$RUN_DIR"/*.pth 2>/dev/null
```

Warnings about the transferred checkpoint's incompatible classifier shape are
expected: the source head has eight garment classes and the new head has 19
part classes. Backbone, pixel-decoder, transformer-decoder, and compatible mask
features still load. A traceback, CUDA out-of-memory message, missing
`model_final.pth`, or missing category-repeat sampler is not expected.

## API Contract

The request model accepts an image, query, and optional subject ROI:

```json
{
  "image_path": "data/raw/fashionpedia/test/example.jpg",
  "query": "这件衣服的领口",
  "auto_subject_roi": true
}
```

A configured runtime returns:

```json
{
  "image_path": "data/raw/fashionpedia/test/example.jpg",
  "query": "这件衣服的领口",
  "regions": [
    {
      "region_label": "collar",
      "matched_text": "领口",
      "confidence": 0.94,
      "box": {
        "x_min": 10.0,
        "y_min": 20.0,
        "x_max": 50.0,
        "y_max": 60.0
      },
      "mask": [[10.0, 20.0, 50.0, 20.0, 50.0, 60.0, 10.0, 60.0]]
    }
  ]
}
```

## Implemented Inference Flow

1. validate the project-relative RGB image path
2. normalize known Chinese/Fashionpedia terms into an English grounding prompt
3. use the accepted person detector and retain a `0.35` context margin
4. run Grounding DINO Swin-T to obtain text-conditioned candidate boxes
5. refine all retained boxes in one SAM-HQ batch
6. discard malformed or tiny masks and derive each final box from its mask
7. map crop-local polygons and boxes back to original-image coordinates

The first baseline uses person ROI constraints but does not yet load the large
Mask2Former garment model alongside Grounding DINO and SAM-HQ. Parent-garment
mask constraints remain an evaluation-driven follow-up if text grounding
produces cross-garment false positives. DINOv2 also remains an optional
candidate re-ranker rather than a standalone text localizer.

The supervised 19-class run reached 10,000 iterations without a traceback,
OOM, NaN, or recurrence of the JIT allocation failure. Formal validation at
score threshold `0.0` selected the 10,000-iteration checkpoint: mask AP is
`8.88`, AP50 is `16.95`, AP75 is `7.81`, AP85 is `4.13`, and AP90 is `1.94`.
Collar, lapel, sleeve, pocket, and neckline AP are `19.87`, `19.03`, `61.04`,
`13.06`, and `9.08`. This is the end-to-end flow baseline, not an accuracy
acceptance result.

### Hybrid Deployment

`configs/localization_mask2former_parts_deployment.yaml` points to the selected
checkpoint through a stable project path. On AutoDL, create the link once:

```bash
cd /root/fashion-semantic-parser

mkdir -p models/checkpoints/localization
ln -sfn \
  /root/autodl-tmp/fashionpedia/outputs/localization_parts_stage1_01000/model_0009999.pth \
  models/checkpoints/localization/mask2former_parts_r50_10000.pth

ls -Llh models/checkpoints/localization/mask2former_parts_r50_10000.pth
```

The default application backend is `hybrid`:

1. directly supervised Fashionpedia part queries use Mask2Former and fall back
   to Grounding DINO + SAM-HQ when the supervised result is empty
2. generic decoration queries retain all predicted decoration subclasses
3. shoulder queries use epaulette as explicitly partial supervision; after an
   epaulette miss, they derive two compact upper-side regions from the current
   `top`, `outerwear`, or `dress` mask instead of accepting a whole-person
   open-vocabulary mask
4. cuff queries retain at most two sleeve masks with confidence at least `0.5`,
   estimate each sleeve's principal axis, and use the distal `8%`; Grounding
   DINO + SAM-HQ remains the fallback when no qualifying sleeve is detected
5. hem queries reuse the current 3.1.1 `top`, `skirt`, `outerwear`, or `dress`
   masks, honor an explicitly named parent garment, suppress overlapping boxes,
   and follow the central lower contour with a `6%`-height band so sleeve ends
   do not become garment hems
6. waist queries reuse the current 3.1.1 garment mask and return one central
   `6%`-height band; explicit parent garments are honored, upper garments use
   an anatomy-informed torso position, and pants or skirts use their top edge
7. pattern queries inspect the selected 3.1.1 garment mask for compact internal
   color outliers, reject garment borders, broad illumination changes, and
   large upper-body occlusions, then use Grounding DINO + SAM-HQ when no stable
   appearance candidate remains
8. custom queries use Grounding DINO + SAM-HQ
9. `/v1/query` invokes localization only for known local-region language and
   keeps the 3.1.1 garment result in the same response; reused ROI coordinates
   retain their original `detected`, `full_image_fallback`, or `manual` source

The deployment score threshold `0.25` is provisional for functional testing.
Thresholded direct-IoU and visual acceptance remain later accuracy work.
The cuff, hem, and waist derivations are explicit geometric approximations, and
pattern extraction is an appearance heuristic. None has direct Fashionpedia
part supervision; labelled or pseudo-labelled evaluation data is still needed
before reporting accuracy for these regions.

### Query API Functional Acceptance

With the API already running, verify all eight PRD region paths using one known
derived-region image plus deterministic large validation annotations for the
directly supervised classes:

```bash
DERIVED_IMAGE="$(python -c '
import json
print(json.load(open(
    "outputs/localization/hybrid_api_smoke/pattern_request.json"
))["image_path"])
')"

python scripts/accept_localization_api.py \
  --base-url http://127.0.0.1:8002 \
  --val-json data/processed/autodl/localization/fashionpedia_parts_validation.json \
  --derived-image "$DERIVED_IMAGE" \
  --output outputs/localization/hybrid_api_smoke/acceptance_report.json \
  --responses-dir outputs/localization/hybrid_api_smoke/acceptance_responses
```

The script prints one progress line per request and exits nonzero if any expected
label, derived source, valid automatic ROI state, segmentation payload, mask,
or box is missing. For collar, pocket, and ruffle, the selected Fashionpedia
ground-truth mask is also compared with the matching API prediction and
requires mask IoU at least `0.50`. Fashionpedia's epaulette mask covers only a
shoulder decoration, so it is displayed as a partial reference and is not used
as full-shoulder IoU ground truth. A valid automatic ROI state is either
`detected` with a person box or an explicit `full_image_fallback` without one;
the stricter all-detected value remains in the report as a diagnostic. Shoulder
and the four unlabelled derived regions remain visual checks and are not
converted into accuracy evidence.

The first AutoDL run returned `accepted=true` under the old structural checks:
all eight labels were present and every response had a detected person ROI,
segmentation payload, non-empty mask, and valid box. Visual review then found
that shoulder covered almost the entire person. That `8/8 PASS` result is
therefore rejected. The stricter runner and three-panel visualization were
introduced to keep label/schema success separate from spatial correctness. In
the first strict rerun, collar, pocket, and ruffle reached `95.1%`, `94.3%`, and
`92.0%` mask IoU. The apparent whole-skirt ruffle result matches the selected
Fashionpedia annotation. Shoulder reached only `3.9%` against an epaulette mask,
which exposed a semantic mismatch in the test rather than a valid full-shoulder
metric. The epaulette is now a partial visual reference, and the derived
shoulder bands are narrower and closer to the garment's upper edge. The final
corrected run passed all eight representative requests in `8.52 s`; all eight
used detected person ROIs. Collar, pocket, and ruffle retained their direct-IoU
evidence, while shoulder, cuff, hem, waist, and pattern are explicitly recorded
as visual-only acceptance. This closes the first end-to-end functional slice,
not the dataset-level `92%` accuracy or `30 ms` latency targets.

Render the saved eight-request result as one visual acceptance sheet:

```bash
python scripts/visualize_localization_api_acceptance.py \
  --report outputs/localization/hybrid_api_smoke/acceptance_report.json \
  --output outputs/localization/hybrid_api_smoke/acceptance_overview.png
```

The overview contains Original / Ground truth / Prediction panels for exactly
labelled cases and marks their best mask IoU. Shoulder displays the epaulette as
a `Partial reference`, while shoulder and other derived cases show `visual only`
because no equivalent direct Fashionpedia mask exists. Every card also includes
the person ROI source, region count, and request time. It is intended for visual
functional review, not dataset-level accuracy reporting.

## Formal Performance Benchmark

The representative `8/8` sheet is functional evidence only. Performance work
uses two separate measurement contracts:

- exact-GT accuracy runs the deployed 19-class model with its committed score
  threshold and automatic person ROI on the full validation image union for
  collar, lapel, neckline, pocket, and all 12 decoration subclasses
- warm service latency includes path validation, image decoding, automatic
  person ROI, inference, and mask/box postprocessing, while excluding model
  loading, HTTP transport, and JSON serialization

The provisional accuracy metric is macro `Recall50` across the 16 exact source
categories. It is compared with `92%`, but the result is explicitly named
`exact_gt_scope_passed`. It cannot establish overall eight-region accuracy
because shoulder has only partial epaulette supervision and cuff, hem, waist,
and pattern have no equivalent Fashionpedia masks.

Run a short accuracy smoke test first:

```bash
cd /root/fashion-semantic-parser

export OMP_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST="8.6"
export PYTHONPATH=$PWD/src:$PWD/external/Mask2Former:$PYTHONPATH

SMOKE_DIR=outputs/localization/performance/exact_gt_smoke

python scripts/benchmark_localization_accuracy.py \
  --image-limit-per-category 2 \
  --progress-every 1 \
  --output-dir "$SMOKE_DIR"
```

After the smoke test succeeds, start the full validation run:

```bash
RUN_DIR=outputs/localization/performance/exact_gt_full
mkdir -p "$RUN_DIR"

nohup python scripts/benchmark_localization_accuracy.py \
  --progress-every 25 \
  --output-dir "$RUN_DIR" \
  > "$RUN_DIR/benchmark.log" 2>&1 &

echo $! | tee "$RUN_DIR/benchmark.pid"
```

Watch it continuously in the current terminal. `Ctrl-C` stops only the monitor,
not the benchmark:

```bash
RUN_DIR=outputs/localization/performance/exact_gt_full
PID="$(cat "$RUN_DIR/benchmark.pid")"

tail --pid="$PID" -F "$RUN_DIR/benchmark.log" \
  | grep --line-buffered -E \
    "^[[]|exact_gt_scope_passed|measured_percent|Traceback|out of memory"
```

Check completion and print the decision without copying the full JSON:

```bash
RUN_DIR=outputs/localization/performance/exact_gt_full
PID="$(cat "$RUN_DIR/benchmark.pid")"

if ps -p "$PID" > /dev/null; then
  echo "状态：完整准确率评估仍在运行"
elif [ -s "$RUN_DIR/metrics.json" ]; then
  echo "状态：完整准确率评估已完成"
else
  echo "状态：进程结束但没有指标，检查 benchmark.log"
fi

python - <<'PY'
import json

path = "outputs/localization/performance/exact_gt_full/metrics.json"
d = json.load(open(path))
contract = d["accuracy_contract"]
metrics = d["evaluation"]["segm_direct_iou"]
print({
    "macro_recall50": round(contract["measured_percent"], 2),
    "target": contract["target_percent"],
    "exact_gt_passed": contract["exact_gt_scope_passed"],
    "micro_P50": round(metrics["Precision50"], 2),
    "micro_R50": round(metrics["Recall50"], 2),
    "micro_F1": round(metrics["F1_50"], 2),
    "AllGTIoU": round(metrics["AllGTMeanIoU"], 2),
})
PY
```

Then run the eight-route warm latency benchmark. It reuses loaded services
within one process and prints one line after each query route:

```bash
LAT_DIR=outputs/localization/performance/latency
mkdir -p "$LAT_DIR"

nohup python scripts/benchmark_localization_latency.py \
  --image-limit 20 \
  --warmup-runs 5 \
  --runs 100 \
  --output "$LAT_DIR/warm_hybrid.json" \
  > "$LAT_DIR/warm_hybrid.log" 2>&1 &

echo $! | tee "$LAT_DIR/warm_hybrid.pid"

PID="$(cat "$LAT_DIR/warm_hybrid.pid")"
tail --pid="$PID" -F "$LAT_DIR/warm_hybrid.log" \
  | grep --line-buffered -E \
    "mean=|all_routes_passed|Traceback|out of memory"
```

After it exits, inspect only the route-level P95 values:

```bash
python - <<'PY'
import json

path = "outputs/localization/performance/latency/warm_hybrid.json"
d = json.load(open(path))
print("all_routes_passed", d["all_routes_passed"])
for route in d["routes"]:
    print(
        route["resolved_region"],
        {
            "mean_ms": round(route["latency_ms"]["mean"], 2),
            "p95_ms": round(route["latency_ms"]["p95"], 2),
            "passed": route["passed"],
        },
    )
PY
```

Decision branch after both runs:

1. If exact-GT macro Recall50 is below `92%`, calibrate score threshold and
   per-class Top-K from the saved `predictions.json`; if recall remains low,
   continue supervised training or class-balanced fine-tuning.
2. If exact-GT passes, build a separately annotated five-region validation set
   before making an overall eight-region accuracy claim.
3. If any warm route P95 exceeds `30 ms`, profile person ROI and each model
   route separately, then evaluate shared ROI reuse, TensorRT/ONNX export, and
   lighter backbones. Do not hide a failing route inside an aggregate mean.

If the exact-GT run misses `92%`, regenerate candidates with the model-output
threshold set to zero. This is different from `--score-threshold`, which is an
extra offline filter applied only after candidates have already been saved:

```bash
AUTO_DIR=outputs/localization/performance/exact_gt_auto_candidates
mkdir -p "$AUTO_DIR"

nohup python scripts/benchmark_localization_accuracy.py \
  --roi-mode auto \
  --inference-score-threshold 0.0 \
  --progress-every 25 \
  --output-dir "$AUTO_DIR" \
  > "$AUTO_DIR/benchmark.log" 2>&1 &

echo $! | tee "$AUTO_DIR/benchmark.pid"

PID="$(cat "$AUTO_DIR/benchmark.pid")"
tail --pid="$PID" -F "$AUTO_DIR/benchmark.log" \
  | grep --line-buffered -E \
    "^[[]|exact_gt_scope_passed|measured_percent|Traceback|out of memory"
```

After candidate generation finishes, scan score thresholds and per-category
Top-K values without rerunning either model:

```bash
AUTO_DIR=outputs/localization/performance/exact_gt_auto_candidates
SCAN_DIR=outputs/localization/performance/operating_points
mkdir -p "$SCAN_DIR"

nohup python scripts/scan_localization_operating_points.py \
  --predictions "$AUTO_DIR/predictions.json" \
  --run-summary "$AUTO_DIR/predictions_summary.json" \
  --output "$SCAN_DIR/auto.json" \
  > "$SCAN_DIR/auto.log" 2>&1 &

echo $! | tee "$SCAN_DIR/auto.pid"

PID="$(cat "$SCAN_DIR/auto.pid")"
tail --pid="$PID" -F "$SCAN_DIR/auto.log" \
  | grep --line-buffered -E \
    "^[[]|any_operating_point_passed|Traceback|out of memory"
```

Interpret the scan before starting another training run:

- high recall only at low thresholds means calibration or postprocessing is
  the primary blocker
- low automatic-ROI recall but materially higher full-image recall indicates
  an ROI crop problem; repeat candidate generation with `--roi-mode full`
- low recall for both ROI modes even with threshold `0.0` and unlimited Top-K
  indicates that class-balanced training or additional labels are required

The first full scan produced `38.53%` macro Recall50 with automatic ROI and
`38.93%` on the full image. The `0.40` point difference rules out ROI cropping
as the primary blocker. The best automatic-ROI F1 was `34.32%`; unlimited
candidates raised recall but reduced precision to `1.71%`. The lowest-recall
classes were tassel, rivet, fringe, flower, bead, zipper, ribbon, and buckle.
Threshold calibration alone therefore cannot meet the accuracy contract.

Continue from the 10,000-iteration checkpoint with the isolated long-tail
profile in `configs/localization_mask2former_parts_long_tail.yaml`. It raises
the repeat-factor threshold from `0.01` to `0.05`, lowers the learning rate to
`5e-6`, and leaves the selected deployment checkpoint unchanged. Evaluate its
intermediate and final checkpoints before promoting any weights.

Use `benchmark_localization_accuracy.py --weights CHECKPOINT` for those
comparisons. The explicit override is written into `predictions_summary.json`
so a result cannot be mistaken for the fixed deployment checkpoint.

The 5,000-iteration long-tail checkpoint improved formal mask AP from `8.88`
to `9.84` and AP50 from `16.95` to `18.40`. Exact-source macro Recall50 rose
from `38.53%` to `41.55%`. Buckle regressed, while bow and ribbon were flat and
tassel remained at zero recall. The next isolated experiment therefore replays
complete images containing buckle, bow, ribbon, rivet, or tassel through
`configs/localization_mask2former_parts_targeted.yaml`; it starts from the
long-tail checkpoint and does not change the deployment profile.
The audit selected `5,291 / 44,898` training images (`11.78%`), so the replay
source uses a conservative `2.0` factor instead of the initial `3.0` proposal.

The targeted replay stage improved formal mask AP from `9.84` to `10.30` at
3,000 iterations (`AP50 18.40 -> 19.03`, `AP75 8.78 -> 9.25`). Its best exact
source macro Recall50 was `42.52%`, compared with `41.55%` for the long-tail
checkpoint. Buckle, bow, ribbon, and rivet gained small formal-AP improvements,
but tassel remained at zero AP and only `2.56%` Recall50. Because the 1,000 and
3,000 targeted checkpoints were close, further replay-only training is treated
as saturated rather than evidence that the `92%` requirement is within reach.

The final isolated fixed-label experiment used
`configs/localization_mask2former_parts_class_weighted.yaml`. It kept the
targeted replay ratio at `2.0`, started from targeted 3,000, and applied modest
classification-loss weights to buckle, bow, ribbon, rivet, and tassel. The
3,000-iteration checkpoint reached mask AP `10.22`, exact-source macro Recall50
`40.86%`, and best F1 `34.06%`; all three were below targeted 3,000 at `10.30`,
`42.52%`, and `35.29%`. The class-weighted checkpoints are therefore rejected,
and further fixed-label loss, replay, or iteration tuning is stopped.

## Open-Language Referring-Expression Correction

The fixed-label experiments above are retained as engineering baselines, not as
the final interpretation of PRD 3.1.2. Mapping a query such as `左边的袖口` to
the fixed label `cuff` discards the spatial modifier. The intended task instead
requires the complete expression to select the target and must distinguish:

- parts, such as buttons or zippers
- spatial modifiers, such as left, right, upper, or lower
- visual attributes, such as floral, striped, red, or silver
- relationships, such as an inner garment underneath a coat
- novel paraphrases, compositions, and parts outside the fixed 19-class head

Do not merge the 19 Fashionpedia classes into another fixed four-class model as
the next main experiment. Keep the selected `targeted_3000` checkpoint only as
an auxiliary known-part candidate source and closed-vocabulary comparison.

### Feasibility Manifest

Start from the committed 20-query template:

```bash
cp \
  data/benchmarks/localization/referring_smoke_v1.template.json \
  data/benchmarks/localization/referring_smoke_v1.json
```

For the current AutoDL Fashionpedia archive, the `1158` records in
`instances_attributes_val2020.json` match the images under
`data/raw/fashionpedia/test/` despite that directory name. Generate a
deterministic candidate manifest directly from those records:

```bash
python scripts/prepare_referring_smoke_fashionpedia.py
```

The selector fills all 20 query rows, keeps the cuff contrast set on one image,
and imports reviewed-source Fashionpedia masks only where the expression has a
direct annotation boundary. It deliberately does not relabel sleeves as cuffs
or decorations as logos. Its summary is written to
`outputs/localization/referring_smoke/fashionpedia_selection.json`.

The generated file is not immediately accuracy-ready. Direct Fashionpedia GT
and all automatically selected images still need a visual review; unsupported
cuff, button, drawstring, color, pattern, and layering expressions remain
`unlabelled` until a human adds a reviewed Box or Mask.

Replace every placeholder image with a project-relative path and review each
case. Version 1 defines `left` and `right` in image coordinates. A spatial case
must record its reference frame. Expressions can contain multiple dimensions,
for example `左边袖子上的碎花图案` is simultaneously spatial, attribute, and
relational. Use `contrast_set_id` for multiple expressions on the same image so
the report can verify that every modifier variant succeeds.

Each case has one explicit annotation boundary:

- `mask`: every target has a reviewed COCO polygon or RLE Mask
- `box`: every target has a reviewed `xyxy` Box
- `negative`: the image was reviewed and contains no valid target
- `unlabelled`: useful for visual exploration but excluded from accuracy

Do not assign `expected_count` to an unlabelled case. For a labelled positive
case, `expected_count` must equal the number of target instances. This keeps
`袖口` with two targets distinct from `左边的袖口` with one target.

One reviewed Box case has this shape:

```json
{
  "id": "spatial_left_cuff_001",
  "contrast_set_id": "cuff_image_001",
  "image_path": "data/raw/referring_smoke/cuffs.jpg",
  "query": "衣服左边的袖口",
  "grounding_prompt": "the cuff on the left side of the garment",
  "dimensions": ["basic", "spatial"],
  "novelty": "novel_composition",
  "reference_frame": "image",
  "annotation_status": "box",
  "expected_count": 1,
  "targets": [
    {
      "label": "left_cuff",
      "box": {"x_min": 10, "y_min": 20, "x_max": 40, "y_max": 60}
    }
  ]
}
```

### Full-Expression Baseline

The batch runner passes each case's complete English grounding prompt without
changing the original Chinese query and reuses one loaded Grounding DINO +
SAM-HQ bundle. Run full-image mode first to isolate language grounding from ROI
cropping:

```bash
python scripts/predict_referring_localization.py \
  --manifest data/benchmarks/localization/referring_smoke_v1.json \
  --roi-mode full \
  --box-threshold 0.15 \
  --max-regions 10 \
  --output-dir outputs/localization/referring_smoke/full
```

Then evaluate the saved per-query responses without rerunning either model:

```bash
python scripts/evaluate_referring_localization.py \
  --manifest data/benchmarks/localization/referring_smoke_v1.json \
  --responses-dir outputs/localization/referring_smoke/full/responses \
  --output outputs/localization/referring_smoke/full/metrics.json \
  --min-iou 0.50
```

The report separates Mask and Box IoU scopes, positive-instance Recall50,
query-level all-target recall, exact expected-count success, reviewed negative
queries, empty predictions, language dimensions, novelty types, and contrast
sets. A correct target among ten candidates can pass target recall but cannot
pass exact query selection. Unlabelled cases never enter accuracy denominators.

The first request includes model loading, so cold first-call wall time is kept
separate from the remaining warm calls. Do not compare the cold value with the
PRD latency target. This 20-query set selects the next model direction only;
`prd_accuracy_passed` remains `null` and cannot establish the `92%` contract.

Decision branch after the first complete run:

1. If basic parts fail, compare a stronger referring/open-vocabulary grounding
   model before adding postprocessing.
2. If basic parts pass but spatial modifiers fail, retain the candidates and add
   explicit spatial reranking.
3. If attributes fail, compare region-attribute or vision-language reranking.
4. If relationships fail, ground the referenced entities separately and test
   containment, overlap, adjacency, and garment-layer reasoning.
5. If selected boxes are correct but Mask boundaries fail, isolate the SAM
   refinement stage. Current returned boxes are Mask-derived, so proposal boxes
   must be saved separately before claiming which stage caused an error.

### First Real Referring Baseline

The first 20-query full-image run completed without empty responses, but only
four cases had imported Fashionpedia GT. Those four scored `0/4` query success,
with `25` predicted instances, `4` GT instances, and no Mask IoU match at
`0.50`. This is a failed accuracy baseline, not evidence that all 20 unlabelled
queries failed.

Per-case diagnostics separated candidate localization from Mask quality:

| Query | Full expression best Mask/Box IoU | Noun-only best Mask/Box IoU | Noun-only + person ROI |
|---|---:|---:|---:|
| neckline | `45.82 / 90.77` | `41.67 / 92.35` | `24.83 / 75.79` |
| zipper | `4.57 / 14.12` | `1.44 / 9.15` | `0.22 / 5.62` |
| right pocket | `1.04 / 3.16` | `0.49 / 1.12` | `0.49 / 1.13` |
| lower zipper | `3.19 / 30.30` | `5.37 / 24.85` | `5.92 / 25.05` |

The neckline result shows a strong candidate Box but weak SAM-HQ Mask. Zipper
and pocket fail before spatial reranking because no useful candidate is present.
Noun-only prompts and person ROI do not fix this, so further Grounding DINO
threshold or ROI tuning is stopped. Warm full-image inference was about `212 ms`
mean and `231 ms` P95; person ROI increased the warm mean to about `250 ms`.
The original `258 s` cold call was dominated by blocked Hugging Face retries;
with cached offline loading it fell to about `6-7 s`.

### Known-Part Candidates with Language Constraints

The next bounded path uses the selected fixed-label model only as an auxiliary
candidate generator. The complete query is retained, and explicit left, right,
upper, and lower modifiers select candidates in image coordinates. A query with
no spatial modifier keeps every matching part. Unknown targets still receive
the original English expression in the Grounding DINO + SAM-HQ fallback.

This does not yet solve color, pattern, or garment-layer relationships; those
require attribute or relation reranking after candidate recall is established.
It also does not turn the fixed 19-class model back into the PRD definition.

Run the four labelled known-part cases with the selected targeted checkpoint:

```bash
python scripts/predict_referring_localization.py \
  --manifest data/benchmarks/localization/referring_noun_only_4.json \
  --backend hybrid \
  --part-config configs/localization_mask2former_parts_targeted_deployment.yaml \
  --roi-mode full \
  --box-threshold 0.15 \
  --max-regions 10 \
  --output-dir outputs/localization/referring_smoke/hybrid_known_part_4
```

If a labelled case has no useful known-part candidate, run one bounded recall
diagnostic with `--part-score-threshold 0.05`. This override applies only to the
Mask2Former auxiliary candidate generator; `--box-threshold` remains the
Grounding DINO threshold. Compare best Mask/Box IoU before changing any spatial
reranking rule. If the unfiltered low-threshold candidates still have poor IoU,
stop threshold tuning and replace or augment the candidate generator.

The bounded `0.05` diagnostic reached that stopping condition. It increased the
right-pocket candidates from `2` to `11` and the lower-zipper candidates from
`1` to `6`, but did not produce a reliable language-selected result:

| Case | Unfiltered best Mask/Box IoU | Spatially selected Mask/Box IoU |
|---|---:|---:|
| right pocket | `40.92 / 78.80` | `39.17 / 39.84` |
| lower zipper | `28.22 / 20.64` | `6.63 / 0.83` |

The best right-pocket candidate had confidence `0.052`, while the slightly more
rightward fragment selected by the coordinate rule had confidence `0.215`.
Therefore neither a confidence cutoff nor an extreme-coordinate rule can select
the correct candidate consistently. The zipper set still contained no usable
candidate. Further fixed-part threshold and handcrafted spatial-rank tuning is
frozen: it would fit these two cases without addressing arbitrary referring
expressions.

The next model-direction smoke must follow the published PRD stack rather than
introducing an unlisted foundation model. The primary path is therefore:

1. use DINOv2 to encode dense image regions
2. encode the complete natural-language expression and learn the cross-modal
   projection required for region-text similarity matching
3. retain multiple high-recall region proposals before applying spatial,
   attribute, and relation constraints
4. pass the selected proposal box to SAM-HQ for the required output Mask and Box

The existing Grounding DINO and fixed-part Hybrid results remain reproducible
historical baselines only. They are not the PRD delivery backend because the
published technical plan names DINOv2 region features, SAM-HQ, Python 3.10.12,
and later TensorRT optimization. No replacement model may enter the main path
without an approved PRD revision.

### PRD Compliance Guardrails

All subsequent 3.1.2 work uses the published PRD as a hard constraint:

- model path: DINOv2 region features, complete-query text features, learned
  region-text similarity matching, and SAM-HQ Mask refinement
- supporting stack: PyTorch, OpenCV, and Mask2Former where a proposal or
  auxiliary segmentation stage is required
- data boundary: DeepFashion2 and Fashionpedia may both support training,
  development validation, part supervision, and query-Mask construction;
  neither dataset's native category metric replaces the independent manually
  annotated referring-expression acceptance set
- runtime boundary: Python 3.10.12 first, followed by ONNX Runtime and
  TensorRT 8.6.1 optimization
- accuracy boundary: at least `92%` on a manually annotated localization test
  set; functional demos, Fashionpedia AP, and four-case smoke results are not
  substitutes
- latency boundary: at most `30 ms` per localization request, with model-load
  time reported separately; the complete system must also meet `60 QPS` on one
  RTX 3090 and at most `400 ms` mean end-to-end response time

The PRD does not name the text encoder or define the exact formula behind
"localization accuracy." Those two implementation details must be fixed inside
the listed stack and written into the acceptance protocol before any `92%`
claim is permitted.

### Fashionpedia Referring Training Index

`scripts/prepare_referring_training_fashionpedia.py` converts official
Fashionpedia annotations into a compact JSONL index for the PRD DINOv2
region-text matching path. Each record preserves the complete query, language,
query dimensions, target boxes, and official source annotation IDs. Masks are
not duplicated in JSONL; the training loader resolves them from the official
Fashionpedia annotation file. This prevents hundreds of thousands of repeated
polygon/RLE payloads from filling the data disk.

The first version generates only deterministic supervision:

- basic queries target every instance of the named part in one image
- left/right/upper/lower queries require a unique extreme separated from the
  next candidate by at least `5%` of the relevant image dimension
- attribute queries use attributes attached directly to the target part; they
  do not inherit a garment attribute and relabel the part
- part-on-garment relation queries require at least `80%` part-box containment
  and exactly one eligible containing garment; nested ambiguous cases are
  skipped
- basic, spatial, and reliable relation queries are generated in Chinese and
  English; official attribute names remain in English rather than using an
  invented translation

Run a small audit before producing the full index:

```bash
python scripts/prepare_referring_training_fashionpedia.py \
  --split train \
  --limit 100 \
  --output data/processed/autodl/localization/fashionpedia_referring_train_smoke.jsonl \
  --summary-output outputs/localization/referring_training/fashionpedia_train_smoke_summary.json
```

Then remove `--limit` and use the default full-output paths. The summary reports
sample, target-reference, dimension, language, category, relation-association,
ambiguous-spatial, invalid-annotation, and missing-image counts. This index is
training-data preparation only: DINOv2 alignment training, independent manual
acceptance data, `92%` accuracy, and `30 ms` latency remain pending.
The full converter prints one compact progress line per `1000` selected images;
use `--progress-every` to change that interval.

The complete Fashionpedia conversion produced `713,059` training expressions
and `17,454` validation expressions. Nine of the `170,341` candidate training
part annotations had invalid zero-area or otherwise non-positive bounding boxes
in the official source file and were excluded before query generation:

```text
23298, 23299, 139952, 262063, 277148, 287029, 290142, 309141, 311943
```

The affected source categories are six rivets, two beads, and one neckline.
Validation had no invalid part annotations, and neither split had a missing
image. These source defects must remain documented exclusions; the pipeline
must not fabricate replacement boxes or Masks.

`FashionpediaReferringDataset` resolves every JSONL target through its official
annotation ID. Before returning a training item it verifies the source image
ID, category, bounding box, image dimensions, non-empty decoded Mask, and safe
project-relative image path. Multi-target queries retain an independent Mask
per target instead of merging broad expressions such as "the pockets" into one
instance. COCO polygon/RLE decoding uses `pycocotools` so training uses the same
Mask edge semantics as evaluation.

Run a bounded PyTorch `DataLoader` smoke before adding model features:

```bash
python -u scripts/smoke_referring_training_loader.py \
  --split train \
  --limit 8 \
  --batch-size 2 \
  --workers 0
```

The initial smoke deliberately uses `workers=0`. A full-dataset multi-worker
memory and throughput profile is still required before choosing the production
training worker count. Successful Mask loading does not yet establish DINOv2
alignment quality or PRD accuracy and latency compliance.

### PRD 3.1.2 Training Environment

The PRD requires Python `3.10.12` exactly. It lists PyTorch but does not assign a
PyTorch version, so this project pins the already validated implementation
combination PyTorch `2.1.2` with CUDA `12.1`; that pin is a reproducibility
decision, not a claim that the PRD mandates PyTorch 2.1.2. Create the isolated
foundation-training environment from the repository root:

```bash
bash scripts/setup_prd_312_training_env.sh
```

The setup script uses a temporary regular Conda configuration so a malformed
user mirror cannot affect dependency resolution. It repairs a partially
created `fashion-prd-312` environment with `conda env update`, or creates it
when absent, and then runs the readiness check through `conda run`. Its internal
error handling exits only the script and cannot close the caller's terminal.

The check requires exact Python `3.10.12`, the pinned CUDA-enabled PyTorch,
OpenCV, and `pycocotools`. This is only the foundation-training gate. Separate
checks must still establish the official DINOv2 model, the selected text encoder
from the PRD stack, SAM-HQ `1.0+`, ONNX Runtime `1.17`, and TensorRT `8.6.1`
before model or deployment compliance can be claimed.

### DINOv2 Region Feature Smoke

The first PRD-main-path model smoke uses Meta's official `dinov2_vits14` Torch
Hub backbone. ViT-S/14 is the latency-oriented starting point: it has 21 million
parameters, patch size 14, and 384-dimensional features. This is an experimental
implementation choice, not an accuracy result or a commitment to the final
backbone. The official model and weights use Apache License 2.0.

Images and source target Masks receive the same aspect-ratio-preserving 518x518
letterbox transform. Padding uses the ImageNet mean rather than black pixels.
Every Mask is converted to a 37x37 occupancy grid by marking any patch touched
by the target, so small parts do not disappear through nearest sampling. The
corresponding normalized DINOv2 patch tokens are mean-pooled and L2-normalized,
with broad multi-target queries retaining one independent feature per target.

Install the pinned official repository and weights. The setup uses Git SSH for
the repository because Python HTTPS access to GitHub may be blocked on AutoDL;
the weights use Meta's official CDN and support a resumed partial download:

```bash
bash scripts/setup_dinov2_region_model.sh
```

The pinned source commit is
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`, and the expected official
ViT-S/14 weights size is `88,283,115` bytes. Runtime loading uses the local
checkout and local weights only. It rejects an unpinned checkout or incomplete
weights instead of falling back to a network download.

Then run the bounded GPU smoke in the exact PRD foundation environment:

```bash
conda run -n fashion-prd-312 \
  python -u scripts/smoke_dinov2_region_features.py \
  --split train \
  --limit 4
```

The first run downloads the official Torch Hub repository and ViT-S/14 weights.
Model download/load time, first inference, and warm inference are reported
separately. The final line deliberately remains
`prd_localization_30ms_passed: not_evaluated`: region feature extraction alone
does not include text encoding, similarity selection, proposal generation, or
SAM-HQ, and therefore cannot be compared with the complete 30 ms requirement.

The first RTX 3090 run in the exact Python 3.10.12 environment loaded the model
in `2.015 s` and returned four normalized `384`-D target features. The cold first
encode took `221.846 ms`; after CUDA/operator warm-up, mean latency was
`11.419 ms` and maximum latency was `11.686 ms` over three requests. This is
valid evidence for the isolated DINOv2 region encoder only. It leaves roughly
`18.6 ms` of the localization budget before text encoding, matching, proposal
generation, and SAM-HQ, so it is not a 30 ms pass.

### BGE-M3 Complete-Query Text Feature Smoke

The PRD lists BGE-M3 and leaves the 3.1.2 text encoder unspecified. BGE-M3 is
therefore the first in-stack implementation choice for complete-query text
features: it supports more than 100 languages, including the Chinese/English
training expressions, and produces normalized 1024-dimensional dense vectors.
The official model does not require retrieval instructions for BGE-M3 queries.
This choice remains conditional on latency and alignment accuracy; merely
embedding text does not align its space with DINOv2's 384-dimensional regions.

Update the existing PRD environment to add the pinned Transformers and
Sentence Transformers runtime, then download only the dense-model files from
the fixed BAAI snapshot:

```bash
bash scripts/setup_prd_312_training_env.sh

conda run -n fashion-prd-312 \
  python scripts/setup_bge_m3_text_model.py
```

The pinned BGE-M3 revision is
`3c06a359c08b8c49f1cab07e3eac8f846eb3a038`; the expected official
`model.safetensors` size is `2,271,064,456` bytes. Runtime loading is local-only
and rejects a missing revision marker or partial weights. Duplicate PyTorch
weights and the sparse/ColBERT heads are not downloaded for this dense smoke.

Measure four complete basic, spatial, attribute, and relation queries:

```bash
conda run -n fashion-prd-312 \
  python -u scripts/smoke_bge_m3_text_features.py
```

The smoke must report four normalized 1024-D vectors and keeps both
`dinov2_text_alignment_trained: false` and
`prd_localization_30ms_passed: not_evaluated`. The next model step is a learned
1024-to-384 projection with contrastive region-text supervision, not direct
cosine comparison between unrelated pretrained spaces.

The first RTX 3090 BGE-M3 smoke loaded in `3.764 s` and produced four complete
query embeddings with shape `4x1024`. Their norm range was
`0.999760..1.000442`; the cold first encode took `156.436 ms`, while the three
warm requests averaged `18.679 ms` and reached `20.667 ms` maximum. Combined
with the isolated DINOv2 warm maximum of `11.686 ms`, the two encoders already
sum to more than the complete `30 ms` localization budget when run
sequentially. This is a latency risk requiring batching, caching, a smaller
text encoder, distillation, or deployment optimization; it is not a PRD pass.

### Frozen-Encoder Region-Text Alignment Smoke

The first alignment stage freezes both official foundation encoders and trains
only a two-layer BGE-M3 `1024 -> 512 -> 384` projection. It uses symmetric
multi-positive InfoNCE: all independent Fashionpedia annotations referenced by
a query are positive, and alternative queries that share a source annotation
do not incorrectly make that region a negative. DINOv2 features are extracted
once per unique source Mask and reused across the bounded optimization steps.

Run the eight-query smoke in the exact PRD environment:

```bash
OMP_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
conda run -n fashion-prd-312 \
  python -u scripts/smoke_region_text_alignment_training.py \
  --split train \
  --limit 8 \
  --steps 20
```

The output is a small projection-only checkpoint plus `metrics.json` under
`outputs/localization/dinov2_bge_alignment_smoke`. Loss decrease and training
top-1 on these eight queries validate only gradient flow and checkpoint
serialization. They are not held-out localization accuracy: candidate-region
generation, negative regions, validation retrieval, SAM-HQ refinement, the
independent manual acceptance set, `92%` accuracy, and complete `30 ms` timing
all remain pending.

Evaluate the saved smoke head on the independent official validation split:

```bash
OMP_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
conda run -n fashion-prd-312 \
  python -u scripts/evaluate_region_text_alignment.py \
  --split validation \
  --limit 8
```

This reports all-query and competitive-only Top-1 and exact-set-at-target-count
rates. A query is competitive only when its selected image has at least one
candidate region outside the ground-truth target set, preventing trivial
single-candidate cases from inflating the result. Dimension and language
breakdowns plus per-query audit rows are saved to `metrics.json` and
`cases.json`.

This first evaluator deliberately marks its candidate scope as
`selected_query_target_union_per_image`: it ranks only the source regions
referenced by the bounded query selection. It also marks full-image candidate
coverage and mask localization as false. The next experiment must construct
all same-image part/proposal negatives before these retrieval numbers can be
used for model selection; even that retrieval experiment remains separate from
manual Mask/Box localization acceptance and the PRD `92%` metric.

The initial 128-query experiment exposed a training/evaluation mismatch. A
global contrastive matrix treated semantically equivalent parts from different
images as negatives, even though localization selects among regions from the
current image. The training smoke now constructs an independent candidate pool
per source image and averages multi-positive losses only across images with at
least one genuine negative pair. Images whose selected candidates are all
targets are excluded from the contrastive loss and counted separately.

Checkpoints created by the corrected path record
`negative_scope: same_image`. Training metrics report both the same-image
negative-pair count used for gradients, the total global negative-pair count,
and their difference as the cross-image pair count that was excluded.
Initial/final Top-1 is also computed by the same per-image retrieval evaluator
used on validation, so the train and validation metrics now have matching
candidate semantics.

The earlier global-negative 128-query checkpoint remains experimental history.
On 128 validation expressions it improved competitive Top-1 from `33.33%` for
the eight-query head to `74.80%`, and competitive exact-set-at-target-count from
`29.27%` to `69.11%`. This confirms an alignment signal but is not the corrected
training objective, full candidate coverage, Mask localization, or PRD
acceptance evidence.

The corrected same-image checkpoint reached `73.17%` competitive Top-1 and
`66.67%` competitive exact-set on the same 128-expression validation prefix.
That is slightly below the global-negative checkpoint by two and three queries,
respectively. The result does not justify selecting either objective because a
query-count prefix can stop midway through one image and therefore omit valid
same-image Fashionpedia part candidates.

Both alignment scripts now accept mutually exclusive `--limit` and
`--image-limit` bounds. `--image-limit` consumes every generated expression for
each selected image before stopping. Because every valid Fashionpedia part Mask
has a basic bilingual expression, the union then covers all valid annotated
part Masks for those images. This is recorded as
`fashionpedia_annotated_part_candidate_coverage: true`, while broader full-image
and open-vocabulary proposal coverage remains false.

Run the next validation audit on ten complete images:

```bash
OMP_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
conda run -n fashion-prd-312 \
  python -u scripts/evaluate_region_text_alignment.py \
  --split validation \
  --image-limit 10 \
  --checkpoint outputs/localization/dinov2_bge_alignment_train128_same_image/alignment_head_smoke.pt \
  --output-dir outputs/localization/dinov2_bge_alignment_validation_images10
```

This image-complete Fashionpedia retrieval audit is stricter than the old query
prefix but still evaluates oracle source Masks as candidates. It does not test
whether an open-vocabulary proposal model finds unseen parts, nor whether
SAM-HQ produces the accepted Mask, so it remains below PRD acceptance.

On the same ten image-complete validation images, the earlier global-negative
checkpoint reached `76.70%` competitive Top-1 and `70.45%` competitive
exact-set, outperforming the same-image checkpoint by six and five queries.
Cross-image supervision is therefore useful, but treating a pocket in another
image as a negative for a basic pocket query is still semantically incorrect.

The next controlled objective is `--negative-scope label_aware_global`. It uses
all same-image pairs so left/right and attribute-modified instances can compete,
keeps cross-image pairs whose Fashionpedia target labels differ, and ignores
cross-image pairs with the same target label. The training audit separates
same-image, cross-image different-label, and excluded cross-image same-label
pair counts. `global` and `same_image` remain available for reproducible
ablation runs.

The label-aware checkpoint reached only `72.16%` competitive Top-1 and `67.05%`
competitive exact-set on the same image-complete development set. The global
checkpoint remains the empirical baseline at `76.70%` and `70.45%`. Its grouped
results show that appearance attributes are comparatively strong (`92.31%`
competitive Top-1), while spatial (`66.07%`), relation (`69.57%`), and Chinese
queries (`73.33%`) remain the main representation gaps.

### Explicit Spatial Reranking

The spatial baseline preserves the complete query and parses only an explicit
Chinese or English left/right/upper/lower modifier. It converts candidate box
centers to image-relative coordinates and adds a bounded directional prior to
the learned cosine score. Queries with no supported modifier, or combined
phrases such as "left and right cuffs", receive no adjustment. The default
weight was zero during selection; the frozen development value and explicit
zero-weight ablation are recorded below.

Use `--spatial-weight` only for Fashionpedia development experiments. The
evaluator reports the spatial-query count and spatial-only competitive Top-1
and exact-set metrics, and stores the parsed modifier with each case. This
weight must be frozen before the independent manual PRD acceptance set; tuning
on the acceptance set is not allowed. Explicit coordinates cover the current
image-frame expressions only. Reference-object and garment-layer relations
still require a separate relation model.

`--image-offset` can be combined with `--image-limit` to select a disjoint,
image-complete group after skipping the requested number of source images. It
cannot be combined with a query-count-only boundary. Spatial weight candidates
are chosen on the first development images, then compared against weight zero
on later images before being frozen. The saved metrics record the offset so the
two image groups remain auditable.

The `0.10` spatial weight was then frozen as the Fashionpedia development
baseline. On the disjoint next ten validation images it raised spatial
competitive Top-1 and exact-set from `83.33%` to `91.67%` (two additional
correct queries out of 24), while overall competitive Top-1 and exact-set rose
from `87.07%`/`84.48%` to `88.79%`/`86.21%`. The project alignment config now
stores `spatial_rerank_weight: 0.10`; `--spatial-weight 0` remains the explicit
ablation path. This is still Fashionpedia development evidence using oracle
part Masks, not the independent manual PRD acceptance result.

The next disjoint evaluation contains several hundred complete queries. BGE-M3
encoding therefore uses an explicit maximum batch size of `32` from
`configs/localization_bge_m3_text.yaml` instead of using the entire query set as
one CUDA batch. Smaller smokes still run as one batch. This is a memory-safety
bound for feature extraction, not a throughput or PRD latency optimization.

The later image-complete audit exposed valid Fashionpedia parts smaller than
one output pixel after whole-image letterboxing. Nearest-neighbor resizing could
otherwise delete such a Mask and abort evaluation. The DINOv2 preprocessing path
now preserves one centroid-mapped pixel only when a non-empty source Mask would
become empty; genuinely empty source Masks are still rejected. This keeps tiny
targets in the scored candidate set instead of silently excluding difficult
examples. It does not establish that the resulting patch feature is accurate.

### Class-Agnostic SAM-HQ Proposal Recall

The 300-image global alignment head is frozen as the current oracle-candidate
development checkpoint. On a previously unused 100-image validation group it
reached `92.32%` competitive Top-1 and `89.26%` competitive exact-set over
`1,406` queries. English and Chinese Top-1 were `92.43%` and `92.16%`, while
spatial Top-1 remained `88.78%`. These numbers assume that every official
Fashionpedia target Mask is already in the candidate bank, so they do not
establish PRD localization accuracy.

The next PRD-main-path gate replaces that oracle bank with class-agnostic
SAM-HQ automatic Masks. `configs/localization_sam_hq_proposals.yaml` starts in a
high-recall diagnostic mode with a 32-by-32 point grid, one crop layer, relaxed
quality thresholds, and at most 200 retained proposals per image. This mode is
deliberately an accuracy-ceiling experiment, not a latency configuration. Each
valid official part Mask remains in the denominator even when SAM-HQ returns no
overlapping proposal.

Run a two-image bounded smoke before expanding the proposal-recall audit:

```bash
bash scripts/setup_sam_hq_proposal_model.sh

OMP_NUM_THREADS=1 \
conda run -n fashion-prd-312 \
  python -u scripts/smoke_sam_hq_proposal_recall.py \
  --split validation \
  --image-limit 2 \
  --output-dir outputs/localization/sam_hq_proposal_recall_images2
```

The setup pins the official SAM-HQ source checkout to
`e696978d60352dc9a26b12631cd91781502c6546`, validates the already downloaded
ViT-B checkpoint against its published SHA256 checksum, and confirms that the
`segment_anything` automatic generator is importable in `fashion-prd-312`.
The official source declares `timm` as an optional all-feature dependency; this
project pins `timm==0.9.16` in the exact environment and the setup repairs a
missing installation before the import check.
Python import caches inside the external checkout are allowed as untracked
runtime files; tracked SAM-HQ source modifications still stop the setup before
the pinned checkout is changed.
SAM-HQ's `predicted_iou` is retained as a finite regression score and is not
clamped to `[0, 1]`; the official quality head can emit values slightly outside
that interval. `stability_score` remains strictly validated in `[0, 1]`.

The primary gate is `proposal_recall50`, accompanied by `proposal_recall75`
and all-GT mean best Mask IoU. This is independent best-proposal recall rather
than one-to-one query accuracy: it answers whether the proposal bank contains a
usable target before DINOv2/BGE-M3 ranking. Language selection, independent
manual acceptance, complete `30 ms` timing, and `60 QPS` remain unevaluated.

The first two-image run completed over five unique official part Masks. It
generated 236 automatic Masks but reached only `20%` proposal Recall50 and
Recall75, with all-GT mean best Mask IoU `27.14%`. The second-image warm runtime
was `4.54 s`. This rejects the automatic Mask generator as an online candidate
path: it is both below the required proposal ceiling and orders of magnitude
above the complete `30 ms` localization target. Expanding this exact setting to
more images would not resolve either failure.

Before designing the DINOv2 dense candidate path, isolate the downstream
refinement ceiling with exact Fashionpedia GT boxes:

```bash
OMP_NUM_THREADS=1 \
conda run -n fashion-prd-312 \
  python -u scripts/smoke_sam_hq_box_prompt_recall.py \
  --split validation \
  --image-limit 2 \
  --output-dir outputs/localization/sam_hq_box_prompt_recall_images2
```

This is deliberately an oracle diagnostic, not a PRD accuracy result. If its
`box_prompt_recall50` is high, proposal generation is the isolated bottleneck
and the next implementation should derive coarse boxes from aligned dense
DINOv2 patch features before SAM-HQ refinement. If it remains low even with
exact boxes, SAM-HQ input resolution, Box expansion, or the selected checkpoint
must be corrected before building the candidate path. In either case,
`prd_accuracy_92_passed` and `prd_localization_30ms_passed` remain `null`.

The first exact-Box run reached only `60%` Recall50/Recall75 over five targets,
with mean Mask IoU `63.19%` and `89.75 ms` warm latency. Both sleeve Masks and
one applique passed, while the two neckline Masks reached only `39.60%` and
`18.26%`. The successful applique occupies only `0.168%` of the image, smaller
than both failed necklines, so target area alone does not explain the misses.
SAM-HQ refinement ambiguity is now a second isolated bottleneck alongside
automatic proposal coverage.

Run one controlled ambiguity sweep on the same images. All Box variants for an
image are batched through one SAM-HQ image embedding:

```bash
OMP_NUM_THREADS=1 \
conda run -n fashion-prd-312 \
  python -u scripts/smoke_sam_hq_box_prompt_recall.py \
  --split validation \
  --image-limit 2 \
  --multimask-output \
  --box-expansion-ratios 0,0.10,0.20 \
  --output-dir outputs/localization/sam_hq_box_prompt_multimask_sweep_images2
```

The report separates score-selected Mask quality, which is usable at inference,
from oracle-best multimask quality, which only tests whether a correct candidate
exists. Improvement only in `oracle_best_recall50` means candidate selection
must be learned or redesigned; improvement from Box expansion identifies prompt
geometry as the issue. Failure in both columns rejects this ViT-B refinement
setting for neckline-like parts. Sweep timing contains three Box variants and
must not be compared directly with the single-variant `89.75 ms` measurement.

On the first sweep, the unexpanded multimask baseline raised Recall50 from
`60%` to `80%`, because one neckline reached `52.68%`, but the other remained at
`36.04%`. Score-selected and oracle-best metrics were identical for every Box
variant, so quality-score selection did not hide a better Mask. Ten-percent
expansion retained `80%` Recall50 while lowering mean IoU; twenty-percent
expansion reduced Recall50 to `40%`. Plain Box expansion is therefore rejected.

The next accuracy-only diagnostic changes image encoding resolution rather than
prompt geometry. Crop each exact Box with two times context before SAM-HQ:

```bash
OMP_NUM_THREADS=1 \
conda run -n fashion-prd-312 \
  python -u scripts/smoke_sam_hq_box_prompt_recall.py \
  --split validation \
  --image-limit 2 \
  --multimask-output \
  --roi-crop-scale 2.0 \
  --output-dir outputs/localization/sam_hq_roi_crop2_multimask_images2
```

ROI mode evaluates each Mask in crop coordinates while preserving the original
GT target area in `cases.json`. It re-encodes each target crop separately, so
its runtime is an accuracy ceiling and is not comparable with batched full-image
latency. If crop scale `2.0` still leaves a neckline below IoU `0.50`, test scale
`4.0`; do not tune more scales on these same five development targets. A useful
scale must then be frozen and verified on a disjoint image-complete group.

Crop scale `2.0` reduced Recall50 to `60%` and mean IoU to `58.43%`; crop scale
`4.0` recovered `80%` Recall50 but reached only `63.27%` mean IoU, still below
the `69.33%` full-image multimask baseline. The difficult neckline fell to
`4.20%` and `2.35%` respectively. ROI crop/upscale is therefore rejected, and
no further crop scales are tuned on this development sample.

The remaining prompt-level ambiguity diagnostic adds one positive point chosen
at the maximum interior distance of each official GT Mask. This is still an
oracle ceiling, but it matches a prompt that a future dense DINOv2 heatmap could
produce without introducing another model family:

```bash
OMP_NUM_THREADS=1 \
conda run -n fashion-prd-312 \
  python -u scripts/smoke_sam_hq_box_prompt_recall.py \
  --split validation \
  --image-limit 2 \
  --multimask-output \
  --oracle-positive-point \
  --output-dir outputs/localization/sam_hq_box_point_multimask_images2
```

If the same difficult neckline remains below IoU `0.50` even with an exact Box
and an interior foreground point, further prompt heuristics on these five
targets stop. The current SAM-HQ ViT-B refinement setting then lacks the Mask
ceiling required by PRD 3.1.2 and must be reconsidered before candidate-path or
latency optimization.

The positive-point run also failed the stop gate: Recall50 was `60%`, Recall75
was `40%`, and the two neckline Masks were `43.30%` and `46.76%`. Exact Box,
multimask output, an interior foreground point, Box expansion, and ROI crop have
now all been exhausted on this development group. No additional prompt
heuristics are tuned on these five targets.

The final model-setting comparison uses a disjoint ten-image group. The default
SAM-HQ combination output and official HQ-token-only output share the same
pinned ViT-B checkpoint and all other settings. The separate config makes the
ablation explicit in every saved `metrics.json`:

```bash
for MODE in combined hq_only; do
  if [ "$MODE" = combined ]; then
    CONFIG=configs/localization_sam_hq_proposals.yaml
  else
    CONFIG=configs/localization_sam_hq_proposals_hq_only.yaml
  fi

  OMP_NUM_THREADS=1 \
  conda run -n fashion-prd-312 \
    python -u scripts/smoke_sam_hq_box_prompt_recall.py \
    --split validation \
    --image-offset 2 \
    --image-limit 10 \
    --multimask-output \
    --config "$CONFIG" \
    --output-dir "outputs/localization/sam_hq_disjoint10_${MODE}"
done
```

This comparison uses exact GT boxes and remains an oracle Mask-ceiling test.
Select a setting only if it improves overall Recall50 and mean IoU without a
systematic category regression. It cannot establish `92%` language-localization
accuracy or `30 ms` latency. If both remain materially below the required Mask
ceiling, current SAM-HQ ViT-B is rejected and the next decision is a PRD-stack
checkpoint/architecture review, not more prompt tuning.

The disjoint comparison completed over 41 official part Masks. The default
combined SAM-HQ output reached `68.29%` Recall50, `29.27%` Recall75, and
`56.46%` mean Mask IoU. HQ-token-only fell to `63.41%`, `24.39%`, and `50.93%`
respectively, with the same `79 ms` warm runtime. The oracle-best and
score-selected Recall50 values were identical, so multimask ranking did not
hide better candidates. Combined output remains the stronger configuration,
but an exact-GT-Box ceiling of `68.29%` rejects this SAM-HQ ViT-B setting as the
sole producer of final local-region Masks. Neckline (`25%` Recall50 over eight
targets) and the available zipper target (`0%`) are the clearest failures.

### Full-Image DINOv2 Dense Mask Baseline

The PRD requires language-conditioned local-region `Mask + Box` output and
names DINOv2 region/text similarity as the localization mechanism. It includes
SAM-HQ in the required computer-vision stack, but does not require every final
Mask to be generated solely by SAM-HQ. After the exact-Box ceiling failure, the
next bounded baseline therefore converts the required DINOv2/text similarity
field directly into a full-image coarse Mask and derives its tight Box. This
does not introduce a model outside the PRD stack.

For each source image, the smoke encodes the complete image once into a
`37 x 37` normalized DINOv2 patch grid. It projects every complete BGE-M3 query
with the frozen 300-image alignment head, computes patch-to-query cosine
similarity, removes letterbox padding, and restores the similarity map to the
source image. Fixed score quantiles form Mask candidates without Fashionpedia
candidate Masks, GT boxes, category lookup, or a fixed-part classifier. The
development sweep retains every failed query in the Mask/Box denominator.

Run the first bounded threshold scan on AutoDL:

```bash
OMP_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
conda run -n fashion-prd-312 \
  python -u scripts/smoke_dinov2_dense_localization.py \
  --split validation \
  --image-limit 2 \
  --checkpoint \
    outputs/localization/dinov2_bge_alignment_train_images300_global/alignment_head_smoke.pt \
  --output-dir outputs/localization/dinov2_dense_localization_images2
```

`metrics.json` reports Mask Recall50/Recall75, mean Mask IoU, Box Recall50,
explicit numerators, startup timing, DINOv2 image-encoding time, and dense
scoring time for every quantile. Select at most one quantile by Mask Recall50,
then mean Mask IoU, freeze it, and verify it on a disjoint image-complete group.
The initial two-image sweep is threshold-development evidence, not the manual
PRD acceptance set, so both PRD pass flags remain `null` regardless of its
score.

If the dense coarse Mask passes the bounded coverage gate, combined SAM-HQ may
be evaluated only as an optional boundary refinement against that frozen coarse
Mask; it must be retained only when it improves the disjoint Mask result.
Mask2Former remains an auxiliary known-part/proposal path. The complete
production request, ONNX/TensorRT optimization, independent manual `92%`
acceptance, and end-to-end `30 ms` timing remain separate later gates.

The raw dense sweep completed over 24 full queries from the first two validation
images. Its best Mask result was quantile `0.95`: `16.67%` Recall50, `0%`
Recall75, and `12.59%` mean Mask IoU. Quantile `0.98` reached the best Box
Recall50 at `29.17%`, but no Mask passed IoU `0.50`. More quantile tuning is
stopped. The existing projection was trained only against mean-pooled oracle
region features, so applying it independently to every patch does not provide
foreground/background or calibrated area supervision.

### Supervised DINOv2 Patch Alignment

The correction stays inside the same PRD mechanism. DINOv2 and BGE-M3 remain
frozen; the existing 300-image text projection initializes training. For each
complete query, its Fashionpedia target Mask is letterboxed with the image and
converted into soft foreground fractions on the `37 x 37` DINOv2 patch grid.
Only the text projection, cosine logit scale, and foreground bias are optimized
with foreground-balanced BCE plus soft Dice loss. This remains region/text
similarity matching and does not add a replacement foundation model or fixed
part classifier.

Train the first image-complete patch head on 100 Fashionpedia training images:

```bash
TRAIN_DIR=outputs/localization/dinov2_dense_patch_alignment_train_images100
mkdir -p "$TRAIN_DIR"

nohup env \
  OMP_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  conda run -n fashion-prd-312 \
  python -u scripts/train_dense_patch_alignment.py \
  --split train \
  --image-limit 100 \
  --steps 300 \
  --initial-checkpoint \
    outputs/localization/dinov2_bge_alignment_train_images300_global/alignment_head_smoke.pt \
  --output-dir "$TRAIN_DIR" \
  > "$TRAIN_DIR/run.log" 2>&1 &
```

Training reports only patch-grid metrics on the training set. They demonstrate
optimization behavior but cannot establish full-image localization accuracy.
The saved checkpoint freezes a learned `0.5` probability threshold; validation
does not scan quantiles or use target areas.

Evaluate that fixed checkpoint on the same two-image architecture-debug set:

```bash
EVAL_DIR=outputs/localization/dinov2_dense_patch_localization_images2
mkdir -p "$EVAL_DIR"

OMP_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
conda run -n fashion-prd-312 \
  python -u scripts/evaluate_dense_patch_localization.py \
  --split validation \
  --image-limit 2 \
  --checkpoint \
    "$TRAIN_DIR/dense_patch_alignment.pt" \
  --output-dir "$EVAL_DIR"
```

The evaluation restores calibrated patch probabilities to source-image
coordinates and produces the final coarse Mask and tight Box directly. It
reports overall and dimension/language/category metrics, retains empty and poor
predictions as misses, and separates first-image model loading from warm image
time. This two-image set has already been used for architecture diagnostics, so
its pass flags remain `null`. Its immediate gate is improvement over the raw
`16.67%` Mask Recall50 baseline at the fixed threshold. A successful result is
then frozen and evaluated on a larger disjoint image-complete set before any
SAM-HQ refinement comparison.
