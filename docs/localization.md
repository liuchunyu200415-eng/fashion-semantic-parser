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
Because foreground-balanced BCE deliberately changes the foreground prior,
`0.5` is only the default probability boundary, not a calibrated target-area
threshold. After optimization, the trainer evaluates a committed list of
thresholds using training patch Masks only. It freezes the value with the
highest training patch Recall50, then mean patch IoU, then the tighter threshold
as deterministic tie-breakers. The selected threshold and every candidate
metric are stored in the checkpoint and `metrics.json`; validation does not
scan thresholds or use target areas.

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

The first uncalibrated `0.5` evaluation did not pass that gate. Training patch
Recall50 rose from `0.21%` to `26.04%` and mean patch IoU from `5.08%` to
`35.46%`, confirming learnable foreground signal. On the 24 validation queries,
however, Mask Recall50 was `12.50%`, mean Mask IoU `21.95%`, and Box Recall50
`25%`. Every category was over-segmented: predicted-to-GT area ranged from
`1.70x` to `4.11x` for sleeves, `5.82x` to `8.54x` for applique, and `11.73x`
to `29.38x` for necklines. This is consistent with an uncalibrated threshold
under foreground-balanced loss. More optimization steps, validation threshold
scans, and connected-component rules are stopped until the training-only
threshold calibration is rerun and independently evaluated.

Training-only calibration selected probability `0.85` and raised the reused
two-image Mask Recall50 to `41.67%`, but this did not generalize. On the next 50
validation images (`754` queries), the 100-image model reached only `9.68%`
Mask Recall50, `0.40%` Recall75, `19.81%` mean Mask IoU, and `25.73%` Box
Recall50. Increasing training coverage from 100 to 300 images while holding the
architecture and optimization steps fixed produced `9.42%`, `0.80%`, `20.27%`,
and `26.26%` respectively. Data scaling on the independent-cosine architecture
is therefore stopped: it did not improve Mask Recall50 and cannot justify a
1,000-image run.

### Query-Conditioned Multiscale Patch Decoder

The next controlled architecture keeps the PRD encoders frozen and adds no new
foundation model. A lightweight PyTorch decoder receives normalized DINOv2
patch features, the projected complete-query vector, their elementwise
interaction and cosine score, plus normalized image-frame coordinates. Dilated
`3 x 3` branches at rates `1`, `2`, and `4` add local and wider spatial context
before producing one logit per patch. This explicitly addresses the two
rejected cosine-path limitations: independent patch scoring and the absence of
coordinates for spatial expressions.

The same soft Fashionpedia patch targets, balanced BCE/Dice loss, training-only
threshold calibration, full-query preservation, and all-miss evaluation remain
unchanged. Checkpoint schema two records `model_type: multiscale_decoder` and
strictly restores the decoder state. Schema-one cosine checkpoints remain
loadable for reproducible comparisons.

Run a 20-image/20-step compatibility smoke before the 300-image comparison:

```bash
SMOKE_DIR=outputs/localization/dinov2_multiscale_decoder_smoke20
mkdir -p "$SMOKE_DIR"

OMP_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
conda run --no-capture-output -n fashion-prd-312 \
  python -u scripts/train_dense_patch_alignment.py \
  --split train \
  --image-limit 20 \
  --steps 20 \
  --model-type multiscale_decoder \
  --initial-checkpoint \
    outputs/localization/dinov2_bge_alignment_train_images300_global/alignment_head_smoke.pt \
  --output-dir "$SMOKE_DIR"
```

Only after the smoke emits a schema-two checkpoint and decreasing finite loss,
train 300 images for 500 steps. Evaluate it unchanged on the same 50-image
development group at validation offset two. The decoder must materially exceed
the frozen cosine reference (`9.42%` Mask Recall50 and `20.27%` mean Mask IoU)
before any larger data run, SAM-HQ comparison, or latency optimization.

The 20-image/20-step compatibility smoke completed successfully: loss decreased
from `1.6608` to `1.2682`, schema-two save/load worked, and the evaluator ran
without missing-query exclusions. Its accuracy was not used for model
selection. The subsequent 300-image/500-step comparison selected probability
`0.70` on the training patch grid and reached `67.83%` training patch Recall50.
On the frozen 50-image development group (`754` queries), the multiscale
decoder reached `21.35%` Mask Recall50, `2.52%` Recall75, `24.61%` mean Mask
IoU, and `36.74%` Box Recall50. This materially exceeds the same-data cosine
reference (`9.42%`, `0.80%`, `20.27%`, and `26.26%` respectively), so the
multiscale decoder is retained and the independent-cosine path remains frozen.

The result establishes an architecture improvement only. It remains far below
the PRD `92%` acceptance target, uses Fashionpedia development supervision, and
does not establish complete-request `30 ms` latency. The next controlled scale
run expands training coverage to 1,000 images while preserving the decoder,
loss, frozen base encoders, training-only threshold calibration, and the same
50-image evaluation group. SAM-HQ refinement remains blocked until coarse Mask
coverage is substantially higher.

The 1,000-image/1,500-step scale run selected probability `0.85` on `16,064`
training queries and reached `69.93%` training patch Recall50. On the
model-selection group at validation offset two, it improved to `25.73%` Mask
Recall50, `1.99%` Recall75, `28.81%` mean Mask IoU, and `41.38%` Box Recall50.
The checkpoint and threshold were then frozen before evaluation on the next 50
previously unused validation images at offset 52 (`762` queries). That unseen
group reached `35.43%` Mask Recall50, `6.17%` Recall75, `34.16%` mean Mask IoU,
and `51.71%` Box Recall50. Differences in image/query composition mean the two
50-image groups are not a paired score comparison, but the unseen result
confirms that the multiscale improvement is not confined to the selection
group.

This 1,000-image multiscale checkpoint is the current coarse-Mask development
baseline. Further blind data scaling is paused. The next analysis separates
basic, spatial, attribute, relation, language, and part-category results on the
unused group before changing architecture. The result is still not the
independent manually annotated PRD acceptance set, is far below `92%`, and does
not establish `30 ms` complete-request latency.

The grouped audit shows that language parsing is not the primary limiting
factor. Attribute and spatial queries reached `41.91%` and `44.51%` Mask
Recall50, while English and Chinese reached `36.53%` and `33.87%`. In contrast,
small local parts were systematically over-segmented: rivet, zipper, and
neckline median predicted-to-target area ratios were `217.24x`, `21.12x`, and
`8.49x`; their Recall50 values were `0%`, `0%`, and `2.72%`. Pocket had `98`
queries but still reached `0%` Recall50 with a `2.19x` median area ratio. This
rules out missing examples alone and supports query-conditioned area control
before more data or spatial rules.

### Query-Conditioned Target-Area Control

Checkpoint schema three adds a lightweight area predictor alongside the
multiscale decoder. It consumes the normalized complete-query feature, the
global full-image DINOv2 feature, and their elementwise interaction. It predicts
a continuous foreground fraction without a category lookup or per-part
threshold. At inference, the multiscale decoder still determines location, but
only the query-specific highest-scoring top-k patches implied by the predicted
area are retained. This preserves arbitrary language descriptions while
directly targeting the observed over-segmentation failure.

The area predictor is supervised only on the training Mask fraction. Evaluation
never uses the target area, does not scan thresholds, and records
`mask_selection_mode: query_area_topk`, predicted area fraction, and selected
patch count for every retained query. Schema-one and schema-two checkpoints
remain strictly loadable.

Run the compatibility smoke first:

```bash
SMOKE_DIR=outputs/localization/dinov2_multiscale_area_smoke20
mkdir -p "$SMOKE_DIR"

nohup env \
  OMP_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  conda run --no-capture-output -n fashion-prd-312 \
  python -u scripts/train_dense_patch_alignment.py \
  --split train \
  --image-limit 20 \
  --steps 20 \
  --model-type multiscale_area_decoder \
  --initial-checkpoint \
    outputs/localization/dinov2_bge_alignment_train_images300_global/alignment_head_smoke.pt \
  --output-dir "$SMOKE_DIR" \
  > "$SMOKE_DIR/run.log" 2>&1 &

TRAIN_PID=$!
tail --pid="$TRAIN_PID" -F "$SMOKE_DIR/run.log" |
  grep --line-buffered -E \
  'dataset_ready:|^\[train|mask_selection_ready:|checkpoint_path:|Traceback|RuntimeError|CUDA out'
wait "$TRAIN_PID"
echo "train_exit_code=$?"
test -s "$SMOKE_DIR/dense_patch_alignment.pt" && echo "checkpoint_ready"
```

The smoke gate is a finite decreasing loss, a schema-three checkpoint, and a
successful two-image evaluation with `query_area_topk`; its accuracy is not a
model-selection result. If it passes, train the same architecture on 300 images
for 500 steps and compare on the frozen offset-two 50-image group. Continue to
1,000 images only if it improves both Mask Recall50 and mean Mask IoU over the
300-image multiscale reference (`21.35%` and `24.61%`).

The schema-three compatibility smoke passed, but the controlled 300-image run
failed the continuation gate. On the same offset-two group it reached only
`9.42%` Mask Recall50, `0%` Recall75, `20.90%` mean Mask IoU, and `24.14%` Box
Recall50. The corresponding schema-two multiscale model reached `21.35%`,
`2.52%`, `24.61%`, and `36.74%`. Scaling the area model to 1,000 images is
therefore stopped.

The evaluator now reports two explicitly GT-dependent oracle diagnostics before
changing the loss. `oracle_pixel_area_topk` selects the decoder's best patches
using the true soft foreground area, while `oracle_support_topk` uses the true
number of touched patches. Neither is an inference result or PRD acceptance
metric. If the pixel oracle fails but support succeeds, full-patch selection is
too coarse for thin parts; if both succeed, the learned area head is the primary
failure; if both fail, decoder ranking/localization remains the bottleneck.

On the 754-query development group, learned-area selection reached `9.42%`
Recall50 and `20.90%` mean IoU. The GT pixel-area oracle improved to only
`21.09%` and `27.38%`; the GT patch-support oracle reached `8.75%` and
`23.36%`. Median predicted-to-target pixel fraction was already `1.14x`.
Therefore better area calibration alone cannot exceed the schema-two baseline
(`21.35%` Recall50), and the area-control route is frozen. Oracle computation
also raised warm image time from about `0.134 s` to `0.326 s`; that diagnostic
CPU overhead is not an inference latency result.

The next resolution-only feasibility check keeps the frozen 1,000-image
schema-two checkpoint and all PRD models unchanged, but evaluates DINOv2 at
`728 x 728` (`52 x 52` patches) instead of `518 x 518` (`37 x 37`). The
fully-convolutional decoder can consume either grid. Its inference Mask does not
retrain, scan a threshold, or use oracle area; separate GT diagnostics remain
reported but do not affect the prediction. The test cannot claim acceptance and determines
whether added patch density improves small-part localization enough to justify a
coarse-to-fine crop implementation. Use
`configs/localization_dinov2_region_728.yaml` through the evaluator's
`--dinov2-config` option, first on two images and then on the frozen 50-image
group only if checkpoint restoration succeeds.

The first 50-image resolution comparison was rejected before model selection.
Its `518` result drifted from the frozen historical checkpoint result because
the schema-three area integration had inadvertently changed schema-one/two Mask
restoration from “continuous probability upsample, then threshold” to “patch
threshold, then binary upsample.” The evaluator now restores the original
continuous-probability order for schema one/two and retains binary top-k
restoration only for schema three. A regression test freezes this compatibility
contract. The rejected `19.89%` versus `23.47%` Recall50 pair is not evidence for
high-resolution training; both resolutions must be rerun after the fix.

After restoring the original probability interpolation order, the paired
754-query rerun recovered the frozen `518` result exactly: `25.73%` Recall50,
`1.99%` Recall75, `28.81%` mean IoU, and `41.38%` Box Recall50. Without
retraining, `728` reached `27.72%`, `5.97%`, `31.62%`, and `38.33%`
respectively. Higher patch density therefore improves Mask overlap and Recall75
but slightly weakens Box recall; its target-discovery improvement remains small.

The next controlled run trains the same decoder on 300 images and 500 steps at
`728`, using `--dinov2-config configs/localization_dinov2_region_728.yaml`, then
evaluates at `728` on the same offset-two group. Checkpoints now record
`dinov2_input_size`; older checkpoints default to their historical `518` value
when loaded. Continue to a 1,000-image high-resolution run only if same-resolution
training materially improves both Recall50 and mean IoU without a further Box
recall collapse. Otherwise stop full-image resolution scaling and implement
query-conditioned coarse-to-fine crops.

The 300-image/500-step `728` run passed that controlled gate. On the same 754
queries it reached `26.13%` Mask Recall50, `2.25%` Recall75, `29.37%` mean Mask
IoU, and `39.39%` Box Recall50. Relative to the same-data `518` decoder
(`21.35%`, `2.52%`, `24.61%`, and `36.74%`), Recall50 and mean IoU improved by
`4.78` and `4.76` percentage points while Box Recall50 improved by `2.65`;
Recall75 was approximately flat. The GT pixel-area oracle reached only `29.97%`
Recall50, so decoder ranking still imposes a low ceiling. One 1,000-image/1,500-
step high-resolution run is allowed as the final full-image scale test. If it
does not materially exceed both the frozen 1,000-image `518` checkpoint
evaluated at `728` (`27.72%` Recall50, `31.62%` mean IoU) and the high-resolution
300-image result, further full-image data/resolution scaling stops in favor of
query-conditioned coarse-to-fine local re-encoding.

The final full-image scale run passed its gate. Training on 1,000 images and
16,064 queries at `728` reached `69.65%` training patch Recall50 and `57.66%`
mean patch IoU. On the frozen offset-two group, the checkpoint reached `30.90%`
Mask Recall50, `6.63%` Recall75, `32.47%` mean Mask IoU, and `42.84%` Box
Recall50. This exceeds both the 300-image `728` checkpoint (`26.13%`, `2.25%`,
`29.37%`, `39.39%`) and the 1,000-image `518` checkpoint evaluated at `728`
(`27.72%`, `5.97%`, `31.62%`, `38.33%`). The GT pixel-area oracle reached only
`32.49%` Recall50, just `1.59` percentage points above inference, so area
calibration no longer contains substantial recoverable recall on this group.

This checkpoint is frozen as the final full-image development model. More data,
steps, target-area tuning, and full-image resolution growth stop. Before a
coarse-to-fine architecture change, evaluate the frozen checkpoint without
threshold changes on the previously unused validation offset-52 group. That
result tests generalization only; it is still not the independent manually
annotated PRD acceptance set and cannot establish the `92%` requirement or
complete-request `30 ms` latency.

The frozen checkpoint generalized on the previously unused offset-52 group
(`762` queries): `41.60%` Mask Recall50, `9.19%` Recall75, `38.25%` mean Mask
IoU, and `55.64%` Box Recall50. Relative to the earlier 1,000-image `518`
checkpoint on the same group (`35.43%`, `6.17%`, `34.16%`, and `51.71%`), the
high-resolution trained model improved every localization metric. The GT
pixel-area oracle reached `43.70%` Recall50, only `2.10` percentage points above
inference, again confirming that target-area calibration is not the principal
remaining limitation. Warm image time was `0.361 s`, but this run includes two
GT-dependent oracle restorations per query and is not a valid production latency
measurement. The model generalizes across held-out images, but remains far below
PRD acceptance; full-image scaling is now frozen and the next architecture must
improve query-to-location ranking through local re-encoding.

The held-out diagnostic confirms that spatial (`50.55%` Recall50) and attribute
queries (`49.26%`) are not the primary failure. Sleeve reached `88.42%` Recall50
with a `0.97x` median predicted-to-target area ratio. In contrast, rivet,
ruffle, zipper, epaulette, and neckline reached `0%`, `0%`, `0%`, `0%`, and
`2.72%`; their median area ratios were `235.01x`, `10.53x`, `9.67x`, `10.49x`,
and `6.27x`. Relation queries reached `38.24%`, so language-relation improvement
remains useful but cannot explain the severe small-part failures.

Before implementing or training local re-encoding, run the category-free coarse
crop audit in `scripts/audit_dense_coarse_crop_coverage.py`. It selects up to
three distinct peaks from the complete-query probability grid and forms fixed
image-relative crops; generation receives neither part labels nor GT. GT is used
only afterward to report target-pixel coverage at `50%`/`90%` and crop-union
image area. Crop fractions `0.20`, `0.30`, and `0.40` measure the trade-off
between recovering small targets and approaching a trivial full-image crop. A
coarse-to-fine implementation is justified only if one or three query-driven
crops cover at least `90%` of targets at a materially smaller image-area cost.

On the unseen offset-52 group, the audit retained all `762` queries. Three
query-driven `20%` crops covered at least `90%` of target pixels for `89.37%`
of queries while occupying `15.27%` of image area. At `30%`, coverage rose to
`97.24%` at `31.44%` area; at `40%`, it reached `100%` but consumed `53.48%`.
The fixed first coarse-to-fine baseline therefore uses `30%` and Top-3: it is
the smallest audited setting with near-complete target coverage.

`scripts/evaluate_dense_local_reencoding.py` evaluates that baseline without
training or GT/category input. For each complete language query it selects the
three coarse peaks, crops the source image, re-encodes every crop with the same
PRD DINOv2 `728` encoder, scores it with the frozen BGE-M3 projection and
multiscale decoder, then restores local probabilities to source coordinates.
It reports three fixed branches at the checkpoint's frozen threshold:
`coarse`, `local_only`, and pixelwise `coarse_local_max`. Predicted-to-target
area ratios remain diagnostics; target Masks never affect crop generation,
local scoring, fusion, or threshold selection. Run two images first. Continue
to the frozen 50-image group only if local re-encoding materially improves Mask
Recall50 or mean Mask IoU without uncontrolled foreground growth. The recorded
offline per-image time is not complete-request latency and cannot be compared
with the PRD `30 ms` requirement.

The fixed offset-two 50-image evaluation retained all `754` queries and passed
the continuation gate. Compared with the unchanged coarse branch, `local_only`
improved Mask Recall50 from `30.90%` to `35.15%`, Recall75 from `6.63%` to
`11.54%`, and mean Mask IoU from `32.47%` to `38.88%`. Its median
predicted-to-target area ratio fell from `1.96x` to `1.29x`, so the gain is not
caused by broader foreground. Box Recall50 fell from `42.84%` to `35.15%`,
which must remain an explicit secondary regression. Pixelwise
`coarse_local_max` reached only `31.17%` Recall50 and enlarged the median area
ratio to `2.28x`; that fusion branch is rejected. The frozen `local_only`
branch now proceeds unchanged to the offset-52 held-out group. That evaluation
tests image generalization only and still cannot establish PRD acceptance.

The unchanged `local_only` branch also generalized on the offset-52 held-out
group (`762` queries). Relative to `coarse`, Mask Recall50 improved from
`41.60%` to `43.44%`, Recall75 from `9.19%` to `18.64%`, and mean Mask IoU from
`38.25%` to `43.43%`; the median predicted-to-target area ratio decreased from
`1.61x` to `1.14x`. This reproduces the development-group Mask-quality gain
without threshold or crop changes, so `30%` Top-3 `local_only` is frozen as the
current coarse-to-fine Mask path. Box Recall50 regressed from `55.64%` to
`41.99%`; therefore the coarse localization Box must remain the auxiliary Box
output instead of deriving it from the local Mask. `coarse_local_max` is
rejected: it did not improve Recall50 and increased the area ratio to `1.96x`.
These Fashionpedia generated-query results establish architecture progress,
not the PRD `92%` acceptance result. Independent manual referring-expression
evaluation and complete-request latency measurement remain required.

### Production Dense Local-Reencoding Service

The application `localization.backend` now defaults to
`dense_local_reencoding`. The runtime receives the complete query verbatim,
uses the frozen BGE-M3 projection and DINOv2 `728` multiscale checkpoint,
selects three `30%` coarse crops, and returns the restored `local_only` Mask.
It does not invoke `resolve_localization_prompt` inside the model path and does
not map expressions to Fashionpedia or PRD part classes. Explicit whole-image
garment inventory/classification questions remain on the 3.1.1 route; unknown
local expressions, including attribute and relation compositions, reach the
open-query backend unchanged.

The default dense service runs on the complete image, matching both frozen
development and held-out evaluations. It does not reuse an automatically
detected 3.1.1 subject ROI. An explicitly supplied manual ROI remains available
as a separate, unvalidated client-controlled mode and is reported as such.

The response records `mask_source: dense_local_reencoding` and
`box_source: dense_coarse_localization`. The Box deliberately remains the
coarse prediction because held-out Box Recall50 was `55.64%` for coarse versus
`41.99%` for the local Mask's tight Box. This mixed output is an evidence-based
contract, not an implicit recomputation. The service returns one query result
whose polygon list can contain multiple disconnected target components.

On AutoDL, link the already frozen checkpoint into the deployment path before
starting the API:

```bash
bash scripts/setup_dense_local_reencoding_model.sh
```

The setup script refuses to overwrite a regular checkpoint file and validates
the resulting link. DINOv2 and BGE-M3 assets are still pinned and validated by
their existing setup/runtime checks. This production wiring does not change the
measured Fashionpedia scores, establish the PRD `92%` metric, or establish the
complete-request `30 ms` requirement. API acceptance, independent manual
accuracy evaluation, and latency benchmarking remain separate gates.

Render one actual complete-query result as an auditable Original / Localized
comparison. This command runs the deployed backend rather than drawing a
hand-authored example:

```bash
python scripts/visualize_localization_prediction.py \
  data/raw/fashionpedia/test/0229bef01efc25f915374d55f59cbfdd.jpg \
  --query "the sleeve on the left side of the garment" \
  --output outputs/localization/dense_local_visual/left_sleeve.png \
  --json-output outputs/localization/dense_local_visual/left_sleeve.json
```

The left panel preserves the source image. The right panel overlays every
returned Mask polygon and the independently sourced coarse Box. The saved JSON
retains the complete query, confidence, Mask source, and Box source. A visible
region proves only that this request returned a spatial result; success against
the requested target still requires GT comparison or manual review.

The first complete-service latency benchmark passed functional inference but
failed the PRD performance target. Across collar, sleeve, and zipper queries,
warm mean latency was `105.87-112.77 ms` and P95 was `107.33-115.85 ms`, versus
the required `30 ms`. The result includes image decode, complete-query BGE-M3,
one coarse DINOv2 pass, three sequential local DINOv2 passes, scoring, and
polygon/Box postprocessing; model loading and HTTP transport are excluded.
The next optimization batches the three fixed local crops into one DINOv2
forward pass without changing the model, query, crops, threshold, or output.

The repeated benchmark after batched crop encoding still failed the target:
warm mean latency was `102.16-106.54 ms` and P95 was `107.33-113.14 ms`.
This is not a material improvement over the sequential baseline, so batching
alone is not an accepted optimization. The benchmark supports an explicit
`--profile-stages` diagnostic mode that reports non-overlapping image decode,
query projection, coarse DINOv2, crop preparation, batched local DINOv2,
scoring/restoration, and polygon/schema times. Diagnostic CUDA boundaries are
synchronized for attribution; this mode is excluded from PRD latency
acceptance and does not add synchronization to the normal service path.

The synchronized profile attributes `99.80-103.12 ms` warm means as follows:
query projection `18.91-19.09 ms`, coarse DINOv2 `16.53-17.19 ms`, and batched
local DINOv2 `40.14-41.99 ms`. Those three model stages alone consume roughly
`76-78 ms`; image decode, score restoration, and output construction cannot
close the gap to `30 ms`. The next deployment gate is therefore the PRD-listed
ONNX Runtime `1.17` and TensorRT `8.6.1` path, followed by numerical-parity,
accuracy, and complete latency reruns. Repeated-query or repeated-image caches
must not be used to claim cold unique-request latency compliance.

Before exporting models, audit the active AutoDL runtime rather than assuming
that an installed package exposes CUDA execution:

```bash
python scripts/check_prd_312_deployment_env.py
```

The audit requires Python `3.10.12`, an active RTX 3090, ONNX Runtime `1.17.x`
with both `CUDAExecutionProvider` and `TensorrtExecutionProvider`, and TensorRT
`8.6.1.x` with a working native Builder. It checks only environment readiness
and cannot establish engine conversion, numerical parity, `92%`, `30 ms`, or
`60 QPS` compliance.

For the isolated AutoDL environment, install the exact CUDA 12 packages with:

```bash
bash scripts/setup_prd_312_deployment_env.sh
```

The setup pins ONNX Runtime GPU `1.17.1` from Microsoft's CUDA 12 feed and
TensorRT `8.6.1.post1` from NVIDIA's package index. Every pip operation disables
the download cache because the AutoDL system disk is constrained. The final
step runs the strict audit above and fails if pip resolved a CPU, CUDA 11, or
non-8.6.1 runtime.

The historical TensorRT metapackage has unbounded CUDA-library dependencies on
the current package index. Installing it with dependency resolution can pull a
newer CUDA 12.9/cuDNN 9 stack even though the TensorRT module itself remains
8.6.1. The setup therefore installs the three TensorRT 8.6.1 modules without
dependencies, then restores the CUDA 12.1/cuDNN 8.9 package versions pinned by
PyTorch 2.1.2. The audit runs a real CUDA tensor operation and rejects any
remaining package-version drift before model export.

### DINOv2 ONNX and TensorRT Parity Gate

After the deployment environment passes, export only the frozen DINOv2
patch-token boundary first:

```bash
python scripts/export_dinov2_onnx.py
```

The artifact fixes the spatial input at `728x728`, retains a dynamic batch range
of `1-3` for the full image and three local crops, uses ONNX opset `17`, and
returns normalized `384`-dimensional patch tokens. Validation compares batch 1
and batch 3 against PyTorch, separately checks CUDA EP FP32 and TensorRT EP FP16,
and parses the ORT profile to prove that TensorRT actually executed a graph
partition. Merely listing `TensorrtExecutionProvider` is not a pass. This gate
does not yet replace the PyTorch service and does not claim complete-request
accuracy or latency compliance.

### Locked PRD 3.1.2 Accuracy Contract

Final accuracy uses a separate acceptance scope from the earlier feasibility,
candidate-ranking, Box, Top-k, and oracle diagnostics. The fixed query-level
definition is:

```text
one complete natural-language query -> first returned region only
success = MaskIoU(top1_mask, query_target_mask) > 0.50
accuracy = successful_query_count / all_reviewed_query_count
required accuracy = 0.92
```

For a query that refers to multiple target instances, the product and project
owners must choose either one query-level union Mask or exclusion from the
acceptance set. The implementation must not choose that policy implicitly.
Likewise, the mutually exclusive primary counts for basic, spatial, attribute,
and relation queries, plus the orthogonal novelty and language counts, must be
approved before model optimization is judged against `92%`.

The versioned draft is
`configs/prd_312_acceptance_contract.json`. Audit it with:

```bash
python scripts/check_prd_312_acceptance_contract.py
```

Exit code `1` is expected while proportions, multi-target handling, or either
owner approval remains unresolved. Setting `status` to `locked` without filling
every decision is rejected by schema validation. Once approved, embed the
locked contract in a reviewed `Prd312AcceptanceManifest` and evaluate saved
responses with:

```bash
python scripts/evaluate_prd_312_acceptance.py \
  --manifest data/benchmarks/localization/prd_312_acceptance_v1.json \
  --responses-dir outputs/localization/prd_312_acceptance/responses
```

The evaluator requires exact response coverage, scores only `regions[0]`,
unions multi-instance GT only when the locked policy permits it, treats exactly
`0.50` IoU as a failure, and retains empty predictions and inference errors in
the denominator. It reports overall accuracy and separate primary-dimension,
all-dimension, novelty, language, and target-label breakdowns. The older
`evaluate_referring_localization.py` remains a feasibility diagnostic and must
not be used for the final `92%` claim.

### Full Fashionpedia and LLM Paraphrase Expansion

Build the deterministic, image-complete Fashionpedia source index before any
language-model rewrite. Omitting `--limit` processes the full selected split:

```bash
python -u scripts/prepare_referring_training_fashionpedia.py \
  --split train \
  --progress-every 1000
```

Then export vendor-neutral rewrite jobs. This command does not contact an
external model or upload any image/Mask data:

```bash
python scripts/select_balanced_referring_training.py

python scripts/export_referring_paraphrase_jobs.py \
  --index \
    data/processed/autodl/localization/fashionpedia_referring_train_balanced_100k.jsonl \
  --selection-policy weak_complex_balanced \
  --limit 20000 \
  --paraphrases-per-sample 3
```

The first command selects exactly 100,000 records by deterministic water-filled
strata over target label, language, and the complete modifier signature. Within
each stratum it uses a stable SHA-256 rank instead of an image prefix, then
restores original source order so image-complete loaders remain valid. The
summary exposes every category, language, dimension, stratum, weak-part, image,
and target-reference count. The second command prioritizes weak-part queries
(`zipper`, `rivet`, `neckline`, and `pocket`) that also contain spatial,
attribute, or relation modifiers, then balances the bounded job batch within
each priority tier. `--limit 20000` is a reviewable starting batch, not an
acceptance constant.

Each JSONL job contains the source query, language, dimensions, target label,
target count, a referent-preserving instruction, and a SHA-256 fingerprint of
the immutable source query plus its image and target annotations. A model or
batch service must return JSONL records with this contract:

```json
{
  "schema_version": 1,
  "source_sample_id": "fashionpedia-train-...",
  "source_fingerprint": "64-lowercase-hex-characters",
  "language": "zh",
  "generator_model": "provider/model/revision",
  "review_status": "reviewed",
  "reviewed_by": "reviewer-id",
  "reviewed_at": "2026-08-14T12:00:00+08:00",
  "paraphrases": ["改写一", "改写二", "改写三"]
}
```

Merge reviewed results and enforce the mentor-directed 100,000-query floor:

```bash
python scripts/merge_referring_paraphrases.py \
  --results outputs/localization/referring_training/paraphrase_results.jsonl
```

The merge changes only query text and augmentation provenance. It reuses the
source image, language, dimensions, target label, target boxes, annotation IDs,
and official Fashionpedia Mask references. Unknown source IDs, changed source
fingerprints, language drift, duplicate rewrites, and unreviewed results fail
closed. `--allow-unreviewed` is an explicit diagnostic override and the summary
still reports those rows separately. The default output is removed if the
combined dataset remains below 100,000 records.

The summary reports template versus LLM counts, language and query-dimension
counts, category balance, and dedicated counts for `zipper`, `rivet`,
`neckline`, and `pocket`. Reaching 100,000 records is only a data-scale gate;
it does not establish semantic rewrite quality, balanced weak-part coverage,
Mask accuracy, or PRD compliance. Review a stratified sample before training.

### Small-Part DINOv2 Backbone Adaptation

The backbone fine-tuning path starts from the validated `728` multiscale dense
checkpoint and consumes the balanced 100,000-query index. It applies three
separately audited changes:

- target unions below `1%` of image pixels receive `2.0x` query-loss weight;
- `zipper`, `rivet`, `neckline`, and `pocket` receive `1.5x`, with the combined
  weight capped at `3.0x`;
- only the final two DINOv2 blocks and terminal normalization are unfrozen, at
  `1e-5`, while the projection/decoder head uses `1e-4`.

For 100k-scale runs, only validated query metadata, text embeddings, loss
weights, and donor indices remain resident. Source images and exact Fashionpedia
Masks are decoded lazily for the current training or clean-audit batch. Exact
target-union area fractions are cached by annotation-ID tuple, so repeated
language variants do not repeatedly decode the same supervision Masks.

Copy-Paste never moves the referent to an arbitrary background position. It
replaces one receiver target's appearance with a resized same-label donor at
the original receiver target box. This preserves spatial and garment-relation
modifiers; attribute queries only accept donors with identical Fashionpedia
attribute IDs. The schema-four checkpoint stores only the explicitly unfrozen
DINOv2 parameter subset, and inference/evaluation restores it on top of the
pinned official pretrained weights.

Run a bounded compatibility smoke before scaling:

```bash
python -u scripts/finetune_dense_patch_backbone.py \
  --image-limit 20 \
  --steps 20 \
  --batch-size 2 \
  --output-dir \
    outputs/localization/dinov2_backbone_finetune_smoke20
```

The smoke checks finite training, augmentation counts, trainable parameter
scope, and schema-four checkpoint creation. Because stochastic weighted and
Copy-Paste batches are not directly comparable, the script also freezes 32
clean, unaugmented source queries and reports their identical before/after
audit loss. Even a decreased clean audit loss is training evidence, not
independent validation, and cannot establish the PRD `92%` target. Only after
a successful smoke should the image count and steps increase, followed by the
same frozen validation split and single-query Top-1 Mask IoU acceptance path.

The first 20-step aggressive smoke increased the fixed clean audit loss from
`0.748404` to `0.978319`, so that checkpoint is rejected and must not be scaled.
The next experiment uses the conservative config: one unfrozen DINOv2 block,
`1e-5` head learning rate, and `1e-6` backbone learning rate. Batch sampling
and Copy-Paste use independent deterministic random streams, so enabling or
disabling Copy-Paste preserves the exact query-batch sequence for a valid A/B.

The conservative 20-image A/B decreased the fixed clean training audit for
both variants, but neither passed the frozen 762-query validation gate. Without
Copy-Paste, Mask Recall50 changed from `41.60%` to `38.58%`; with Copy-Paste it
changed to `38.71%`. Mean Mask IoU also fell from `38.25%` to `37.12%` and
`37.08%`, respectively. Both checkpoints are therefore rejected and must not
be scaled. Their Box Recall50 rose from `55.64%` to about `58.2%`, isolating
Mask generation as the next bottleneck rather than justifying backbone tuning.

After Mask2Former refinement became the fixed validation route, a balanced
1,000-query, 250-step conservative run again failed the independent gate.
Clean audit loss fell from `0.8962` to `0.8422`, but refined Mask Recall50 fell
from `57.48%` to `56.82%`, mean Mask IoU from `51.16%` to `50.87%`, and Box
Recall50 from `55.64%` to `52.89%`. Zipper, rivet, and pocket did not improve;
the checkpoint is rejected. The next 10k gate uses the scale-safe config with
lower head/backbone learning rates and no Copy-Paste so data scale remains the
only major changed factor.

Training batches now use a deterministic shuffled epoch iterator. Every query
is visited once before reshuffling, while batch selection and Copy-Paste retain
independent seeds. The earlier independent per-step sampling could revisit
queries before covering the selected set and was unsuitable for a 100k scale
claim.

### DINOv2 Box-Guided Mask2Former Refinement

The staged refinement backend keeps the complete query on the category-free
DINOv2 path. For a directly supervised Fashionpedia part only, its DINOv2 Box
geometrically gates query-compatible Mask2Former outputs; the selected Mask
replaces the dense Mask while the DINOv2 Box and original query remain intact.
An unqualified multi-target query unions all overlapping part instances, while
the existing spatial parser reduces an explicit left/right/up/down query to one
candidate first. Unknown parts or non-overlapping part predictions retain the
dense output. Candidate selection does not use ground truth.

Run a two-image compatibility smoke with the retained frozen checkpoint before
the 50-image comparison:

```bash
python -u scripts/evaluate_dense_patch_localization.py \
  --split validation \
  --image-limit 2 \
  --checkpoint \
    outputs/localization/dinov2_multiscale_728_train1000_steps1500/dense_patch_alignment.pt \
  --dinov2-config configs/localization_dinov2_region_728.yaml \
  --mask2former-part-config \
    configs/localization_mask2former_parts_targeted_deployment.yaml \
  --refinement-minimum-box-iou 0.05 \
  --output-dir \
    outputs/localization/dinov2_mask2former_refinement_smoke2
```

The backend name is `dense_mask2former_refinement`, and it is now the default
API route after the frozen validation comparison demonstrated a Mask
improvement. Mask2Former is a domain-specific refinement/fallback here;
it is not allowed to replace full-query selection or claim open-query coverage.
Do not prepend `external/Mask2Former` through `PYTHONPATH`: its top-level
`datasets` directory shadows Hugging Face `datasets` and breaks BGE-M3. The
runtime appends the checkout after installed packages when Mask2Former loads.
Detectron2 must be installed inside `fashion-prd-312`; borrowing the base
environment would violate the pinned Python runtime. Run
`scripts/setup_prd_312_detectron2.sh` to build official Detectron2 `v0.6` at
commit `d1e04565d3bec8719335b88be9e9b961bf3ec464` for CUDA architecture `8.6`,
then verify Detectron2 CUDA, Mask2Former, and BGE-M3 in one process.
The installer pins `setuptools==80.9.0` because PyTorch 2.1.2 still imports
`pkg_resources` from its C++/CUDA extension helper; newer Setuptools releases
remove that compatibility module and fail before compilation starts.
It builds and installs a regular wheel rather than using editable mode: modern
Setuptools delegates editable installation to a nested isolated build that
cannot import the PRD environment's PyTorch CUDA extension toolchain.
The setup also retains Detectron2 0.6's declared `black==21.4b2`, `future`, and
`pydot` runtime metadata so the isolated environment passes `pip check` before
the joint Detectron2, Mask2Former, and BGE-M3 import gate.
The joint gate uses Detectron2 0.6's actual `has_cuda()` and
`get_cuda_version()` extension APIs, then inspects the compiled shared object
with `cuobjdump` to require `sm_86`; later-version-only extension APIs are not
used.
Pillow is pinned to `9.5.0` because Detectron2 0.6 still references legacy
resampling constants such as `PIL.Image.LINEAR`, which Pillow 12 removes.
The same setup script builds Mask2Former's bundled
`MultiScaleDeformableAttention` CUDA op as a non-isolated wheel in the PRD
environment. A Python-only Mask2Former import without this op does not pass the
joint readiness gate.

On the frozen 50-image, 762-query validation subset, box-guided Mask2Former
refinement raised Mask Recall50 from `41.60%` to `57.48%`, Mask Recall75 from
`9.19%` to `31.76%`, and mean Mask IoU from `38.25%` to `51.16%`. Box Recall50
remained `55.64%`, which isolates the gain to Mask refinement. Refinement was
applied to `632/762` known-part queries without GT-based selection. Warm mean
image time rose from about `0.361 s` to `0.542 s`, so this validates the
accuracy direction but does not satisfy the PRD `30 ms` deployment target or
the final `92%` acceptance target.
