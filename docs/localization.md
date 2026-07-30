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

The next required work is to run the candidate-ranking validation, select
Top-K and score settings from full-category evidence, then establish per-part
mask IoU and recall. Only then should missing PRD regions, fine-tuning,
TensorRT, and the `30 ms` target be addressed.
