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

The code path and injected unit tests are complete, but the external models
have not yet been run on AutoDL. Missing repositories or weights therefore
produce HTTP `503` with an explicit setup message. This stage does not prove
the PRD `92%` accuracy or `30 ms` latency targets. Existing `/v1/segment` and
`/v1/query` behavior is unchanged.

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

After the first real-image smoke test, the next required work is to generate
Fashionpedia validation predictions, establish per-part mask IoU and recall,
and tune `box_threshold`/`text_threshold`. Only then should missing PRD regions,
fine-tuning, TensorRT, and the `30 ms` target be addressed.
