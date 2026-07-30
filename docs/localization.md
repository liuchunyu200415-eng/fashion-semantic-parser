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

The first engineering slice is implemented:

- a separate Fashionpedia local-part COCO conversion path
- 19 directly annotated part categories with Chinese and English prompt terms
- explicit PRD coverage reporting instead of relabeling weak proxies
- typed request and response models for local-region masks and boxes
- an injectable `POST /v1/localize` API route

The Grounding DINO and SAM-HQ inference runtime is not implemented yet. The
default route returns HTTP `503` with a model-setup message rather than
inventing a localization result. Existing `/v1/segment` and `/v1/query`
behavior is unchanged.

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

The audit should report `4,016` selected part annotations for the official
validation annotations and approximately `167,406` for training. The converter
itself remains the source of truth if the official release differs.

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

These outputs are independent from
`data/processed/autodl/segmentation/fashionpedia_*.json`; converting local parts
does not change the accepted PRD 3.1.1 training data.

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

## Next Model Slice

The first executable model baseline will use:

1. the accepted automatic person ROI to remove unrelated subjects and scenery
2. the accepted garment segmentation result to constrain the parent garment
3. Grounding DINO to convert the normalized text phrase into candidate boxes
4. SAM-HQ to refine the selected box into a local-region mask
5. overlap and confidence checks before returning a typed result

DINOv2 may be evaluated for candidate re-ranking, but it is not used as a
standalone text localizer because it does not directly encode natural-language
queries. Accuracy will be established before TensorRT and the `30 ms` latency
target are addressed.
