# Dataset Layout

Large datasets are not committed to GitHub. Store raw datasets, processed data,
model checkpoints, and experiment outputs on the server data disk, then expose
them through project-relative paths.

## Server Paths

On AutoDL, keep large files under:

```text
/root/autodl-tmp/data/raw/fashionai
/root/autodl-tmp/data/raw/deepfashion2
/root/autodl-tmp/data/processed
/root/autodl-tmp/models/checkpoints
/root/autodl-tmp/outputs
```

The project should access them through:

```text
data/raw/fashionai
data/raw/deepfashion2
data/processed/autodl
models/checkpoints/autodl
outputs/autodl
```

## Create Links

From the project root on the server:

```bash
python scripts/setup_data_links.py --data-root /root/autodl-tmp
```

If old symlinks created during setup already exist, first restore the Git-tracked
placeholder directories:

```bash
rm data/raw data/processed models/checkpoints outputs
git restore data/raw/.gitkeep data/processed/.gitkeep models/checkpoints/.gitkeep outputs/.gitkeep
python scripts/setup_data_links.py --data-root /root/autodl-tmp
```

## Current Raw Dataset Structure

```text
data/raw/fashionai/round1_fashionAI_attributes_test_a
data/raw/deepfashion2/train
data/raw/deepfashion2/validation
data/raw/deepfashion2/test
data/raw/deepfashion2/json_for_validation
data/raw/deepfashion2/json_for_test
```

## Build Lightweight Indexes

After raw datasets are linked, build JSONL indexes on the server:

```bash
python scripts/build_dataset_indexes.py
```

Generated files are stored under:

```text
data/processed/autodl/indexes
```

Use a small limit for smoke tests:

```bash
python scripts/build_dataset_indexes.py --limit 10
```

Index records use project-relative paths so they remain portable across local
development and AutoDL server environments.

Preview generated indexes:

```bash
python scripts/preview_dataset_index.py --index-name deepfashion2_train --limit 3
python scripts/preview_dataset_index.py --index-name deepfashion2_train --category-name trousers --limit 3
```

Compute dataset statistics:

```bash
python scripts/analyze_dataset_indexes.py
python scripts/analyze_dataset_indexes.py --output data/processed/autodl/indexes/statistics.json
```

The statistics JSON includes FashionAI attribute group counts, DeepFashion2
category counts, and DeepFashion2 source counts for each split.
