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

