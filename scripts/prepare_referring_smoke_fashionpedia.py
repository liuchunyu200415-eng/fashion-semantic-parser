"""Prepare the 20-query referring smoke manifest from Fashionpedia validation."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def add_src_to_python_path() -> None:
    """Add the local src directory when the package is not installed yet."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Select Fashionpedia validation images for the 20-query PRD 3.1.2 "
            "referring-expression smoke set."
        )
    )
    parser.add_argument(
        "--annotations",
        default=(
            "data/raw/fashionpedia/annotations/" "instances_attributes_val2020.json"
        ),
    )
    parser.add_argument(
        "--image-root",
        default="data/raw/fashionpedia/test",
    )
    parser.add_argument(
        "--template",
        default=("data/benchmarks/localization/" "referring_smoke_v1.template.json"),
    )
    parser.add_argument(
        "--output",
        default="data/benchmarks/localization/referring_smoke_v1.json",
    )
    parser.add_argument(
        "--summary-output",
        default=("outputs/localization/referring_smoke/" "fashionpedia_selection.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Select deterministic candidates and write a validated manifest."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path

    manifest, summary = prepare_referring_smoke_manifest(
        annotations_path=resolve_project_path(args.annotations),
        image_root=resolve_project_path(args.image_root),
        template_path=resolve_project_path(args.template),
    )
    output_path = resolve_project_path(args.output)
    summary_path = resolve_project_path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def prepare_referring_smoke_manifest(
    *,
    annotations_path: Path,
    image_root: Path,
    template_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fill the standard template with deterministic Fashionpedia candidates."""
    add_src_to_python_path()
    from fashion_semantic_parser.common.paths import to_project_relative_path
    from fashion_semantic_parser.dao.fashionpedia import (
        dict_records,
        is_integer,
        normalize_coco_bbox_xywh,
        normalize_coco_segmentation,
    )
    from fashion_semantic_parser.dao.localization.referring_smoke import (
        ReferringSmokeManifest,
    )

    source = json.loads(annotations_path.read_text(encoding="utf-8"))
    template = json.loads(template_path.read_text(encoding="utf-8"))
    images = dict_records(source.get("images"))
    annotations = dict_records(source.get("annotations"))
    categories = dict_records(source.get("categories"))
    category_name_by_id = {
        category["id"]: category["name"]
        for category in categories
        if is_integer(category.get("id")) and isinstance(category.get("name"), str)
    }
    image_by_id = {
        image["id"]: image
        for image in images
        if is_integer(image.get("id")) and isinstance(image.get("file_name"), str)
    }
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        image_id = annotation.get("image_id")
        if is_integer(image_id) and image_id in image_by_id:
            category_name = category_name_by_id.get(annotation.get("category_id"))
            if category_name is not None:
                annotation = dict(annotation)
                annotation["_category_name"] = category_name
                annotations_by_image[image_id].append(annotation)

    available_image_ids = [
        image_id
        for image_id, image in image_by_id.items()
        if (image_root / Path(str(image["file_name"])).name).is_file()
    ]
    available_image_ids.sort(
        key=lambda image_id: str(image_by_id[image_id]["file_name"])
    )
    if not available_image_ids:
        raise FileNotFoundError(
            f"No annotated Fashionpedia images were found under {image_root}."
        )

    selection_rules = _selection_rules()
    selections: dict[str, int] = {}
    missing_groups: list[str] = []
    for group_name, alternatives in selection_rules.items():
        selected = _select_image(
            available_image_ids,
            annotations_by_image,
            alternatives,
        )
        if selected is None:
            missing_groups.append(group_name)
        else:
            selections[group_name] = selected
    if missing_groups:
        raise ValueError(
            "Fashionpedia has no candidate image for selection group(s): "
            + ", ".join(missing_groups)
        )

    case_to_group = _case_selection_groups()
    automatically_scored = {
        "basic_collar_001": ("neckline", "all"),
        "basic_zipper_001": ("zipper", "all"),
        "spatial_right_pocket_001": ("pocket", "rightmost"),
        "spatial_lower_zipper_001": ("zipper", "lowest"),
    }
    prepared_cases: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for original_case in template.get("cases", []):
        case = dict(original_case)
        case_id = case.get("id")
        if not isinstance(case_id, str) or case_id not in case_to_group:
            raise ValueError(f"No Fashionpedia selection rule for case {case_id!r}.")
        group_name = case_to_group[case_id]
        image_id = selections[group_name]
        image = image_by_id[image_id]
        image_path = image_root / Path(str(image["file_name"])).name
        case["image_path"] = to_project_relative_path(image_path)

        status = "manual_review_required"
        score_rule = automatically_scored.get(case_id)
        if score_rule is not None:
            category_name, selection = score_rule
            targets = _targets_for_category(
                annotations_by_image[image_id],
                category_name=category_name,
                selection=selection,
                normalize_bbox=normalize_coco_bbox_xywh,
                normalize_segmentation=normalize_coco_segmentation,
            )
            if targets:
                case["annotation_status"] = "mask"
                case["targets"] = targets
                case["expected_count"] = len(targets)
                case["notes"] = (
                    "GT imported from Fashionpedia; visually review the selected "
                    "image and expression before reporting metrics."
                )
                status = "fashionpedia_gt_imported_review_required"
        if status == "manual_review_required":
            case["annotation_status"] = "unlabelled"
            case["targets"] = []
            case.pop("expected_count", None)
            case["notes"] = (
                "Candidate image selected automatically; manually review and add "
                "a Box or Mask before this case can enter accuracy metrics."
            )

        prepared_cases.append(case)
        selection_rows.append(
            {
                "case_id": case_id,
                "selection_group": group_name,
                "image_path": case["image_path"],
                "annotation_status": case["annotation_status"],
                "selection_status": status,
            }
        )

    prepared = dict(template)
    prepared["description"] = (
        "Fashionpedia-selected PRD 3.1.2 feasibility cases. Imported GT and "
        "automatic candidate choices both require visual review before metrics."
    )
    prepared["cases"] = prepared_cases
    validated = ReferringSmokeManifest.model_validate(prepared).model_dump(
        mode="json",
        exclude_none=True,
    )
    labelled_count = sum(
        case["annotation_status"] != "unlabelled" for case in validated["cases"]
    )
    summary = {
        "annotation_path": str(annotations_path),
        "image_root": str(image_root),
        "source_image_count": len(images),
        "available_annotated_image_count": len(available_image_ids),
        "case_count": len(prepared_cases),
        "fashionpedia_gt_imported_count": labelled_count,
        "manual_review_required_count": len(prepared_cases) - labelled_count,
        "accuracy_ready": False,
        "accuracy_boundary": (
            "No metric is reportable until every scored case and imported target "
            "has been visually reviewed."
        ),
        "cases": selection_rows,
    }
    return validated, summary


def _selection_rules() -> dict[str, list[dict[str, int]]]:
    """Return ordered category-count alternatives for each visual concept."""
    return {
        "collar": [{"neckline": 1}],
        "cuffs": [{"sleeve": 2}],
        "zipper": [{"zipper": 1}],
        "buttons": [{"rivet": 2}, {"jacket": 1}],
        "hood": [{"hood": 1}],
        "two_pockets": [{"pocket": 2}],
        "two_zippers": [{"zipper": 2}],
        "floral": [{"flower": 1}, {"applique": 1}],
        "striped_collar": [{"collar": 1}],
        "logo": [{"applique": 1}],
        "outer_inner": [
            {"jacket": 1, "top, t-shirt, sweatshirt": 1},
            {"coat": 1, "top, t-shirt, sweatshirt": 1},
            {"jacket": 1, "shirt, blouse": 1},
        ],
        "zipper_pocket": [{"zipper": 1, "pocket": 1}],
        "collar_decoration": [
            {"collar": 1, "applique": 1},
            {"collar": 1, "flower": 1},
            {"collar": 1, "bead": 1},
            {"collar": 1},
        ],
        "floral_sleeve": [
            {"sleeve": 1, "flower": 1},
            {"sleeve": 1, "applique": 1},
            {"sleeve": 1},
        ],
    }


def _case_selection_groups() -> dict[str, str]:
    """Map every committed template case to one reusable candidate image."""
    return {
        "basic_collar_001": "collar",
        "basic_cuffs_001": "cuffs",
        "basic_zipper_001": "zipper",
        "basic_button_001": "buttons",
        "basic_drawstring_001": "hood",
        "spatial_left_cuff_001": "cuffs",
        "spatial_right_cuff_001": "cuffs",
        "spatial_right_pocket_001": "two_pockets",
        "spatial_top_button_001": "buttons",
        "spatial_lower_zipper_001": "two_zippers",
        "attribute_floral_pattern_001": "floral",
        "attribute_silver_zipper_001": "zipper",
        "attribute_red_pocket_001": "two_pockets",
        "attribute_striped_collar_001": "striped_collar",
        "attribute_logo_001": "logo",
        "relation_inner_garment_001": "outer_inner",
        "relation_zipper_beside_pocket_001": "zipper_pocket",
        "relation_decoration_on_collar_001": "collar_decoration",
        "relation_left_floral_sleeve_001": "floral_sleeve",
        "relation_drawstring_at_hood_001": "hood",
    }


def _select_image(
    image_ids: list[int],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    alternatives: list[dict[str, int]],
) -> int | None:
    """Choose the first deterministic image satisfying one alternative."""
    for requirements in alternatives:
        for image_id in image_ids:
            counts: dict[str, int] = defaultdict(int)
            for annotation in annotations_by_image[image_id]:
                counts[str(annotation["_category_name"])] += 1
            if all(counts[name] >= count for name, count in requirements.items()):
                return image_id
    return None


def _targets_for_category(
    annotations: list[dict[str, Any]],
    *,
    category_name: str,
    selection: str,
    normalize_bbox: Any,
    normalize_segmentation: Any,
) -> list[dict[str, Any]]:
    """Convert selected official annotations into referring targets."""
    candidates: list[tuple[dict[str, Any], list[float]]] = []
    for annotation in annotations:
        if annotation.get("_category_name") != category_name:
            continue
        bbox = normalize_bbox(annotation.get("bbox"))
        segmentation = normalize_segmentation(annotation.get("segmentation"))
        if bbox is not None and segmentation is not None:
            candidates.append((annotation, bbox))
    if selection == "rightmost" and candidates:
        candidates = [max(candidates, key=lambda item: item[1][0] + item[1][2] / 2)]
    elif selection == "lowest" and candidates:
        candidates = [max(candidates, key=lambda item: item[1][1] + item[1][3] / 2)]

    targets: list[dict[str, Any]] = []
    for annotation, bbox in candidates:
        x, y, width, height = bbox
        targets.append(
            {
                "label": category_name,
                "box": {
                    "x_min": x,
                    "y_min": y,
                    "x_max": x + width,
                    "y_max": y + height,
                },
                "segmentation": normalize_segmentation(annotation["segmentation"]),
            }
        )
    return targets


if __name__ == "__main__":
    main()
