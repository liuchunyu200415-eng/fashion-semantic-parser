"""Build visual acceptance artifacts from saved COCO segmentation predictions."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

COLORS_BY_CATEGORY = {
    "top": (46, 204, 113),
    "pants": (52, 152, 219),
    "skirt": (155, 89, 182),
    "outerwear": (241, 196, 15),
    "dress": (231, 76, 60),
    "shoes": (230, 126, 34),
    "bag": (26, 188, 156),
    "accessory": (149, 165, 166),
}


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
            "Create original/ground-truth/prediction acceptance comparisons from "
            "saved COCO segmentation results."
        )
    )
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--score-threshold", type=float, default=0.8)
    parser.add_argument("--samples-per-category", type=int, default=2)
    parser.add_argument("--misses-per-category", type=int, default=1)
    parser.add_argument("--seed", type=int, default=311)
    parser.add_argument("--panel-width", type=int, default=360)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    """Create comparison images, one contact sheet, and a JSON manifest."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path

    try:
        from pycocotools.coco import COCO
    except ImportError as error:
        raise RuntimeError(
            "pycocotools is required for segmentation acceptance visualization."
        ) from error

    _validate_args(args)
    validation_path = _resolve_path(args.val_json, resolve_project_path)
    prediction_path = _resolve_path(args.predictions, resolve_project_path)
    output_dir = _resolve_path(args.output_dir, resolve_project_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = _read_prediction_list(prediction_path)
    filtered_predictions = _filter_predictions(predictions, args.score_threshold)
    if not filtered_predictions:
        raise ValueError(
            "No predictions remain after applying --score-threshold "
            f"{args.score_threshold}."
        )

    coco_ground_truth = COCO(str(validation_path))
    coco_predictions = coco_ground_truth.loadRes(filtered_predictions)
    image_ids = sorted(coco_ground_truth.getImgIds())
    ground_truth_by_image = {
        image_id: coco_ground_truth.imgToAnns.get(image_id, [])
        for image_id in image_ids
    }
    predictions_by_image = {
        image_id: coco_predictions.imgToAnns.get(image_id, []) for image_id in image_ids
    }
    category_ids = sorted(
        {
            int(annotation["category_id"])
            for annotations in ground_truth_by_image.values()
            for annotation in annotations
        }
    )
    selected_image_ids, selection_reasons = _select_acceptance_images(
        ground_truth_by_image=ground_truth_by_image,
        predictions_by_image=predictions_by_image,
        category_ids=category_ids,
        samples_per_category=args.samples_per_category,
        misses_per_category=args.misses_per_category,
        seed=args.seed,
    )
    if not selected_image_ids:
        raise ValueError("No annotated validation images are available to visualize.")

    category_names = {
        int(category_id): str(category["name"])
        for category_id, category in coco_ground_truth.cats.items()
    }
    comparison_images: list[np.ndarray] = []
    manifest_images: list[dict[str, Any]] = []
    for index, image_id in enumerate(selected_image_ids, start=1):
        image_record = coco_ground_truth.imgs[image_id]
        image_path = _resolve_image_path(
            image_record["file_name"],
            args.image_root,
            resolve_project_path,
        )
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read validation image: {image_path}")

        ground_truth = ground_truth_by_image[image_id]
        image_predictions = predictions_by_image[image_id]
        comparison = _render_comparison(
            image=image,
            ground_truth=ground_truth,
            predictions=image_predictions,
            ground_truth_coco=coco_ground_truth,
            prediction_coco=coco_predictions,
            category_names=category_names,
            score_threshold=args.score_threshold,
            panel_width=args.panel_width,
        )
        readable_reasons = _readable_selection_reasons(
            selection_reasons[image_id], category_names
        )
        comparison = _add_review_caption(
            comparison,
            f"Image {int(image_id)} | {' | '.join(readable_reasons)}",
        )
        output_name = f"{index:02d}_image_{int(image_id):06d}.png"
        output_path = output_dir / output_name
        if not cv2.imwrite(str(output_path), comparison):
            raise ValueError(f"Unable to write comparison image: {output_path}")
        comparison_images.append(comparison)
        manifest_images.append(
            {
                "image_id": int(image_id),
                "file_name": str(image_record["file_name"]),
                "selection_reasons": readable_reasons,
                "ground_truth_categories": sorted(
                    {
                        category_names[int(annotation["category_id"])]
                        for annotation in ground_truth
                    }
                ),
                "prediction_count": len(image_predictions),
                "output": output_path.as_posix(),
            }
        )

    contact_sheet = _build_contact_sheet(comparison_images, columns=args.columns)
    contact_sheet_path = output_dir / "contact_sheet.jpg"
    if not cv2.imwrite(str(contact_sheet_path), contact_sheet):
        raise ValueError(f"Unable to write contact sheet: {contact_sheet_path}")

    manifest = {
        "validation_json": validation_path.as_posix(),
        "predictions_json": prediction_path.as_posix(),
        "score_threshold": args.score_threshold,
        "prediction_count_before_filter": len(predictions),
        "prediction_count_after_filter": len(filtered_predictions),
        "samples_per_category": args.samples_per_category,
        "misses_per_category": args.misses_per_category,
        "seed": args.seed,
        "selected_image_count": len(selected_image_ids),
        "covered_categories": [
            category_names[category_id] for category_id in category_ids
        ],
        "contact_sheet": contact_sheet_path.as_posix(),
        "images": manifest_images,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid sampling, layout, and threshold arguments."""
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1.")
    if args.samples_per_category < 0:
        raise ValueError("--samples-per-category must be non-negative.")
    if args.misses_per_category < 0:
        raise ValueError("--misses-per-category must be non-negative.")
    if args.samples_per_category + args.misses_per_category < 1:
        raise ValueError("At least one sample or missed example must be requested.")
    if args.panel_width < 1:
        raise ValueError("--panel-width must be at least one.")
    if args.columns < 1:
        raise ValueError("--columns must be at least one.")


def _read_prediction_list(path: Path) -> list[dict[str, Any]]:
    """Read the flat COCO result list written by Detectron2."""
    with path.open("r", encoding="utf-8") as file:
        predictions = json.load(file)
    if not isinstance(predictions, list):
        raise ValueError(f"Expected a COCO prediction list: {path}")
    if not all(isinstance(prediction, dict) for prediction in predictions):
        raise ValueError(f"COCO predictions must be JSON objects: {path}")
    return predictions


def _filter_predictions(
    predictions: list[dict[str, Any]], score_threshold: float
) -> list[dict[str, Any]]:
    """Keep predictions at or above the deployment score threshold."""
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1.")
    return [
        prediction
        for prediction in predictions
        if float(prediction.get("score", 0.0)) >= score_threshold
    ]


def _select_acceptance_images(
    ground_truth_by_image: dict[int, list[dict[str, Any]]],
    predictions_by_image: dict[int, list[dict[str, Any]]],
    category_ids: list[int],
    samples_per_category: int,
    misses_per_category: int,
    seed: int,
) -> tuple[list[int], dict[int, list[str]]]:
    """Select deterministic category samples plus explicit category misses."""
    random_generator = random.Random(seed)
    selected_image_ids: list[int] = []
    selection_reasons: dict[int, list[str]] = defaultdict(list)

    def add_image(image_id: int, reason: str) -> None:
        if image_id not in selected_image_ids:
            selected_image_ids.append(image_id)
        if reason not in selection_reasons[image_id]:
            selection_reasons[image_id].append(reason)

    for category_id in category_ids:
        candidates = [
            image_id
            for image_id, annotations in ground_truth_by_image.items()
            if _annotations_include_category(annotations, category_id)
        ]
        detected_candidates = [
            image_id
            for image_id in candidates
            if _annotations_include_category(
                predictions_by_image.get(image_id, []), category_id
            )
        ]
        random_generator.shuffle(detected_candidates)
        for image_id in detected_candidates[:samples_per_category]:
            add_image(image_id, f"sample:category_{category_id}")

        missed_candidates = [
            image_id
            for image_id in candidates
            if not _annotations_include_category(
                predictions_by_image.get(image_id, []), category_id
            )
        ]
        random_generator.shuffle(missed_candidates)
        for image_id in missed_candidates[:misses_per_category]:
            add_image(image_id, f"miss:category_{category_id}")

    return selected_image_ids, dict(selection_reasons)


def _annotations_include_category(
    annotations: list[dict[str, Any]], category_id: int
) -> bool:
    """Return whether an annotation collection contains one category."""
    return any(
        int(annotation.get("category_id", -1)) == category_id
        for annotation in annotations
    )


def _readable_selection_reasons(
    reasons: list[str], category_names: dict[int, str]
) -> list[str]:
    """Replace category ids in selection reasons with readable labels."""
    readable_reasons = []
    for reason in reasons:
        reason_type, separator, category_text = reason.partition(":category_")
        if not separator:
            readable_reasons.append(reason)
            continue
        category_name = category_names.get(int(category_text), category_text)
        readable_reasons.append(f"{reason_type}:{category_name}")
    return readable_reasons


def _render_comparison(
    image: np.ndarray,
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    ground_truth_coco: Any,
    prediction_coco: Any,
    category_names: dict[int, str],
    score_threshold: float,
    panel_width: int,
) -> np.ndarray:
    """Render original, ground-truth, and prediction panels side by side."""
    ground_truth_panel = _draw_coco_annotations(
        image,
        ground_truth,
        ground_truth_coco,
        category_names,
        show_score=False,
    )
    prediction_panel = _draw_coco_annotations(
        image,
        predictions,
        prediction_coco,
        category_names,
        show_score=True,
    )
    if not predictions:
        _draw_empty_message(prediction_panel, f"No prediction >= {score_threshold:.2f}")

    panels = [
        _add_panel_header(_resize_to_width(image, panel_width), "Original"),
        _add_panel_header(
            _resize_to_width(ground_truth_panel, panel_width), "Ground truth"
        ),
        _add_panel_header(
            _resize_to_width(prediction_panel, panel_width),
            f"Prediction >= {score_threshold:.2f}",
        ),
    ]
    return np.hstack(panels)


def _draw_coco_annotations(
    image: np.ndarray,
    annotations: list[dict[str, Any]],
    coco: Any,
    category_names: dict[int, str],
    show_score: bool,
    alpha: float = 0.45,
) -> np.ndarray:
    """Draw COCO masks, boxes, and category labels on one image."""
    result = image.copy()
    overlay = image.copy()
    masks: list[tuple[dict[str, Any], np.ndarray, tuple[int, int, int]]] = []
    for annotation in annotations:
        category_id = int(annotation["category_id"])
        category_name = category_names.get(category_id, str(category_id))
        color = COLORS_BY_CATEGORY.get(category_name, (255, 255, 255))
        mask = np.asarray(coco.annToMask(annotation), dtype=bool)
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}."
            )
        overlay[mask] = color
        masks.append((annotation, mask, color))

    result = cv2.addWeighted(overlay, alpha, result, 1.0 - alpha, 0)
    for annotation, _mask, color in masks:
        category_name = category_names.get(
            int(annotation["category_id"]), str(annotation["category_id"])
        )
        label = category_name
        if show_score and "score" in annotation:
            label = f'{label} {float(annotation["score"]):.2f}'
        _draw_annotation_box(result, annotation, label, color)
    return result


def _draw_annotation_box(
    image: np.ndarray,
    annotation: dict[str, Any],
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Draw one COCO xywh box and compact label."""
    x, y, width, height = [float(value) for value in annotation["bbox"]]
    x_min = max(0, int(round(x)))
    y_min = max(0, int(round(y)))
    x_max = min(image.shape[1] - 1, int(round(x + width)))
    y_max = min(image.shape[0] - 1, int(round(y + height)))
    if x_max <= x_min or y_max <= y_min:
        return
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (label_width, label_height), baseline = cv2.getTextSize(
        label, font, scale, thickness
    )
    label_y = max(y_min, label_height + baseline + 4)
    cv2.rectangle(
        image,
        (x_min, label_y - label_height - baseline - 4),
        (min(image.shape[1] - 1, x_min + label_width + 6), label_y),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (x_min + 3, label_y - baseline - 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _draw_empty_message(image: np.ndarray, message: str) -> None:
    """Draw a visible empty-prediction message."""
    cv2.rectangle(image, (8, 8), (min(image.shape[1] - 8, 280), 42), (30, 30, 30), -1)
    cv2.putText(
        image,
        message,
        (14, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    """Resize one image to a stable panel width while preserving aspect ratio."""
    scale = width / image.shape[1]
    height = max(1, int(round(image.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _add_panel_header(image: np.ndarray, title: str) -> np.ndarray:
    """Add a compact dark header above one image panel."""
    header_height = 38
    panel = np.full(
        (image.shape[0] + header_height, image.shape[1], 3),
        (28, 28, 28),
        dtype=np.uint8,
    )
    panel[header_height:] = image
    cv2.putText(
        panel,
        title,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def _add_review_caption(image: np.ndarray, caption: str) -> np.ndarray:
    """Add the image id and sampling reason above one full comparison."""
    caption_height = 36
    result = np.full(
        (image.shape[0] + caption_height, image.shape[1], 3),
        (18, 18, 18),
        dtype=np.uint8,
    )
    result[caption_height:] = image
    cv2.putText(
        result,
        caption,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return result


def _build_contact_sheet(
    images: list[np.ndarray], columns: int, gap: int = 12
) -> np.ndarray:
    """Arrange comparison images into a padded contact sheet."""
    if not images:
        raise ValueError("At least one comparison image is required.")
    if columns < 1:
        raise ValueError("columns must be at least one.")
    cell_width = max(image.shape[1] for image in images)
    cell_height = max(image.shape[0] for image in images)
    rows = (len(images) + columns - 1) // columns
    sheet_width = columns * cell_width + max(0, columns - 1) * gap
    sheet_height = rows * cell_height + max(0, rows - 1) * gap
    sheet = np.full((sheet_height, sheet_width, 3), (18, 18, 18), dtype=np.uint8)

    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        x = column * (cell_width + gap)
        y = row * (cell_height + gap)
        sheet[y : y + image.shape[0], x : x + image.shape[1]] = image
    return sheet


def _resolve_path(path: str | Path, resolver: Callable[[str | Path], Path]) -> Path:
    """Use absolute paths directly and resolve project-relative paths."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolver(candidate)


def _resolve_image_path(
    file_name: str,
    image_root: str,
    resolver: Callable[[str | Path], Path],
) -> Path:
    """Resolve one COCO image path against the configured image root."""
    image_path = Path(file_name)
    if image_path.is_absolute():
        return image_path
    rooted_path = Path(image_root) / image_path
    return rooted_path if rooted_path.is_absolute() else resolver(rooted_path)


if __name__ == "__main__":
    main()
