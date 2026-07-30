"""Compare one PRD 3.1.2 localization result with COCO ground truth."""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

GROUND_TRUTH_COLOR = (46, 204, 113)
PREDICTION_COLORS = (
    (52, 152, 219),
    (230, 126, 34),
    (155, 89, 182),
)
SUBJECT_ROI_COLOR = (255, 255, 255)
TITLE_HEIGHT = 42


def add_src_to_python_path() -> None:
    """Add the local package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse saved predictions and visualization output paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Render Original / Ground Truth / Prediction panels and direct mask IoU."
        )
    )
    parser.add_argument(
        "--val-json",
        required=True,
        help="Project-relative Fashionpedia parts COCO validation JSON.",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Saved localization JSON. Repeat to compare multiple inference modes.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Project-relative PNG/JPG comparison output.",
    )
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Optional project-relative direct-IoU JSON output.",
    )
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    """Load saved results, evaluate one image, and render comparison panels."""
    args = parse_args()
    add_src_to_python_path()

    from pycocotools.coco import COCO  # type: ignore[import-untyped]

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.localization.taxonomy import (
        resolve_localization_prompt,
    )

    if args.panel_width < 160:
        raise ValueError("--panel-width must be at least 160")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    if not 0.0 <= args.match_iou <= 1.0:
        raise ValueError("--match-iou must be between 0 and 1")

    prediction_specs = _parse_prediction_specs(args.prediction)
    prediction_payloads = [
        (label, _read_json(resolve_project_path(path)))
        for label, path in prediction_specs
    ]
    image_path, query = _validate_prediction_payloads(prediction_payloads)
    prompt = resolve_localization_prompt(query)

    coco = COCO(str(resolve_project_path(args.val_json)))
    image_id = _find_image_id(coco, image_path)
    image_record = coco.loadImgs([image_id])[0]
    image = cv2.imread(str(resolve_project_path(image_path)))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    ground_truth, evaluation_scope = _load_ground_truth(
        coco,
        image_id=image_id,
        target_label=prompt.region_label,
    )
    ground_truth_masks = [
        coco.annToMask(annotation).astype(bool) for annotation in ground_truth
    ]
    category_names = {
        int(category_id): str(category["name"])
        for category_id, category in coco.cats.items()
    }

    panels = [_add_title(_resize_panel(image, args.panel_width), "Original")]
    ground_truth_panel = _draw_ground_truth(
        image,
        ground_truth,
        ground_truth_masks,
        category_names,
        alpha=args.alpha,
    )
    panels.append(
        _add_title(
            _resize_panel(ground_truth_panel, args.panel_width),
            f"Ground truth: {prompt.region_label} ({len(ground_truth)})",
        )
    )

    metrics_by_prediction: dict[str, dict[str, Any]] = {}
    for index, (label, payload) in enumerate(prediction_payloads):
        prediction_masks = _prediction_masks(
            payload,
            height=image.shape[0],
            width=image.shape[1],
            decoder=_coco_polygons_to_mask,
        )
        metrics = _direct_mask_iou_metrics(
            prediction_masks,
            ground_truth_masks,
            min_iou=args.match_iou,
        )
        metrics_by_prediction[label] = metrics
        prediction_panel = _draw_prediction(
            image,
            payload,
            prediction_masks,
            color=PREDICTION_COLORS[index % len(PREDICTION_COLORS)],
            alpha=args.alpha,
        )
        title = (
            f"{label}: P50={_format_metric(metrics['Precision50'])} "
            f"R50={_format_metric(metrics['Recall50'])}"
        )
        panels.append(
            _add_title(_resize_panel(prediction_panel, args.panel_width), title)
        )

    comparison = _join_panels(panels)
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), comparison):
        raise ValueError(f"Unable to write visualization: {output_path}")

    result = {
        "image_path": image_path,
        "image_id": int(image_record["id"]),
        "query": query,
        "target_label": prompt.region_label,
        "evaluation_scope": evaluation_scope,
        "ground_truth_categories": [
            category_names[int(annotation["category_id"])]
            for annotation in ground_truth
        ],
        "ground_truth_count": len(ground_truth),
        "match_iou_threshold": args.match_iou,
        "predictions": metrics_by_prediction,
        "visualization": args.output,
    }
    if args.metrics_output:
        metrics_path = resolve_project_path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


def _parse_prediction_specs(values: list[str]) -> list[tuple[str, str]]:
    """Parse repeatable LABEL=PATH prediction arguments."""
    parsed: list[tuple[str, str]] = []
    labels: set[str] = set()
    for value in values:
        label, separator, path = value.partition("=")
        label = label.strip()
        path = path.strip()
        if not separator or not label or not path:
            raise ValueError("--prediction must use LABEL=PATH")
        if label in labels:
            raise ValueError(f"Duplicate prediction label: {label}")
        labels.add(label)
        parsed.append((label, path))
    return parsed


def _validate_prediction_payloads(
    payloads: list[tuple[str, dict[str, Any]]],
) -> tuple[str, str]:
    """Require every comparison payload to describe the same image and query."""
    if not payloads:
        raise ValueError("At least one prediction is required.")
    first_payload = payloads[0][1]
    image_path = str(first_payload.get("image_path", "")).strip()
    query = str(first_payload.get("query", "")).strip()
    if not image_path or not query:
        raise ValueError("Prediction JSON must contain image_path and query.")
    for label, payload in payloads:
        if payload.get("image_path") != image_path:
            raise ValueError(f"Prediction {label} uses a different image_path.")
        if payload.get("query") != query:
            raise ValueError(f"Prediction {label} uses a different query.")
        if not isinstance(payload.get("regions"), list):
            raise ValueError(f"Prediction {label} must contain a regions list.")
    return image_path, query


def _find_image_id(coco: Any, image_path: str) -> int:
    """Find one COCO image by its normalized project-relative path."""
    normalized_path = Path(image_path).as_posix().removeprefix("./")
    matches = [
        int(image_id)
        for image_id, image in coco.imgs.items()
        if Path(str(image.get("file_name", ""))).as_posix().removeprefix("./")
        == normalized_path
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one COCO image for {image_path}, found {len(matches)}."
        )
    return matches[0]


def _load_ground_truth(
    coco: Any,
    image_id: int,
    target_label: str,
) -> tuple[list[dict[str, Any]], str]:
    """Prefer an exact category, otherwise evaluate its broader semantic group."""
    exact_category_ids = [
        int(category_id)
        for category_id, category in coco.cats.items()
        if category.get("name") == target_label
    ]
    if exact_category_ids:
        category_ids = exact_category_ids
        evaluation_scope = "exact_category"
    else:
        category_ids = [
            int(category_id)
            for category_id, category in coco.cats.items()
            if category.get("supercategory") == target_label
        ]
        evaluation_scope = "region_group" if category_ids else "unmapped"
    if not category_ids:
        return [], evaluation_scope
    annotation_ids = coco.getAnnIds(
        imgIds=[image_id],
        catIds=category_ids,
        iscrowd=False,
    )
    return list(coco.loadAnns(annotation_ids)), evaluation_scope


def _prediction_masks(
    payload: dict[str, Any],
    height: int,
    width: int,
    decoder: Callable[[list[list[float]], int, int], np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Rasterize every saved prediction polygon into one binary mask."""
    decode = decoder or _polygons_to_mask
    return [
        decode(region.get("mask", []), height, width) for region in payload["regions"]
    ]


def _polygons_to_mask(
    polygons: list[list[float]],
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize possibly multipart polygons using OpenCV."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) < 6 or len(polygon) % 2:
            continue
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        points[:, 0] = np.clip(points[:, 0], 0, max(width - 1, 0))
        points[:, 1] = np.clip(points[:, 1], 0, max(height - 1, 0))
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], (1,))
    return cast(np.ndarray, mask.astype(bool))


def _coco_polygons_to_mask(
    polygons: list[list[float]],
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize polygons with pycocotools for metric-compatible mask edges."""
    from pycocotools import mask as mask_utils  # type: ignore[import-untyped]

    valid_polygons = [
        polygon for polygon in polygons if len(polygon) >= 6 and len(polygon) % 2 == 0
    ]
    if not valid_polygons:
        return cast(np.ndarray, np.zeros((height, width), dtype=bool))
    rles = mask_utils.frPyObjects(valid_polygons, height, width)
    merged = mask_utils.merge(rles)
    return cast(np.ndarray, np.asarray(mask_utils.decode(merged), dtype=bool))


def _mask_iou_matrix(
    prediction_masks: list[np.ndarray],
    ground_truth_masks: list[np.ndarray],
) -> np.ndarray:
    """Build a prediction-by-ground-truth mask IoU matrix."""
    matrix = np.zeros(
        (len(prediction_masks), len(ground_truth_masks)),
        dtype=float,
    )
    for prediction_index, prediction_mask in enumerate(prediction_masks):
        for ground_truth_index, ground_truth_mask in enumerate(ground_truth_masks):
            union = np.logical_or(prediction_mask, ground_truth_mask).sum()
            if union:
                intersection = np.logical_and(prediction_mask, ground_truth_mask).sum()
                matrix[prediction_index, ground_truth_index] = intersection / union
    return cast(np.ndarray, matrix)


def _direct_mask_iou_metrics(
    prediction_masks: list[np.ndarray],
    ground_truth_masks: list[np.ndarray],
    min_iou: float = 0.50,
) -> dict[str, Any]:
    """Summarize one-to-one mask matches using the PRD 3.1.1 metric contract."""
    iou_matrix = _mask_iou_matrix(prediction_masks, ground_truth_masks)
    matches = _greedy_match_iou_pairs(iou_matrix, min_iou=min_iou)
    matched_ious = [iou for _, _, iou in matches]
    ground_truth_count = len(ground_truth_masks)
    prediction_count = len(prediction_masks)
    target_count = sum(iou >= 0.85 for iou in matched_ious)
    summary = {
        "MatchedCount": float(len(matches)),
        "GroundTruthCount": float(ground_truth_count),
        "PredictionCount": float(prediction_count),
        "MatchIoUThreshold": min_iou * 100.0,
        "TargetIoUThreshold": 85.0,
        "MatchedMeanIoU": _percentage(
            float(np.mean(matched_ious)) if matched_ious else None
        ),
        "MatchedMedianIoU": _percentage(
            float(np.median(matched_ious)) if matched_ious else None
        ),
        "AllGTMeanIoU": _percentage(
            sum(matched_ious) / ground_truth_count if ground_truth_count else None
        ),
        "Precision50": _percentage(
            len(matches) / prediction_count if prediction_count else None
        ),
        "Recall50": _percentage(
            len(matches) / ground_truth_count if ground_truth_count else None
        ),
        "MatchedIoU85Rate": _percentage(
            target_count / len(matches) if matches else None
        ),
        "AllGTIoU85Rate": _percentage(
            target_count / ground_truth_count if ground_truth_count else None
        ),
    }
    return {
        key: _finite_or_none(value)
        for key, value in {
            **summary,
            "matches": [
                {
                    "prediction_index": prediction_index,
                    "ground_truth_index": ground_truth_index,
                    "mask_iou": iou * 100.0,
                }
                for prediction_index, ground_truth_index, iou in matches
            ],
        }.items()
    }


def _greedy_match_iou_pairs(
    iou_matrix: np.ndarray,
    min_iou: float,
) -> list[tuple[int, int, float]]:
    """Greedily select one-to-one mask matches above the minimum IoU."""
    if iou_matrix.ndim != 2 or iou_matrix.size == 0:
        return []
    candidates = np.argwhere(iou_matrix >= min_iou)
    if candidates.size == 0:
        return []
    candidate_ious = iou_matrix[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(-candidate_ious, kind="stable")
    used_predictions: set[int] = set()
    used_ground_truth: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for candidate_index in order:
        prediction_index = int(candidates[candidate_index, 0])
        ground_truth_index = int(candidates[candidate_index, 1])
        if (
            prediction_index in used_predictions
            or ground_truth_index in used_ground_truth
        ):
            continue
        used_predictions.add(prediction_index)
        used_ground_truth.add(ground_truth_index)
        matches.append(
            (
                prediction_index,
                ground_truth_index,
                float(iou_matrix[prediction_index, ground_truth_index]),
            )
        )
    return matches


def _percentage(value: float | None) -> float:
    """Convert a zero-to-one value to percent while preserving missing values."""
    return float("nan") if value is None else float(value * 100.0)


def _finite_or_none(value: Any) -> Any:
    """Replace non-finite metric values so the JSON remains standards compliant."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _draw_ground_truth(
    image: np.ndarray,
    annotations: list[dict[str, Any]],
    masks: list[np.ndarray],
    category_names: dict[int, str],
    alpha: float,
) -> np.ndarray:
    """Draw grouped ground-truth masks, boxes, and category labels."""
    result = _blend_masks(image, masks, GROUND_TRUTH_COLOR, alpha)
    for annotation in annotations:
        x_min, y_min, width, height = annotation["bbox"]
        _draw_box_and_label(
            result,
            {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_min + width,
                "y_max": y_min + height,
            },
            category_names[int(annotation["category_id"])],
            GROUND_TRUTH_COLOR,
        )
    return result


def _draw_prediction(
    image: np.ndarray,
    payload: dict[str, Any],
    masks: list[np.ndarray],
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Draw saved localization polygons, boxes, scores, and subject ROI."""
    result = _blend_masks(image, masks, color, alpha)
    for region in payload["regions"]:
        label = (
            f"{region.get('region_label', 'region')} "
            f"{float(region.get('confidence', 0.0)):.2f}"
        )
        _draw_box_and_label(result, region["box"], label, color)
    subject_roi = payload.get("subject_roi")
    if subject_roi:
        _draw_box_and_label(result, subject_roi, "subject ROI", SUBJECT_ROI_COLOR)
    return result


def _blend_masks(
    image: np.ndarray,
    masks: list[np.ndarray],
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Blend binary masks over a BGR image."""
    overlay = image.copy()
    for mask in masks:
        overlay[mask] = color
    return cast(
        np.ndarray,
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0),
    )


def _draw_box_and_label(
    image: np.ndarray,
    box: dict[str, float],
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Draw one clipped xyxy box with a compact readable label."""
    height, width = image.shape[:2]
    x_min = int(np.clip(round(box["x_min"]), 0, max(width - 1, 0)))
    y_min = int(np.clip(round(box["y_min"]), 0, max(height - 1, 0)))
    x_max = int(np.clip(round(box["x_max"]), 0, max(width - 1, 0)))
    y_max = int(np.clip(round(box["y_max"]), 0, max(height - 1, 0)))
    if x_max <= x_min or y_max <= y_min:
        return
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    (label_width, label_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    label_x = min(x_min, max(width - label_width - 8, 0))
    label_bottom = max(y_min, label_height + baseline + 5)
    cv2.rectangle(
        image,
        (label_x, label_bottom - label_height - baseline - 5),
        (min(label_x + label_width + 7, width - 1), label_bottom),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (label_x + 3, label_bottom - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _resize_panel(image: np.ndarray, panel_width: int) -> np.ndarray:
    """Resize a panel to a stable width while preserving aspect ratio."""
    scale = panel_width / image.shape[1]
    panel_height = max(1, int(round(image.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cast(
        np.ndarray,
        cv2.resize(
            image,
            (panel_width, panel_height),
            interpolation=interpolation,
        ),
    )


def _add_title(image: np.ndarray, title: str) -> np.ndarray:
    """Add a fixed title strip without covering image content."""
    result = np.zeros(
        (image.shape[0] + TITLE_HEIGHT, image.shape[1], 3),
        dtype=np.uint8,
    )
    result[TITLE_HEIGHT:] = image
    cv2.putText(
        result,
        title,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return cast(np.ndarray, result)


def _join_panels(panels: list[np.ndarray]) -> np.ndarray:
    """Pad panels to one height and concatenate them horizontally."""
    maximum_height = max(panel.shape[0] for panel in panels)
    padded_panels = []
    for panel in panels:
        if panel.shape[0] == maximum_height:
            padded_panels.append(panel)
            continue
        padded = np.zeros(
            (maximum_height, panel.shape[1], 3),
            dtype=np.uint8,
        )
        padded[: panel.shape[0]] = panel
        padded_panels.append(padded)
    return cast(np.ndarray, np.concatenate(padded_panels, axis=1))


def _format_metric(value: float | None) -> str:
    """Format an optional percentage for a compact panel title."""
    return "N/A" if value is None else f"{value:.1f}"


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


if __name__ == "__main__":
    main()
