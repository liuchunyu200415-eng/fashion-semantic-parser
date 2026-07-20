"""Experiment with conservative garment category-conflict suppression."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


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
            "Filter saved COCO predictions with conservative dress-versus-separates "
            "conflict suppression."
        )
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.8)
    parser.add_argument("--min-union-iou", type=float, default=0.8)
    parser.add_argument("--min-component-coverage", type=float, default=0.8)
    parser.add_argument("--score-margin", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    """Write filtered COCO predictions and a compact diagnostic report."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.service.segmentation_postprocessing import (
        find_composite_dress_conflicts,
    )

    prediction_path = _resolve_path(args.predictions, resolve_project_path)
    output_path = _resolve_path(args.output, resolve_project_path)
    report_path = _resolve_path(args.report, resolve_project_path)
    predictions = _read_prediction_list(prediction_path)
    output_predictions, report = _postprocess_predictions(
        predictions,
        score_threshold=args.score_threshold,
        min_union_iou=args.min_union_iou,
        min_component_coverage=args.min_component_coverage,
        score_margin=args.score_margin,
        conflict_finder=find_composite_dress_conflicts,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output_predictions, file, ensure_ascii=False)
        file.write("\n")
    report.update(
        {
            "predictions_json": prediction_path.as_posix(),
            "output_json": output_path.as_posix(),
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    report_path.write_text(report_json + "\n", encoding="utf-8")
    print(report_json)


def _postprocess_predictions(
    predictions: list[dict[str, Any]],
    *,
    score_threshold: float,
    min_union_iou: float,
    min_component_coverage: float,
    score_margin: float,
    conflict_finder: Callable[..., list[Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter by score, then remove only diagnosed composite dress conflicts."""
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be between zero and one")

    retained = [
        prediction
        for prediction in predictions
        if float(prediction.get("score", 0.0)) >= score_threshold
    ]
    predictions_by_image: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(
        list
    )
    for index, prediction in enumerate(retained):
        predictions_by_image[int(prediction["image_id"])].append((index, prediction))

    suppressed_indices: set[int] = set()
    examples: list[dict[str, Any]] = []
    affected_image_ids: set[int] = set()
    for image_id, image_predictions in predictions_by_image.items():
        scored_boxes = [
            _to_scored_box(index, prediction) for index, prediction in image_predictions
        ]
        conflicts = conflict_finder(
            scored_boxes,
            min_union_iou=min_union_iou,
            min_component_coverage=min_component_coverage,
            score_margin=score_margin,
        )
        for conflict in conflicts:
            suppressed_indices.add(conflict.dress_index)
            affected_image_ids.add(image_id)
            if len(examples) < 50:
                examples.append(
                    {
                        "image_id": image_id,
                        "dress_score": conflict.dress_score,
                        "separates_score": conflict.separates_score,
                        "union_iou": conflict.union_iou,
                        "top_coverage": conflict.top_coverage,
                        "lower_coverage": conflict.lower_coverage,
                    }
                )

    output_predictions = [
        prediction
        for index, prediction in enumerate(retained)
        if index not in suppressed_indices
    ]
    report = {
        "score_threshold": score_threshold,
        "min_union_iou": min_union_iou,
        "min_component_coverage": min_component_coverage,
        "score_margin": score_margin,
        "prediction_count_before_score_filter": len(predictions),
        "prediction_count_after_score_filter": len(retained),
        "suppressed_dress_count": len(suppressed_indices),
        "affected_image_count": len(affected_image_ids),
        "prediction_count_after_conflict_filter": len(output_predictions),
        "examples": examples,
    }
    return output_predictions, report


def _to_scored_box(index: int, prediction: dict[str, Any]) -> Any:
    """Convert one COCO xywh prediction to the shared xyxy helper type."""
    from fashion_semantic_parser.service.segmentation_postprocessing import (
        ScoredCategoryBox,
    )

    x_min, y_min, width, height = [float(value) for value in prediction["bbox"]]
    return ScoredCategoryBox(
        index=index,
        category_id=int(prediction["category_id"]),
        score=float(prediction.get("score", 0.0)),
        x_min=x_min,
        y_min=y_min,
        x_max=x_min + width,
        y_max=y_min + height,
    )


def _read_prediction_list(path: Path) -> list[dict[str, Any]]:
    """Read the flat COCO prediction list."""
    with path.open("r", encoding="utf-8") as file:
        predictions = json.load(file)
    if not isinstance(predictions, list):
        raise ValueError(f"Expected a COCO prediction list: {path}")
    if not all(isinstance(prediction, dict) for prediction in predictions):
        raise ValueError(f"COCO predictions must be JSON objects: {path}")
    return predictions


def _resolve_path(path: str, resolver: Callable[[str | Path], Path]) -> Path:
    """Use absolute paths directly and resolve project-relative paths."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolver(candidate)


if __name__ == "__main__":
    main()
