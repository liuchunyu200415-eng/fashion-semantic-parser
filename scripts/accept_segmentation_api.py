"""Run repeatable eight-class acceptance against the query API."""

import argparse
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


def add_src_to_python_path() -> None:
    """Add the local src directory when the package is not installed yet."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse API acceptance arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Select one saved high-confidence example per category and verify "
            "the default /v1/query segmentation pipeline."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    """Call the query API once per category and write an acceptance report."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path

    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1.")
    if args.timeout_seconds <= 0.0:
        raise ValueError("--timeout-seconds must be positive.")

    validation_path = resolve_project_path(args.val_json)
    prediction_path = resolve_project_path(args.predictions)
    output_path = resolve_project_path(args.output)
    validation = _read_json(validation_path)
    predictions = _read_json(prediction_path)
    cases = select_high_confidence_cases(
        validation,
        predictions,
        score_threshold=args.score_threshold,
    )

    endpoint = f"{args.base_url.rstrip('/')}/v1/query"
    rows: list[dict[str, Any]] = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        response = _post_json(
            endpoint,
            {
                "image_path": case["image_path"],
                "query": "What garments are visible?",
            },
            timeout_seconds=args.timeout_seconds,
        )
        row = summarize_query_response(case, response)
        rows.append(row)
        print(
            f"[{index}/{total}] expected={row['expected_category']} "
            f"hit={row['expected_detected']} "
            f"roi={row['subject_roi_source']} "
            f"instances={row['instance_count']}",
            flush=True,
        )

    report = build_acceptance_report(
        base_url=args.base_url,
        validation_json=str(validation_path),
        predictions_json=str(prediction_path),
        score_threshold=args.score_threshold,
        rows=rows,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_json = json.dumps(report, ensure_ascii=False, indent=2)
    output_path.write_text(output_json + "\n", encoding="utf-8")
    print(output_json, flush=True)
    if not report["accepted"]:
        raise RuntimeError("Segmentation query API acceptance failed.")


def select_high_confidence_cases(
    validation: Any,
    predictions: Any,
    *,
    score_threshold: float,
) -> list[dict[str, Any]]:
    """Select the highest-confidence saved prediction for every category."""
    if not isinstance(validation, dict):
        raise ValueError("Validation JSON must be a COCO mapping.")
    if not isinstance(predictions, list):
        raise ValueError("Predictions JSON must be a COCO result list.")

    categories = {
        int(category["id"]): str(category["name"])
        for category in validation.get("categories", [])
    }
    images = {
        int(image["id"]): str(image["file_name"])
        for image in validation.get("images", [])
    }
    if not categories:
        raise ValueError("Validation JSON does not define categories.")
    if not images:
        raise ValueError("Validation JSON does not define images.")
    selected: dict[int, dict[str, Any]] = {}
    for prediction in predictions:
        category_id = int(prediction.get("category_id", -1))
        image_id = int(prediction.get("image_id", -1))
        score = float(prediction.get("score", 0.0))
        if (
            category_id not in categories
            or image_id not in images
            or score < score_threshold
        ):
            continue
        current = selected.get(category_id)
        if current is None or score > float(current["selected_score"]):
            selected[category_id] = {
                "category_id": category_id,
                "expected_category": categories[category_id],
                "image_id": image_id,
                "image_path": images[image_id],
                "selected_score": score,
            }

    missing_categories = [
        categories[category_id]
        for category_id in sorted(categories)
        if category_id not in selected
    ]
    if missing_categories:
        raise ValueError(
            "No prediction meets the score threshold for: "
            + ", ".join(missing_categories)
        )
    return [selected[category_id] for category_id in sorted(selected)]


def summarize_query_response(
    case: dict[str, Any],
    response: Any,
) -> dict[str, Any]:
    """Validate one query response and return compact acceptance details."""
    if not isinstance(response, dict):
        raise ValueError("Query API response must be a JSON object.")
    segmentation = response.get("segmentation")
    if not isinstance(segmentation, dict):
        raise ValueError("Query API response is missing segmentation.")
    instances = segmentation.get("instances")
    if not isinstance(instances, list):
        raise ValueError("Segmentation response is missing instances.")

    labels = [str(instance.get("category_label", "")) for instance in instances]
    all_masks_present = bool(instances) and all(
        bool(instance.get("mask")) for instance in instances
    )
    all_boxes_valid = bool(instances) and all(
        _valid_box(instance.get("box")) for instance in instances
    )
    return {
        **case,
        "expected_detected": case["expected_category"] in labels,
        "subject_roi_source": segmentation.get("subject_roi_source"),
        "subject_roi_present": isinstance(segmentation.get("subject_roi"), dict),
        "instance_count": len(instances),
        "detected_categories": labels,
        "all_masks_present": all_masks_present,
        "all_boxes_valid": all_boxes_valid,
    }


def build_acceptance_report(
    *,
    base_url: str,
    validation_json: str,
    predictions_json: str,
    score_threshold: float,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-category API checks into one acceptance decision."""
    category_counts = Counter(
        label for row in rows for label in row["detected_categories"]
    )
    checks = {
        "all_expected_detected": all(row["expected_detected"] for row in rows),
        "all_subject_rois_detected": all(
            row["subject_roi_source"] == "detected" and row["subject_roi_present"]
            for row in rows
        ),
        "all_masks_present": all(row["all_masks_present"] for row in rows),
        "all_boxes_valid": all(row["all_boxes_valid"] for row in rows),
    }
    return {
        "base_url": base_url,
        "endpoint": "/v1/query",
        "roi_request_mode": "config_default",
        "validation_json": validation_json,
        "predictions_json": predictions_json,
        "score_threshold": score_threshold,
        "request_count": len(rows),
        **checks,
        "accepted": all(checks.values()),
        "category_counts": dict(sorted(category_counts.items())),
        "requests": rows,
    }


def _valid_box(box: Any) -> bool:
    """Return whether an API xyxy box has positive area."""
    if not isinstance(box, dict):
        return False
    try:
        return float(box["x_max"]) > float(box["x_min"]) and float(
            box["y_max"]
        ) > float(box["y_min"])
    except (KeyError, TypeError, ValueError):
        return False


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> Any:
    """POST one JSON object using the Python standard library."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_json(path: Path) -> Any:
    """Read one JSON artifact."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    main()
