"""Run repeatable PRD 3.1.2 acceptance against the query API."""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

DIRECT_CASES = (
    ("collar", "collar", "这件衣服的衣领在哪里？", ("collar",)),
    ("pocket", "pocket", "这件衣服的口袋在哪里？", ("pocket",)),
    (
        "shoulder",
        "epaulette",
        "这件衣服的肩部在哪里？",
        ("epaulette", "shoulder"),
    ),
    ("decoration", "ruffle", "荷叶边在哪里？", ("ruffle",)),
)
DERIVED_CASES = (
    ("cuff", "这件上衣的袖口在哪里？", "derived from sleeve"),
    ("hem", "这件上衣的下摆在哪里？", "derived from top mask"),
    ("waist", "这件上衣的腰部在哪里？", "derived from top mask"),
    ("pattern", "这件上衣的图案在哪里？", "derived from top appearance"),
)


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
            "Select direct-supervision examples, add one derived-region image, "
            "and verify all eight PRD 3.1.2 query paths."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--derived-image", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--responses-dir",
        default=None,
        help="Optional directory for saving each complete query API response.",
    )
    return parser.parse_args()


def main() -> None:
    """Call the query API once per PRD region and write an acceptance report."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import (
        resolve_project_path,
        to_project_relative_path,
    )

    if args.timeout_seconds <= 0.0:
        raise ValueError("--timeout-seconds must be positive.")
    validation_path = resolve_project_path(args.val_json)
    derived_image_path = resolve_project_path(args.derived_image)
    output_path = resolve_project_path(args.output)
    responses_dir = (
        resolve_project_path(args.responses_dir) if args.responses_dir else None
    )
    if not derived_image_path.is_file():
        raise FileNotFoundError(f"Derived-region image not found: {args.derived_image}")
    if responses_dir is not None:
        responses_dir.mkdir(parents=True, exist_ok=True)

    validation = _read_json(validation_path)
    cases = build_acceptance_cases(validation, args.derived_image)
    endpoint = f"{args.base_url.rstrip('/')}/v1/query"
    rows: list[dict[str, Any]] = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        started_at = time.perf_counter()
        response = _post_json(
            endpoint,
            {
                "image_path": case["image_path"],
                "query": case["query"],
                "auto_subject_roi": True,
            },
            timeout_seconds=args.timeout_seconds,
        )
        elapsed_seconds = time.perf_counter() - started_at
        row = summarize_query_response(
            case,
            response,
            elapsed_seconds=elapsed_seconds,
        )
        if responses_dir is not None:
            response_path = responses_dir / (
                f"{index:02d}_{case['target_region']}.json"
            )
            response_path.write_text(
                json.dumps(response, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            row["response_json"] = to_project_relative_path(response_path)
        rows.append(row)
        print(
            f"[{index}/{total}] region={row['target_region']} "
            f"hit={row['expected_detected']} source={row['source_matched']} "
            f"regions={row['region_count']} elapsed={elapsed_seconds:.2f}s",
            flush=True,
        )

    report = build_acceptance_report(
        base_url=args.base_url,
        validation_json=str(validation_path),
        derived_image=args.derived_image,
        rows=rows,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_json = json.dumps(report, ensure_ascii=False, indent=2)
    output_path.write_text(output_json + "\n", encoding="utf-8")
    print(output_json, flush=True)
    if not report["accepted"]:
        raise RuntimeError("Localization query API acceptance failed.")


def build_acceptance_cases(
    validation: Any,
    derived_image_path: str,
) -> list[dict[str, Any]]:
    """Select deterministic direct examples and append derived-region cases."""
    if not isinstance(validation, dict):
        raise ValueError("Validation JSON must be a COCO mapping.")
    categories = {
        int(category["id"]): str(category["name"])
        for category in validation.get("categories", [])
    }
    category_ids = {name: category_id for category_id, name in categories.items()}
    images = {
        int(image["id"]): str(image["file_name"])
        for image in validation.get("images", [])
    }
    if not categories or not images:
        raise ValueError("Validation JSON must define categories and images.")

    cases = []
    annotations = validation.get("annotations", [])
    for target_region, source_category, query, expected_labels in DIRECT_CASES:
        category_id = category_ids.get(source_category)
        if category_id is None:
            raise ValueError(f"Validation JSON is missing category: {source_category}")
        candidates = [
            annotation
            for annotation in annotations
            if int(annotation.get("category_id", -1)) == category_id
            and int(annotation.get("image_id", -1)) in images
        ]
        if not candidates:
            raise ValueError(f"No validation annotation for: {source_category}")
        selected = max(candidates, key=_annotation_area)
        image_id = int(selected["image_id"])
        cases.append(
            {
                "target_region": target_region,
                "query": query,
                "image_id": image_id,
                "image_path": images[image_id],
                "expected_labels": list(expected_labels),
                "expected_source_contains": None,
                "case_source": f"largest_{source_category}_annotation",
            }
        )

    for target_region, query, expected_source in DERIVED_CASES:
        cases.append(
            {
                "target_region": target_region,
                "query": query,
                "image_id": None,
                "image_path": derived_image_path,
                "expected_labels": [target_region],
                "expected_source_contains": expected_source,
                "case_source": "derived_region_image",
            }
        )
    return cases


def summarize_query_response(
    case: dict[str, Any],
    response: Any,
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Validate one localization query response and return compact details."""
    if not isinstance(response, dict):
        raise ValueError("Query API response must be a JSON object.")
    localization = response.get("localization")
    if not isinstance(localization, dict):
        raise ValueError("Query API response is missing localization.")
    regions = localization.get("regions")
    if not isinstance(regions, list):
        raise ValueError("Localization response is missing regions.")

    expected_labels = set(case["expected_labels"])
    matching_regions = [
        region
        for region in regions
        if str(region.get("region_label", "")) in expected_labels
    ]
    expected_source = case.get("expected_source_contains")
    source_matched = expected_source is None or any(
        expected_source in str(region.get("matched_text", ""))
        for region in matching_regions
    )
    segmentation = response.get("segmentation")
    return {
        **case,
        "expected_detected": bool(matching_regions),
        "source_matched": source_matched,
        "subject_roi_source": localization.get("subject_roi_source"),
        "subject_roi_present": isinstance(localization.get("subject_roi"), dict),
        "segmentation_present": isinstance(segmentation, dict),
        "region_count": len(regions),
        "detected_labels": [str(region.get("region_label", "")) for region in regions],
        "matched_texts": [str(region.get("matched_text", "")) for region in regions],
        "all_masks_present": bool(regions)
        and all(bool(region.get("mask")) for region in regions),
        "all_boxes_valid": bool(regions)
        and all(_valid_box(region.get("box")) for region in regions),
        "elapsed_seconds": elapsed_seconds,
    }


def build_acceptance_report(
    *,
    base_url: str,
    validation_json: str,
    derived_image: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate all localization requests into one functional decision."""
    all_subject_rois_detected = all(
        row["subject_roi_source"] == "detected" and row["subject_roi_present"]
        for row in rows
    )
    checks = {
        "all_expected_detected": all(row["expected_detected"] for row in rows),
        "all_sources_matched": all(row["source_matched"] for row in rows),
        "all_roi_modes_valid": all(_valid_automatic_roi_state(row) for row in rows),
        "all_segmentations_present": all(row["segmentation_present"] for row in rows),
        "all_masks_present": all(row["all_masks_present"] for row in rows),
        "all_boxes_valid": all(row["all_boxes_valid"] for row in rows),
    }
    return {
        "base_url": base_url,
        "endpoint": "/v1/query",
        "validation_json": validation_json,
        "derived_image": derived_image,
        "request_count": len(rows),
        "total_elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
        "all_subject_rois_detected": all_subject_rois_detected,
        "roi_source_counts": {
            source if source is not None else "missing": sum(
                row["subject_roi_source"] == source for row in rows
            )
            for source in ("detected", "full_image_fallback", "manual", None)
            if any(row["subject_roi_source"] == source for row in rows)
        },
        **checks,
        "accepted": all(checks.values()),
        "requests": rows,
    }


def _valid_automatic_roi_state(row: dict[str, Any]) -> bool:
    """Accept a detected person box or an explicit full-image fallback."""
    source = row["subject_roi_source"]
    roi_present = row["subject_roi_present"]
    return (source == "detected" and roi_present) or (
        source == "full_image_fallback" and not roi_present
    )


def _annotation_area(annotation: dict[str, Any]) -> float:
    """Return COCO area with a bounding-box fallback."""
    area = float(annotation.get("area", 0.0))
    if area > 0.0:
        return area
    bbox = annotation.get("bbox", [])
    if isinstance(bbox, list) and len(bbox) == 4:
        return max(0.0, float(bbox[2])) * max(0.0, float(bbox[3]))
    return 0.0


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
