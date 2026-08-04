"""Render a compact eight-region PRD 3.1.2 API acceptance overview."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.visualize_localization_comparison import (
    PREDICTION_COLORS,
    _draw_prediction,
    _prediction_masks,
)

HEADER_HEIGHT = 58
CARD_HEADER_HEIGHT = 42
PANEL_LABEL_HEIGHT = 24
CARD_IMAGE_HEIGHT = 270
GRID_GAP = 12


def add_src_to_python_path() -> None:
    """Add the local package when the project is not installed."""
    src_path = PROJECT_ROOT / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse acceptance report and overview output paths."""
    parser = argparse.ArgumentParser(
        description="Render the eight PRD 3.1.2 API acceptance cases."
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--card-width", type=int, default=720)
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    """Load saved API responses and render the acceptance overview."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path

    if args.columns < 1:
        raise ValueError("--columns must be positive")
    if args.card_width < 480:
        raise ValueError("--card-width must be at least 480")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    report_path = resolve_project_path(args.report)
    report = _read_json(report_path)
    rows = report.get("requests")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Acceptance report must contain non-empty requests.")

    cards = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError("Every acceptance request must be an object.")
        response_path_text = str(row.get("response_json", "")).strip()
        if not response_path_text:
            raise ValueError(
                "Acceptance requests are missing response_json. "
                "Rerun accept_localization_api.py with --responses-dir."
            )
        response = _read_json(resolve_project_path(response_path_text))
        localization = response.get("localization")
        if not isinstance(localization, dict):
            raise ValueError(f"Response is missing localization: {response_path_text}")
        image_path = str(localization.get("image_path") or row.get("image_path", ""))
        image = cv2.imread(str(resolve_project_path(image_path)))
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        regions = localization.get("regions")
        if not isinstance(regions, list):
            raise ValueError(f"Localization is missing regions: {response_path_text}")

        masks = _prediction_masks(
            localization,
            height=image.shape[0],
            width=image.shape[1],
        )
        overlay = _draw_prediction(
            image,
            localization,
            masks,
            color=PREDICTION_COLORS[(index - 1) % len(PREDICTION_COLORS)],
            alpha=args.alpha,
        )
        cards.append(
            _build_case_card(
                image,
                overlay,
                row,
                index=index,
                card_width=args.card_width,
            )
        )

    overview = _build_overview(
        cards,
        columns=args.columns,
        accepted=bool(report.get("accepted")),
        passed_count=sum(_row_passed(row) for row in rows),
        total_seconds=float(report.get("total_elapsed_seconds", 0.0)),
    )
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overview):
        raise ValueError(f"Unable to write visualization: {output_path}")

    print(
        json.dumps(
            {
                "accepted": bool(report.get("accepted")),
                "request_count": len(rows),
                "total_elapsed_seconds": float(
                    report.get("total_elapsed_seconds", 0.0)
                ),
                "visualization": args.output,
                "width": int(overview.shape[1]),
                "height": int(overview.shape[0]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _build_case_card(
    original: np.ndarray,
    prediction: np.ndarray,
    row: dict[str, Any],
    *,
    index: int,
    card_width: int,
) -> np.ndarray:
    """Build one stable Original/Localized acceptance card."""
    panel_width = card_width // 2
    card_height = CARD_HEADER_HEIGHT + PANEL_LABEL_HEIGHT + CARD_IMAGE_HEIGHT
    card = np.full((card_height, card_width, 3), 18, dtype=np.uint8)
    passed = _row_passed(row)
    status = "PASS" if passed else "FAIL"
    status_color = (70, 190, 105) if passed else (65, 65, 220)
    region = str(row.get("target_region", "region")).upper()
    roi_source = str(row.get("subject_roi_source", "missing"))
    region_count = int(row.get("region_count", 0))
    elapsed = float(row.get("elapsed_seconds", 0.0))
    title = (
        f"{index}. {region}  |  {status}  |  ROI: {roi_source}  |  "
        f"regions: {region_count}  |  {elapsed:.2f}s"
    )
    cv2.putText(
        card,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        status_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        card,
        "Original",
        (12, CARD_HEADER_HEIGHT + 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        card,
        "Localized mask + box + subject ROI",
        (panel_width + 12, CARD_HEADER_HEIGHT + 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    image_top = CARD_HEADER_HEIGHT + PANEL_LABEL_HEIGHT
    card[
        image_top : image_top + CARD_IMAGE_HEIGHT,
        :panel_width,
    ] = _fit_to_canvas(original, panel_width, CARD_IMAGE_HEIGHT)
    card[
        image_top : image_top + CARD_IMAGE_HEIGHT,
        panel_width:,
    ] = _fit_to_canvas(prediction, card_width - panel_width, CARD_IMAGE_HEIGHT)
    cv2.line(
        card,
        (panel_width, CARD_HEADER_HEIGHT),
        (panel_width, card_height - 1),
        (60, 60, 60),
        1,
    )
    return cast(np.ndarray, card)


def _fit_to_canvas(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Letterbox one image into a fixed panel without distortion."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Panel image must have three color channels.")
    scale = min(width / image.shape[1], height / image.shape[0])
    target_width = max(1, int(round(image.shape[1] * scale)))
    target_height = max(1, int(round(image.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=interpolation,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x_offset = (width - target_width) // 2
    y_offset = (height - target_height) // 2
    canvas[
        y_offset : y_offset + target_height,
        x_offset : x_offset + target_width,
    ] = resized
    return cast(np.ndarray, canvas)


def _build_overview(
    cards: list[np.ndarray],
    *,
    columns: int,
    accepted: bool,
    passed_count: int,
    total_seconds: float,
) -> np.ndarray:
    """Arrange fixed-size acceptance cards under one summary header."""
    if not cards:
        raise ValueError("At least one acceptance card is required.")
    card_height, card_width = cards[0].shape[:2]
    if any(card.shape[:2] != (card_height, card_width) for card in cards):
        raise ValueError("All acceptance cards must have the same dimensions.")
    rows = math.ceil(len(cards) / columns)
    width = columns * card_width + (columns + 1) * GRID_GAP
    height = HEADER_HEIGHT + rows * card_height + (rows + 1) * GRID_GAP
    overview = np.full((height, width, 3), 10, dtype=np.uint8)
    status = "PASS" if accepted else "FAIL"
    color = (70, 190, 105) if accepted else (65, 65, 220)
    cv2.putText(
        overview,
        (
            f"PRD 3.1.2 Localization API Acceptance  |  {status}  |  "
            f"{passed_count}/{len(cards)} passed  |  {total_seconds:.2f}s total"
        ),
        (GRID_GAP, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )
    for index, card in enumerate(cards):
        row = index // columns
        column = index % columns
        x = GRID_GAP + column * (card_width + GRID_GAP)
        y = HEADER_HEIGHT + GRID_GAP + row * (card_height + GRID_GAP)
        overview[y : y + card_height, x : x + card_width] = card
    return cast(np.ndarray, overview)


def _row_passed(row: dict[str, Any]) -> bool:
    """Apply the per-request functional checks used by the acceptance runner."""
    roi_source = row.get("subject_roi_source")
    roi_present = bool(row.get("subject_roi_present"))
    roi_valid = (roi_source == "detected" and roi_present) or (
        roi_source == "full_image_fallback" and not roi_present
    )
    return bool(
        row.get("expected_detected")
        and row.get("source_matched")
        and roi_valid
        and row.get("segmentation_present")
        and row.get("all_masks_present")
        and row.get("all_boxes_valid")
    )


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


if __name__ == "__main__":
    main()
