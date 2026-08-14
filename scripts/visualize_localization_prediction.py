"""Run the deployed PRD 3.1.2 backend and render one localization result."""

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import cv2
import numpy as np

try:
    _comparison = __import__(
        "scripts.visualize_localization_comparison",
        fromlist=["visualize_localization_comparison"],
    )
except ModuleNotFoundError:
    _comparison = __import__("visualize_localization_comparison")

PREDICTION_COLOR = (52, 152, 219)
HEADER_HEIGHT = 54


def add_src_to_python_path() -> None:
    """Add the local package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse one image, complete query, and visualization output path.

    Returns:
        Validated command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run dense local re-encoding and render Original / Localized panels."
        )
    )
    parser.add_argument("image", help="Project-relative product image path.")
    parser.add_argument("--query", required=True, help="Complete language query.")
    parser.add_argument(
        "--config",
        default="configs/localization_dense_local_reencoding.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    """Run one complete-query localization request and save its visual proof.

    Raises:
        ValueError: If visual settings or image/output paths are invalid.
    """
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.service.dense_local_reencoding import (
        DenseLocalReencodingRegionLocalizationService,
    )

    if args.panel_width < 320:
        raise ValueError("--panel-width must be at least 320.")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between zero and one.")

    prediction = DenseLocalReencodingRegionLocalizationService(args.config).localize(
        args.image,
        args.query,
        auto_subject_roi=False,
    )
    payload = prediction.model_dump(mode="json")
    image_path = cast(Path, resolve_project_path(args.image))
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {args.image}")

    visualization = build_visualization(
        image,
        payload,
        query=args.query,
        panel_width=args.panel_width,
        alpha=args.alpha,
    )
    output_path = cast(Path, resolve_project_path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), visualization):
        raise ValueError(f"Unable to write visualization: {output_path}")

    if args.json_output:
        json_path = cast(Path, resolve_project_path(args.json_output))
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "query": args.query,
                "region_count": len(prediction.regions),
                "visualization": args.output,
                "prediction": args.json_output,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


def build_visualization(
    image: np.ndarray,
    payload: dict[str, object],
    *,
    query: str,
    panel_width: int,
    alpha: float,
) -> np.ndarray:
    """Render the original image and the query-localized Mask and Box.

    Args:
        image: Source BGR image.
        payload: Serialized region-localization prediction.
        query: Complete language expression used by inference.
        panel_width: Width of each comparison panel in pixels.
        alpha: Predicted Mask overlay opacity.

    Returns:
        Side-by-side Original / Localized BGR visualization.

    Raises:
        ValueError: If the source image is not three-channel BGR.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Localization visualization requires an HxWx3 image.")
    masks = _comparison._prediction_masks(
        payload,
        height=image.shape[0],
        width=image.shape[1],
    )
    localized = _comparison._draw_prediction(
        image,
        payload,
        masks,
        color=PREDICTION_COLOR,
        alpha=alpha,
    )
    regions = payload.get("regions")
    region_count = len(regions) if isinstance(regions, list) else 0
    original_panel = _comparison._add_title(
        _comparison._resize_panel(image, panel_width),
        "Original",
    )
    localized_panel = _comparison._add_title(
        _comparison._resize_panel(localized, panel_width),
        f"Localized | regions={region_count}",
    )
    joined = _comparison._join_panels([original_panel, localized_panel])
    return _add_query_header(joined, query)


def _add_query_header(image: np.ndarray, query: str) -> np.ndarray:
    """Add the complete query when OpenCV can render every character."""
    result = np.zeros(
        (image.shape[0] + HEADER_HEIGHT, image.shape[1], 3),
        dtype=np.uint8,
    )
    result[HEADER_HEIGHT:] = image
    display_query = query if query.isascii() else "Complete query saved in prediction JSON"
    cv2.putText(
        result,
        f"Query: {display_query}",
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return cast(np.ndarray, result)


if __name__ == "__main__":
    main()
