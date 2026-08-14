"""Tests for one-query localization visual proof generation."""

import numpy as np

from scripts.visualize_localization_prediction import build_visualization


def test_visualization_draws_localized_mask_and_box() -> None:
    """A predicted region must be visible beside the unchanged source image."""
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    payload = {
        "image_path": "images/example.jpg",
        "query": "the left sleeve",
        "regions": [
            {
                "region_label": "open_query_region",
                "matched_text": "the left sleeve",
                "confidence": 0.9,
                "box": {
                    "x_min": 20.0,
                    "y_min": 20.0,
                    "x_max": 60.0,
                    "y_max": 60.0,
                },
                "mask": [[20.0, 20.0, 60.0, 20.0, 60.0, 60.0, 20.0, 60.0]],
            }
        ],
        "subject_roi": None,
    }

    result = build_visualization(
        image,
        payload,
        query="the sleeve on the left side of the garment",
        panel_width=400,
        alpha=0.5,
    )

    assert result.shape[1] == 800
    assert result.shape[0] > 80
    assert not np.any(result[250, 100:300])
    assert np.any(result[250, 500:700])


def test_visualization_retains_empty_localization_case() -> None:
    """A miss must still produce an auditable Original / Localized artifact."""
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    payload = {
        "image_path": "images/example.jpg",
        "query": "silver zipper",
        "regions": [],
        "subject_roi": None,
    }

    result = build_visualization(
        image,
        payload,
        query="silver zipper",
        panel_width=320,
        alpha=0.45,
    )

    assert result.shape[1] == 640
    assert result.shape[0] > 40
