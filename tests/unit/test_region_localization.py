"""Tests for Grounding DINO + SAM-HQ localization orchestration."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from fashion_semantic_parser.common.exceptions import ModelNotReadyError
from fashion_semantic_parser.models.segmentation import SegmentationSubjectROI
from fashion_semantic_parser.service.grounded_sam_hq import (
    GroundedMaskCandidate,
    GroundedSAMHQSettings,
    _rank_grounding_results,
)
from fashion_semantic_parser.service.region_localization import (
    GroundedSAMHQRegionLocalizationService,
)


class _FakeGroundedPredictor:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], str]] = []

    def predict(
        self,
        image_bgr: np.ndarray,
        prompt: str,
    ) -> list[GroundedMaskCandidate]:
        self.calls.append((image_bgr.shape, prompt))
        mask = np.zeros(image_bgr.shape[:2], dtype=bool)
        mask[3:8, 2:6] = True
        return [
            GroundedMaskCandidate(
                box=(1.0, 2.0, 8.0, 10.0),
                confidence=0.87,
                phrase="neckline",
                mask=mask,
                mask_quality=0.91,
            )
        ]


class _FakeSubjectROIDetector:
    def __init__(self, roi: SegmentationSubjectROI | None) -> None:
        self.roi = roi
        self.calls: list[Path] = []

    def detect(self, image_path: Path) -> SegmentationSubjectROI | None:
        self.calls.append(image_path)
        return self.roi


def _service(
    predictor: _FakeGroundedPredictor,
    *,
    detector: _FakeSubjectROIDetector | None = None,
    margin: float = 0.0,
) -> GroundedSAMHQRegionLocalizationService:
    return GroundedSAMHQRegionLocalizationService(
        predictor=predictor,
        subject_roi_detector=detector,
        settings=GroundedSAMHQSettings(
            device="cpu",
            precision="fp32",
            subject_roi_margin=margin,
            min_mask_area=4,
        ),
    )


def test_localization_crops_roi_and_restores_mask_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mask polygons and boxes must return to original image coordinates."""
    image_path = tmp_path / "image.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    monkeypatch.setattr(
        "fashion_semantic_parser.service.region_localization.resolve_project_path",
        lambda _: image_path,
    )
    predictor = _FakeGroundedPredictor()
    subject_roi = SegmentationSubjectROI(
        x_min=5,
        y_min=4,
        x_max=20,
        y_max=16,
    )

    result = _service(predictor).localize(
        "data/image.jpg",
        "这件衣服的领口",
        subject_roi=subject_roi,
        auto_subject_roi=False,
    )

    assert predictor.calls == [((12, 15, 3), "neckline")]
    assert result.subject_roi == subject_roi
    assert result.subject_roi_source == "manual"
    assert len(result.regions) == 1
    region = result.regions[0]
    assert region.region_label == "neckline"
    assert region.matched_text == "neckline"
    assert region.box.model_dump() == {
        "x_min": 7.0,
        "y_min": 7.0,
        "x_max": 11.0,
        "y_max": 12.0,
    }
    assert min(region.mask[0][::2]) >= 5.0
    assert min(region.mask[0][1::2]) >= 4.0


def test_localization_uses_detected_roi_and_reports_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic person detection should precede text localization."""
    image_path = tmp_path / "image.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    monkeypatch.setattr(
        "fashion_semantic_parser.service.region_localization.resolve_project_path",
        lambda _: image_path,
    )
    roi = SegmentationSubjectROI(x_min=5, y_min=4, x_max=20, y_max=16)
    detector = _FakeSubjectROIDetector(roi)
    predictor = _FakeGroundedPredictor()

    result = _service(predictor, detector=detector).localize(
        "data/image.jpg",
        "口袋",
        auto_subject_roi=True,
    )

    assert detector.calls == [image_path]
    assert predictor.calls == [((12, 15, 3), "pocket")]
    assert result.subject_roi == roi
    assert result.subject_roi_source == "detected"


def test_localization_falls_back_to_full_image_without_person(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No person detection should remain an explicit full-image fallback."""
    image_path = tmp_path / "image.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    monkeypatch.setattr(
        "fashion_semantic_parser.service.region_localization.resolve_project_path",
        lambda _: image_path,
    )
    detector = _FakeSubjectROIDetector(None)
    predictor = _FakeGroundedPredictor()

    result = _service(predictor, detector=detector).localize(
        "data/image.jpg",
        "口袋",
        auto_subject_roi=True,
    )

    assert predictor.calls == [((20, 30, 3), "pocket")]
    assert result.subject_roi is None
    assert result.subject_roi_source == "full_image_fallback"


def test_localization_rejects_wrong_sam_mask_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed external model result must not produce invalid coordinates."""
    image_path = tmp_path / "image.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    monkeypatch.setattr(
        "fashion_semantic_parser.service.region_localization.resolve_project_path",
        lambda _: image_path,
    )

    class _WrongShapePredictor:
        def predict(
            self,
            image_bgr: np.ndarray,
            prompt: str,
        ) -> list[GroundedMaskCandidate]:
            return [
                GroundedMaskCandidate(
                    box=(0, 0, 2, 2),
                    confidence=0.9,
                    phrase=prompt,
                    mask=np.ones((2, 2), dtype=bool),
                    mask_quality=0.9,
                )
            ]

    service = GroundedSAMHQRegionLocalizationService(
        predictor=_WrongShapePredictor(),
        settings=GroundedSAMHQSettings(device="cpu", precision="fp32"),
    )
    with pytest.raises(ModelNotReadyError, match="dimensions"):
        service.localize(
            "data/image.jpg",
            "pocket",
            auto_subject_roi=False,
        )


def test_grounding_results_are_clamped_ranked_and_limited() -> None:
    """External detections should be normalized before SAM receives them."""

    class _Detections:
        xyxy = np.asarray(
            [
                [-5, 1, 8, 9],
                [2, 2, 7, 8],
                [9, 9, 3, 3],
            ]
        )
        confidence = np.asarray([0.7, 0.9, 0.99])

    boxes, scores, phrases = _rank_grounding_results(
        _Detections(),
        ["first", "second", "invalid"],
        image_width=10,
        image_height=10,
        limit=1,
    )

    assert boxes == [[2.0, 2.0, 7.0, 8.0]]
    assert scores == [0.9]
    assert phrases == ["second"]
