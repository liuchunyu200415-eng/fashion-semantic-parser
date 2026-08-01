"""Tests for Grounding DINO + SAM-HQ localization orchestration."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from fashion_semantic_parser.common.exceptions import ModelNotReadyError
from fashion_semantic_parser.models.localization import RegionLocalizationPrediction
from fashion_semantic_parser.models.segmentation import (
    SegmentationBoundingBox,
    SegmentationInstance,
    SegmentationPrediction,
    SegmentationSubjectROI,
)
from fashion_semantic_parser.service.grounded_sam_hq import (
    GroundedMaskCandidate,
    GroundedSAMHQSettings,
    _rank_grounding_results,
)
from fashion_semantic_parser.service.region_localization import (
    GroundedSAMHQRegionLocalizationService,
    HybridRegionLocalizationService,
    Mask2FormerPartLocalizationService,
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


class _FakePartSegmentationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, SegmentationSubjectROI | None, bool]] = []

    def segment(
        self,
        image_path: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = False,
    ) -> SegmentationPrediction:
        self.calls.append((image_path, subject_roi, auto_subject_roi))
        return SegmentationPrediction(
            image_path=image_path,
            instances=[
                _part_instance("collar", 2, 0.91),
                _part_instance("pocket", 6, 0.83),
                _part_instance("ruffle", 17, 0.72),
            ],
            subject_roi=subject_roi,
            subject_roi_source="manual" if subject_roi is not None else None,
        )


class _FakeFallbackLocalizationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, SegmentationSubjectROI | None, bool]] = []

    def localize(
        self,
        image_path: str,
        query: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = True,
    ) -> RegionLocalizationPrediction:
        self.calls.append((image_path, query, subject_roi, auto_subject_roi))
        return RegionLocalizationPrediction(
            image_path=image_path,
            query=query,
            regions=[],
        )


def _part_instance(
    label: str,
    category_id: int,
    confidence: float,
) -> SegmentationInstance:
    return SegmentationInstance(
        category_id=category_id,
        category_label=label,
        confidence=confidence,
        box=SegmentationBoundingBox(
            x_min=10.0,
            y_min=20.0,
            x_max=40.0,
            y_max=60.0,
        ),
        mask=[[10.0, 20.0, 40.0, 20.0, 40.0, 60.0, 10.0, 60.0]],
    )


def _service(
    predictor: _FakeGroundedPredictor,
    *,
    detector: _FakeSubjectROIDetector | None = None,
    margin: float = 0.0,
    grounding_prompt_override: str | None = None,
) -> GroundedSAMHQRegionLocalizationService:
    return GroundedSAMHQRegionLocalizationService(
        predictor=predictor,
        subject_roi_detector=detector,
        grounding_prompt_override=grounding_prompt_override,
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


def test_localization_can_override_grounding_prompt_without_changing_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt experiments should retain the query-derived API category."""
    image_path = tmp_path / "image.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    monkeypatch.setattr(
        "fashion_semantic_parser.service.region_localization.resolve_project_path",
        lambda _: image_path,
    )
    predictor = _FakeGroundedPredictor()

    result = _service(
        predictor,
        grounding_prompt_override=" shirt collar .  clothing collar ",
    ).localize(
        "data/image.jpg",
        "这件衣服的衣领",
        auto_subject_roi=False,
    )

    assert predictor.calls == [((20, 30, 3), "shirt collar . clothing collar")]
    assert result.regions[0].region_label == "collar"


def test_localization_rejects_empty_grounding_prompt_override() -> None:
    """An explicit prompt override must carry usable grounding text."""
    with pytest.raises(ValueError, match="cannot be empty"):
        _service(
            _FakeGroundedPredictor(),
            grounding_prompt_override="   ",
        )


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


def test_supervised_part_localization_filters_predictions_by_query() -> None:
    """Known part queries should retain only the matching supervised class."""
    segmentation = _FakePartSegmentationService()
    service = Mask2FormerPartLocalizationService(
        segmentation_service=segmentation,
    )
    subject_roi = SegmentationSubjectROI(
        x_min=1.0,
        y_min=2.0,
        x_max=100.0,
        y_max=200.0,
    )

    result = service.localize(
        "data/example.jpg",
        "这件衣服的衣领",
        subject_roi=subject_roi,
        auto_subject_roi=False,
    )

    assert segmentation.calls == [("data/example.jpg", subject_roi, False)]
    assert [region.region_label for region in result.regions] == ["collar"]
    assert result.regions[0].mask
    assert result.subject_roi == subject_roi


def test_supervised_part_localization_expands_generic_decoration_query() -> None:
    """Generic decoration language should retain every supervised decoration."""
    service = Mask2FormerPartLocalizationService(
        segmentation_service=_FakePartSegmentationService(),
    )

    result = service.localize(
        "data/example.jpg",
        "衣服上有什么装饰？",
        auto_subject_roi=False,
    )

    assert [region.region_label for region in result.regions] == ["ruffle"]


def test_hybrid_localization_routes_only_uncovered_queries_to_fallback() -> None:
    """Missing direct supervision should use Grounded SAM without double loading."""
    segmentation = _FakePartSegmentationService()
    supervised = Mask2FormerPartLocalizationService(
        segmentation_service=segmentation,
    )
    fallback = _FakeFallbackLocalizationService()
    service = HybridRegionLocalizationService(supervised, fallback)

    collar = service.localize(
        "data/example.jpg",
        "衣领",
        auto_subject_roi=False,
    )
    cuff = service.localize(
        "data/example.jpg",
        "袖口",
        auto_subject_roi=False,
    )

    assert [region.region_label for region in collar.regions] == ["collar"]
    assert cuff.regions == []
    assert len(segmentation.calls) == 1
    assert fallback.calls == [("data/example.jpg", "袖口", None, False)]


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
