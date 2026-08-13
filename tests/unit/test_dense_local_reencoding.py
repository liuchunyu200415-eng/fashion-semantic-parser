"""Tests for the frozen complete-query local re-encoding service."""

from pathlib import Path

import cv2
import numpy as np

from fashion_semantic_parser.models.segmentation import SegmentationSubjectROI
from fashion_semantic_parser.service.dense_local_reencoding import (
    DenseLocalMaskResult,
    DenseLocalReencodingEngine,
    DenseLocalReencodingRegionLocalizationService,
    DenseLocalReencodingSettings,
    DenseLocalRuntimeBundle,
    load_dense_local_reencoding_settings,
)
from fashion_semantic_parser.service.dinov2_region_encoder import (
    DinoV2DenseFeatureMap,
    DinoV2LetterboxGeometry,
)


class _FakeProjector:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def project(self, query: str) -> np.ndarray:
        self.queries.append(query)
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


class _FakeEncoder:
    def encode_dense(self, image_rgb: np.ndarray) -> DinoV2DenseFeatureMap:
        height, width = image_rgb.shape[:2]
        return DinoV2DenseFeatureMap(
            features=np.ones((2, 2, 2), dtype=np.float32),
            geometry=DinoV2LetterboxGeometry(
                original_height=height,
                original_width=width,
                resized_height=max(height, width),
                resized_width=max(height, width),
                top=0,
                left=0,
                output_size=max(height, width),
            ),
        )


class _SequenceScorer:
    threshold = 0.5

    def __init__(self) -> None:
        self.call_count = 0

    def score(
        self,
        patch_features: np.ndarray,
        projected_query: np.ndarray,
    ) -> np.ndarray:
        self.call_count += 1
        if self.call_count == 1:
            return np.asarray([[0.0, 0.0], [0.0, 0.9]], dtype=np.float32)
        return np.ones((2, 2), dtype=np.float32)


class _FakeEngine:
    def __init__(self, result: DenseLocalMaskResult) -> None:
        self.result = result
        self.queries: list[str] = []

    def predict(self, image_rgb: np.ndarray, query: str) -> DenseLocalMaskResult:
        self.queries.append(query)
        return self.result


def test_engine_preserves_query_and_uses_local_mask_with_coarse_box() -> None:
    """The full query must drive local re-encoding without replacing its Box."""
    projector = _FakeProjector()
    scorer = _SequenceScorer()
    engine = DenseLocalReencodingEngine(
        DenseLocalReencodingSettings(
            crop_fraction=0.30,
            max_crops=1,
        ),
        runtime=DenseLocalRuntimeBundle(
            projector=projector,
            image_encoder=_FakeEncoder(),
            scorer=scorer,
        ),
    )

    result = engine.predict(
        np.zeros((10, 10, 3), dtype=np.uint8),
        "衣服左侧的银色拉链",
    )

    assert projector.queries == ["衣服左侧的银色拉链"]
    assert scorer.call_count == 2
    assert result.local_mask.sum() == 9
    assert result.coarse_box is not None
    assert result.confidence == 1.0


def test_service_offsets_local_mask_and_marks_independent_sources(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Retain ROI offsets and auditable Mask/Box provenance.

    Args:
        monkeypatch: Pytest patch helper for project path isolation.
        tmp_path: Temporary directory containing the test image.
    """
    image_path = tmp_path / "fashion.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((12, 14, 3), dtype=np.uint8))
    local_mask = np.zeros((8, 8), dtype=bool)
    local_mask[2:6, 1:5] = True
    engine = _FakeEngine(
        DenseLocalMaskResult(
            local_mask=local_mask,
            coarse_box=(0.0, 1.0, 7.0, 8.0),
            confidence=0.85,
        )
    )
    service = DenseLocalReencodingRegionLocalizationService(
        engine=engine,  # type: ignore[arg-type]
        settings=DenseLocalReencodingSettings(),
    )
    monkeypatch.setattr(
        "fashion_semantic_parser.service.dense_local_reencoding._resolve_image_path",
        lambda _: image_path,
    )

    prediction = service.localize(
        "fashion.jpg",
        "the silver zipper on the left side",
        subject_roi=SegmentationSubjectROI(
            x_min=2.0,
            y_min=3.0,
            x_max=10.0,
            y_max=11.0,
        ),
        auto_subject_roi=False,
    )

    assert engine.queries == ["the silver zipper on the left side"]
    assert prediction.query == "the silver zipper on the left side"
    assert prediction.subject_roi_source == "manual"
    assert len(prediction.regions) == 1
    region = prediction.regions[0]
    assert region.region_label == "open_query_region"
    assert region.mask_source == "dense_local_reencoding"
    assert region.box_source == "dense_coarse_localization"
    assert region.box.model_dump() == {
        "x_min": 2.0,
        "y_min": 4.0,
        "x_max": 9.0,
        "y_max": 11.0,
    }
    assert min(region.mask[0][::2]) >= 3.0
    assert min(region.mask[0][1::2]) >= 5.0


def test_service_routes_open_expressions_but_rejects_inventory_questions() -> None:
    """Open localization must not replace whole-image garment classification."""
    service = DenseLocalReencodingRegionLocalizationService(
        engine=_FakeEngine(  # type: ignore[arg-type]
            DenseLocalMaskResult(
                local_mask=np.ones((2, 2), dtype=bool),
                coarse_box=(0.0, 0.0, 2.0, 2.0),
                confidence=1.0,
            )
        )
    )

    assert service.accepts_query("外套里面带银色装饰的内搭区域") is True
    assert service.accepts_query("图中有哪些服饰？") is False


def test_deployment_settings_freeze_the_validated_crop_path() -> None:
    """Production config must preserve the held-out evaluation conditions."""
    settings = load_dense_local_reencoding_settings()

    assert settings.crop_fraction == 0.30
    assert settings.max_crops == 3
    assert settings.dinov2_config_path.endswith("dinov2_region_728.yaml")
    assert settings.checkpoint_path.endswith("train1000_steps1500.pt")
