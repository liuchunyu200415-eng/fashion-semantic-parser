"""Tests for the frozen complete-query local re-encoding service."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

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
from fashion_semantic_parser.service.dense_local_runtime import _BatchedDinoV2Encoder
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
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

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

    def encode_dense_batch(
        self,
        images_rgb: list[np.ndarray],
    ) -> tuple[DinoV2DenseFeatureMap, ...]:
        self.batch_sizes.append(len(images_rgb))
        return tuple(self.encode_dense(image) for image in images_rgb)


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


class _FakeDinoModel:
    def __init__(self, torch_module) -> None:
        self.torch = torch_module

    def forward_features(self, image_tensor):
        batch_size = image_tensor.shape[0]
        values = np.arange(
            batch_size * 4 * 2,
            dtype=np.float32,
        ).reshape(batch_size, 4, 2)
        values[:, :, 0] += 1.0
        return {"x_norm_patchtokens": _FakeTensor(values)}


class _FakeBaseDinoEncoder:
    def __init__(self, torch_module) -> None:
        self._torch = torch_module
        self._model = _FakeDinoModel(torch_module)
        self.settings = SimpleNamespace(
            input_size=4,
            patch_size=2,
            feature_dimension=2,
            precision="fp32",
        )

    def load(self) -> None:
        pass

    def encode_dense(self, image_rgb: np.ndarray) -> DinoV2DenseFeatureMap:
        raise AssertionError("Single-image delegate is not used in this test.")

    def _normalized_image_tensor(self, image_rgb: np.ndarray):
        return np.zeros((1, 3, 4, 4), dtype=np.float32)


class _FakeTorch:
    float16 = np.float16
    float32 = np.float32

    def __init__(self) -> None:
        self.nn = SimpleNamespace(functional=SimpleNamespace(normalize=self._normalize))

    @staticmethod
    def cat(values, dim=0):
        return np.concatenate(values, axis=dim)

    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def _normalize(values, dim):
        array = values.values if isinstance(values, _FakeTensor) else values
        norms = np.linalg.norm(array, axis=dim, keepdims=True)
        return _FakeTensor(array / norms)


class _FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def cpu(self):
        return self

    @property
    def shape(self):
        return self.values.shape

    def float(self):
        return self

    def numpy(self) -> np.ndarray:
        return self.values


def test_engine_preserves_query_and_uses_local_mask_with_coarse_box() -> None:
    """The full query must drive local re-encoding without replacing its Box."""
    projector = _FakeProjector()
    scorer = _SequenceScorer()
    encoder = _FakeEncoder()
    engine = DenseLocalReencodingEngine(
        DenseLocalReencodingSettings(
            crop_fraction=0.30,
            max_crops=1,
        ),
        runtime=DenseLocalRuntimeBundle(
            projector=projector,
            image_encoder=encoder,
            scorer=scorer,
        ),
    )

    result = engine.predict(
        np.zeros((10, 10, 3), dtype=np.uint8),
        "衣服左侧的银色拉链",
    )

    assert projector.queries == ["衣服左侧的银色拉链"]
    assert scorer.call_count == 2
    assert encoder.batch_sizes == [1]
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


def test_batched_adapter_preserves_crop_order_and_geometry() -> None:
    """One forward pass must return one normalized dense grid per crop."""
    torch = _FakeTorch()
    encoder = _BatchedDinoV2Encoder(  # type: ignore[arg-type]
        _FakeBaseDinoEncoder(torch)
    )
    images = [
        np.zeros((2, 4, 3), dtype=np.uint8),
        np.zeros((4, 2, 3), dtype=np.uint8),
        np.zeros((3, 3, 3), dtype=np.uint8),
    ]

    results = encoder.encode_dense_batch(images)

    assert len(results) == 3
    assert all(result.features.shape == (2, 2, 2) for result in results)
    assert results[0].geometry.original_height == 2
    assert results[1].geometry.original_width == 2
    assert not np.array_equal(results[0].features, results[1].features)
    assert all(
        np.allclose(np.linalg.norm(result.features, axis=2), 1.0) for result in results
    )
