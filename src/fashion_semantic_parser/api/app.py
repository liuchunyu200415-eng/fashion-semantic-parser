"""FastAPI application factory."""

from fastapi import FastAPI, HTTPException

from fashion_semantic_parser.common.exceptions import (
    FashionParserError,
    InvalidImageInputError,
)
from fashion_semantic_parser.config import Settings, load_settings
from fashion_semantic_parser.models.localization import (
    RegionLocalizationPrediction,
    RegionLocalizationRequest,
)
from fashion_semantic_parser.models.schemas import (
    MultimodalQueryRequest,
    MultimodalQueryResponse,
)
from fashion_semantic_parser.models.segmentation import (
    SegmentationPrediction,
    SegmentationRequest,
)
from fashion_semantic_parser.service.dense_local_reencoding import (
    DenseLocalReencodingRegionLocalizationService,
)
from fashion_semantic_parser.service.parser_service import FashionParserService
from fashion_semantic_parser.service.region_localization import (
    GroundedSAMHQRegionLocalizationService,
    HybridRegionLocalizationService,
    Mask2FormerPartLocalizationService,
    RegionLocalizationRuntime,
)
from fashion_semantic_parser.service.segmentation_runtime import (
    GarmentSegmentationService,
    SegmentationRuntime,
)


def create_app(
    *,
    settings: Settings | None = None,
    segmentation_service: SegmentationRuntime | None = None,
    localization_service: RegionLocalizationRuntime | None = None,
    parser_service: FashionParserService | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="Fashion Semantic Parser", version="0.1.0")
    app_settings = settings
    if segmentation_service is None:
        app_settings = app_settings or load_settings()
        segmentation_service = GarmentSegmentationService(
            app_settings.segmentation.config_path
        )
    if localization_service is None:
        app_settings = app_settings or load_settings()
        localization_settings = app_settings.localization
        if localization_settings.backend == "dense_local_reencoding":
            localization_service = DenseLocalReencodingRegionLocalizationService(
                localization_settings.dense_config_path
            )
        elif localization_settings.backend == "grounded_sam_hq":
            localization_service = GroundedSAMHQRegionLocalizationService(
                localization_settings.fallback_config_path
            )
        elif localization_settings.backend == "mask2former_parts":
            localization_service = Mask2FormerPartLocalizationService(
                localization_settings.config_path
            )
        else:
            localization_service = HybridRegionLocalizationService(
                Mask2FormerPartLocalizationService(localization_settings.config_path),
                GroundedSAMHQRegionLocalizationService(
                    localization_settings.fallback_config_path
                ),
                garment_segmentation_service=segmentation_service,
            )
    if parser_service is None:
        query_auto_subject_roi = (
            app_settings.segmentation.query_auto_subject_roi
            if app_settings is not None
            else True
        )
        parser_service = FashionParserService(
            segmentation_service,
            localization_service=localization_service,
            default_auto_subject_roi=query_auto_subject_roi,
        )

    @app.get("/health")
    def health_check() -> dict[str, str]:
        """Return service health status."""
        return {"status": "ok"}

    @app.post("/v1/query", response_model=MultimodalQueryResponse)
    def query_fashion_image(
        request: MultimodalQueryRequest,
    ) -> MultimodalQueryResponse:
        """Answer a fashion image query."""
        try:
            return parser_service.answer_query(request)
        except FashionParserError as error:
            raise _http_error(error) from error

    @app.post("/v1/segment", response_model=SegmentationPrediction)
    def segment_fashion_image(
        request: SegmentationRequest,
    ) -> SegmentationPrediction:
        """Return PRD 3.1.1 masks, boxes, labels, and confidence scores."""
        try:
            return segmentation_service.segment(
                request.image_path,
                subject_roi=request.subject_roi,
                auto_subject_roi=request.auto_subject_roi,
            )
        except FashionParserError as error:
            raise _http_error(error) from error

    @app.post("/v1/localize", response_model=RegionLocalizationPrediction)
    def localize_fashion_region(
        request: RegionLocalizationRequest,
    ) -> RegionLocalizationPrediction:
        """Return the mask and box for a natural-language fashion-part query."""
        auto_subject_roi = request.auto_subject_roi
        if auto_subject_roi is None:
            auto_subject_roi = request.subject_roi is None
        try:
            return localization_service.localize(
                request.image_path,
                request.query,
                subject_roi=request.subject_roi,
                auto_subject_roi=auto_subject_roi,
            )
        except FashionParserError as error:
            raise _http_error(error) from error

    return app


def _http_error(error: FashionParserError) -> HTTPException:
    """Map domain errors to stable inference API status codes."""
    status_code = 400 if isinstance(error, InvalidImageInputError) else 503
    return HTTPException(status_code=status_code, detail=str(error))


app = create_app()
