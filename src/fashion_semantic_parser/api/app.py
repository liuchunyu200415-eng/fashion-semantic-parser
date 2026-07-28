"""FastAPI application factory."""

from fastapi import FastAPI, HTTPException

from fashion_semantic_parser.common.exceptions import (
    FashionParserError,
    InvalidImageInputError,
)
from fashion_semantic_parser.config import Settings, load_settings
from fashion_semantic_parser.models.schemas import (
    MultimodalQueryRequest,
    MultimodalQueryResponse,
)
from fashion_semantic_parser.models.segmentation import (
    SegmentationPrediction,
    SegmentationRequest,
)
from fashion_semantic_parser.service.parser_service import FashionParserService
from fashion_semantic_parser.service.segmentation_runtime import (
    GarmentSegmentationService,
    SegmentationRuntime,
)


def create_app(
    *,
    settings: Settings | None = None,
    segmentation_service: SegmentationRuntime | None = None,
    parser_service: FashionParserService | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="Fashion Semantic Parser", version="0.1.0")
    if segmentation_service is None:
        app_settings = settings or load_settings()
        segmentation_service = GarmentSegmentationService(
            app_settings.segmentation.config_path
        )
    if parser_service is None:
        parser_service = FashionParserService(segmentation_service)

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

    return app


def _http_error(error: FashionParserError) -> HTTPException:
    """Map domain errors to stable inference API status codes."""
    status_code = 400 if isinstance(error, InvalidImageInputError) else 503
    return HTTPException(status_code=status_code, detail=str(error))


app = create_app()
