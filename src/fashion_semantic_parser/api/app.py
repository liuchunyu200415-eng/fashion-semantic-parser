"""FastAPI application factory."""

from fastapi import FastAPI, HTTPException

from fashion_semantic_parser.common.exceptions import FashionParserError
from fashion_semantic_parser.models.schemas import (
    MultimodalQueryRequest,
    MultimodalQueryResponse,
)
from fashion_semantic_parser.service.parser_service import FashionParserService


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="Fashion Semantic Parser", version="0.1.0")
    parser_service = FashionParserService()

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
            raise HTTPException(status_code=503, detail=str(error)) from error

    return app


app = create_app()

