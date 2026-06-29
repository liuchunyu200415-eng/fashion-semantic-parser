"""High-level orchestration service for multimodal fashion parsing."""

from fashion_semantic_parser.common.exceptions import ModelNotReadyError
from fashion_semantic_parser.models.schemas import (
    MultimodalQueryRequest,
    MultimodalQueryResponse,
)


class FashionParserService:
    """Coordinate visual parsing, RAG retrieval, and answer generation."""

    def answer_query(
        self,
        request: MultimodalQueryRequest,
    ) -> MultimodalQueryResponse:
        """Answer a multimodal fashion query.

        Args:
            request: User image path and natural language query.

        Returns:
            Structured answer with localized regions and references.

        Raises:
            ModelNotReadyError: Always raised until model adapters are implemented.
        """
        raise ModelNotReadyError(
            "Model adapters are not implemented yet. "
            f"Received query for image: {request.image_path}"
        )
