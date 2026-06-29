"""Custom exception hierarchy for the fashion semantic parser."""


class FashionParserError(Exception):
    """Base error for project-specific exceptions."""


class ConfigurationError(FashionParserError):
    """Raised when project configuration is invalid."""


class ModelNotReadyError(FashionParserError):
    """Raised when a model-dependent service is called before initialization."""
