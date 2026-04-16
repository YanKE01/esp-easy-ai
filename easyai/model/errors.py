"""Model translation and validation errors."""


class ModelTranslationError(Exception):
    """Raised when unified model layers cannot be translated to a backend."""


class ModelDefinitionError(Exception):
    """Raised when a translated model cannot be instantiated or executed."""
