"""Top-level package for Document Tools."""

__author__ = """deeptools.ai"""
__email__ = "contact@deeptools.ai"
__version__ = "0.0.1"


from .encoders import TARGET_MODELS, LayoutLMv2Encoder, LayoutLMv3Encoder, LayoutXLMEncoder
from .tokenize import tokenize_dataset

__all__ = ["LayoutLMv2Encoder", "LayoutLMv3Encoder", "LayoutXLMEncoder", "TARGET_MODELS", "tokenize_dataset"]
