"""Computer-vision scope — this pipeline only supports image-based CV models."""

from __future__ import annotations

from core.types import ModelProfile

# Layer name substrings that indicate spatial (image) processing
SPATIAL_LAYER_MARKERS = ("Conv", "DepthwiseConv", "Conv2D", "SeparableConv", "Conv1d")

# Layers typical of NLP / audio / time-series — rejected when no spatial ops present
NON_CV_LAYER_MARKERS = ("LSTM", "GRU", "RNN", "Embedding", "TransformerEncoder")

VALID_IMAGE_CHANNELS = (1, 3, 4)  # grayscale, RGB, RGBA


def has_spatial_layers(layer_inventory: dict[str, int]) -> bool:
    return any(
        any(marker in layer_name for marker in SPATIAL_LAYER_MARKERS)
        for layer_name in layer_inventory
    )


def has_vision_transformer_layers(layer_inventory: dict[str, int]) -> bool:
    return any(
        "Attention" in layer_name or "Multihead" in layer_name
        for layer_name in layer_inventory
    )


def is_non_cv_model(layer_inventory: dict[str, int]) -> bool:
    """True when the graph looks like NLP / audio / time-series, not vision."""
    if has_spatial_layers(layer_inventory):
        return False
    if has_vision_transformer_layers(layer_inventory):
        return False
    return any(
        any(marker in layer_name for marker in NON_CV_LAYER_MARKERS)
        for layer_name in layer_inventory
    )


def validate_image_input_shape(shape: tuple[int, ...]) -> None:
    if len(shape) != 3:
        raise ValueError(
            f"Computer-vision pipeline expects image input (C, H, W); got {shape!r}. "
            "Sequence or tabular inputs are not supported."
        )
    channels = shape[0]
    if channels not in VALID_IMAGE_CHANNELS:
        raise ValueError(
            f"Expected 1, 3, or 4 input channels (grayscale/RGB/RGBA); got {channels}."
        )


def validate_cv_model(profile: ModelProfile) -> None:
    """Raise ValueError if the profile does not represent a CV image model."""
    if is_non_cv_model(profile.layer_type_inventory):
        raise ValueError(
            "This pipeline only supports computer-vision models (classification, detection, "
            "segmentation, etc.). The loaded graph appears to be NLP, audio, or time-series."
        )
    if not has_spatial_layers(profile.layer_type_inventory) and not has_vision_transformer_layers(
        profile.layer_type_inventory
    ):
        raise ValueError(
            "No vision layers detected (Conv/DepthwiseConv or ViT attention). "
            "Only image-based CV models are supported."
        )
    validate_image_input_shape(profile.input_resolution)
