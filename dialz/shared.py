"""Shared utilities for extracting model layer lists."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from .vector import SteeringModel


def model_layer_list(model: SteeringModel | PreTrainedModel) -> torch.nn.ModuleList:
    """Extract the ``ModuleList`` of transformer blocks from a model.

    Supports Mistral-like (``model.layers``) and GPT-2-like (``transformer.h``)
    architectures. Unwraps :class:`SteeringModel` automatically.

    Args:
        model: A HuggingFace ``PreTrainedModel`` or a ``SteeringModel`` wrapper.

    Returns:
        The ``ModuleList`` containing the transformer layers.

    Raises:
        ValueError: If the model architecture is not recognized.
    """
    from .vector import SteeringModel as _SteeringModel

    if isinstance(model, _SteeringModel):
        model = model.model

    if hasattr(model, "model"):  # mistral-like
        return model.model.layers  # type: ignore[union-attr,return-value]
    elif hasattr(model, "transformer"):  # gpt-2-like
        return model.transformer.h  # type: ignore[union-attr,return-value]
    else:
        raise ValueError(f"don't know how to get layer list for {type(model)}")
