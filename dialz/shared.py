from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from .vector import SteeringModel


def model_layer_list(model: SteeringModel | PreTrainedModel) -> torch.nn.ModuleList:
    from .vector import SteeringModel as _SteeringModel

    if isinstance(model, _SteeringModel):
        model = model.model

    if hasattr(model, "model"):  # mistral-like
        return model.model.layers  # type: ignore[union-attr,return-value]
    elif hasattr(model, "transformer"):  # gpt-2-like
        return model.transformer.h  # type: ignore[union-attr,return-value]
    else:
        raise ValueError(f"don't know how to get layer list for {type(model)}")
