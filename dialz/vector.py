import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

import gguf
import numpy as np
import pyarrow as pa
import torch
import tqdm
from sklearn.decomposition import PCA
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .dataset import Dataset
from .shared import model_layer_list
from .types import Method


class SteeringStrategy(Protocol):
    def __call__(self, h: np.ndarray) -> np.ndarray: ...


class PCASteeringStrategy:
    def __call__(self, h: np.ndarray) -> np.ndarray:
        train = h[::2] - h[1::2]
        pca = PCA(n_components=1, whiten=False).fit(train)
        return pca.components_.astype(np.float32).squeeze(axis=0)


class PCACenterSteeringStrategy:
    def __call__(self, h: np.ndarray) -> np.ndarray:
        center = (h[::2] + h[1::2]) / 2
        h = h.copy()
        h[::2] -= center
        h[1::2] -= center
        pca = PCA(n_components=1, whiten=False).fit(h)
        return pca.components_.astype(np.float32).squeeze(axis=0)


class UMAPSteeringStrategy:
    def __call__(self, h: np.ndarray) -> np.ndarray:
        import umap  # type: ignore[import-untyped]

        model = umap.UMAP(n_components=1)
        embedding = model.fit_transform(h).astype(np.float32)
        return np.sum(h * embedding, axis=0) / np.sum(embedding)


class MeanDiffSteeringStrategy:
    def __call__(self, h: np.ndarray) -> np.ndarray:
        train = h[::2] - h[1::2]
        return np.mean(train, axis=0).astype(np.float32)


_strategy_map: dict[Method, type[SteeringStrategy]] = {
    Method.PCA: PCASteeringStrategy,
    Method.PCA_CENTER: PCACenterSteeringStrategy,
    Method.UMAP: UMAPSteeringStrategy,
    Method.MEAN_DIFF: MeanDiffSteeringStrategy,
}


def _resolve_strategy(method: Method | SteeringStrategy | str) -> SteeringStrategy:
    if isinstance(method, str):
        method = Method(method)
    if not isinstance(method, Method):
        return method
    return _strategy_map[method]()


def _flatten_dataset(
    dataset: Dataset,
) -> tuple[list[str], list[int], list[str]]:
    """Flatten Dataset into interleaved positive/negative strings with metadata."""
    train_strs: list[str] = []
    example_ids: list[int] = []
    example_classes: list[str] = []
    for idx, ex in enumerate(dataset.entries):
        train_strs.append(ex.positive)
        example_ids.append(idx)
        example_classes.append("positive")
        train_strs.append(ex.negative)
        example_ids.append(idx)
        example_classes.append("negative")
    return train_strs, example_ids, example_classes


def _batched_forward(
    model: "PreTrainedModel | SteeringModel",
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[str],
    hidden_layers: list[int],
    token_indices: list[int],
    batch_size: int,
) -> dict[tuple[int, int], list[np.ndarray]]:
    """
    Core forward pass: tokenize, run model, extract hidden states per (layer, token_index).

    Returns a dict mapping (layer, token_index) to a list of numpy arrays (one per input).
    """
    batched_inputs = [
        inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)
    ]
    activations: dict[tuple[int, int], list[np.ndarray]] = {
        (layer, ti): [] for layer in hidden_layers for ti in token_indices
    }
    with torch.no_grad():
        for batch in tqdm.tqdm(batched_inputs):
            encoded_batch = tokenizer(batch, padding=True, return_tensors="pt")
            encoded_batch = encoded_batch.to(model.device)
            out = model(**encoded_batch, output_hidden_states=True)
            attention_mask = encoded_batch["attention_mask"]
            for i in range(len(batch)):
                # todo: add type annotations for `encoded_batch`
                non_padding_indices = attention_mask[i].nonzero(as_tuple=True)[0]  # type: ignore
                for ti in token_indices:
                    token_pos = non_padding_indices[ti].item()
                    for layer in hidden_layers:
                        hidden_idx = layer + 1 if layer >= 0 else layer
                        hidden_state = (
                            out.hidden_states[hidden_idx][i][token_pos]
                            .cpu()
                            .float()
                            .numpy()
                        )
                        activations[(layer, ti)].append(hidden_state)
            del out

    return activations


@dataclass
class SteeringVector:
    model_type: str
    directions: dict[int, np.ndarray]

    @classmethod
    def train(
        cls,
        model: "PreTrainedModel | SteeringModel",
        dataset: Dataset,
        method: Method | SteeringStrategy | str = Method.PCA,
        **kwargs,
    ) -> "SteeringVector":
        """
        Train a SteeringVector for a given model and tokenizer using the provided dataset.

        Args:
            model (PreTrainedModel | SteeringModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            dataset (Dataset): The dataset used for training.
            method (Method | SteeringStrategy | str): The training method to use.
                Can be a Method enum (Method.PCA), a string ("pca"), or a custom
                SteeringStrategy instance. Defaults to Method.PCA.
            **kwargs: Additional keyword arguments.
                max_batch_size (int, optional): The maximum batch size for training.
                    Defaults to 32. Try reducing this if you're running out of memory.
                token_index (int, optional): Index into non-padding tokens to extract
                    activations from. -1 selects the last non-padding token (default),
                    0 the first, etc.

        Returns:
            SteeringVector: The trained vector.
        """
        tokenizer = AutoTokenizer.from_pretrained(model.model_name, token=model.token)
        tokenizer.pad_token_id = 0

        with torch.inference_mode():
            dirs = read_representations(
                model,
                tokenizer,
                dataset,
                method=method,
                **kwargs,
            )
        return cls(model_type=model.config.model_type, directions=dirs)

    def export_gguf(self, path: os.PathLike[str] | str) -> None:
        """
        Export a trained SteeringVector to a llama.cpp .gguf file.
        Note: This file can't be used with llama.cpp yet. WIP!

        ```python
        vector = SteeringVector.train(...)
        vector.export_gguf("path/to/write/vector.gguf")
        ```

        """

        arch = "steeringvector"
        writer = gguf.GGUFWriter(path, arch)
        writer.add_string(f"{arch}.model_hint", self.model_type)
        writer.add_uint32(f"{arch}.layer_count", len(self.directions))
        for layer in self.directions.keys():
            writer.add_tensor(f"direction.{layer}", self.directions[layer])
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    @classmethod
    def import_gguf(cls, path: os.PathLike[str] | str) -> "SteeringVector":
        reader = gguf.GGUFReader(path)

        archf = reader.get_field("general.architecture")
        if not archf or not len(archf.parts):
            warnings.warn(".gguf file missing architecture field")
        else:
            arch = str(bytes(archf.parts[-1]), encoding="utf-8", errors="replace")
            if arch != "steeringvector":
                warnings.warn(
                    f".gguf file with architecture {arch!r} does not appear to be a steering vector!"
                )

        modelf = reader.get_field("steeringvector.model_hint")
        if not modelf or not len(modelf.parts):
            raise ValueError(".gguf file missing steeringvector.model_hint field")
        model_hint = str(bytes(modelf.parts[-1]), encoding="utf-8")

        directions = {}
        for tensor in reader.tensors:
            if not tensor.name.startswith("direction."):
                continue
            try:
                layer = int(tensor.name.split(".")[1])
            except Exception:  # todo: fix with correct error type.
                raise ValueError(
                    f".gguf file has invalid direction field name: {tensor.name}"
                )
            directions[layer] = tensor.data

        return cls(model_type=model_hint, directions=directions)

    def _helper_combine(
        self, other: "SteeringVector", other_scalar: float
    ) -> "SteeringVector":
        if self.model_type != other.model_type:
            warnings.warn(
                "Trying to add vectors with mismatched model_types together, this may produce unexpected results."
            )

        model_type = self.model_type
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = self.directions[layer]
        for layer in other.directions:
            other_layer = other_scalar * other.directions[layer]
            if layer in directions:
                directions[layer] = directions[layer] + other_layer
            else:
                directions[layer] = other_layer
        return SteeringVector(model_type=model_type, directions=directions)

    def __eq__(self, other: object) -> bool:

        if not isinstance(other, SteeringVector):
            return False

        if self is other:
            return True

        if self.model_type != other.model_type:
            return False
        if self.directions.keys() != other.directions.keys():
            return False
        for k in self.directions.keys():
            if (self.directions[k] != other.directions[k]).any():
                return False
        return True

    def __add__(self, other: "SteeringVector") -> "SteeringVector":
        if not isinstance(other, SteeringVector):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'SteeringVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, 1)

    def __sub__(self, other: "SteeringVector") -> "SteeringVector":
        if not isinstance(other, SteeringVector):
            raise TypeError(
                f"Unsupported operand type(s) for -: 'SteeringVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, -1)

    def __neg__(self) -> "SteeringVector":
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = -self.directions[layer]
        return SteeringVector(model_type=self.model_type, directions=directions)

    def __mul__(self, other: int | float | np.int_ | np.float64) -> "SteeringVector":
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = other * self.directions[layer]
        return SteeringVector(model_type=self.model_type, directions=directions)

    def __rmul__(self, other: int | float | np.int_ | np.float64) -> "SteeringVector":
        return self.__mul__(other)

    def __truediv__(
        self, other: int | float | np.int_ | np.float64
    ) -> "SteeringVector":
        return self.__mul__(1 / other)


def read_representations(
    model: "PreTrainedModel | SteeringModel",
    tokenizer: PreTrainedTokenizerBase,
    inputs: Dataset,
    hidden_layers: Iterable[int] | None = None,
    batch_size: int = 32,
    method: Method | SteeringStrategy | str = Method.PCA,
    transform_hiddens: (
        Callable[[dict[int, np.ndarray]], dict[int, np.ndarray]] | None
    ) = None,
    token_index: int = -1,
) -> dict[int, np.ndarray]:
    """
    Extract the representations based on the contrast dataset.
    """
    if not hidden_layers:
        hidden_layers = range(-1, -model.config.num_hidden_layers, -1)

    # Normalize the layer indexes if they're negative
    n_layers = len(model_layer_list(model))
    hidden_layers = [i if i >= 0 else n_layers + i for i in hidden_layers]

    # The order is [positive, negative, positive, negative, ...]
    train_strs, _, _ = _flatten_dataset(inputs)

    layer_hiddens = batched_get_hiddens(
        model, tokenizer, train_strs, hidden_layers, batch_size, token_index
    )

    if transform_hiddens is not None:
        layer_hiddens = transform_hiddens(layer_hiddens)

    # Get directions for each layer using the specified strategy
    strategy = _resolve_strategy(method)
    directions: dict[int, np.ndarray] = {}
    for layer in tqdm.tqdm(hidden_layers):
        h = layer_hiddens[layer]
        assert h.shape[0] == len(inputs.entries) * 2

        directions[layer] = strategy(h)

        # Calculate sign to ensure the direction aligns with the sentiment order.
        projected_hiddens = project_onto_direction(h, directions[layer])
        positive_smaller_mean = float(
            np.mean(
                [
                    projected_hiddens[i] < projected_hiddens[i + 1]
                    for i in range(0, len(inputs.entries) * 2, 2)
                ]
            )
        )
        positive_larger_mean = float(
            np.mean(
                [
                    projected_hiddens[i] > projected_hiddens[i + 1]
                    for i in range(0, len(inputs.entries) * 2, 2)
                ]
            )
        )

        if positive_smaller_mean > positive_larger_mean:
            directions[layer] *= -1

    return directions


def batched_get_hiddens(
    model: "PreTrainedModel | SteeringModel",
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[str],
    hidden_layers: list[int],
    batch_size: int,
    token_index: int = -1,
) -> dict[int, np.ndarray]:
    """
    Using the given model and tokenizer, pass the inputs through the model and get the hidden
    states for each layer in `hidden_layers` for the given token position.

    Args:
        token_index: Index into non-padding tokens to extract activations from.
            -1 selects the last non-padding token, 0 the first, etc.
            Clamped to the last non-padding token if out of range.

    Returns a dictionary from `hidden_layers` layer id to an numpy array of shape `(n_inputs, hidden_dim)`
    """
    raw = _batched_forward(
        model, tokenizer, inputs, hidden_layers, [token_index], batch_size
    )
    return {layer: np.vstack(raw[(layer, token_index)]) for layer in hidden_layers}


def extract_activations(
    model: "PreTrainedModel | SteeringModel",
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    hidden_layers: list[int],
    token_indices: list[int],
    batch_size: int = 32,
) -> pa.Table:
    """
    Run the model on a dataset and extract activations for a list of layers and token positions.

    Args:
        model: The model to run.
        tokenizer: The tokenizer to use.
        dataset: Dataset of positive/negative pairs.
        hidden_layers: List of layer indices to extract activations from.
        token_indices: List of token position indices into non-padding tokens
            (e.g., [-1, 0] for last and first tokens).
        batch_size: Batch size for inference.

    Returns:
        pyarrow.Table with columns:
            - activation: fixed_size_list[float32] — the activation vector
            - layer: int32 — layer index
            - token_index: int32 — token position index
            - example_id: int32 — index in the original dataset
            - example_class: string — "positive" or "negative"
    """

    n_layers = len(model_layer_list(model))
    hidden_layers_norm = [i if i >= 0 else n_layers + i for i in hidden_layers]

    train_strs, example_ids, example_classes = _flatten_dataset(dataset)
    raw = _batched_forward(
        model, tokenizer, train_strs, hidden_layers_norm, token_indices, batch_size
    )

    # Accumulators — one row per (example, layer, token_index)
    activations: list[np.ndarray] = []
    layers_out: list[int] = []
    token_indices_out: list[int] = []
    ids_out: list[int] = []
    classes_out: list[str] = []

    for (layer, ti), arr_list in raw.items():
        for idx, act in enumerate(arr_list):
            activations.append(act)
            layers_out.append(layer)
            token_indices_out.append(ti)
            ids_out.append(example_ids[idx])
            classes_out.append(example_classes[idx])

    hidden_dim = activations[0].shape[0]
    activation_array = pa.FixedSizeListArray.from_arrays(
        pa.concat_arrays([pa.array(a, type=pa.float32()) for a in activations]),
        hidden_dim,
    )

    return pa.table(
        {
            "activation": activation_array,
            "layer": pa.array(layers_out, type=pa.int32()),
            "token_index": pa.array(token_indices_out, type=pa.int32()),
            "example_id": pa.array(ids_out, type=pa.int32()),
            "example_class": pa.array(classes_out, type=pa.string()),
        }
    )


def project_onto_direction(H: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    mag = np.linalg.norm(direction)
    assert not np.isinf(mag)
    return (H @ direction) / mag


class SteeringModel(torch.nn.Module):
    """
    **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

    A wrapped language model that can have controls set on its layers with `self.set_control`.
    """

    def __init__(
        self,
        model_name: str,
        layer_ids: Iterable[int],
        token: str | None = None,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        """
        **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

        Build a new SteeringModel around a model instance, initializing control on
        the layers specified in `layer_ids`.
        """

        super().__init__()
        self.model_name = model_name

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, token=token, torch_dtype=torch_dtype
        )
        self.token = token

        device = (
            "cuda:0"
            if torch.cuda.is_available()
            else ("mps:0" if torch.backends.mps.is_available() else "cpu")
        )

        self.model = self.model.to(device)  # type: ignore[arg-type]

        layers = model_layer_list(self.model)
        self.layer_ids = [i if i >= 0 else len(layers) + i for i in layer_ids]
        for layer_id in layer_ids:
            layer = layers[layer_id]
            if not isinstance(layer, SteeringModule):
                layers[layer_id] = SteeringModule(layer)
            else:
                warnings.warn(
                    "Trying to rewrap a wrapped model! Probably not what you want! Try calling .unwrap first."
                )

    @property
    def config(self) -> PretrainedConfig:
        return self.model.config

    @property
    def device(self) -> torch.device:
        return self.model.device

    def unwrap(self) -> PreTrainedModel:
        """
        Removes the mutations done to the wrapped model and returns it.
        After using this method, `set_control` and `reset` will not work.
        """

        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layer = layers[layer_id]
            assert isinstance(layer, SteeringModule)
            layers[layer_id] = layer.block
        return self.model

    def set_control(
        self, control: "SteeringVector", scalar: float = 1.0, **kwargs
    ) -> None:
        """
        Set a `SteeringVector` for the layers this SteeringModel handles, with a strength given
        by `scalar`. (Negative `scalar` values invert the control vector, e.g. happiness→sadness.)
        `scalar` defaults to `1.0`.

        Additional kwargs:
        - `normalize: bool`: track the magnitude of the non-modified activation, and rescale the
          activation to that magnitude after control (default: `False`)
        - `operator: Callable[[Tensor, Tensor], Tensor]`: how to combine the base output and control
          (default: +)
        """

        raw_control = {}
        for layer_id in self.layer_ids:
            raw_control[layer_id] = torch.tensor(
                scalar * control.directions[layer_id]
            ).to(self.model.device, dtype=self.model.dtype)
        self.set_raw_control(raw_control, **kwargs)

    def reset(self) -> None:
        """
        Resets the control for all layer_ids, returning the model to base behavior.
        """
        self.set_raw_control(None)

    def set_raw_control(
        self, control: dict[int, torch.Tensor] | None, **kwargs
    ) -> None:
        """
        Set or remove control parameters to the layers this ControlModel handles.
        The keys of `control` should be equal to or a superset of the `layer_ids` passed to __init__.
        Only those layers will be controlled, any others in `control` will be ignored.

        Passing `control=None` will reset the control tensor for all layer_ids, making the model act
        like a non-control model.

        Additional kwargs:
        - `normalize: bool`: track the magnitude of the non-modified activation, and rescale the
          activation to that magnitude after control (default: `False`)
        - `operator: Callable[[Tensor, Tensor], Tensor]`: how to combine the base output and control
          (default: +)
        """

        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layer = layers[layer_id]
            assert isinstance(layer, SteeringModule)
            if control is None:
                layer.reset()
            else:
                layer.set_control(BlockControlParams(control[layer_id], **kwargs))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.forward(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)


@dataclass
class BlockControlParams:
    control: torch.Tensor | None = None
    normalize: bool = False
    operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        lambda current, control: current + control
    )

    @classmethod
    def default(cls) -> "BlockControlParams":
        return cls()


class SteeringModule(torch.nn.Module):
    def __init__(self, block: torch.nn.Module) -> None:
        super().__init__()
        self.block: torch.nn.Module = block
        self.params: BlockControlParams = BlockControlParams.default()

        if hasattr(block, "attention_type"):
            self.attention_type = block.attention_type

    def set_control(self, params: BlockControlParams) -> None:
        self.params = params

    def reset(self) -> None:
        self.set_control(BlockControlParams.default())

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        output = self.block(*args, **kwargs)

        control = self.params.control

        if control is None:
            return output
        elif len(control.shape) == 1:
            control = control.reshape(1, 1, -1)

        if isinstance(output, tuple):
            modified = output[0]
        else:
            modified = output

        assert len(control.shape) == len(modified.shape)
        control = control.to(modified.device)

        norm_pre = torch.norm(modified, dim=-1, keepdim=True)

        # we should ignore the padding tokens when doing the activation addition
        # mask has ones for non padding tokens and zeros at padding tokens.
        # only tested this on left padding
        if "position_ids" in kwargs:
            pos = kwargs["position_ids"]
            zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
            col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
            target_shape = modified.shape
            mask = (
                (col_indices >= zero_indices)
                .float()
                .reshape(target_shape[0], target_shape[1], 1)
            )
            mask = mask.to(modified.dtype).to(modified.device)
        else:
            mask = 1.0

        modified = self.params.operator(modified, control * mask)

        if self.params.normalize:
            norm_post = torch.norm(modified, dim=-1, keepdim=True)
            modified = modified / norm_post * norm_pre

        if isinstance(output, tuple):
            output = (modified,) + output[1:]
        else:
            output = modified

        return output
