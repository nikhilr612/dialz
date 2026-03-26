"""Module to load and create contrastive-pair training datasets."""

import json
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from transformers import AutoTokenizer, PreTrainedTokenizerBase

load_dotenv()
hf_token = os.getenv("HF_TOKEN")


@dataclass
class DatasetEntry:
    """
    Represents a single entry in the dataset, consisting of a positive
    and a negative example.
    """

    positive: str
    negative: str


class Dataset:
    """A collection of contrastive (positive, negative) example pairs.

    Datasets are the primary input to :meth:`SteeringVector.train` and
    :func:`extract_activations`.  Each entry is a :class:`DatasetEntry`
    containing a positive and a negative string.  The class supports
    construction from prompt templates (:meth:`create_dataset`),
    loading from bundled corpora (:meth:`load_dataset`), and
    serialization to/from JSON files.
    """

    def __init__(self) -> None:
        """
        Initializes an empty dataset.
        """
        self.entries: list[DatasetEntry] = []

    def add_entry(self, positive: str, negative: str) -> None:
        """
        Adds a new DatasetEntry to the dataset.

        Args:
            positive (str): The positive example.
            negative (str): The negative example.
        """
        self.entries.append(DatasetEntry(positive=positive, negative=negative))

    def add_from_saved(self, saved_entries: list[dict[str, str]]) -> None:
        """
        Adds entries from a pre-saved dataset.

        Args:
            saved_entries (list[dict[str, str]]): A list of dictionaries, each containing
                                        "positive" and "negative" keys.
        """
        for entry in saved_entries:
            if "positive" in entry and "negative" in entry:
                self.add_entry(entry["positive"], entry["negative"])
            else:
                raise ValueError("Each entry must have 'positive' and 'negative' keys.")

    def view_dataset(self) -> list[DatasetEntry]:
        """
        Returns the current dataset as a list of DatasetEntry objects.

        Returns:
            list[DatasetEntry]: The list of all entries in the dataset.
        """
        return self.entries

    def save_to_file(self, file_path: str) -> None:
        """
        Saves the dataset to a JSON file.

        Args:
            file_path (str): The path to the file where the dataset will be
                saved.
        """
        with open(file_path, "w") as file:
            json.dump([entry.__dict__ for entry in self.entries], file, indent=4)

    @staticmethod
    def _apply_chat_template(
        tokenizer: PreTrainedTokenizerBase,
        system_role: str,
        content1: str,
        content2: str,
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Apply the model's chat template to produce a formatted prompt string.

        Args:
            tokenizer: HuggingFace tokenizer with ``apply_chat_template`` support.
            system_role: System message prefix.  If empty, no system message is added.
            content1: Content appended to the system message (e.g. the contrastive
                concept).  Ignored when ``system_role`` is empty.
            content2: User message content (the actual prompt text).
            add_generation_prompt: If ``True``, append the assistant turn prefix
                so the model is prompted to generate.

        Returns:
            The rendered prompt string.
        """
        messages = []

        # Only add system message if system_role is non-empty
        if system_role:
            messages.append({"role": "system", "content": f"{system_role}{content1}."})

        messages.append({"role": "user", "content": content2})

        tokenized: str = tokenizer.apply_chat_template(  # type: ignore[attr-defined,assignment]
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
        return tokenized

    @staticmethod
    def _extract_choice_marker(
        text: str,
    ) -> tuple[str, str | None]:
        """Strip a trailing ``(A)`` / ``(B)`` choice marker and return the body.

        Many of the bundled *load* datasets encode the target answer as a
        trailing choice letter, e.g.::

            "…question text…\\n(B)"

        or::

            "…question text…\\n\\nAnswer:\\n (A)"

        When applying a chat template, the marker should live in the
        *model's* turn (as the start of its continuation), not inside the
        user message.  This helper extracts the marker so callers can
        template the body separately and reattach the marker afterwards.

        Args:
            text: Raw dataset string that may end with a choice marker.

        Returns:
            A ``(body, marker)`` tuple.  *marker* is ``None`` when no
            trailing choice letter is detected.
        """
        match = re.search(r"\n\n?Answer:\s*\(([A-Z])\)\s*$", text)
        if match:
            letter = match.group(1)
            return text[: match.start()].rstrip(), f"({letter})"
        match = re.search(r"\(([A-Z])\)\s*$", text)
        if not match:
            return text, None
        return text[: match.start()].rstrip(), match.group(0).strip()

    @classmethod
    def create_dataset(
        cls,
        model_name: str,
        contrastive_pair: list[str],
        system_role: str = "Act as if you are extremely ",
        prompt_type: str = "sentence-starters",
        num_sents: int = 300,
    ) -> "Dataset":
        """
        Creates a dataset by generating positive and negative examples based on a given model,
        contrastive pairs, and prompt variations.
        This function uses a tokenizer to process input prompts and applies a chat template
        to generate positive and negative examples for each variation. The resulting examples
        are added to a dataset object.

        Args:
            cls: The class instance (used for accessing class methods).
            model_name (str): The name of the pre-trained model to use for tokenization.
            contrastive_pair (list[str]): A list containing two elements representing the positive and negative contrastive pairs.
            system_role (str, optional): A string representing the system's role in the chat template. Defaults to "Act as if you are extremely ".
            prompt_type (str, optional): The type of prompt variations to use. Defaults to "sentence-starters".
            num_sents (int, optional): The number of prompt variations to process. Defaults to 300.
        Returns:
            Dataset: A dataset object containing the generated positive and negative examples.
        Raises:
            FileNotFoundError: If the specified prompt variations file does not exist.
            json.JSONDecodeError: If the prompt variations file is not a valid JSON file.
        """

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        file_path = os.path.join(
            os.path.dirname(__file__), "datasets", "create", f"{prompt_type}.json"
        )
        with open(file_path, "r", encoding="utf-8") as file:
            variations = json.load(file)

        dataset = Dataset()

        for variation in variations[:num_sents]:
            # Use the helper function for both positive and negative
            positive_decoded = cls._apply_chat_template(
                tokenizer, system_role, contrastive_pair[0], variation
            )
            negative_decoded = cls._apply_chat_template(
                tokenizer, system_role, contrastive_pair[1], variation
            )

            # Add to dataset
            dataset.add_entry(positive_decoded, negative_decoded)

        return dataset

    @classmethod
    def load_from_file(cls, file_path: str) -> "Dataset":
        """
        Loads a dataset from a JSON file.

        Args:
            file_path (str): The path to the JSON file containing the dataset.

        Returns:
            Dataset: A new Dataset instance loaded from the file.
        """
        with open(file_path, "r") as file:
            data = json.load(file)
        dataset = cls()
        dataset.add_from_saved(data)
        return dataset

    @classmethod
    def load_dataset(
        cls, model_name: str, name: str, num_sents: int = 300
    ) -> "Dataset":
        """
        Loads a default pre-saved corpus included in the package,
        re-applies chat templates to each entry, and limits to num_sents.

        Args:
            model_name (str): The name of the model to use for tokenization.
            name (str): The name of the dataset to load.
            num_sents (int, optional): The maximum number of sentences to limit the dataset to.

        Returns:
            Dataset: A processed dataset with chat templates applied.

        Raises:
            FileNotFoundError: If the specified dataset file does not exist.
        """
        base_path = os.path.join(os.path.dirname(__file__), "datasets", "load")
        file_path = os.path.join(base_path, f"{name}.json")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset '{name}' not found.")

        # 1. Load the raw data (list of dicts with "positive" and "negative")
        with open(file_path, "r", encoding="utf-8") as file:
            raw_entries = json.load(file)

        # 2. Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # 3. Create a new dataset to store the transformed entries
        processed_dataset = cls()

        # 4. Iterate through the first num_sents entries, apply templates
        for entry in raw_entries[:num_sents]:
            pos_body, pos_marker = cls._extract_choice_marker(entry["positive"])
            neg_body, neg_marker = cls._extract_choice_marker(entry["negative"])

            positive_transformed = cls._apply_chat_template(
                tokenizer, system_role="", content1="", content2=pos_body
            )
            negative_transformed = cls._apply_chat_template(
                tokenizer, system_role="", content1="", content2=neg_body
            )

            # Reattach the choice marker *after* the template so it falls
            # inside the model's turn rather than the user's.
            if pos_marker:
                positive_transformed += pos_marker
            if neg_marker:
                negative_transformed += neg_marker

            processed_dataset.add_entry(positive_transformed, negative_transformed)

        return processed_dataset

    def __str__(self) -> str:
        """
        Returns a string representation of the dataset for easy viewing.
        """
        return "\n".join(
            [
                f"Positive: {entry.positive}\nNegative: {entry.negative}"
                for entry in self.entries
            ]
        )

    def __getitem__(self, index: int) -> DatasetEntry:
        """
        Allows indexing into the dataset to retrieve a specific entry.

        Args:
            index (int): The index of the entry to retrieve.

        Returns:
            DatasetEntry: The dataset entry at the specified index.
        """
        return self.entries[index]

    def __len__(self) -> int:
        """
        Returns the number of entries in the dataset.

        Returns:
            int: The number of entries in the dataset.
        """
        return len(self.entries)
