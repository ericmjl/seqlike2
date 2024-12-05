"""A class for working with sequences of symbols.

The SequenceLike class provides a flexible interface for working with sequences
of arbitrary symbols, while preserving the ability to convert between string,
index-encoded, and one-hot encoded representations.

Unlike the SeqLike class which assumes biological sequences,
this class makes no assumptions about the nature of the symbols,
making it suitable for any sequence data that can be represented as discrete symbols.

Key features:
- Flexible alphabet specification
- Conversion between string, index and one-hot encodings
- Sequence search and counting functionality
- Integration with numpy/xarray data structures
"""

from collections.abc import Sequence
from typing import Any, Iterable

import numpy as np
import xarray as xr

from .encoders import (
    OneHotEncoder,
    OrdinalEncoder,
)


class SequenceLike(Sequence):
    """A class for working with sequences of arbitrary symbols.

    :param sequence: The input sequence of symbols
    :param alphabet: The set of valid symbols. If None, will be inferred from sequence
    :param encoding: The encoding format of the input sequence,
        either "onehot", "index" or None.
        If specified, the sequence will be converted from that encoding to symbols.
    """

    def __init__(self, sequence: Iterable, alphabet=None, encoding=None):
        if alphabet is None:
            alphabet = set([x for x in sequence])
        alphabet = sorted(alphabet)

        self._index_encoder = OrdinalEncoder(categories=list(alphabet))
        self._onehot_encoder = OneHotEncoder(categories=list(alphabet))

        # Create the xarray dataset
        self.sequence_data = xr.Dataset(
            data_vars={
                "sequence": ("position", list(sequence)),
                "alphabet": ("alphabet_position", list(alphabet)),
            },
            coords={
                "position": np.arange(len(sequence)),
                "alphabet_index": np.arange(len(alphabet)),
            },
        )

        # Create index encoding
        index_encoded = self._index_encoder.transform(sequence)
        self.sequence_data["index_encoding"] = ("position", index_encoded)

        # Create onehot encoding
        onehot_encoded = self._onehot_encoder.transform(sequence)
        self.sequence_data["onehot_encoding"] = (
            ("position", "alphabet_position"),
            onehot_encoded,
        )

    def __len__(self) -> int:
        """Get the length of the sequence.

        :returns: The length of the sequence
        """
        return len(self.sequence_data.position)

    def __str__(self) -> str:
        """Get string representation of the sequence.

        :returns: The sequence as a string
        """
        return "".join(self.sequence.values.tolist())

    def __getattr__(self, name) -> Any:
        """Attribute access delegation.

        We first check if the attribute exists on the underlying xarray Dataset,
        and if so, we delegate to it. Otherwise, we check the SeqLike object.

        :param name: Name of the attribute to access
        :returns: The requested attribute
        :raises AttributeError: If the attribute doesn't exist
        """
        # Protect against recursion for special attributes
        if name.startswith("_"):
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{name}'"
            )

        # Only delegate to sequence_data if it exists
        if hasattr(self, "sequence_data") and hasattr(self.sequence_data, name):
            return getattr(self.sequence_data, name)
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Attribute setting delegation.

        We first check if the attribute exists on the underlying xarray Dataset,
        and if so, we delegate to it. Otherwise, we set it on the SeqLike object.

        :param name: Name of the attribute to set
        :param value: Value to set the attribute to
        """
        # Always allow setting protected attributes directly
        if name.startswith("_") or name == "sequence_data":
            super().__setattr__(name, value)
            return

        # Only delegate to sequence_data if it exists
        if hasattr(self, "sequence_data") and hasattr(self.sequence_data, name):
            setattr(self.sequence_data, name, value)
        else:
            super().__setattr__(name, value)
