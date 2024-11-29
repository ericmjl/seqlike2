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
from copy import deepcopy
from typing import Iterable

import numpy as np
import xarray as xr

from .encoders import (
    array_to_symbols,
    index_encoder_from_alphabet,
    onehot_encoder_from_alphabet,
)


class SequenceLike(xr.Dataset, Sequence):
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

        # Get the encoders - both one-hot and index.
        _index_encoder = index_encoder_from_alphabet(alphabet)
        _onehot_encoder = onehot_encoder_from_alphabet(alphabet)

        if encoding in ["onehot", "index"]:
            sequence = array_to_symbols(sequence, _index_encoder, _onehot_encoder)

        # Set properties all in one block for readability.
        self.alphabet = alphabet
        self._index_encoder = _index_encoder
        self._onehot_encoder = _onehot_encoder
        self.sequence = sequence

    def to_str(self) -> str:
        """Convert the sequence to a string representation.

        :returns: The sequence as a concatenated string of symbols
        """
        return "".join(self.sequence)

    def to_index(self, dtype=int, encoder=None) -> np.ndarray:
        """Convert the sequence to an index-encoded array.

        Each symbol is converted to its corresponding integer index in the alphabet.
        Uses the internal index encoder by default,
        but a custom encoder can be provided.

        :param dtype: The numpy dtype for the output array, defaults to int
        :param encoder: Optional custom sklearn-style encoder
            to use instead of the default
        :returns: A 1D numpy array containing the index encoding
        """
        seq_as_array = [[x] for x in self]

        if encoder is not None:
            return encoder.transform(seq_as_array).squeeze().astype(dtype)
        return self._index_encoder.transform(seq_as_array).squeeze().astype(dtype)

    def to_onehot(self, dtype=int, encoder=None) -> np.ndarray:
        """Convert the sequence to a one-hot encoded array.

        Each symbol is converted to a binary vector where only one element is 1.
        Uses the internal one-hot encoder by default,
        but a custom encoder can be provided.

        :param dtype: The numpy dtype for the output array, defaults to int
        :param encoder: Optional custom sklearn-style encoder
            to use instead of the default
        :returns: A 2D numpy array containing the one-hot encoding
        """
        seq_as_array = [[x] for x in self]

        if encoder is not None:
            return encoder.transform(seq_as_array).squeeze().astype(dtype)
        return self._onehot_encoder.transform(seq_as_array).squeeze().astype(dtype)

    def apply(self, func, **kwargs):
        """Apply a function to this sequence and return the result.

        Enables method chaining through a fluent interface pattern.
        The function must take a SequenceLike object as the 'seq' parameter.

        :param func: The function to apply
        :param kwargs: Keyword arguments to pass to the function
        :returns: The result of calling func(seq=self, **kwargs)
        """
        return func(seq=deepcopy(self), **kwargs)

    def find(self, sub, start=None, end=None) -> int:
        """Find the first occurrence of a subsequence.

        :param sub: The subsequence to search for
        :param start: Optional start position for the search
        :param end: Optional end position for the search
        :returns: Index of first match, or -1 if not found
        :raises AssertionError: If subsequence contains invalid symbols
        """
        assert set(sub).issubset(self.alphabet)

        if start is None:
            start = 0
        if end is None:
            end = len(self) - len(sub)

        for i in range(start, end + 1):
            if "".join(self[i : i + len(sub)]) == "".join(sub):
                return i
        return -1

    def count(self, sub, start=None, end=None) -> int:
        """Count occurrences of a subsequence.

        :param sub: The subsequence to count
        :param start: Optional start position for counting
        :param end: Optional end position for counting
        :returns: Number of non-overlapping occurrences
        :raises ValueError: If subsequence contains invalid symbols
        """
        if not set(sub).issubset(self.alphabet):
            raise ValueError(
                f"Subsequence set {set(sub)} "
                f"is not part of the alphabet {self.alphabet}"
            )

        if start is None:
            start = 0
        if end is None:
            end = len(self) - len(sub)

        count = 0
        for i in range(start, end + 1):
            if "".join(self[i : i + len(sub)]) == "".join(sub):
                count += 1
        return count

    def __len__(self):
        """Get length of sequence.

        :returns: Length of the sequence
        """
        return len(self.sequence)

    def __getitem__(self, index):
        """Get item at specified index.

        :param index: Integer index or slice
        :returns: Element(s) at specified position(s)
        """
        return self.sequence[index]

    def __contains__(self, x: object) -> bool:
        """Check if item exists in sequence.

        :param x: Item to check for
        :returns: True if item exists in sequence, False otherwise
        """
        return x in self.sequence

    def __iter__(self):
        """Get iterator over sequence.

        :returns: Iterator over sequence elements
        """
        return iter(self.sequence)

    def __str__(self) -> str:
        """Convert to string representation.

        :returns: The sequence as a string
        """
        return self.to_str()

    def __deepcopy__(self, memo):
        """Create a deep copy of this object.

        :param memo: Memo dictionary used by deepcopy
        :returns: A new SequenceLike instance with copied data
        """
        return SequenceLike(self.sequence, self.alphabet)

    def __repr__(self) -> str:
        """Get string representation of the sequence.

        :returns: String representation of the sequence
        """
        return self.sequence.__repr__()
