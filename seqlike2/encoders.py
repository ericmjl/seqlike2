"""Encoders for sequence data."""

from typing import Iterable, Union

import numpy as np


class OrdinalEncoder:
    """Encode categorical sequence data as ordinal integers.

    This encoder transforms sequence symbols into integer indices based on the provided
    categories. Each unique symbol is mapped to a unique integer index.

    :param categories: Iterable of valid symbols that can appear in sequences
    """

    def __init__(self, categories: Iterable):
        self.categories = list(categories)  # Convert to list for indexing
        self._category_map = {cat: idx for idx, cat in enumerate(self.categories)}

    def transform(self, sequence: Iterable) -> np.ndarray:
        """Transform a sequence of symbols into integer indices.

        :param sequence: Input sequence of symbols to encode
        :returns: Numpy array of integer indices corresponding to the input symbols
        :raises KeyError: If sequence contains symbols not in the encoder's categories
        """
        return np.array([self._category_map[i] for i in sequence])


class OneHotEncoder:
    """Encode categorical sequence data as one-hot vectors.

    This encoder transforms sequence symbols into one-hot encoded vectors where each
    symbol is represented by a binary vector with a single 1 and all other elements 0.

    :param categories: Iterable of valid symbols that can appear in sequences
    """

    def __init__(self, categories: Iterable):
        self.categories = list(categories)  # Convert to list for indexing
        self._category_map = {cat: idx for idx, cat in enumerate(self.categories)}
        self.identity_matrix = np.eye(len(self.categories))

    def transform(self, sequence: Iterable) -> np.ndarray:
        """Transform a sequence of symbols into one-hot encoded vectors.

        :param sequence: Input sequence of symbols to encode
        :returns: 2D numpy array where each row is a one-hot encoded vector
        :raises KeyError: If sequence contains symbols not in the encoder's categories
        """
        # Get index encoding first
        index_encoded = np.array([self._category_map[i] for i in sequence])
        return self.identity_matrix[index_encoded]


def array_to_symbols(
    sequence: Union[list, np.ndarray], _index_encoder, _onehot_encoder
) -> str:
    """Convert array-like sequence representations to a list of symbols.

    Handles both index-encoded (1D) and one-hot encoded (2D) sequence representations,
    converting them back to their original symbolic form.

    :param sequence: Input sequence as either index-encoded or one-hot encoded array
    :param _index_encoder: OrdinalEncoder instance for handling index encoding
    :param _onehot_encoder: OneHotEncoder instance for handling one-hot encoding
    :returns: List of symbols corresponding to the input encoding
    :raises IndexError: If the sequence contains values that map to indices
        larger than the size of the encoder's alphabet
    :raises ValueError: If input array dimensions are neither 1D nor 2D
    """
    sequence = np.asarray(sequence, dtype=float)
    if sequence.ndim == 1:
        try:
            sequence = _index_encoder.inverse_transform(
                sequence.reshape(-1, 1)
            ).flatten()
        except IndexError:
            raise IndexError(
                "The encoder encountered a value "
                "that is outside the range of valid indices. "
                "This usually happens when the sequence contains values "
                "that map to indices "
                "larger than the size of the alphabet. "
                "The encoder's alphabet "
                f"has {len(_index_encoder.categories_[0])} symbols. "
                "Please check that your sequence values are within the valid range "
                "and that you're using an alphabet which contains all needed symbols."
            )
    elif sequence.ndim == 2:
        sequence = _onehot_encoder.inverse_transform(sequence).flatten()

    # NOTE: We do not need to check for other dim sizes
    # because we assume that validate_sequence will take care of it.
    return sequence


def array_to_string(
    sequence: Union[list, np.ndarray], _index_encoder, _onehot_encoder
) -> str:
    """Convert array-like sequence representations to a concatenated string.

    A convenience wrapper around array_to_symbols that joins the resulting
    symbols into a single string.

    :param sequence: Input sequence as either index-encoded or one-hot encoded array
    :param _index_encoder: OrdinalEncoder instance for handling index encoding
    :param _onehot_encoder: OneHotEncoder instance for handling one-hot encoding
    :returns: String representation of the sequence
    :raises IndexError: If the sequence contains values that map to indices
        larger than the size of the encoder's alphabet
    :raises ValueError: If input array dimensions are neither 1D nor 2D
    """
    return "".join(array_to_symbols(sequence, _index_encoder, _onehot_encoder))
