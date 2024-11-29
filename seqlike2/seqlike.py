"""SeqLike class for working with biological sequences."""

from enum import Enum
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import xarray as xr
from Bio.Seq import Seq

from .alphabets import AA
from .encoders import OneHotEncoder, OrdinalEncoder


class SeqType(Enum):
    """Enumeration of sequence types supported by SeqLike.

    :cvar NT: Nucleotide sequence type
    :cvar AA: Amino acid sequence type
    """

    NT = "nt"
    AA = "aa"


RESERVED_ATTRS = ["sequence", "alphabet", "onehot_encoding", "index_encoding"]


class SeqLike:
    """A class for working with biological sequences.

    SeqLike provides a unified interface for working with biological sequences,
    combining string-based sequence representation with numerical encodings
    and annotation capabilities.

    :param sequence: The input sequence string
    :param alphabet: The alphabet to use for encoding the sequence.
        Must be an iterable of valid characters.
    :param seq_type: The type of sequence (nucleotide or amino acid).
        Defaults to nucleotide.
    """

    def __init__(
        self,
        sequence: str,
        alphabet: Optional[Iterable] = None,
        seq_type: SeqType = SeqType.NT,
    ):
        # Create xarray dataset with sequence data and alphabet
        self._seq_type = seq_type

        # Create the xarray dataset
        self.ds = xr.Dataset(
            data_vars={
                "sequence": ("position", list(sequence)),
                "alphabet": ("alphabet_position", list(alphabet)),
            },
            coords={
                "position": np.arange(len(sequence)),
                "alphabet_index": np.arange(len(alphabet)),
            },
        )

        # Initialize encoders
        self._index_encoder = OrdinalEncoder(categories=list(alphabet))
        self._onehot_encoder = OneHotEncoder(categories=list(alphabet))

        # Create index encoding
        index_encoded = self._index_encoder.transform(sequence)
        self.ds["index_encoding"] = ("position", index_encoded)

        # Create onehot encoding
        onehot_encoded = self._onehot_encoder.transform(sequence)
        self.ds["onehot_encoding"] = (("position", "alphabet_position"), onehot_encoded)

    def sel(self, *args, **kwargs) -> "SeqLike":
        """Select positions from the sequence based on criteria.

        This method allows filtering the sequence based on arbitrary criteria.
        The selection criteria are passed to xarray's where() function.

        :param args: Positional arguments passed to xarray.Dataset.where()
        :param kwargs: Keyword arguments passed to xarray.Dataset.where()
        :returns: A new SeqLike object containing only the selected positions
        """
        drop = kwargs.pop("drop", True)
        positions = self.ds.position.where(*args, **kwargs, drop=drop)
        selected_ds = self.ds.sel(position=positions)

        # Create new SeqLike instance
        new_seqlike = SeqLike(
            sequence="".join(selected_ds.sequence.values),
            alphabet=self.ds.alphabet.values,
            seq_type=self._seq_type,
        )

        # Copy over any additional data variables
        for var in selected_ds.data_vars:
            new_seqlike.ds[var] = (selected_ds[var].dims, selected_ds[var].values)

        return new_seqlike

    def annotate(self, **kwargs) -> "SeqLike":
        """Add annotation tracks to the sequence.

        Each annotation track must be an iterable of the same length as the sequence.
        The values can be of any type that can be coerced to a numpy array.

        :param kwargs: Keyword arguments where keys are annotation names
            and values are iterables.
        :returns: Self for method chaining
        """
        for k, v in kwargs.items():
            # Coerce to numpy array
            v = np.array(v, dtype=type(v[0]))
            self.ds[k] = ("position", v)
        return self

    @property
    def annotations(self) -> Dict[str, xr.DataArray]:
        """Get all annotation tracks.

        :returns: Dictionary mapping annotation names
            to their xarray.DataArray representations
        """
        return {
            k: v
            for k, v in self.ds.data_vars.items()
            if k not in ["sequence", "alphabet", "onehot_encoding", "index_encoding"]
        }

    def aa(self) -> "SeqLike":
        """Convert nucleotide sequence to amino acid sequence.

        This method translates the nucleotide sequence to amino acids and
        appropriately transforms all annotation tracks. Annotations are
        aggregated over codons according to their data type:

        1. Boolean annotations: Take the OR over each codon
        2. Numeric annotations: Take the mean over each codon
        3. Categorical/string annotations: Take the mode over each codon

        :returns: A new SeqLike object containing the amino acid sequence
        :raises ValueError: If the sequence is not a nucleotide sequence
        """
        # Only works if the sequence is of type NT
        if self._seq_type != SeqType.NT:
            raise ValueError("This method only works for nucleotide sequences!")

        # Translate sequence into amino acids
        sequence = str(Seq("".join(i for i in self.sequence.values)).translate())
        new_seqlike = SeqLike(sequence, alphabet=AA, seq_type=SeqType.AA)

        # Copy over all annotation tracks but coarsen them to triplets
        for k, v in self.ds.data_vars.items():
            if k not in RESERVED_ATTRS and v.dims == ("position",):
                # for booleans, take the OR over each codon
                if v.dtype == bool:
                    new_seqlike.ds[k] = (
                        "position",
                        v.coarsen({"position": 3}).any().values,
                    )
                elif v.dtype.kind in ["i", "f"]:  # integer or float
                    new_seqlike.ds[k] = (
                        "position",
                        v.coarsen({"position": 3}).mean().values,
                    )
                else:
                    # For categorical/string data,
                    # take mode (most common value) over each codon
                    values = v.values
                    new_values = []
                    for i in range(0, len(values), 3):
                        triplet = values[i : i + 3]
                        # Get most common value in triplet
                        if np.issubdtype(v.dtype, np.integer):
                            counts = np.bincount(triplet)
                            mode_val = np.argmax(counts)
                        else:
                            counts = pd.Series(triplet).value_counts()
                            mode_val = counts.index[0]
                        new_values.append(mode_val)
                    new_seqlike.ds[k] = ("position", np.array(new_values))
        return new_seqlike

    def __str__(self) -> str:
        """Get string representation of the sequence.

        :returns: The sequence as a string
        """
        return "".join(i for i in self.sequence.values)

    def __getattr__(self, name) -> Any:
        """Delegate attribute access to the underlying xarray Dataset.

        :param name: Name of the attribute to access
        :returns: The requested attribute
        :raises AttributeError: If the attribute doesn't exist
        """
        if hasattr(self.ds, name):
            return getattr(self.ds, name)
        raise AttributeError(f"'SeqLike' object has no attribute '{name}'")
