"""SeqLike class for working with biological sequences."""

from enum import Enum
from typing import Any, Dict, Iterable, Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from Bio.Seq import Seq
from jax import random, vmap

from .alphabets import AA
from .encoders import OneHotEncoder, OrdinalEncoder
from .utils.orf import find_longest_orf


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
        self.live_dataset = xr.Dataset(
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
        self.live_dataset["index_encoding"] = ("position", index_encoded)

        # Create onehot encoding
        onehot_encoded = self._onehot_encoder.transform(sequence)
        self.live_dataset["onehot_encoding"] = (
            ("position", "alphabet_position"),
            onehot_encoded,
        )

        self.live_dataset["longest_orf"] = (
            "position",
            np.zeros(len(sequence), dtype=bool),
        )
        # If this is a nucleotide sequence, try to find and annotate the longest ORF
        if self._seq_type == SeqType.NT:
            orf_coords = find_longest_orf(sequence)
            if orf_coords is not None:
                start, end = orf_coords
                # Create boolean array marking ORF positions
                in_orf = np.zeros(len(sequence), dtype=bool)
                in_orf[start:end] = True
                self.live_dataset["longest_orf"] = ("position", in_orf)

        if seq_type == SeqType.AA:
            self.aa_dataset = self.live_dataset
        else:
            self.nt_dataset = self.live_dataset

    def sel(self, *args, **kwargs) -> "SeqLike":
        """Select positions from the sequence based on criteria.

        This method allows filtering the sequence based on arbitrary criteria.
        The selection criteria are passed to xarray's where() function.

        :param args: Positional arguments passed to xarray.Dataset.where()
        :param kwargs: Keyword arguments passed to xarray.Dataset.where()
        :returns: A new SeqLike object containing only the selected positions
        """
        drop = kwargs.pop("drop", True)
        positions = self.live_dataset.position.where(*args, **kwargs, drop=drop)
        selected_ds = self.live_dataset.sel(position=positions)

        # Create new SeqLike instance
        new_seqlike = SeqLike(
            sequence="".join(selected_ds.sequence.values),
            alphabet=self.live_dataset.alphabet.values,
            seq_type=self._seq_type,
        )

        # Copy over any additional data variables
        for var in selected_ds.data_vars:
            new_seqlike.live_dataset[var] = (
                selected_ds[var].dims,
                selected_ds[var].values,
            )

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
            if not isinstance(v, np.ndarray):
                v = np.array(v, dtype=type(v[0]))
            # Check that annotation is not a reserved attribute
            if k in RESERVED_ATTRS:
                raise ValueError(
                    f"Attribute {k} is reserved! Please use a different name."
                )
            if len(v) != len(self.live_dataset.position):
                raise ValueError(
                    f"Annotation {k} must be the same length as the sequence!"
                )
            self.live_dataset[k] = ("position", v)
        return self

    @property
    def annotations(self) -> Dict[str, xr.DataArray]:
        """Get all annotation tracks.

        :returns: Dictionary mapping annotation names
            to their xarray.DataArray representations
        """
        return {
            k: v
            for k, v in self.live_dataset.data_vars.items()
            if k not in ["sequence", "alphabet", "onehot_encoding", "index_encoding"]
        }

    def aa(self) -> "SeqLike":
        """Swap representations from the nucleotide to the amino acid domain.

        This imply switches self.live_dataset to self.aa_dataset.
        Will error out if self.aa_dataset is not initialized
        and instead suggest running .translate() first.
        """
        self._seq_type = SeqType.AA
        self.live_dataset = self.aa_dataset
        return self

    def nt(self) -> "SeqLike":
        """Swap representations from the amino acid to the nucleotide domain.

        This imply switches self.live_dataset to self.nt_dataset.
        Will error out if self.nt_dataset is not initialized
        and instead suggest running .translate() first.
        """
        self._seq_type = SeqType.NT
        self.live_dataset = self.nt_dataset
        return self

    def translate(self) -> "SeqLike":
        """Translate the nucleotide sequence into an amino acid sequence.

        :returns: A new SeqLike object containing the amino acid sequence
        """
        if self._seq_type != SeqType.NT:
            raise ValueError("This method only works for nucleotide sequences!")

        # Translate sequence into amino acids
        sequence = str(Seq("".join(i for i in self.sequence.values)).translate())
        aa_seqlike = SeqLike(sequence, alphabet=AA, seq_type=SeqType.AA)

        # Copy over all annotation tracks but coarsen them to triplets
        for k, v in self.nt_dataset.data_vars.items():
            if k not in RESERVED_ATTRS and v.dims == ("position",):
                # for booleans, take the OR over each codon
                if v.dtype == bool:
                    aa_seqlike.aa_dataset[k] = (
                        "position",
                        v.coarsen({"position": 3}).any().values,
                    )
                elif v.dtype.kind in ["i", "f"]:  # integer or float
                    aa_seqlike.aa_dataset[k] = (
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
                    aa_seqlike.aa_dataset[k] = ("position", np.array(new_values))
        aa_seqlike.nt_dataset = self.nt_dataset
        return aa_seqlike

    def back_translate(
        self, codon_matrix: Optional[jnp.ndarray] = None, random_seed: int = 42
    ) -> "SeqLike":
        """Back translate the amino acid sequence into a nucleotide sequence.

        Uses JAX for efficient sampling from codon probabilities.
        If no codon matrix is provided, uses uniform probabilities
        for synonymous codons.

        :param codon_matrix: Optional codon probability matrix.
            Shape should be (len(AA), 64).
            If None, uses uniform probabilities.
        :returns: A new SeqLike object containing the nucleotide sequence
        :raises ValueError: If sequence is not amino acid type
        """
        if self._seq_type != SeqType.AA:
            raise ValueError("Can only back-translate amino acid sequences!")

        # Import here to avoid circular imports
        from .utils.codons import CODONS, create_uniform_codon_matrix

        # Use uniform matrix if none provided
        if codon_matrix is None:
            codon_matrix = create_uniform_codon_matrix()

        # Get per-position codon probabilities based on amino acid indices
        aa_indices = self.index_encoding.values
        pos_codon_probs = codon_matrix[aa_indices]

        # Create a random key for each position
        key = random.PRNGKey(random_seed)
        keys = random.split(key, len(aa_indices))

        # Sample codons for each position
        sample_codons = vmap(lambda k, p: random.categorical(k, jnp.log(p)))
        codon_indices = sample_codons(keys, pos_codon_probs)

        # Convert codon indices to sequence
        nt_sequence = "".join(CODONS[idx] for idx in codon_indices)

        # Create new SeqLike object
        from .alphabets import NT

        nt_seqlike = SeqLike(nt_sequence, alphabet=NT, seq_type=SeqType.NT)

        # Copy over relevant annotations, expanding each position to 3 nucleotides
        for k, v in self.live_dataset.data_vars.items():
            if k not in RESERVED_ATTRS and v.dims == ("position",):
                expanded = np.repeat(v.values, 3)
                nt_seqlike.live_dataset[k] = ("position", expanded)

        # Store both representations
        nt_seqlike.aa_dataset = self.live_dataset

        return nt_seqlike

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
        if hasattr(self.live_dataset, name):
            return getattr(self.live_dataset, name)
        raise AttributeError(f"'SeqLike' object has no attribute '{name}'")
