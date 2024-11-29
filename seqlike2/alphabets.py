"""SeqLike catalog of alphabets.

This module contains the definitions of nucleotide and amino acid alphabets
used throughout the SeqLike package, along with utility functions for
checking sequence types and parsing alphabet specifications.
"""

import string

from Bio.Data.IUPACData import (
    ambiguous_dna_values,
    ambiguous_rna_values,
    protein_letters,
)

# Special characters
gap_letter = "-"
stop_letter = "*"
generic_protein_letter = "X"
generic_nt_letter = "N"

# Default alphabet for functions that require one
every_letter_alphabet = string.ascii_uppercase

# The rationale for this ordering
# is that the gap character and standard symbols (4 bases / 20 amino acids)
# should come first, followed by the extra letters.
# If we were to use something like alphanumeric ordering,
# then the standard and full alphabets
# would be mutually incompatible.

STANDARD_NT = gap_letter + "ACGTU" + generic_nt_letter
NT = STANDARD_NT + "BDHKMRSVWY"
STANDARD_AA = stop_letter + gap_letter + protein_letters + generic_protein_letter
AA = STANDARD_AA + "BJOUZ"

STANDARD_NT_SET = set(STANDARD_NT)
NT_SET = set(NT)
STANDARD_AA_SET = set(STANDARD_AA)
AA_SET = set(AA)


def merge_dicts_of_str(d1: dict, d2: dict, ignore_keys: list = None) -> dict:
    """Merge two dictionaries of strings.

    :param d1: First dictionary to merge
    :param d2: Second dictionary to merge
    :param ignore_keys: List of keys to ignore during merging
    :returns: Merged dictionary with unique sorted characters for each key
    """
    if ignore_keys is None:
        ignore_keys = list()
    keys = set(d1).union(d2) - set(ignore_keys)
    return dict((k, "".join(sorted(set(d1.get(k, "") + d2.get(k, ""))))) for k in keys)


# combine ambiguous_dna_values and ambiguous_rna_values into one dict
ambiguous_nt_values = merge_dicts_of_str(
    ambiguous_dna_values, ambiguous_rna_values, ignore_keys="X"
)


def is_NT(sequence) -> bool:
    """Check if a sequence contains only nucleotide characters.

    :param sequence: Input sequence (str, Seq, SeqRecord or SeqLike)
    :returns: True if sequence contains only valid nucleotide characters
    """
    return _is_seqtype(sequence, NT_SET)


def is_AA(sequence) -> bool:
    """Check if a sequence contains only amino acid characters.

    :param sequence: Input sequence (str, Seq, SeqRecord or SeqLike)
    :returns: True if sequence contains only valid amino acid characters
    """
    return _is_seqtype(sequence, AA_SET)


def is_STANDARD_AA(sequence) -> bool:
    """Check if a sequence contains only standard amino acid characters.

    :param sequence: Input sequence (str, Seq, SeqRecord or SeqLike)
    :returns: True if sequence contains only standard amino acid characters
    """
    return _is_seqtype(sequence, STANDARD_AA_SET)


def is_STANDARD_NT(sequence) -> bool:
    """Check if a sequence contains only standard nucleotide characters.

    :param sequence: Input sequence (str, Seq, SeqRecord or SeqLike)
    :returns: True if sequence contains only standard nucleotide characters
    """
    return _is_seqtype(sequence, STANDARD_NT_SET)


def _is_seqtype(sequence, seq_letters: set) -> bool:
    """Internal function to check if a sequence matches a given alphabet.

    Handles various sequence types (SeqLike, SeqRecord, Seq, str) and
    performs case-insensitive comparison against the provided alphabet.

    :param sequence: Input sequence in any supported format
    :param seq_letters: Set of valid characters to check against
    :returns: True if all characters in sequence are in seq_letters
    """
    # seqlike
    if hasattr(sequence, "_seqrecord"):
        sequence = sequence._seqrecord.seq._data

    # seqrecord
    elif hasattr(sequence, "seq"):
        # seqrecord was initialized from a Seq
        try:
            sequence = sequence.seq._data
        # seqrecord was initialized from a string
        except AttributeError:
            sequence = sequence.seq
    # seq
    elif hasattr(sequence, "_data"):
        sequence = sequence._data

    if isinstance(sequence, bytes):
        sequence = sequence.decode()
    sequence = sequence.upper()

    # The meat of the logic lies here.
    return set(sequence).issubset(seq_letters)


def parse_alphabet(alphabet: str) -> str:
    """Parse and validate an alphabet specification.

    Converts various common alphabet specifications into the standardized
    NT or AA alphabet strings used by SeqLike.

    :param alphabet: String specifying 'NT', 'DNA', 'RNA', or 'AA' (case insensitive)
    :returns: Either the NT or AA alphabet string
    :raises AssertionError: If alphabet specification is invalid
    """
    # parse string designation to desired alphabet
    if isinstance(alphabet, str):
        alphabet = alphabet.upper()
        assert alphabet in ["NT", "DNA", "RNA", "AA"], "Invalid alphabet!"
    if alphabet in ["DNA", "NT", "RNA"]:
        return NT
    else:
        return AA
