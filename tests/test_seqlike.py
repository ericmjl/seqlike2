"""Tests for the SeqLike class."""

import numpy as np
import pytest
import xarray as xr
from Bio.Seq import Seq

from seqlike2.alphabets import AA, NT
from seqlike2.seqlike import SeqLike, SeqType


@pytest.fixture
def nt_sequence() -> str:
    """Return a sample nucleotide sequence."""
    return "ATGCCCGGGTAA"


@pytest.fixture
def aa_sequence() -> str:
    """Return a sample amino acid sequence."""
    return "MPGX"


@pytest.fixture
def translatable_aa_sequence() -> str:
    """Return a sample amino acid sequence that can be back-translated."""
    return "MPG*"


@pytest.fixture
def non_translatable_aa_sequence() -> str:
    """Return a sample amino acid sequence that cannot be back-translated."""
    return "MPGXBZ"


@pytest.fixture
def nt_seqlike(nt_sequence: str) -> SeqLike:
    """Return a SeqLike instance with nucleotide sequence."""
    return SeqLike(sequence=nt_sequence, alphabet=NT, seq_type=SeqType.NT)


@pytest.fixture
def aa_seqlike(aa_sequence: str) -> SeqLike:
    """Return a SeqLike instance with amino acid sequence."""
    return SeqLike(sequence=aa_sequence, alphabet=AA, seq_type=SeqType.AA)


@pytest.fixture
def translatable_aa_seqlike(translatable_aa_sequence: str) -> SeqLike:
    """Return a SeqLike instance with translatable amino acid sequence."""
    return SeqLike(sequence=translatable_aa_sequence, alphabet=AA, seq_type=SeqType.AA)


@pytest.fixture
def non_translatable_aa_seqlike(non_translatable_aa_sequence: str) -> SeqLike:
    """Return a SeqLike instance with non-translatable amino acid sequence."""
    return SeqLike(
        sequence=non_translatable_aa_sequence, alphabet=AA, seq_type=SeqType.AA
    )


def test_seqlike_initialization(nt_sequence: str, nt_seqlike: SeqLike):
    """Test basic initialization of SeqLike object."""
    assert str(nt_seqlike) == nt_sequence
    assert isinstance(nt_seqlike.sequence_data, xr.Dataset)
    assert "".join(nt_seqlike.alphabet.values) == "".join(sorted(NT))


def test_seqlike_encodings(nt_seqlike: SeqLike):
    """Test that encodings are created correctly."""
    # Test index encoding
    assert nt_seqlike.index_encoding.dims == ("position",)
    assert len(nt_seqlike.index_encoding) == len(str(nt_seqlike))

    # Test onehot encoding
    assert nt_seqlike.onehot_encoding.dims == ("position", "alphabet_position")
    assert nt_seqlike.onehot_encoding.shape == (len(str(nt_seqlike)), len(NT))


def test_seqlike_selection(nt_seqlike: SeqLike):
    """Test position selection functionality."""
    # Test selection with boolean mask
    mask = nt_seqlike.sequence == "A"
    selected = nt_seqlike.select(mask)
    assert str(selected) == "AAA"
    assert len(selected.sequence) == 3


def test_seqlike_annotation(nt_seqlike: SeqLike):
    """Test annotation functionality."""
    # Test adding boolean annotation
    is_gc = np.array([c in "GC" for c in str(nt_seqlike)])
    nt_seqlike.annotate(is_gc=is_gc)
    assert "is_gc" in nt_seqlike.annotations
    assert nt_seqlike.is_gc.dtype == bool

    # Test adding numeric annotation
    position_scores = np.random.rand(len(str(nt_seqlike)))
    nt_seqlike.annotate(scores=position_scores)
    assert "scores" in nt_seqlike.annotations
    assert nt_seqlike.scores.dtype == float

    # Test error on wrong length annotation
    with pytest.raises(ValueError):
        nt_seqlike.annotate(wrong_length=[1, 2])

    # Test error on reserved attribute names
    with pytest.raises(ValueError):
        nt_seqlike.annotate(sequence=[1] * len(str(nt_seqlike)))


def test_translation(nt_seqlike: SeqLike):
    """Test translation from nucleotide to amino acid sequence."""
    aa_seq = nt_seqlike.translate()
    assert aa_seq._seq_type == SeqType.AA
    assert str(aa_seq) == str(Seq(str(nt_seqlike)).translate())

    # Test annotation transfer
    nt_seqlike.annotate(
        is_gc=np.array([c in "GC" for c in str(nt_seqlike)]),
        scores=np.random.rand(len(str(nt_seqlike))),
    )
    aa_seq = nt_seqlike.translate()
    assert "is_gc" in aa_seq.annotations
    assert "scores" in aa_seq.annotations
    assert len(aa_seq.is_gc) == len(str(aa_seq))


def test_back_translation(
    translatable_aa_seqlike: SeqLike, non_translatable_aa_seqlike: SeqLike
):
    """Test back translation from amino acid to nucleotide sequence."""
    # Test successful back translation
    nt_seq = translatable_aa_seqlike.back_translate()
    assert nt_seq._seq_type == SeqType.NT
    assert str(Seq(str(nt_seq)).translate()) == str(translatable_aa_seqlike)

    # Test annotation transfer for successful back translation
    translatable_aa_seqlike.annotate(
        test_bool=np.array([True] * len(str(translatable_aa_seqlike))),
        test_float=np.random.rand(len(str(translatable_aa_seqlike))),
    )
    nt_seq = translatable_aa_seqlike.back_translate()
    assert "test_bool" in nt_seq.annotations
    assert "test_float" in nt_seq.annotations
    assert len(nt_seq.test_bool) == len(str(nt_seq))

    # Test failure with non-translatable sequence
    with pytest.raises(
        ValueError, match="Sequence contains invalid amino acids for back-translation!"
    ):
        non_translatable_aa_seqlike.back_translate()


def test_domain_switching(nt_seqlike: SeqLike, aa_seqlike: SeqLike):
    """Test switching between nucleotide and amino acid domains."""
    # Original test cases
    aa_seq = nt_seqlike.translate()
    nt_seq = aa_seq.nt()
    assert nt_seq._seq_type == SeqType.NT
    assert str(nt_seq) == str(nt_seqlike)
    aa_seq = nt_seq.aa()
    assert aa_seq._seq_type == SeqType.AA

    # Additional test cases from the second implementation
    # Create a nucleotide sequence that's translatable
    nt_seq = SeqLike("ATGGCCTAA", alphabet=NT)

    # Translate to amino acid
    aa_seq = nt_seq.translate()

    # Test error when trying to switch to non-existent domain
    nt_only = SeqLike("ATGGCCTAA", alphabet=NT)
    with pytest.raises(AttributeError):
        nt_only.aa()  # Should fail because aa_dataset doesn't exist yet


def test_longest_orf_annotation(nt_seqlike: SeqLike):
    """Test that longest ORF is correctly annotated."""
    assert "longest_orf" in nt_seqlike.annotations
    assert nt_seqlike.longest_orf.dtype == bool

    # For our test sequence "ATGCCCGGGTAA", the entire sequence is an ORF
    assert np.all(nt_seqlike.longest_orf.values)


def test_error_conditions():
    """Test various error conditions."""
    # Test invalid sequence type for translation
    aa_seq = SeqLike("MPGX", alphabet=AA, seq_type=SeqType.AA)
    with pytest.raises(ValueError):
        aa_seq.translate()

    # Test invalid sequence type for back translation
    nt_seq = SeqLike("ATGCCCGGGTAA", alphabet=NT, seq_type=SeqType.NT)
    with pytest.raises(ValueError):
        nt_seq.back_translate()


def test_attribute_access(nt_seqlike: SeqLike):
    """Test attribute access delegation to xarray Dataset."""
    # Test access to existing attribute
    assert hasattr(nt_seqlike, "sequence")

    # Test access to non-existent attribute
    with pytest.raises(AttributeError):
        nt_seqlike.nonexistent_attribute


def test_sequence_slicing(nt_seqlike: SeqLike):
    """Test sequence slicing functionality."""
    # Test basic slicing
    assert str(nt_seqlike[0:3]) == "ATG"
    assert str(nt_seqlike[:3]) == "ATG"
    assert str(nt_seqlike[-3:]) == "TAA"
    assert str(nt_seqlike[::2]) == "AGCGGA"

    # Test single position selection
    assert str(nt_seqlike[0]) == "A"
    assert str(nt_seqlike[-1]) == "A"

    # Test boolean mask
    mask = np.array([c == "A" for c in str(nt_seqlike)])
    assert str(nt_seqlike[mask]) == "AAA"

    # Test integer array indexing
    indices = [0, 2, 4]
    assert str(nt_seqlike[indices]) == "AGC"

    # Test error conditions
    with pytest.raises(ValueError):
        nt_seqlike[np.array([True, False])]  # Wrong length boolean mask

    with pytest.raises(TypeError):
        nt_seqlike["invalid"]  # Invalid key type

    # Test that annotations are preserved
    nt_seqlike.annotate(is_gc=np.array([c in "GC" for c in str(nt_seqlike)]))
    sliced = nt_seqlike[0:3]
    assert "is_gc" in sliced.annotations
    assert len(sliced.is_gc) == 3


def test_annotation_preservation():
    """Test preservation of annotations during translation/back-translation."""
    # Create sequence with annotations
    nt_seq = SeqLike("ATGGCCTAA", alphabet=NT)
    nt_seq.annotate(
        is_start=np.array(
            [True, False, False, False, False, False, False, False, False]
        ),
        quality=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
        category=np.array(["A", "B", "C", "A", "B", "C", "A", "B", "C"]),
    )

    # Translate and check annotation preservation
    aa_seq = nt_seq.translate()

    # Check boolean annotation (should use OR for codons)
    assert aa_seq.aa_dataset.is_start.values[0]
    assert all(~aa_seq.aa_dataset.is_start.values[1:])

    # Check numeric annotation (should use mean for codons)
    np.testing.assert_almost_equal(
        aa_seq.aa_dataset.quality.values[0], (0.9 + 0.8 + 0.7) / 3
    )

    # Check categorical annotation (should use mode for codons)
    assert aa_seq.aa_dataset.category.values[0] in ["A", "B", "C"]


def test_slicing_with_domains():
    """Test slicing behavior with domain preservation."""
    # Create a sequence with both domains
    nt_seq = SeqLike("ATGGCCTAA", alphabet=NT, seq_type=SeqType.NT)
    aa_seq = nt_seq.translate()

    # Test valid codon-preserving slice in NT domain
    valid_slice = aa_seq.nt()[0:6]  # Slice 2 complete codons
    assert len(valid_slice.sequence) == 6

    # Test invalid codon-breaking slice
    with pytest.raises(ValueError):
        _ = aa_seq.nt()[0:7]  # This breaks a codon

    # Test AA domain slicing
    aa_slice = aa_seq[0:2]
    assert len(aa_slice.sequence) == 2
    assert len(aa_slice.nt_dataset.sequence) == 6  # Should have 6 NT positions


def test_back_translation_validation():
    """Test back-translation input validation and error handling."""
    # Test with invalid amino acids
    invalid_aa = SeqLike("ABZX", alphabet=AA, seq_type=SeqType.AA)
    with pytest.raises(ValueError, match="invalid amino acids"):
        invalid_aa.back_translate()

    # Test with non-AA sequence
    nt_seq = SeqLike("ATGGCC", alphabet=NT)
    with pytest.raises(ValueError, match="Can only back-translate"):
        nt_seq.back_translate()


def test_annotation_validation():
    """Test annotation validation."""
    seq = SeqLike("ATGGCC", alphabet=NT)

    # Test invalid annotation length
    with pytest.raises(ValueError, match="must be the same length"):
        seq.annotate(test=[1, 2, 3])  # Wrong length

    # Test reserved attribute names
    with pytest.raises(ValueError, match="reserved"):
        seq.annotate(sequence=[1, 2, 3, 4, 5, 6])


def test_select_method():
    """Test the select method with various conditions."""
    seq = SeqLike("ATGGCC", alphabet=NT)
    seq.annotate(
        quality=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
        is_start=np.array([True, False, False, False, False, False]),
    )

    # Select high quality positions
    high_qual = seq.select(seq.quality > 0.7)
    assert len(high_qual.sequence) == 2
    assert high_qual.quality.values.tolist() == [0.9, 0.8]

    # Select with multiple conditions
    selected = seq.select((seq.quality > 0.7) & seq.is_start)
    assert len(selected.sequence) == 1
    assert selected.sequence.values[0] == "A"
