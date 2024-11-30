"""Utilities for finding open reading frames in nucleotide sequences."""

from typing import Optional, Tuple

from Bio.Seq import Seq


def find_longest_orf(sequence: str) -> Optional[Tuple[int, int]]:
    """Find the longest open reading frame in a nucleotide sequence.

    :param sequence: Nucleotide sequence string
    :returns: Tuple of (start, end) positions of the longest ORF,
        or None if no ORF is found
    """
    # Convert to Seq object for translation
    seq = Seq(sequence)

    # Look for start codons
    start_codons = ["ATG"]
    stop_codons = ["TAA", "TAG", "TGA"]

    longest_orf = None
    max_length = 0

    # Check all three reading frames
    for frame in range(3):
        # Get sequence in current frame
        framed_seq = seq[frame:]
        # Ensure length is multiple of 3
        framed_seq = framed_seq[: len(framed_seq) - (len(framed_seq) % 3)]

        # Look for ORFs
        for i in range(0, len(framed_seq), 3):
            codon = str(framed_seq[i : i + 3])
            if codon in start_codons:
                # Found start codon, look for stop codon
                for j in range(i + 3, len(framed_seq), 3):
                    codon = str(framed_seq[j : j + 3])
                    if codon in stop_codons:
                        # Found stop codon, check if this is longest ORF
                        orf_length = j - i + 3
                        if orf_length > max_length:
                            max_length = orf_length
                            # Convert back to original sequence coordinates
                            longest_orf = (i + frame, j + frame + 3)
                        break

    return longest_orf
