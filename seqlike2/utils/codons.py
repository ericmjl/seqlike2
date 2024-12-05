"""Utilities for working with codons and codon usage."""

from typing import Dict

import jax.numpy as jnp
import numpy as np

from ..alphabets import AA

# Define all possible codons
NUCLEOTIDES = ["A", "C", "G", "T"]
CODONS = [
    n1 + n2 + n3 for n1 in NUCLEOTIDES for n2 in NUCLEOTIDES for n3 in NUCLEOTIDES
]

# Standard genetic code mapping
GENETIC_CODE: Dict[str, str] = {
    "ATA": "I",
    "ATC": "I",
    "ATT": "I",
    "ATG": "M",
    "ACA": "T",
    "ACC": "T",
    "ACG": "T",
    "ACT": "T",
    "AAC": "N",
    "AAT": "N",
    "AAA": "K",
    "AAG": "K",
    "AGC": "S",
    "AGT": "S",
    "AGA": "R",
    "AGG": "R",
    "CTA": "L",
    "CTC": "L",
    "CTG": "L",
    "CTT": "L",
    "CCA": "P",
    "CCC": "P",
    "CCG": "P",
    "CCT": "P",
    "CAC": "H",
    "CAT": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGA": "R",
    "CGC": "R",
    "CGG": "R",
    "CGT": "R",
    "GTA": "V",
    "GTC": "V",
    "GTG": "V",
    "GTT": "V",
    "GCA": "A",
    "GCC": "A",
    "GCG": "A",
    "GCT": "A",
    "GAC": "D",
    "GAT": "D",
    "GAA": "E",
    "GAG": "E",
    "GGA": "G",
    "GGC": "G",
    "GGG": "G",
    "GGT": "G",
    "TCA": "S",
    "TCC": "S",
    "TCG": "S",
    "TCT": "S",
    "TTC": "F",
    "TTT": "F",
    "TTA": "L",
    "TTG": "L",
    "TAC": "Y",
    "TAT": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGC": "C",
    "TGT": "C",
    "TGA": "*",
    "TGG": "W",
}


def create_uniform_codon_matrix() -> jnp.ndarray:
    """Create a uniform codon probability matrix.

    The matrix has amino acids as rows and codons as columns.
    Each row contains uniform probabilities for the codons
    that encode that amino acid.

    :returns: JAX array of shape (len(AA), 64) containing codon probabilities
    """
    # Initialize matrix with zeros
    matrix = np.zeros((len(AA), len(CODONS)))

    # For each amino acid, count its codons and assign uniform probabilities
    for aa_idx, aa in enumerate(sorted(AA)):
        # Get all codons coding for this AA
        aa_codons = [i for i, v in enumerate(CODONS) if GENETIC_CODE.get(v, "X") == aa]
        if aa_codons:  # Skip if no codons found (e.g., for special characters)
            matrix[aa_idx, aa_codons] = 1.0 / len(aa_codons)

    return jnp.array(matrix)


def get_codon_indices() -> Dict[str, int]:
    """Get mapping of codons to their indices in the codon matrix.

    :returns: Dictionary mapping codon strings to column indices
    """
    return {codon: idx for idx, codon in enumerate(CODONS)}
