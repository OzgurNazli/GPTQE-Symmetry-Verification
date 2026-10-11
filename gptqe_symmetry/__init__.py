"""
gptqe_symmetry
==============

Particle-number symmetry verification for the Generative Quantum Eigensolver
(GPT-QE), on CUDA-Q.

A transformer assembles molecular circuits from a symmetry-preserving operator
pool; under depolarising noise, particle-number leakage is mitigated by
post-selection onto the correct symmetry sector (symmetry verification).

Public API
----------
Chemistry / pool / circuits:
    build_molecule, build_number_operator, molecule_geometry
    excitations, PoolEntry, build_operator_pool
    apply_pool_entry, build_kernel_for_sequence

Oracles:
    energies_for_sequence, n_expectation_for_sequence, syndrome_violation_rate

Mitigation (the headline technique):
    symmetry_verified_energy, energies_for_sequence_mitigated, parity_operator

Noise / model / training:
    build_noise_model
    GPT, GPTQE, GPTConfig
    train_online
"""

from .chemistry import build_molecule, build_number_operator, molecule_geometry
from .pool import excitations, PoolEntry, build_operator_pool, default_angles
from .circuits import apply_pool_entry, build_kernel_for_sequence
from .noise import build_noise_model
from .oracles import (
    energies_for_sequence,
    get_subsequence_energies,
    n_expectation_for_sequence,
    get_n_violations,
    syndrome_violation_rate,
    build_parity_syndrome_kernel,
)
from .mitigation import (
    symmetry_verified_energy,
    energies_for_sequence_mitigated,
    parity_operator,
)
from .model import GPT, GPTQE, GPTConfig
from .training import train_online

__version__ = "0.1.0"

__all__ = [
    "build_molecule", "build_number_operator", "molecule_geometry",
    "excitations", "PoolEntry", "build_operator_pool", "default_angles",
    "apply_pool_entry", "build_kernel_for_sequence",
    "build_noise_model",
    "energies_for_sequence", "get_subsequence_energies",
    "n_expectation_for_sequence", "get_n_violations",
    "syndrome_violation_rate", "build_parity_syndrome_kernel",
    "symmetry_verified_energy", "energies_for_sequence_mitigated", "parity_operator",
    "GPT", "GPTQE", "GPTConfig", "train_online",
]
