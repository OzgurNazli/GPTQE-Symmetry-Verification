"""
Symmetry verification as error mitigation (the headline technique).

Post-select the energy onto the correct particle-number parity sector:

    Pi      = (I + s * Z^n) / 2          s = +1 even electrons, -1 odd
    <Pi>    = (1 + s <Z^n>) / 2          surviving weight in the sector
    <H Pi>  = (<H> + s <H Z^n>) / 2
    E_mit   = <H Pi> / <Pi>              variationally bounded below by E0

Because H conserves particle number, [H, Pi] = 0, so E_mit is the energy of a
valid projected-and-renormalised state and therefore >= the true ground-state
energy (exact for expectation values; finite sampling can dip slightly when
<Pi> is tiny). This is "symmetry verification as error mitigation"
(Bonet-Monroig et al. 2018) realised by post-selection.

The clean entry point ``symmetry_verified_energy`` is the Mitiq-style technique:
give it a circuit-defining sequence and it returns the mitigated energy.
"""

from __future__ import annotations

import operator
from functools import reduce
from typing import List

import numpy as np
import cudaq

from .pool import PoolEntry
from .circuits import build_kernel_for_sequence


def parity_operator(num_qubits: int) -> "cudaq.SpinOperator":
    """Global parity operator Z^{(x)n} on the data register."""
    return reduce(operator.mul, [cudaq.spin.z(i) for i in range(num_qubits)])


def _mitigated_from_expectations(E: float, HZ: float, Z: float, s: float,
                                 floor: float = 1e-3) -> float:
    """Combine <H>, <H Z^n>, <Z^n> into E_mit = <H Pi>/<Pi>.

    Falls back to the raw energy when <Pi> < ``floor`` (deep leakage: nothing
    left in the sector to post-select onto), so the value never blows up.
    """
    num = 0.5 * (E + s * HZ)       # <H Pi>
    den = 0.5 * (1.0 + s * Z)      # <Pi>
    return E if den < floor else num / den


def energies_for_sequence_mitigated(
    num_qubits: int,
    hf_state: np.ndarray,
    hamiltonian: "cudaq.SpinOperator",
    sequence: List[PoolEntry],
    *,
    n_elec: int,
    floor: float = 1e-3,
) -> List[float]:
    """Symmetry-verified prefix energies E_mit(k) for every prefix of ``sequence``.

    Drop-in mitigated analogue of :func:`oracles.energies_for_sequence`.
    """
    parity = parity_operator(num_qubits)
    H_parity = hamiltonian * parity
    s = 1.0 if (n_elec % 2 == 0) else -1.0

    energies: List[float] = []
    for k in range(1, len(sequence) + 1):
        kernel_k = build_kernel_for_sequence(num_qubits, hf_state, sequence[:k])
        E = float(cudaq.observe(kernel_k, hamiltonian).expectation())
        HZ = float(cudaq.observe(kernel_k, H_parity).expectation())
        Z = float(cudaq.observe(kernel_k, parity).expectation())
        energies.append(_mitigated_from_expectations(E, HZ, Z, s, floor))
    return energies


def symmetry_verified_energy(
    num_qubits: int,
    hf_state: np.ndarray,
    hamiltonian: "cudaq.SpinOperator",
    sequence: List[PoolEntry],
    *,
    n_elec: int,
    floor: float = 1e-3,
) -> float:
    """Return the symmetry-verified energy E_mit of the full ``sequence``.

    This is the public one-call mitigation technique: build the circuit, measure
    <H>, <H Z^n>, <Z^n>, and return <H Pi>/<Pi>. Mirrors a Mitiq-style
    ``execute_with_<technique>`` entry point.
    """
    return energies_for_sequence_mitigated(
        num_qubits, hf_state, hamiltonian, sequence, n_elec=n_elec, floor=floor
    )[-1]
