"""
Oracles: run a circuit and measure an observable.

These are the energy and particle-number "oracles" the training loop scores
candidate circuits with. All expectation values are computed analytically with
``cudaq.observe`` (exact, infinite-shot limit on the statevector /
density-matrix backends), so there is no sampling noise beyond the simulated
gate noise.
"""

from __future__ import annotations

import operator
from functools import reduce
from typing import List

import numpy as np
import cudaq
from tqdm import tqdm

from .pool import PoolEntry
from .circuits import build_kernel_for_sequence


# ---------------------------------------------------------------------------
# Energy oracle
# ---------------------------------------------------------------------------
def energies_for_sequence(
    num_qubits: int,
    hf_state: np.ndarray,
    hamiltonian: "cudaq.SpinOperator",
    sequence: List[PoolEntry],
) -> List[float]:
    """Prefix energies <H> after every operator in ``sequence``."""
    energies: List[float] = []
    for k in range(1, len(sequence) + 1):
        kernel_k = build_kernel_for_sequence(num_qubits, hf_state, sequence[:k])
        e = cudaq.observe(kernel_k, hamiltonian).expectation()
        energies.append(float(e))
    return energies


def get_subsequence_energies(
    num_qubits: int,
    hf_state: np.ndarray,
    hamiltonian: "cudaq.SpinOperator",
    pool: List[PoolEntry],
    token_indices: np.ndarray,
) -> np.ndarray:
    """For (N, seq_len) token indices, return (N, seq_len) prefix energies."""
    N, seq_len = token_indices.shape
    out = np.zeros((N, seq_len), dtype=np.float32)
    for i in tqdm(range(N), desc="Energies", leave=False):
        seq = [pool[int(idx)] for idx in token_indices[i]]
        out[i, :] = energies_for_sequence(num_qubits, hf_state, hamiltonian, seq)
    return out


# ---------------------------------------------------------------------------
# Particle-number oracle (leakage)
# ---------------------------------------------------------------------------
def n_expectation_for_sequence(
    num_qubits: int,
    hf_state: np.ndarray,
    N_op: "cudaq.SpinOperator",
    sequence: List[PoolEntry],
) -> float:
    """<N> at the end of the sequence."""
    kernel = build_kernel_for_sequence(num_qubits, hf_state, sequence)
    return float(cudaq.observe(kernel, N_op).expectation())


def get_n_violations(
    num_qubits: int,
    hf_state: np.ndarray,
    N_op: "cudaq.SpinOperator",
    n_elec: int,
    pool: List[PoolEntry],
    token_indices: np.ndarray,
) -> np.ndarray:
    """For each sequence, return dN = <N> - n_elec at the final step."""
    N = token_indices.shape[0]
    out = np.zeros(N, dtype=np.float32)
    for i in tqdm(range(N), desc="N violations", leave=False):
        seq = [pool[int(idx)] for idx in token_indices[i]]
        out[i] = n_expectation_for_sequence(num_qubits, hf_state, N_op, seq) - n_elec
    return out


# ---------------------------------------------------------------------------
# Parity syndrome (observe-based; segfault-proof under noise)
# ---------------------------------------------------------------------------
def syndrome_violation_rate(
    num_qubits: int,
    hf_state: np.ndarray,
    sequence: List[PoolEntry],
    n_elec: int,
    shots=None,  # accepted for interface compatibility; unused (analytic)
) -> float:
    """Parity-detectable leakage fraction via <Z^n>, no sampling.

        P(odd parity) = (1 - <Z^n>) / 2

    Preferred over the ancilla + ``cudaq.sample`` syndrome, which segfaults on
    the density-matrix backend when noise is active (partial measurement +
    noise + density matrix). This uses the same ``observe`` machinery as the
    energy oracle, so it is deterministic and robust.
    """
    kernel = build_kernel_for_sequence(num_qubits, hf_state, sequence)
    parity = reduce(operator.mul, [cudaq.spin.z(i) for i in range(num_qubits)])
    exp = float(cudaq.observe(kernel, parity).expectation())
    p_odd = 0.5 * (1.0 - exp)
    return p_odd if (n_elec % 2 == 0) else (1.0 - p_odd)


def build_parity_syndrome_kernel(
    num_qubits: int, hf_state: np.ndarray, sequence: List[PoolEntry]
):
    """Ancilla-based parity syndrome kernel (hardware-style, QEC analogue).

    HF prep + sequence + CNOT-chain into one ancilla + measure only the
    ancilla, which then holds N mod 2. Kept for reference / hardware use; do
    not use under the density-matrix backend with noise active (it segfaults --
    use ``syndrome_violation_rate`` instead).
    """
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(num_qubits)
    ancilla = kernel.qalloc(1)
    for i, occ in enumerate(hf_state):
        if int(occ) == 1:
            kernel.x(qubits[i])
    from .circuits import apply_pool_entry

    for op in sequence:
        apply_pool_entry(kernel, qubits, op)
    for i in range(num_qubits):
        kernel.cx(qubits[i], ancilla[0])
    kernel.mz(ancilla[0])
    return kernel
