"""
Operator pool (the GPT-QE "vocabulary").

Enumerates particle-number- and spin-preserving single and double excitations
under the Jordan-Wigner mapping, then discretises each excitation over a fixed
set of rotation angles to form the discrete operator pool the transformer
samples from.

This is the *problem-specific* layer: to apply GPT-QE to a different domain
(e.g. QAOA), replace ``excitations`` with that domain's move set and keep the
rest of the pipeline unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple

import numpy as np


def _spin(i: int) -> int:
    """+1 for alpha (even qubit index), -1 for beta (odd qubit index)."""
    return 1 if (i % 2 == 0) else -1


def excitations(
    num_electrons: int, num_qubits: int
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
    """Enumerate all spin-preserving singles and doubles.

    Uses even=alpha / odd=beta spin-orbital ordering and the Hartree-Fock
    reference (occupied = lowest ``num_electrons`` qubits). Each returned
    excitation conserves particle number and total spin by construction, so a
    circuit built from them never leaves the correct symmetry sector in the
    noiseless limit.

    Returns
    -------
    (singles, doubles)
        singles : list of (q, p)         occupied q -> virtual p
        doubles : list of (q, r, p, s)   occupied q,r -> virtual p,s
    """
    occ = list(range(num_electrons))
    vir = list(range(num_electrons, num_qubits))

    singles: List[Tuple[int, int]] = []
    for q in occ:
        for p in vir:
            if _spin(p) == _spin(q):
                singles.append((q, p))

    doubles: List[Tuple[int, int, int, int]] = []
    for q, r in combinations(occ, 2):
        for p, s in combinations(vir, 2):
            if _spin(p) + _spin(s) == _spin(q) + _spin(r):
                doubles.append((q, r, p, s))

    return singles, doubles


@dataclass
class PoolEntry:
    """One token of the operator pool (one discrete excitation move).

    This is the shared data type that couples the whole pipeline: the pool
    produces ``PoolEntry`` objects, the circuit builder consumes them, and the
    transformer refers to them only by their integer index in the pool.
    """

    kind: str  # 'single', 'double', or 'identity'
    wires: Tuple[int, ...]
    theta: float

    def describe(self) -> str:
        if self.kind == "single":
            q, p = self.wires
            return f"S({self.theta:+.4f}; {q}->{p})"
        if self.kind == "double":
            q, r, p, s = self.wires
            return f"D({self.theta:+.4f}; {q},{r}->{p},{s})"
        return f"I({self.theta:+.4f})"


def default_angles() -> np.ndarray:
    """The eight discrete rotation angles used in the thesis.

    +-2^k / 160 for k = 1..4, sorted. Small, symmetric about zero.
    """
    return np.sort(
        np.array([-(2 ** k) for k in range(1, 5)] + [2 ** k for k in range(1, 5)])
        / 160.0
    )


def build_operator_pool(num_electrons: int, num_qubits: int) -> List[PoolEntry]:
    """Build the full discrete operator pool (the model's vocabulary).

    For each double, then each single, then the identity, emit one
    ``PoolEntry`` per discrete angle. For H2 (2 electrons, 4 qubits) this gives
    2 singles x 8 + 1 double x 8 + 8 identity = 32 entries.
    """
    singles, doubles = excitations(num_electrons, num_qubits)
    angles = default_angles()

    pool: List[PoolEntry] = []
    for d in doubles:
        for t in angles:
            pool.append(PoolEntry("double", tuple(int(x) for x in d), float(t)))
    for s in singles:
        for t in angles:
            pool.append(PoolEntry("single", tuple(int(x) for x in s), float(t)))
    for t in angles:
        pool.append(PoolEntry("identity", (), float(t)))
    return pool
