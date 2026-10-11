"""
Circuit construction: turn a sequence of pool entries into a CUDA-Q kernel.

This is the *fixed skeleton* of the pipeline. ``apply_pool_entry`` converts one
``PoolEntry`` into gates; ``build_kernel_for_sequence`` is the orchestrator that
prepares the Hartree-Fock reference and applies a whole sequence.

The double excitation uses the compact 14-CNOT Givens decomposition
(Anselmetti et al. 2021 / PennyLane ``DoubleExcitation``), verified to
reproduce the exact |1100>-|0011> rotation. Under a depolarising noise model
the CNOT count is the error budget, so this is ~3.4x cleaner than the naive
48-CNOT Pauli-staircase form.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .pool import PoolEntry


def apply_pool_entry(kernel, qubits, op: PoolEntry) -> None:
    """Apply one pool entry to ``kernel`` in place (side effect; no return).

    identity -> nothing
    single   -> 4 CNOT + 2 RY (Givens single excitation)
    double   -> compact 14 CNOT + 8 RY + 6 H (Givens double excitation)
    """
    if op.kind == "identity":
        return

    if op.kind == "single":
        q, p = op.wires
        theta = op.theta
        kernel.cx(qubits[q], qubits[p])
        kernel.ry(theta / 2.0, qubits[q])
        kernel.cx(qubits[p], qubits[q])
        kernel.ry(-theta / 2.0, qubits[q])
        kernel.cx(qubits[p], qubits[q])
        kernel.cx(qubits[q], qubits[p])
        return

    if op.kind == "double":
        q, r, p, s = op.wires
        a = op.theta / 8.0
        w0, w1, w2, w3 = qubits[q], qubits[r], qubits[p], qubits[s]
        # Compact 14-CNOT DoubleExcitation (Anselmetti 2021 / PennyLane).
        kernel.cx(w2, w3); kernel.cx(w0, w2); kernel.h(w3); kernel.h(w0)
        kernel.cx(w2, w3); kernel.cx(w0, w1)
        kernel.ry(a, w1);  kernel.ry(-a, w0)
        kernel.cx(w0, w3); kernel.h(w3); kernel.cx(w3, w1)
        kernel.ry(a, w1);  kernel.ry(-a, w0)
        kernel.cx(w2, w1); kernel.cx(w2, w0)
        kernel.ry(-a, w1); kernel.ry(a, w0)
        kernel.cx(w3, w1); kernel.h(w3); kernel.cx(w0, w3)
        kernel.ry(-a, w1); kernel.ry(a, w0)
        kernel.cx(w0, w1); kernel.cx(w2, w0); kernel.h(w0); kernel.h(w3)
        kernel.cx(w0, w2); kernel.cx(w2, w3)
        return

    raise ValueError(f"Unknown pool entry kind {op.kind!r}")


def build_kernel_for_sequence(
    num_qubits: int, hf_state: np.ndarray, sequence: List[PoolEntry]
):
    """Orchestrator: allocate qubits, prepare HF, apply the sequence.

    Binds the pool/circuit layers together: produces a runnable CUDA-Q kernel
    for ``sequence`` starting from the Hartree-Fock reference ``hf_state``.
    """
    import cudaq

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(num_qubits)

    for i, occ in enumerate(hf_state):
        if int(occ) == 1:
            kernel.x(qubits[i])

    for op in sequence:
        apply_pool_entry(kernel, qubits, op)

    return kernel
