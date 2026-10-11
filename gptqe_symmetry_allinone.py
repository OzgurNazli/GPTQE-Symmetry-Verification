"""
gptqe_symmetry_allinone.py
==========================

Single-file, monkey-patch-free consolidation of the whole GPT-QE + particle-
number symmetry-verification pipeline. Functionally identical to the
``gptqe_symmetry`` package; kept as one file so the pipeline reads end to end.

Run:
    python gptqe_symmetry_allinone.py --molecule H2 --bond_length 0.7414 --cudaq_target qpp-cpu
"""


from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import reduce
from itertools import combinations
from pathlib import Path
from torch.nn import functional as F
from tqdm import tqdm
from typing import List
from typing import List, Tuple
from typing import Optional
from typing import Optional, Tuple
import argparse
import cudaq
import inspect
import math
import numpy as np
import operator
import pandas as pd
import time
import torch
import torch.nn as nn

# ======================================================================
# module: pool
# ======================================================================
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

# ======================================================================
# module: chemistry
# ======================================================================
"""
Molecular Hamiltonian, particle-number operator, and geometries.

The physics boundary of the pipeline: PySCF is hidden inside CUDA-Q's
``create_molecular_hamiltonian``, which returns the second-quantised,
Jordan-Wigner-mapped Hamiltonian as a sparse ``SpinOperator``. Everything
downstream works with that operator and never materialises a 2^n x 2^n matrix.
"""





def build_molecule(
    symbols: List[str],
    coords: np.ndarray,
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
) -> Tuple["cudaq.SpinOperator", int, int, np.ndarray]:
    """Build the molecular Hamiltonian and the Hartree-Fock reference.

    Returns
    -------
    (hamiltonian, num_electrons, num_qubits, hf_state)
        hamiltonian  : qubit Hamiltonian H as a CUDA-Q SpinOperator
        num_electrons: number of electrons (target particle-number sector)
        num_qubits   : 2 * number of spatial orbitals
        hf_state     : int array, 1 on the lowest ``num_electrons`` qubits
    """
    geometry = [
        (s, tuple(float(c) for c in coord)) for s, coord in zip(symbols, coords)
    ]
    hamiltonian, data = cudaq.chemistry.create_molecular_hamiltonian(
        geometry, basis, multiplicity, charge
    )
    num_electrons = data.n_electrons
    num_qubits = 2 * data.n_orbitals

    hf_state = np.zeros(num_qubits, dtype=np.int64)
    hf_state[:num_electrons] = 1
    return hamiltonian, num_electrons, num_qubits, hf_state


def build_number_operator(num_qubits: int) -> "cudaq.SpinOperator":
    """Build the total particle-number operator N = sum_i (I - Z_i)/2.

    For any state in the ``n_elec``-electron sector <N> = n_elec exactly;
    the deviation dN = <N> - n_elec is the (noise-induced) leakage signal.
    Built directly as a sum of O(n) Pauli strings.
    """
    N_op = cudaq.spin.i(0) * (0.5 * num_qubits)
    for i in range(num_qubits):
        N_op -= 0.5 * cudaq.spin.z(i)
    return N_op


def molecule_geometry(molecule: str, bond_length: float) -> Tuple[List[str], np.ndarray]:
    """Return (symbols, coords) for the built-in test molecules."""
    m = molecule.upper()
    if m == "H2":
        return ["H", "H"], np.array([[0, 0, 0], [0, 0, bond_length]], float)
    if m == "H4":
        return ["H"] * 4, np.array([[0, 0, i * bond_length] for i in range(4)], float)
    if m == "H6":
        return ["H"] * 6, np.array([[0, 0, i * bond_length] for i in range(6)], float)
    if m == "LIH":
        return ["Li", "H"], np.array([[0, 0, 0], [0, 0, bond_length]], float)
    raise ValueError(f"Unknown molecule {molecule!r} (add its geometry here)")

# ======================================================================
# module: circuits
# ======================================================================
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

# ======================================================================
# module: noise
# ======================================================================
"""
Gate-level noise models.

``build_noise_model`` attaches a per-gate channel to every gate in the circuit.
The depolarising model is implemented and matches the thesis convention:

    p2 = noise_prob          (two-qubit gate rate)
    p1 = noise_prob / 10     (single-qubit gate rate, 10x cleaner)

The dispatch over ``noise_type`` is the single extension point for future work
(the "three noise models" study): add an ``amplitude`` or ``phase`` channel
factory below and every gate registration is reused unchanged.
"""




# Gate names the channels are attached to.
_SINGLE_QUBIT_GATES = ("x", "y", "z", "h", "s", "t", "rx", "ry", "rz", "r1")
_CONTROLLED_BASES = ("x", "y", "z")  # cx, cy, cz via num_controls=1


def _add_depolarizing(noise: "cudaq.NoiseModel", p1: float, p2: float) -> None:
    depol1 = cudaq.DepolarizationChannel(p1)      # 1-qubit, 2x2 Kraus
    depol2 = cudaq.Depolarization2(p2)            # 2-qubit, 4x4 Kraus
    for g in _SINGLE_QUBIT_GATES:
        noise.add_all_qubit_channel(g, depol1)
    for g in _CONTROLLED_BASES:
        noise.add_all_qubit_channel(g, depol2, 1)


def _add_amplitude(noise: "cudaq.NoiseModel", p1: float, p2: float) -> None:
    # EXTENSION POINT (three-noise-models study).
    # Amplitude damping is single-qubit; attach AmplitudeDampingChannel(p1) to
    # the single-qubit gates, and decide how to handle two-qubit gates (e.g.
    # damping on each involved qubit, or leave the 2-qubit part depolarising).
    raise NotImplementedError(
        "amplitude-damping channel not implemented yet; see noise.py extension point"
    )


def _add_phase(noise: "cudaq.NoiseModel", p1: float, p2: float) -> None:
    # EXTENSION POINT (three-noise-models study).
    # Phase errors keep the particle-number sector, so a particle-number filter
    # is expected to be blind to them -- the key hypothesis to test here.
    raise NotImplementedError(
        "phase-damping / phase-flip channel not implemented yet; see noise.py extension point"
    )


_CHANNEL_FACTORIES = {
    "depolarizing": _add_depolarizing,
    "amplitude": _add_amplitude,
    "phase": _add_phase,
}


def build_noise_model(
    prob: float, noise_type: str = "depolarizing"
) -> Optional["cudaq.NoiseModel"]:
    """Build a gate-level noise model, or ``None`` for the noiseless case.

    Parameters
    ----------
    prob : float
        Two-qubit gate error rate. Single-qubit gates use ``prob / 10``.
        ``prob <= 0`` returns ``None`` (caller then skips ``cudaq.set_noise``).
    noise_type : str
        One of ``_CHANNEL_FACTORIES``. Only ``depolarizing`` is implemented.
    """
    if prob <= 0.0:
        return None

    p2 = min(prob, 1.0)
    p1 = min(prob / 10.0, 1.0)

    try:
        add_channels = _CHANNEL_FACTORIES[noise_type]
    except KeyError:
        raise ValueError(
            f"Unknown noise_type {noise_type!r}; "
            f"choose from {sorted(_CHANNEL_FACTORIES)}"
        )

    noise = cudaq.NoiseModel()
    add_channels(noise, p1, p2)
    return noise

# ======================================================================
# module: oracles
# ======================================================================
"""
Oracles: run a circuit and measure an observable.

These are the energy and particle-number "oracles" the training loop scores
candidate circuits with. All expectation values are computed analytically with
``cudaq.observe`` (exact, infinite-shot limit on the statevector /
density-matrix backends), so there is no sampling noise beyond the simulated
gate noise.
"""






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

# ======================================================================
# module: mitigation
# ======================================================================
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

# ======================================================================
# module: model
# ======================================================================
"""
Transformer backbone and the GPT-QE head.

``GPT`` is a trimmed decoder-only nanoGPT backbone (Karpathy nanoGPT; GPT-2
architecture). ``GPTQE`` adds the logit-matching head of Nakaji et al. 2024:
the cumulative next-token logit is regressed onto the circuit energy, and
generation samples with a *negative*-logit Boltzmann softmax so low-energy
circuits are produced. An optional soft particle-number penalty is available
in the loss.
"""





class LayerNorm(nn.Module):
    """LayerNorm with an optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class GPT(nn.Module):
    """Trimmed decoder-only transformer backbone."""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None and config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay = [p for p in param_dict.values() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra = dict(fused=True) if (fused_ok and device_type == "cuda") else dict()
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)


class GPTQE(GPT):
    """Logit-matching GPT for circuit construction (Nakaji et al. 2024),
    with an optional symmetry-verification (particle-number) penalty."""

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        device = idx.device
        _, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    def calculate_loss(
        self,
        tokens: torch.Tensor,
        energies: torch.Tensor,
        n_violations: Optional[torch.Tensor] = None,
        penalty_weight: float = 0.0,
    ) -> torch.Tensor:
        """Logit-matching MSE plus an optional lambda * dN^2 penalty (final step)."""
        current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
        logits = self(current_tokens)
        mask = F.one_hot(next_tokens, num_classes=self.config.vocab_size)
        next_token_logits = (logits * mask).sum(axis=2)
        cumsum_logits = torch.cumsum(next_token_logits, dim=1)

        target = energies
        if n_violations is not None and penalty_weight > 0.0:
            penalty_final = penalty_weight * (n_violations ** 2)
            pad = torch.zeros_like(target)
            pad[:, -1] = penalty_final
            target = target + pad
        return torch.mean(torch.square(cumsum_logits - target))

    @torch.no_grad()
    def generate(
        self, n_sequences: int, max_new_tokens: int,
        temperature: float = 1.0, device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively sample sequences with a negative-logit Boltzmann softmax."""
        idx = torch.zeros((n_sequences, 1), dtype=torch.long, device=device)
        total_logits = torch.zeros((n_sequences, 1), device=device)
        for _ in range(max_new_tokens):
            idx_cond = (idx if idx.size(1) <= self.config.block_size
                        else idx[:, -self.config.block_size:])
            logits = self(idx_cond)[:, -1, :]
            logits[:, 0] = float("inf")                 # forbid the BOS token
            probs = F.softmax(-logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            total_logits += torch.gather(logits, index=idx_next, dim=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx, total_logits

# ======================================================================
# module: training
# ======================================================================
"""
Online GPT-QE training loop with experience replay.

The transformer proposes circuits, the CUDA-Q oracle scores them, and the model
is trained on its own samples drawn from a FIFO replay buffer. This is what the
thesis used: a uniform random dataset almost never contains the good
multi-double sequences needed to reach the correlated ground state, so the model
must *discover* them on-policy.

Two toggles select the three strategies compared in the thesis, replacing the
old monkey-patched variants:
  * ``penalty_weight > 0`` with ``verify_symmetry`` -> soft dN^2 penalty in the loss
  * ``mitigate=True``                               -> train on symmetry-verified E_mit
  * neither                                         -> raw-energy baseline
"""






def train_online(args: argparse.Namespace) -> None:
    # ---- Backend + noise -------------------------------------------------
    target = args.cudaq_target
    if args.noise_prob > 0.0 and target not in ("density-matrix-cpu", "nvidia"):
        print(f"[noise] forcing target -> density-matrix-cpu (was {target!r})")
        target = "density-matrix-cpu"
    cudaq.set_target(target)
    print(f"[CUDA-Q target] {target}")

    noise_model = build_noise_model(args.noise_prob, getattr(args, "noise_type", "depolarizing"))
    if noise_model is not None:
        cudaq.set_noise(noise_model)
        print(f"[noise] {getattr(args, 'noise_type', 'depolarizing')}, "
              f"p1={min(args.noise_prob / 10, 1.0):g}, p2={min(args.noise_prob, 1.0):g}")
    else:
        print("[noise] none (noiseless training)")

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[Torch device] {dev}")

    # ---- Molecule / Hamiltonian / pool -----------------------------------
    symbols, coords = molecule_geometry(args.molecule, args.bond_length)
    print(f"[Molecule] {args.molecule} / {args.basis}, bond={args.bond_length} A")
    hamiltonian, n_elec, n_qubits, hf_state = build_molecule(symbols, coords, basis=args.basis)
    N_op = build_number_operator(n_qubits)

    hf_kernel = build_kernel_for_sequence(n_qubits, hf_state, sequence=[])
    e_hf = float(cudaq.observe(hf_kernel, hamiltonian).expectation())
    print(f"  -> n_electrons={n_elec}, n_qubits={n_qubits}, E_HF={e_hf:.6f} Ha")

    pool = build_operator_pool(n_elec, n_qubits)
    pool_size = len(pool)
    vocab_size = pool_size + 1
    sg, db = excitations(n_elec, n_qubits)
    print(f"[Operator pool] |singles|={len(sg)}, |doubles|={len(db)}, "
          f"pool_size={pool_size} (vocab={vocab_size})")

    mitigate = bool(getattr(args, "mitigate", False))
    if mitigate:
        print("[oracle] training on symmetry-verified energies E_mit = <H Pi>/<Pi>")

    # ---- Model + optimizer ----------------------------------------------
    gpt = GPTQE(GPTConfig(
        vocab_size=vocab_size, block_size=args.seq_len,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout, bias=False)).to(dev)
    optimizer = gpt.configure_optimizers(
        weight_decay=0.01, learning_rate=args.lr, betas=(0.9, 0.999), device_type=dev.type)

    rng = np.random.default_rng(args.seed)

    # ---- Oracle with a cache (deterministic <H> -> memoise) --------------
    cache: dict = {}

    def energy_of(seq):
        if mitigate:
            return np.asarray(
                energies_for_sequence_mitigated(
                    n_qubits, hf_state, hamiltonian, seq, n_elec=n_elec),
                dtype=np.float32)
        return np.asarray(
            energies_for_sequence(n_qubits, hf_state, hamiltonian, seq), dtype=np.float32)

    def evaluate(pool_row: np.ndarray) -> Tuple[np.ndarray, float]:
        key = tuple(int(x) for x in pool_row)
        if key in cache:
            return cache[key]
        seq = [pool[i] for i in key]
        E = energy_of(seq)
        dN = (float(n_expectation_for_sequence(n_qubits, hf_state, N_op, seq) - n_elec)
              if args.verify_symmetry else 0.0)
        cache[key] = (E, dN)
        return E, dN

    # ---- Experience-replay buffer ---------------------------------------
    buffer: deque = deque(maxlen=args.buffer_size)

    def add_samples(pool_inds: np.ndarray) -> None:
        for row in pool_inds:
            E, dN = evaluate(row)
            tok = np.concatenate([[0], row + 1]).astype(np.int64)  # BOS + tokens
            buffer.append((tok, E, dN))

    # ---- Logging ---------------------------------------------------------
    save_dir = Path(args.save_dir) / f"seq_len={args.seq_len}"
    save_dir.mkdir(parents=True, exist_ok=True)
    losses: List[float] = []
    pred_Es_t: List[np.ndarray] = []
    true_Es_t: List[np.ndarray] = []
    dN_eval: List[float] = []
    best_min_E = float("inf")
    step = 0

    print(f"\n[Online GQE] {args.n_epochs} epochs x {args.n_iter} updates, "
          f"{args.n_sample} samples/epoch, buffer<={args.buffer_size}\n")
    t_start = time.time()

    for epoch in range(args.n_epochs):
        # 1) Collect: sample from the model + epsilon-exploration
        gpt.eval()
        n_explore = int(round(args.explore_frac * args.n_sample))
        n_model = args.n_sample - n_explore
        rows = []
        if n_model > 0:
            gen_tokens, _ = gpt.generate(
                n_sequences=n_model, max_new_tokens=args.seq_len,
                temperature=args.sample_temperature, device=dev)
            rows.append((gen_tokens[:, 1:] - 1).cpu().numpy())
        if n_explore > 0:
            rows.append(rng.integers(0, pool_size, size=(n_explore, args.seq_len)))
        add_samples(np.concatenate(rows, axis=0))

        # 2) Update: train on minibatches from the buffer
        gpt.train()
        buf = list(buffer)
        epoch_loss = 0.0
        for _ in range(args.n_iter):
            idx = rng.integers(0, len(buf), size=min(args.n_batch, len(buf)))
            toks = torch.from_numpy(np.stack([buf[i][0] for i in idx])).to(dev)
            ens = torch.from_numpy(np.stack([buf[i][1] for i in idx])).to(dev)
            nvs = (torch.from_numpy(np.array([buf[i][2] for i in idx], np.float32)).to(dev)
                   if args.verify_symmetry else None)
            optimizer.zero_grad()
            loss = gpt.calculate_loss(toks, ens, n_violations=nvs,
                                      penalty_weight=args.penalty_weight)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            epoch_loss += float(loss.item()) / args.n_iter
            step += 1

        # 3) Periodic evaluation (greedy sampling)
        if (epoch + 1) % args.eval_freq == 0:
            gpt.eval()
            gt, pred = gpt.generate(
                n_sequences=args.eval_size, max_new_tokens=args.seq_len,
                temperature=args.eval_temperature, device=dev)
            gen_inds = (gt[:, 1:] - 1).cpu().numpy()
            true_E = np.array([evaluate(r)[0][-1] for r in gen_inds]).reshape(-1, 1)
            pred_np = pred.cpu().numpy().reshape(-1, 1)
            mae = float(np.mean(np.abs(pred_np - true_E)))
            min_E, mean_E = float(true_E.min()), float(true_E.mean())
            pred_Es_t.append(pred_np)
            true_Es_t.append(true_E)
            if args.verify_symmetry:
                dN_eval.append(float(np.mean([abs(evaluate(r)[1]) for r in gen_inds])))

            print(f"  epoch {epoch + 1:>4d} | step {step:>6d} | loss={epoch_loss:.5f} "
                  f"| MAE={mae:.5f} | min(E)={min_E:.6f} | mean(E)={mean_E:.6f} "
                  f"| buf={len(buffer)} | cache={len(cache)} | {time.time() - t_start:.0f}s")

            if min_E < best_min_E:
                best_min_E = min_E
                torch.save(gpt.state_dict(), save_dir / "gptqe_best.pt")
                print(f"      new best min(E)={min_E:.6f}  (saved)")

    # ---- Save artefacts (plot.py compatible) ----------------------------
    torch.save(gpt.state_dict(), save_dir / "gptqe_last.pt")
    pd.DataFrame(losses).to_csv(save_dir / "losses.csv", index=False)
    if pred_Es_t:
        pd.DataFrame(np.concatenate(pred_Es_t, axis=1)).to_csv(save_dir / "pred_Es_t.csv", index=False)
        pd.DataFrame(np.concatenate(true_Es_t, axis=1)).to_csv(save_dir / "true_Es_t.csv", index=False)
    if dN_eval:
        pd.DataFrame({"mean_abs_dN": dN_eval}).to_csv(save_dir / "n_violations_eval.csv", index=False)

    # ---- Summary: random vs latest vs best ------------------------------
    rand = rng.integers(0, pool_size, size=(args.eval_size, args.seq_len))
    rand_E = np.array([evaluate(r)[0][-1] for r in rand])
    gpt.eval()
    gt, _ = gpt.generate(args.eval_size, args.seq_len, temperature=args.eval_temperature, device=dev)
    latest_E = np.array([evaluate(r)[0][-1] for r in (gt[:, 1:] - 1).cpu().numpy()])
    best = GPTQE(GPTConfig(
        vocab_size=vocab_size, block_size=args.seq_len,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout, bias=False)).to(dev)
    best.load_state_dict(torch.load(save_dir / "gptqe_best.pt", weights_only=True))
    best.eval()
    gb, _ = best.generate(args.eval_size, args.seq_len, temperature=args.eval_temperature, device=dev)
    best_E = np.array([evaluate(r)[0][-1] for r in (gb[:, 1:] - 1).cpu().numpy()])

    summary = pd.DataFrame({
        "Source": ["Random", "Latest", "Best (lowest E)"],
        "Mean E": [rand_E.mean(), latest_E.mean(), best_E.mean()],
        "Min E": [rand_E.min(), latest_E.min(), best_E.min()],
        "Max E": [rand_E.max(), latest_E.max(), best_E.max()],
        "E_HF": [e_hf, e_hf, e_hf],
    })
    print("\n" + summary.to_string(index=False))
    summary.to_csv(save_dir / "summary.csv", index=False)

    if args.compare_noisy:
        _compare_noisy(args, best, pool, n_qubits, hf_state, hamiltonian, N_op, n_elec, dev, save_dir)

    n_eval_steps = args.eval_freq * args.n_iter
    print(f"\nResults saved under {save_dir}/")
    print(f"Plot with:  python -m scripts.plot --run_dir {save_dir} "
          f"--e_hf {e_hf:.6f} --eval_freq {n_eval_steps}")


def _compare_noisy(args, best, pool, n_qubits, hf_state, hamiltonian, N_op, n_elec, dev, save_dir):
    """Re-score the best model's circuits with noise OFF then ON."""
    gb, _ = best.generate(args.eval_size, args.seq_len, temperature=args.eval_temperature, device=dev)
    seqs = [[pool[i] for i in row] for row in (gb[:, 1:] - 1).cpu().numpy()]

    def score(seqs):
        E = np.array([energies_for_sequence(n_qubits, hf_state, hamiltonian, s)[-1] for s in seqs])
        dN = np.array([n_expectation_for_sequence(n_qubits, hf_state, N_op, s) - n_elec for s in seqs])
        sv = np.array([syndrome_violation_rate(n_qubits, hf_state, s, n_elec) for s in seqs])
        return E, np.abs(dN), sv

    cudaq.unset_noise()
    En, dNn, svn = score(seqs)
    nm = build_noise_model(args.noise_prob_eval, getattr(args, "noise_type", "depolarizing"))
    if nm is not None:
        cudaq.set_noise(nm)
    Ey, dNy, svy = score(seqs)
    cudaq.unset_noise()

    tbl = pd.DataFrame({
        "Setting": ["noiseless", f"noisy(p={args.noise_prob_eval:g})"],
        "Mean E": [En.mean(), Ey.mean()],
        "Min E": [En.min(), Ey.min()],
        "Mean |dN|": [dNn.mean(), dNy.mean()],
        "Max |dN|": [dNn.max(), dNy.max()],
        "Mean syndrome viol. rate": [svn.mean(), svy.mean()],
    })
    print("\n== Noiseless vs. Noisy (best model) ==")
    print(tbl.to_string(index=False))
    tbl.to_csv(save_dir / "noiseless_vs_noisy.csv", index=False)

# ======================================================================
# module: cli
# ======================================================================
"""Command-line entry point for online GPT-QE training.

Examples
--------
Noiseless correctness check:
    python -m gptqe_symmetry --molecule H2 --bond_length 0.7414 --cudaq_target qpp-cpu

Noisy baseline (leakage logged, no penalty):
    python -m gptqe_symmetry --molecule H2 --noise_prob 0.01 --verify_symmetry \
        --penalty_weight 0.0 --cudaq_target density-matrix-cpu --compare_noisy

Training-integrated mitigation (train on E_mit):
    python -m gptqe_symmetry --molecule H2 --noise_prob 0.01 --mitigate \
        --cudaq_target nvidia
"""






def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Online GPT-QE with particle-number symmetry verification (CUDA-Q)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Molecule
    p.add_argument("--molecule", default="H2", choices=["H2", "H4", "H6", "LiH"])
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--bond_length", type=float, default=0.7414)
    # Backend / noise
    p.add_argument("--cudaq_target", default="qpp-cpu",
                   choices=["qpp-cpu", "nvidia", "nvidia-mqpu", "density-matrix-cpu"])
    p.add_argument("--noise_prob", type=float, default=0.0,
                   help="Two-qubit depolarising rate p2 (single-qubit uses p2/10). 0 = noiseless.")
    p.add_argument("--noise_type", default="depolarizing",
                   choices=["depolarizing", "amplitude", "phase"],
                   help="Noise channel family (amplitude/phase are extension points).")
    # Mitigation strategies
    p.add_argument("--verify_symmetry", action="store_true",
                   help="Soft syndrome: add lambda*dN^2 penalty to the loss.")
    p.add_argument("--penalty_weight", type=float, default=1.0)
    p.add_argument("--mitigate", action="store_true",
                   help="Train on symmetry-verified energies E_mit (post-selection).")
    # Online loop
    p.add_argument("--seq_len", type=int, default=4)
    p.add_argument("--n_epochs", type=int, default=300)
    p.add_argument("--n_sample", type=int, default=200)
    p.add_argument("--n_iter", type=int, default=20)
    p.add_argument("--n_batch", type=int, default=64)
    p.add_argument("--buffer_size", type=int, default=5000)
    p.add_argument("--sample_temperature", type=float, default=1.0)
    p.add_argument("--explore_frac", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=5e-4)
    # Architecture
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    # Evaluation
    p.add_argument("--eval_freq", type=int, default=10, help="In epochs.")
    p.add_argument("--eval_size", type=int, default=100)
    p.add_argument("--eval_temperature", type=float, default=0.001)
    # compare_noisy
    p.add_argument("--compare_noisy", action="store_true")
    p.add_argument("--noise_prob_eval", type=float, default=0.01)
    p.add_argument("--shots", type=int, default=1024)
    # IO
    p.add_argument("--save_dir", default="runs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_online(args)


if __name__ == "__main__":
    main()
