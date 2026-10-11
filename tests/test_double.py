"""
Correctness tests.

The pool / excitation tests are pure Python and run anywhere. The circuit test
needs CUDA-Q and is skipped automatically when it is not installed (e.g. on a
Windows checkout), so ``pytest`` stays green on any machine.

Run::

    pytest -q
    pytest -q -k double      # just the circuit decomposition check
"""

from __future__ import annotations

import numpy as np
import pytest

# gptqe_symmetry is a CUDA-Q library: importing the package pulls in cudaq.
# Skip the whole module cleanly when it is not installed (e.g. a Windows
# checkout) so `pytest -q` stays green everywhere; it runs in full wherever
# CUDA-Q is available (HPC / Colab).
pytest.importorskip("cudaq", reason="CUDA-Q not installed")

from gptqe_symmetry.pool import excitations, build_operator_pool, PoolEntry, default_angles

FCI_H2_EQ = -1.137270  # Ha, sto-3g, R = 0.7414 A
CHEM_ACC = 1.6e-3      # "chemical accuracy", 1 kcal/mol in Hartree


# --------------------------------------------------------------------------
# Pure-Python structural tests (no simulator)
# --------------------------------------------------------------------------
def test_h2_excitations():
    singles, doubles = excitations(num_electrons=2, num_qubits=4)
    # occ = {0,1}, vir = {2,3}; spin-preserving singles: 0->2 (alpha), 1->3 (beta)
    assert set(singles) == {(0, 2), (1, 3)}
    # the one spin-allowed double: (0,1 -> 2,3)
    assert doubles == [(0, 1, 2, 3)]


def test_h2_pool_size():
    pool = build_operator_pool(num_electrons=2, num_qubits=4)
    # 1 double*8 + 2 singles*8 + 1 identity*8 = 32
    assert len(pool) == 32
    kinds = {e.kind for e in pool}
    assert kinds == {"double", "single", "identity"}


def test_default_angles_symmetric():
    a = default_angles()
    assert len(a) == 8
    assert np.allclose(np.sort(a), a)          # sorted
    assert np.allclose(a, -a[::-1])            # symmetric about 0
    assert np.isclose(a.max(), 16 / 160.0)     # 2^4 / 160 = 0.1


# --------------------------------------------------------------------------
# Circuit decomposition test (needs CUDA-Q)
# --------------------------------------------------------------------------
def test_double_reaches_fci():
    """A single double-excitation gate must be able to drive H2 HF -> FCI.

    This validates the 14-CNOT Givens decomposition in circuits.apply_pool_entry:
    scanning the one allowed double over its angle, the minimum energy must reach
    the full-CI ground state to within chemical accuracy.
    """
    import cudaq
    from gptqe_symmetry.chemistry import build_molecule, molecule_geometry
    from gptqe_symmetry.circuits import build_kernel_for_sequence

    cudaq.set_target("qpp-cpu")
    symbols, coords = molecule_geometry("H2", 0.7414)
    ham, n_elec, n_qubits, hf = build_molecule(symbols, coords)
    assert (n_elec, n_qubits) == (2, 4)

    thetas = np.linspace(-0.5, 0.5, 101)
    energies = []
    for th in thetas:
        seq = [PoolEntry("double", (0, 1, 2, 3), float(th))]
        k = build_kernel_for_sequence(n_qubits, hf, seq)
        energies.append(float(cudaq.observe(k, ham).expectation()))

    e_min = min(energies)
    assert e_min == pytest.approx(FCI_H2_EQ, abs=CHEM_ACC), \
        f"min scanned energy {e_min:.6f} did not reach FCI {FCI_H2_EQ:.6f}"


def test_identity_is_hf_energy():
    """An empty / identity-only circuit must sit exactly at the HF energy."""
    import cudaq
    from gptqe_symmetry.chemistry import build_molecule, molecule_geometry
    from gptqe_symmetry.circuits import build_kernel_for_sequence

    cudaq.set_target("qpp-cpu")
    symbols, coords = molecule_geometry("H2", 0.7414)
    ham, n_elec, n_qubits, hf = build_molecule(symbols, coords)

    e_empty = float(cudaq.observe(build_kernel_for_sequence(n_qubits, hf, []), ham).expectation())
    e_ident = float(cudaq.observe(
        build_kernel_for_sequence(n_qubits, hf, [PoolEntry("identity", (), 0.05)]), ham
    ).expectation())
    assert e_empty == pytest.approx(e_ident, abs=1e-9)
