"""
Molecular Hamiltonian, particle-number operator, and geometries.

The physics boundary of the pipeline: PySCF is hidden inside CUDA-Q's
``create_molecular_hamiltonian``, which returns the second-quantised,
Jordan-Wigner-mapped Hamiltonian as a sparse ``SpinOperator``. Everything
downstream works with that operator and never materialises a 2^n x 2^n matrix.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import cudaq


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
