"""
H2 quickstart: the whole pipeline in ~30 lines, no training.

Shows the three moving parts the library exposes, end to end:

    1. build the molecule (Hamiltonian + HF reference + number operator)
    2. build the operator pool and a short circuit from a couple of its tokens
    3. read out the energy, the particle-number leakage dN, and -- under noise
       -- the symmetry-verified energy E_mit

Run noiseless (exact statevector)::

    python examples/h2_quickstart.py

Run with depolarising noise and watch mitigation recover the energy::

    python examples/h2_quickstart.py --noise_prob 0.02

Requires CUDA-Q (``pip install cudaq``; Linux / WSL2 / Colab).
"""

from __future__ import annotations

import argparse

import numpy as np
import cudaq

from gptqe_symmetry import (
    build_molecule, build_number_operator, molecule_geometry,
    build_operator_pool, build_kernel_for_sequence,
    energies_for_sequence, n_expectation_for_sequence,
    symmetry_verified_energy, build_noise_model,
)

FCI_H2_EQ = -1.137270  # Ha, sto-3g, R = 0.7414 A (reference value)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bond_length", type=float, default=0.7414)
    ap.add_argument("--noise_prob", type=float, default=0.0,
                    help="Two-qubit depolarising rate (0 = noiseless).")
    args = ap.parse_args(argv)

    # Backend: noiseless uses the statevector sim, noisy needs density matrices.
    target = "density-matrix-cpu" if args.noise_prob > 0 else "qpp-cpu"
    cudaq.set_target(target)
    noise = build_noise_model(args.noise_prob)
    if noise is not None:
        cudaq.set_noise(noise)
    print(f"[backend] {target}   noise_prob={args.noise_prob:g}")

    # 1) Molecule -----------------------------------------------------------
    symbols, coords = molecule_geometry("H2", args.bond_length)
    ham, n_elec, n_qubits, hf = build_molecule(symbols, coords)
    N_op = build_number_operator(n_qubits)

    e_hf = float(cudaq.observe(build_kernel_for_sequence(n_qubits, hf, []), ham).expectation())
    print(f"[H2] n_elec={n_elec}  n_qubits={n_qubits}  E_HF={e_hf:.6f} Ha")

    # 2) Pool + a tiny hand-picked circuit ---------------------------------
    pool = build_operator_pool(n_elec, n_qubits)
    print(f"[pool] {len(pool)} tokens (vocab {len(pool) + 1})")

    # Two copies of the strongest double excitation ~ reaches the correlated
    # ground state for H2. Pick the double with the largest positive angle.
    doubles = [i for i, e in enumerate(pool) if e.kind == "double"]
    strong = max(doubles, key=lambda i: pool[i].theta)
    seq = [pool[strong], pool[strong]]
    print(f"[circuit] 2 x {pool[strong].describe()}")

    # 3) Read out -----------------------------------------------------------
    E = energies_for_sequence(n_qubits, hf, ham, seq)[-1]
    dN = n_expectation_for_sequence(n_qubits, hf, N_op, seq) - n_elec
    print(f"\n  E            = {E:.6f} Ha   (FCI = {FCI_H2_EQ:.6f})")
    print(f"  dN = <N>-n   = {dN:+.3e}   (leakage out of the N={n_elec} sector)")

    if args.noise_prob > 0:
        E_mit = symmetry_verified_energy(n_qubits, hf, ham, seq, n_elec=n_elec)
        print(f"  E_mit        = {E_mit:.6f} Ha   (symmetry-verified)")
        print(f"  recovered    = {E - E_mit:+.6f} Ha of the noise bias")

    print("\nDone.")


if __name__ == "__main__":
    main()
