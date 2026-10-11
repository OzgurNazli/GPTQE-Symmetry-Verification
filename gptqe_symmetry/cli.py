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

from __future__ import annotations

import argparse

import numpy as np
import torch

from .training import train_online


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
