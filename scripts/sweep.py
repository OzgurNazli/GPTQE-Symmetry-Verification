"""
Noise sweep orchestrator.

Drives ``python -m gptqe_symmetry`` once per (noise level x strategy) cell as a
subprocess, each into its own run directory, then collates the per-run
``summary.csv`` into a single tidy table. This reproduces the thesis's
energy-vs-noise and leakage-vs-noise curves.

Each subprocess is a fresh interpreter, which keeps CUDA-Q's global target /
noise state from leaking between cells -- the reason the thesis ran the sweep
out-of-process in the first place.

Strategies (columns of the comparison):
    baseline   -- raw-energy training
    penalty    -- soft dN^2 penalty (--verify_symmetry --penalty_weight)
    mitigate   -- train on symmetry-verified E_mit (--mitigate)

Usage::

    python scripts/sweep.py --molecule H2 --bond_length 0.7414 \\
        --cudaq_target density-matrix-cpu --seq_len 4 \\
        --noise_levels 0 0.002 0.005 0.01 \\
        --strategies baseline penalty mitigate \\
        --out_root sweep_h2 --n_epochs 300

The heavy training flags are forwarded verbatim; anything after ``--`` is
passed straight through to the trainer, e.g.::

    python scripts/sweep.py ... -- --n_sample 400 --lr 3e-4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

STRATEGY_FLAGS = {
    "baseline": [],
    "penalty": ["--verify_symmetry", "--penalty_weight", "1.0"],
    "mitigate": ["--mitigate"],
}


def _run_cell(args, p, strategy, passthrough):
    run_dir = Path(args.out_root) / f"{strategy}_p{p:g}"
    cmd = [
        sys.executable, "-m", "gptqe_symmetry",
        "--molecule", args.molecule,
        "--basis", args.basis,
        "--bond_length", str(args.bond_length),
        "--cudaq_target", args.cudaq_target,
        "--noise_prob", str(p),
        "--noise_type", args.noise_type,
        "--seq_len", str(args.seq_len),
        "--n_epochs", str(args.n_epochs),
        "--seed", str(args.seed),
        "--save_dir", str(run_dir),
    ] + STRATEGY_FLAGS[strategy] + passthrough
    print(f"\n=== {strategy}  p={p:g} -> {run_dir} ===")
    print("  " + " ".join(cmd))
    if args.dry_run:
        return run_dir
    subprocess.run(cmd, check=True)
    return run_dir


def _collect(run_dir, molecule, p, strategy):
    """Pull the 'Best (lowest E)' row out of a cell's summary.csv.

    The trainer writes it to ``<run_dir>/seq_len=<seq_len>/summary.csv``; glob
    for it so this stays correct regardless of the seq_len subdir name.
    """
    hits = list(Path(run_dir).rglob("summary.csv"))
    if not hits:
        return None
    df = pd.read_csv(hits[0])
    best = df[df["Source"].str.startswith("Best")]
    row = (best.iloc[0] if len(best) else df.iloc[-1]).to_dict()
    row.update({"molecule": molecule, "noise_prob": p, "strategy": strategy})
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--molecule", default="H2", choices=["H2", "H4", "H6", "LiH"])
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--bond_length", type=float, default=0.7414)
    ap.add_argument("--cudaq_target", default="density-matrix-cpu")
    ap.add_argument("--noise_type", default="depolarizing")
    ap.add_argument("--noise_levels", type=float, nargs="+",
                    default=[0.0, 0.002, 0.005, 0.01])
    ap.add_argument("--strategies", nargs="+", default=["baseline", "mitigate"],
                    choices=list(STRATEGY_FLAGS))
    ap.add_argument("--seq_len", type=int, default=4)
    ap.add_argument("--n_epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_root", default="sweep")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the commands without running them.")
    args, passthrough = ap.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    rows = []
    for p in args.noise_levels:
        for strat in args.strategies:
            run_dir = _run_cell(args, p, strat, passthrough)
            if args.dry_run:
                continue
            row = _collect(run_dir, args.molecule, p, strat)
            if row is not None:
                rows.append(row)

    if not rows:
        print("\nNo results collected (dry run or no summary.csv found).")
        return

    out = pd.DataFrame(rows)
    front = ["molecule", "noise_prob", "strategy"]
    out = out[front + [c for c in out.columns if c not in front]]
    out = out.sort_values(["strategy", "noise_prob"]).reset_index(drop=True)
    csv = Path(args.out_root) / "sweep_summary.csv"
    out.to_csv(csv, index=False)
    print(f"\nCollated {len(out)} cells -> {csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
