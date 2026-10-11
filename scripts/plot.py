"""
Plot the CSV artefacts written by a training run.

Reads whatever CSVs exist in a run directory and writes PNG figures; missing
files are skipped. Works for noiseless baselines and full noisy + verification
runs alike.

    python scripts/plot.py --run_dir runs/seq_len=4 --e_hf -1.116684 \
        --exact_energy -1.137270 --eval_freq 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _read(run_dir, name):
    p = Path(run_dir) / name
    return pd.read_csv(p) if p.exists() else None


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(Path(out) / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png")


def _steps(n, ef):
    return ef * np.arange(1, n + 1)


def main():
    ap = argparse.ArgumentParser(description="Plot training run artefacts")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--eval_freq", type=int, default=200)
    ap.add_argument("--e_hf", type=float, default=None)
    ap.add_argument("--exact_energy", type=float, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    loss = _read(run_dir, "losses.csv")
    if loss is not None:
        y = loss.iloc[:, 0].values
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.arange(1, len(y) + 1), y, lw=1.2)
        ax.set_yscale("log"); ax.set_xlabel("training step")
        ax.set_ylabel("logit-matching loss"); ax.set_title("Training loss")
        _save(fig, out, "fig_loss")

    te = _read(run_dir, "true_Es_t.csv")
    if te is not None:
        E = te.values
        x = _steps(E.shape[1], args.eval_freq)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, E.mean(axis=0), "-o", ms=4, label="mean generated E")
        ax.plot(x, E.min(axis=0), "-s", ms=4, label="min generated E")
        if args.e_hf is not None:
            ax.axhline(args.e_hf, ls="--", color="#7f7f7f", lw=1, label="HF")
        if args.exact_energy is not None:
            ax.axhline(args.exact_energy, ls=":", color="#d62728", lw=1.2, label="exact")
        ax.set_xlabel("training step"); ax.set_ylabel("energy (Ha)")
        ax.set_title("Generated-circuit energy vs. training"); ax.legend(frameon=False)
        _save(fig, out, "fig_energy_convergence")

    lk = _read(run_dir, "n_violations_eval.csv")
    if lk is not None:
        y = lk["mean_abs_dN"].values if "mean_abs_dN" in lk else lk.iloc[:, 0].values
        x = _steps(len(y), args.eval_freq)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y, "-o", ms=4, color="#ff7f0e")
        ax.set_xlabel("training step"); ax.set_ylabel(r"$\langle |\Delta_N| \rangle$")
        ax.set_title("Particle-number leakage under noise")
        _save(fig, out, "fig_leakage")

    print("Done.")


if __name__ == "__main__":
    main()
