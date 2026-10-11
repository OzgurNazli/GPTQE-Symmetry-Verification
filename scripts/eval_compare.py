"""
Post-hoc vs. training-integrated symmetry verification (strategy A vs. B).

Two runs are trained beforehand with ``python -m gptqe_symmetry`` on the same
molecule and noise level:

    baseline  : raw-energy training        (no --mitigate, no --verify_symmetry)
    mitigated : E_mit training             (--mitigate)

Their ``true_Es_t.csv`` are not directly comparable -- one logs raw energies,
the other logs mitigated ones -- so this script reloads each run's
``gptqe_best.pt``, regenerates the circuits it actually selected, and re-scores
them with BOTH metrics on a common footing, at each requested noise level:

    E_raw = <H>
    E_mit = <H Pi> / <Pi>,   Pi = (I + s Z^n)/2,  s = +1 even electrons

It then reports, per noise level:

    baseline raw   -> noisy energy, no mitigation          (reference)
    baseline + mit -> POST-HOC mitigation          (strategy A)
    mitigated mit  -> TRAINING-INTEGRATED mitigation (strategy B)
    mitigated raw  -> the mitigated run's circuits, unmitigated

A vs. B answers the thesis question: does folding mitigation into the training
objective help beyond simply post-processing the baseline model's circuits? If
A ~= B, symmetry verification is effectively a pure post-processing step
(consistent with the energy-leakage correlation result).

Usage (same environment as training, from the repo root)::

    python scripts/eval_compare.py \\
        --baseline runs_baseline --mitigated runs_mitigated \\
        --molecule H2 --bond_length 0.7414 \\
        --cudaq_target density-matrix-cpu --seq_len 4 \\
        --noise_levels 0.002 0.005 0.01

Pass the SAME model hyperparameters (--n_layer/--n_head/--n_embd/--seq_len)
used during training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import cudaq

from gptqe_symmetry.chemistry import build_molecule, molecule_geometry
from gptqe_symmetry.pool import build_operator_pool
from gptqe_symmetry.circuits import build_kernel_for_sequence
from gptqe_symmetry.mitigation import parity_operator, _mitigated_from_expectations
from gptqe_symmetry.noise import build_noise_model
from gptqe_symmetry.model import GPTQE, GPTConfig


def _score_sequence(nq, hf, ham, seq, parity, H_parity, s, floor=1e-3):
    """Return (E_raw, E_mit) for the full sequence, sharing one circuit build."""
    kk = build_kernel_for_sequence(nq, hf, seq)
    E = float(cudaq.observe(kk, ham).expectation())
    HZ = float(cudaq.observe(kk, H_parity).expectation())
    Z = float(cudaq.observe(kk, parity).expectation())
    return E, _mitigated_from_expectations(E, HZ, Z, s, floor)


def _load_and_generate(run_dir, vocab_size, args, dev):
    """Load gptqe_best.pt and regenerate the circuits the model selects."""
    ckpt = Path(run_dir) / f"seq_len={args.seq_len}" / "gptqe_best.pt"
    if not ckpt.exists():
        alt = Path(run_dir) / "gptqe_best.pt"
        ckpt = alt if alt.exists() else ckpt
    gpt = GPTQE(GPTConfig(
        vocab_size=vocab_size, block_size=args.seq_len,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=0.0, bias=False)).to(dev)
    gpt.load_state_dict(torch.load(ckpt, weights_only=True))
    gpt.eval()
    gb, _ = gpt.generate(args.eval_size, args.seq_len,
                         temperature=args.eval_temperature, device=dev)
    return (gb[:, 1:] - 1).cpu().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True,
                    help="Run dir of the raw-energy-trained model.")
    ap.add_argument("--mitigated", required=True,
                    help="Run dir of the --mitigate-trained model.")
    ap.add_argument("--molecule", default="H2", choices=["H2", "H4", "H6", "LiH"])
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--bond_length", type=float, default=0.7414)
    ap.add_argument("--cudaq_target", default="density-matrix-cpu")
    ap.add_argument("--noise_levels", type=float, nargs="+",
                    default=[0.002, 0.005, 0.01])
    ap.add_argument("--seq_len", type=int, default=4)
    ap.add_argument("--eval_size", type=int, default=100)
    ap.add_argument("--eval_temperature", type=float, default=0.001)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudaq.set_target(args.cudaq_target)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    symbols, coords = molecule_geometry(args.molecule, args.bond_length)
    ham, n_elec, n_qubits, hf = build_molecule(symbols, coords, basis=args.basis)
    pool = build_operator_pool(n_elec, n_qubits)
    vocab_size = len(pool) + 1
    parity = parity_operator(n_qubits)
    H_parity = ham * parity
    s = 1.0 if (n_elec % 2 == 0) else -1.0

    print(f"\n{args.molecule}  n_elec={n_elec}  n_qubits={n_qubits}")
    print(f"{'p':>7s} | {'base raw':>10s} {'base+mit(A)':>11s} | "
          f"{'mit(B)':>10s} {'mit raw':>10s} | {'A gain':>8s} {'B-A':>8s}")
    print("-" * 80)

    runs = {"base": args.baseline, "mit": args.mitigated}
    for p in args.noise_levels:
        cudaq.unset_noise()
        nm = build_noise_model(p)
        if nm is not None:
            cudaq.set_noise(nm)

        results = {}
        for tag, run_dir in runs.items():
            try:
                idx = _load_and_generate(run_dir, vocab_size, args, dev)
            except Exception as e:  # missing checkpoint etc.
                print(f"  [skip] {tag} ({run_dir}): {e}")
                results[tag] = None
                continue
            raws, mits = [], []
            for row in idx:
                seq = [pool[int(i)] for i in row]
                Er, Em = _score_sequence(n_qubits, hf, ham, seq, parity, H_parity, s)
                raws.append(Er)
                mits.append(Em)
            results[tag] = (np.array(raws), np.array(mits))

        cudaq.unset_noise()
        if results.get("base") is None or results.get("mit") is None:
            continue

        base_raw = results["base"][0].min()
        base_mit = results["base"][1].min()   # strategy A (post-hoc)
        mit_mit = results["mit"][1].min()      # strategy B (integrated)
        mit_raw = results["mit"][0].min()
        A_gain = base_raw - base_mit
        B_minus_A = base_mit - mit_mit
        print(f"{p:7.3f} | {base_raw:10.5f} {base_mit:11.5f} | "
              f"{mit_mit:10.5f} {mit_raw:10.5f} | {A_gain:+8.4f} {B_minus_A:+8.4f}")

    print("\nLegend:")
    print("  base raw    : baseline circuits, raw energy      = noisy baseline")
    print("  base+mit(A) : baseline circuits, mitigated       = POST-HOC (A)")
    print("  mit(B)      : mitigated-run circuits, mitigated   = INTEGRATED (B)")
    print("  A gain      : what post-hoc mitigation recovers  (base raw - base mit)")
    print("  B-A         : does integrating beat post-hoc?    (>0 = yes)")


if __name__ == "__main__":
    main()
