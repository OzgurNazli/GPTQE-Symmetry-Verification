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

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import cudaq

from .model import GPTQE, GPTConfig
from .chemistry import build_molecule, build_number_operator, molecule_geometry
from .pool import excitations, build_operator_pool
from .circuits import build_kernel_for_sequence
from .noise import build_noise_model
from .oracles import (
    energies_for_sequence,
    n_expectation_for_sequence,
    syndrome_violation_rate,
)
from .mitigation import energies_for_sequence_mitigated


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
