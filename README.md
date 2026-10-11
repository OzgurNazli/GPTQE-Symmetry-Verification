# gptqe-symmetry

**Particle-number symmetry verification for the Generative Quantum Eigensolver (GPT-QE), on CUDA-Q.**

A transformer assembles molecular circuits from a symmetry-preserving operator
pool and learns to generate low-energy ansätze. Under depolarising noise those
circuits leak out of the correct particle-number sector; this library mitigates
that leakage by post-selecting the energy onto the right symmetry sector
(*symmetry verification*), and lets you fold the mitigation into the training
objective itself.

It packages the code behind an METU MS thesis (Özgür Nazlı, Department of
Physics) as a reusable library: the generation stack is the original
contribution, and the mitigation entry point mirrors a
[Mitiq](https://github.com/unitaryfoundation/mitiq)-style
`execute_with_<technique>` so it can sit alongside standard mitigation
techniques.

---

## The idea in one screen

For an `n_elec`-electron molecule the Hamiltonian conserves particle number, so
the global parity operator commutes with it:

```
Π      = (I + s·Zⁿ) / 2            s = +1 (even electrons), −1 (odd)
⟨Π⟩    = (1 + s·⟨Zⁿ⟩) / 2          surviving weight in the correct sector
E_mit  = ⟨H·Π⟩ / ⟨Π⟩              variationally bounded:  E_mit ≥ E₀
```

Because `[H, Π] = 0`, `E_mit` is the energy of a valid projected-and-renormalised
state, so it can only sit **above** the true ground state — noise that kicks the
state out of the sector is removed rather than rewarded. When `⟨Π⟩` falls below a
floor (deep leakage, nothing left to post-select onto) the library falls back to
the raw energy so the estimate never blows up.

The leakage signal itself is `ΔN = ⟨N̂⟩ − n_elec`. Three strategies are
compared out of the box:

| strategy        | what it does                                   | flag |
|-----------------|------------------------------------------------|------|
| baseline        | train on raw noisy energy ⟨H⟩                  | *(none)* |
| soft penalty    | add `λ·ΔN²` to the training loss               | `--verify_symmetry --penalty_weight λ` |
| **mitigation**  | train on symmetry-verified `E_mit`             | `--mitigate` |

---

## Install

CUDA-Q is **Linux-only** (works on WSL2 and Google Colab). On Windows, run
inside WSL2 or Colab.

```bash
git clone https://github.com/ozgurnazli/gptqe-symmetry
cd gptqe-symmetry

# everything except CUDA-Q:
pip install -e .

# CUDA-Q itself (Linux / WSL2 / Colab):
pip install cudaq
# or:  pip install -e ".[cudaq]"
```

A single-file drop-in (`gptqe_symmetry_allinone.py`) is also provided for
clusters where installing a package is awkward — copy that one file next to your
job script and `import gptqe_symmetry_allinone as g`.

---

## Quickstart (library)

```python
import cudaq
from gptqe_symmetry import (
    build_molecule, molecule_geometry, build_operator_pool,
    energies_for_sequence, symmetry_verified_energy, build_noise_model,
)

cudaq.set_target("density-matrix-cpu")          # density matrices for noise
cudaq.set_noise(build_noise_model(0.02))        # 2-qubit depol p=0.02, 1-qubit p/10

symbols, coords = molecule_geometry("H2", 0.7414)
ham, n_elec, n_qubits, hf = build_molecule(symbols, coords)

pool = build_operator_pool(n_elec, n_qubits)    # the transformer's "vocabulary"
seq  = [pool[i], pool[j]]                        # any circuit = list of pool tokens

E     = energies_for_sequence(n_qubits, hf, ham, seq)[-1]          # noisy ⟨H⟩
E_mit = symmetry_verified_energy(n_qubits, hf, ham, seq, n_elec=n_elec)  # mitigated
```

`examples/h2_quickstart.py` runs this end to end and prints the recovered
energy:

```bash
python examples/h2_quickstart.py --noise_prob 0.02
```

## Quickstart (CLI / training)

```bash
# noiseless correctness check
python -m gptqe_symmetry --molecule H2 --bond_length 0.7414 --cudaq_target qpp-cpu

# train ON the symmetry-verified energy under noise
python -m gptqe_symmetry --molecule H2 --noise_prob 0.01 --mitigate \
    --cudaq_target density-matrix-cpu

# noisy baseline, logging leakage, re-scored noiseless vs noisy at the end
python -m gptqe_symmetry --molecule H2 --noise_prob 0.01 --compare_noisy \
    --cudaq_target density-matrix-cpu
```

Each run writes CSVs and checkpoints under `runs/seq_len=<L>/`.

---

## Public API

```python
# chemistry / pool / circuits
build_molecule, build_number_operator, molecule_geometry
excitations, PoolEntry, build_operator_pool, default_angles
apply_pool_entry, build_kernel_for_sequence

# oracles (score a circuit)
energies_for_sequence, n_expectation_for_sequence, syndrome_violation_rate

# mitigation  ← the headline technique (Mitiq-style one-call entry point)
symmetry_verified_energy          # E_mit of a sequence, one call
energies_for_sequence_mitigated   # mitigated prefix energies (drop-in for energies_for_sequence)
parity_operator

# model / training
GPT, GPTQE, GPTConfig
build_noise_model
train_online
```

`symmetry_verified_energy(...)` is deliberately shaped like Mitiq's
`execute_with_zne` / `execute_with_pec`: hand it a circuit-defining sequence and
it returns the mitigated expectation. That makes "symmetry verification" a
technique you can slot next to ZNE, PEC, CDR, etc.

---

## Reproducing the thesis figures

```bash
# 1. sweep noise × strategy (each cell is a fresh subprocess)
python scripts/sweep.py --molecule H2 --bond_length 0.7414 \
    --cudaq_target density-matrix-cpu --seq_len 4 \
    --noise_levels 0 0.002 0.005 0.01 \
    --strategies baseline penalty mitigate --out_root sweep_h2

# 2. per-run figures (loss, energy convergence, leakage)
python scripts/plot.py --run_dir sweep_h2/mitigate_p0.01/seq_len=4 \
    --e_hf -1.116684 --exact_energy -1.137270

# 3. post-hoc (A) vs training-integrated (B) mitigation
python scripts/eval_compare.py \
    --baseline sweep_h2/baseline_p0.01 --mitigated sweep_h2/mitigate_p0.01 \
    --molecule H2 --cudaq_target density-matrix-cpu --seq_len 4 \
    --noise_levels 0.002 0.005 0.01
```

---

## Noise model

Depolarising, with the two-qubit gates **10× noisier** than single-qubit gates
(`p2 = noise_prob`, `p1 = noise_prob/10`) — entangling gates are the dominant
error source on NISQ hardware. `p = 0` returns `None` (noiseless; use the
statevector backend). `noise.py` leaves `amplitude` and `phase` channel families
as documented `NotImplementedError` extension points for future work.

```
qpp-cpu            statevector, noiseless (fast)
density-matrix-cpu density matrices, required for noise on CPU
nvidia             GPU backend (statevector or density matrix)
```

---

## Repository layout

```
gptqe_symmetry/
  chemistry.py    molecular Hamiltonian, number operator, geometries
  pool.py         symmetry-preserving excitations → discrete operator pool
  circuits.py     pool token → CUDA-Q kernel (14-CNOT Givens double excitation)
  oracles.py      energy / particle-number / syndrome oracles
  mitigation.py   symmetry verification  (symmetry_verified_energy, E_mit)
  noise.py        depolarising noise model (+ extension points)
  model.py        nanoGPT backbone + GPTQE head (logit matching, Boltzmann sampling)
  training.py     online GQE loop with experience replay (3 strategies)
  cli.py          `python -m gptqe_symmetry`
scripts/          sweep.py, eval_compare.py, plot.py
examples/         h2_quickstart.py
tests/            test_double.py  (structural + decomposition correctness)
gptqe_symmetry_allinone.py        single-file consolidation (for clusters)
```

Run the tests with `pytest` (CUDA-Q-dependent tests skip automatically where
CUDA-Q is absent):

```bash
pytest -q
```

---

## Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{nazli2026gptqe,
  author = {Özgür Nazlı},
  title  = {SYMMETRY-VERIFIED GENERATIVE QUANTUM EIGENSOLVER:
PARTICLE-NUMBER ERROR SYNDROMES FOR NOISE-RESILIENT
MOLECULAR CIRCUIT GENERATION IN CUDA-Q},
  school = {Middle East Technical University, Department of Physics},
  year   = {2026}
}
```

Key references: Nakaji et al. (GPT-QE), Bonet-Monroig et al. (symmetry
verification as error mitigation), Anselmetti et al. 2021 (compact
double-excitation circuit), Wang et al. 2021 (noise-induced barren plateaus).

## License

MIT — see [LICENSE](LICENSE).
