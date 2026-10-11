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

from __future__ import annotations

from typing import Optional

import cudaq

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
