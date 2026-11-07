"""
Step 3: the QuTiP source of signals (3-level quantum system with dephasing).

TOY MODEL. Three energy levels stand in for the electronic states of a
tryptophan ring. This is NOT a faithful model of tryptophan or of a
microtubule; the levels are scaled placeholders.

What it simulates:
  - H            : the three levels (energies 0, E1, E2)
  - collapse ops : "dephasing" - the environment randomly bumps each level,
                   which erases the phase relationship between levels.
                   Rate gamma = 1/tau (tau = coherence time).
  - start state  : equal mix of all three levels, random relative phases
  - what we read : x = sum of all level-to-level coherences. It wiggles at the
                   three transition frequencies and the wiggles fade at rate gamma.

Run this file directly for a self-check against the exact math:
    python qutip_sim.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from common import (AMPLITUDE, E1, E2, PEAK_JITTER, SEED, TAU_DECOHERED,
                    TAU_ORDERED, TIME)

# Equal mix of 3 levels gives each coherence size 1/3, so x has peaks of
# height 2/3. Scale so each peak has height AMPLITUDE (same as the toy source).
SCALE = 1.5 * AMPLITUDE
KETS = [qt.basis(3, k) for k in range(3)]


def simulate_signal_qutip(tau, rng):
    """Return (signal, (E1_used, E2_used, phases)) for one random 3-level system."""
    e1 = E1 + rng.uniform(-PEAK_JITTER, PEAK_JITTER)
    e2 = E2 + rng.uniform(-PEAK_JITTER, PEAK_JITTER)
    phases = rng.uniform(0, 2 * np.pi, 3)

    H = 2 * np.pi * (e1 * KETS[1] * KETS[1].dag() + e2 * KETS[2] * KETS[2].dag())
    gamma = 1 / tau
    c_ops = [np.sqrt(gamma) * k * k.dag() for k in KETS]

    psi0 = (np.exp(1j * phases[0]) * KETS[0]
            + np.exp(1j * phases[1]) * KETS[1]
            + np.exp(1j * phases[2]) * KETS[2]).unit()

    x_op = 0 * KETS[0] * KETS[0].dag()
    for j in range(3):
        for k in range(j + 1, 3):
            x_op = x_op + KETS[j] * KETS[k].dag() + KETS[k] * KETS[j].dag()

    res = qt.mesolve(H, psi0, TIME, c_ops, e_ops=[x_op],
                     options={"nsteps": 20000, "atol": 1e-10, "rtol": 1e-8})
    return SCALE * np.array(res.expect[0]), (e1, e2, phases)


def analytic_signal(tau, e1, e2, phases):
    """The same signal from exact pen-and-paper math (no QuTiP). Used as a check."""
    energies = np.array([0.0, e1, e2])
    c = np.exp(1j * phases) / np.sqrt(3)
    x = np.zeros_like(TIME)
    for j in range(3):
        for k in range(j + 1, 3):
            rho_jk = (c[j] * np.conj(c[k])
                      * np.exp(-1j * 2 * np.pi * (energies[j] - energies[k]) * TIME)
                      * np.exp(-TIME / tau))
            x += 2 * np.real(rho_jk)
    return SCALE * x


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for ax, (name, tau) in zip(axes, [("ordered", TAU_ORDERED), ("decohered", TAU_DECOHERED)]):
        rng = np.random.default_rng(SEED)
        sig, (e1, e2, ph) = simulate_signal_qutip(tau, rng)
        err = np.max(np.abs(sig - analytic_signal(tau, e1, e2, ph)))
        print(f"{name:10s} tau={tau:<5} max |QuTiP - exact math| = {err:.2e}  "
              f"({'OK' if err < 1e-4 else 'MISMATCH - do not trust the simulator'})")
        ax.plot(TIME, sig, lw=0.8, color="#1f6f78" if name == "ordered" else "#d98324")
        ax.set_title(f"{name} (tau = {tau} s), no noise added", loc="left")
        ax.set_ylabel("signal")
    axes[1].set_xlabel("Time (scaled s)")
    fig.tight_layout()
    fig.savefig(out / "qutip_traces.png", dpi=150)
    print(f"Figure saved to {out / 'qutip_traces.png'}")
