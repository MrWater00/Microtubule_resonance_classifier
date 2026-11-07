"""
Step 4: the MHz band, with literature peak positions and decoherence set from
measured peak sharpness (Q).

TOY MODEL. Peak positions come from the Bandyopadhyay-group papers (see
docs/literature_notes.md; values read off a figure, verify before quoting).
Nothing here shows anything about consciousness.

Your decisions:
  1A  each band gets its own window. Here: 1 MHz -> 1 Hz (scaled)
  2A  MHz band only, first
  3A  decoherence time from Q:   tau = Q / (pi * f)

Two signal sources, same peaks and same tau:
  toy    - sine waves with randomly drifting phase
  qutip  - 8 independent two-level systems with dephasing, solved by QuTiP

Run this file directly for a self-check and an example spectrum:
    python mhz_band.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

# =====================================================================
# YOUR KNOBS
# =====================================================================
SEED = 42
FS = 1000.0                   # scaled sampling rate
DURATION = 4.0                # scaled seconds (long enough to resolve the 20/22 pair)
NPERSEG = 2000                # FFT window -> 0.5 Hz resolution

# MHz-band peaks from the literature (1 MHz = 1 scaled Hz). From a figure axis.
PEAKS = [12.0, 20.0, 22.0, 30.0, 101.0, 113.0, 185.0, 204.0]
PEAK_JITTER = 0.3             # random shift per signal (small: 20 and 22 are close)
BAND_HALFWIDTH = 1.0          # feature window around each peak

# Peak sharpness Q = f / width. Sets how long each peak keeps its rhythm.
Q_ORDERED = 200.0             # middle of the reported MHz range, 100-300
Q_DECOHERED = 4.0             # my assumption: low end seen in the kHz band (3-5)

AMPLITUDE = 1.0               # height of each peak before noise
NOISE_STD = 10.0              # headline noise level (same as your earlier choice)
# =====================================================================

TIME = np.arange(0, DURATION, 1 / FS)
BANDS = [(p - BAND_HALFWIDTH, p + BAND_HALFWIDTH) for p in PEAKS]


def tau_for(Q, f):
    """Coherence time that gives a peak of sharpness Q at frequency f."""
    return Q / (np.pi * f)


def toy_signal(Q, rng):
    sig = np.zeros_like(TIME)
    for f0 in PEAKS:
        f = f0 + rng.uniform(-PEAK_JITTER, PEAK_JITTER)
        step_std = np.sqrt(2 / (tau_for(Q, f) * FS))
        phase = (2 * np.pi * f * TIME + np.cumsum(rng.normal(0, step_std, TIME.size))
                 + rng.uniform(0, 2 * np.pi))
        sig += AMPLITUDE * np.sin(phase)
    return sig


def qutip_signal(Q, rng, with_params=False):
    """Sum of 8 independent two-level systems, each dephasing at rate 1/tau."""
    import qutip as qt
    k0, k1 = qt.basis(2, 0), qt.basis(2, 1)
    P0, P1 = k0 * k0.dag(), k1 * k1.dag()
    sig, params = np.zeros_like(TIME), []
    for f0 in PEAKS:
        f = f0 + rng.uniform(-PEAK_JITTER, PEAK_JITTER)
        tau = tau_for(Q, f)
        phi = rng.uniform(0, 2 * np.pi)
        H = 2 * np.pi * f * P1
        c_ops = [np.sqrt(1 / tau) * P0, np.sqrt(1 / tau) * P1]
        psi0 = (k0 + np.exp(1j * phi) * k1).unit()
        res = qt.mesolve(H, psi0, TIME, c_ops, e_ops=[qt.sigmax()],
                         options={"nsteps": 100000, "atol": 1e-10, "rtol": 1e-8})
        sig += AMPLITUDE * np.array(res.expect[0])
        params.append((f, tau, phi))
    return (sig, params) if with_params else sig


def analytic_signal(params):
    """Exact pen-and-paper answer for the same systems (check for QuTiP)."""
    return sum(AMPLITUDE * np.cos(2 * np.pi * f * TIME - phi) * np.exp(-TIME / tau)
               for f, tau, phi in params)


def features(sig):
    """Per peak window: log power and sharpness (tallest bin / average bin)."""
    freqs, pxx = welch(sig, fs=FS, nperseg=NPERSEG)
    feats = []
    for lo, hi in BANDS:
        band = pxx[(freqs >= lo) & (freqs <= hi)]
        feats.append(np.log10(band.sum()))
        feats.append(band.max() / band.mean())
    return feats


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)

    print("Peak mapping and decoherence times (1 MHz = 1 scaled Hz)")
    print(f"{'MHz':>6} {'tau ordered (s)':>16} {'tau decohered (s)':>18} "
          f"{'width ordered (Hz)':>19} {'width decohered (Hz)':>21}")
    for f in PEAKS:
        print(f"{f:>6.0f} {tau_for(Q_ORDERED, f):>16.3f} {tau_for(Q_DECOHERED, f):>18.4f} "
              f"{f / Q_ORDERED:>19.2f} {f / Q_DECOHERED:>21.1f}")

    print("\nQuTiP self-check against exact math:")
    try:
        for name, Q in [("ordered", Q_ORDERED), ("decohered", Q_DECOHERED)]:
            sig, params = qutip_signal(Q, np.random.default_rng(SEED), with_params=True)
            err = np.max(np.abs(sig - analytic_signal(params)))
            print(f"  {name:10s} Q={Q:<6} max |QuTiP - exact| = {err:.2e}  "
                  f"({'OK' if err < 1e-4 else 'MISMATCH - do not trust'})")
    except ImportError:
        print("  qutip not installed, skipped")

    fig, ax = plt.subplots(figsize=(9, 4.4))
    for name, Q, c in [("ordered (Q=%g)" % Q_ORDERED, Q_ORDERED, "#1f6f78"),
                       ("decohered (Q=%g)" % Q_DECOHERED, Q_DECOHERED, "#d98324")]:
        rng = np.random.default_rng(SEED)
        noisy = toy_signal(Q, rng) + rng.normal(0, NOISE_STD, TIME.size)
        fr, p = welch(noisy, fs=FS, nperseg=NPERSEG)
        ax.semilogy(fr, p, color=c, lw=1.1, label=name)
    for p in PEAKS:
        ax.axvline(p, color="#bbbbbb", lw=0.7, zorder=0)
    ax.set_xlim(0, 250)
    ax.set_xlabel("Frequency (scaled; 1 Hz = 1 MHz of the real band)")
    ax.set_ylabel("Power")
    ax.set_title(f"MHz band, noise = {NOISE_STD:g} (toy source). Grey lines = literature peaks",
                 loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "mhz_spectra.png", dpi=150)
    print(f"\nFigure saved to {out / 'mhz_spectra.png'}")
