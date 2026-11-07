"""
Step 5: experiments A and B on the MHz band.

A) How similar can "decohered" be to "ordered" before the model fails?
   Sweep Q_DECOHERED from 4 up to 200. At 200 the two classes are identical
   by construction, so accuracy MUST fall to about 50% there (built-in check).

B) Equal-energy fairness check. Every clean signal is rescaled to the same
   total power, so the model cannot win just by noticing "signal present vs
   almost nothing" (which is what the fading QuTiP signal allowed).

Signal sources:
  toy    - random-phase sine waves
  exact  - the exact fading-wave formula. mhz_band.py verifies it equals QuTiP
           to about 4e-5, and it uses the same random draws, so these are the
           same signals as QuTiP's, about 1000x faster. Used in A.
  qutip  - the real QuTiP solver (slow). Used in B.

Run:   python mhz_ab.py A      (about 2-3 min)
       python mhz_ab.py B      (about 6-9 min, QuTiP)
"""

import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import add_noise, evaluate
from mhz_band import (AMPLITUDE, PEAK_JITTER, PEAKS, Q_DECOHERED, Q_ORDERED,
                      SEED, analytic_signal, features, qutip_signal, tau_for,
                      toy_signal)

# =====================================================================
# YOUR KNOBS
# =====================================================================
# A
Q_DEC_LIST = [4, 10, 25, 50, 100, 150, 200]
A_NOISE = [5, 10]
A_N_PER_CLASS, A_SEEDS = 100, 5
# B
B_NOISE = [2, 5, 8, 10, 12, 15, 20, 30]
B_TOY = (100, 5)       # (signals per class, seeds)  same as the earlier sweep
B_QUTIP = (40, 3)      # same as the earlier sweep, so the comparison is paired
# =====================================================================

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
TARGET_POWER = len(PEAKS) * AMPLITUDE ** 2 / 2     # 4.0, what the toy source has


def exact_signal(Q, rng):
    params = []
    for f0 in PEAKS:
        f = f0 + rng.uniform(-PEAK_JITTER, PEAK_JITTER)
        params.append((f, tau_for(Q, f), rng.uniform(0, 2 * np.pi)))
    return analytic_signal(params)


def equalize(sig):
    """Rescale so every signal has the same average power."""
    return sig * np.sqrt(TARGET_POWER / np.mean(sig ** 2))


def run_A():
    sources = {"toy": toy_signal, "exact": exact_signal}
    acc = {}
    t0 = time.time()
    for s in range(A_SEEDS):
        for name, fn in sources.items():
            rng = np.random.default_rng(SEED + s)
            ordered = [equalize(fn(Q_ORDERED, rng)) for _ in range(A_N_PER_CLASS)]
            dec = {q: [equalize(fn(q, rng)) for _ in range(A_N_PER_CLASS)] for q in Q_DEC_LIST}
            y = np.array([1] * A_N_PER_CLASS + [0] * A_N_PER_CLASS)
            for nl in A_NOISE:
                Xo = np.array([features(add_noise(x, nl, rng)) for x in ordered])
                for q in Q_DEC_LIST:
                    Xd = np.array([features(add_noise(x, nl, rng)) for x in dec[q]])
                    for m, a in evaluate(np.vstack([Xo, Xd]), y, SEED + s).items():
                        acc.setdefault((name, nl, q, m), []).append(a)
        print(f"  seed {s + 1}/{A_SEEDS} done ({time.time() - t0:.0f} s)", flush=True)

    with open(OUT / "mhz_qsweep_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "noise_std", "q_decohered", "model", "mean_accuracy", "std_accuracy"])
        for (name, nl, q, m), v in sorted(acc.items()):
            w.writerow([name, nl, q, m, round(np.mean(v), 4), round(np.std(v), 4)])

    for nl in A_NOISE:
        print(f"\nA) noise = {nl}: random-forest test accuracy (mean +/- std over {A_SEEDS} seeds)")
        print(f"{'Q_dec':>6} | {'ordered is x sharper':>20} | {'toy':>15} | {'exact (=QuTiP)':>15}")
        for q in Q_DEC_LIST:
            cells = [f"{np.mean(acc[(n, nl, q, 'random forest')]):.3f} +/- "
                     f"{np.std(acc[(n, nl, q, 'random forest')]):.3f}" for n in sources]
            print(f"{q:>6} | {Q_ORDERED / q:>20.1f} | {cells[0]:>15} | {cells[1]:>15}")
    shuf = np.mean([np.mean(v) for k, v in acc.items() if k[3] == "shuffled labels"])
    print(f"\nShuffled-label check, averaged over everything: {shuf:.3f} (expect ~0.5)")

    colors = {"toy": "#1f6f78", "exact": "#d98324"}
    styles = {A_NOISE[0]: "-", A_NOISE[1]: "--"}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name in sources:
        for nl in A_NOISE:
            mean = np.array([np.mean(acc[(name, nl, q, "random forest")]) for q in Q_DEC_LIST])
            std = np.array([np.std(acc[(name, nl, q, "random forest")]) for q in Q_DEC_LIST])
            ax.plot(Q_DEC_LIST, mean, styles[nl], color=colors[name], lw=1.8,
                    label=f"{name}, noise {nl}")
            ax.fill_between(Q_DEC_LIST, mean - std, mean + std, color=colors[name], alpha=0.10, lw=0)
    ax.set_xscale("log")
    ax.set_xticks(Q_DEC_LIST, [str(q) for q in Q_DEC_LIST])
    ax.axhline(0.5, color="#888888", lw=1, ls=":")
    ax.text(Q_DEC_LIST[0], 0.505, "chance (coin flip)", color="#666666", fontsize=9, va="bottom")
    ax.axvline(Q_ORDERED, color="#bbbbbb", lw=1)
    ax.text(Q_ORDERED, 0.98, "identical to\nordered", ha="right", va="top", color="#666666", fontsize=8)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("Q of the 'decohered' class (higher = less decohered; ordered is fixed at 200)")
    ax.set_ylabel("Test accuracy (random forest)")
    ax.set_title("A) How similar can decohered be to ordered? (equal-energy signals)", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(OUT / "mhz_accuracy_vs_Q.png", dpi=150)
    print(f"\nSaved: {OUT / 'mhz_accuracy_vs_Q.png'} and mhz_qsweep_results.csv")


def run_B():
    def sweep(make_clean, n, seeds):
        acc = {nl: {} for nl in B_NOISE}
        for s in range(seeds):
            rng = np.random.default_rng(SEED + s)       # same draws as mhz_sweep.py
            clean = [equalize(make_clean(Q, rng)) for Q in [Q_ORDERED] * n + [Q_DECOHERED] * n]
            y = np.array([1] * n + [0] * n)
            for nl in B_NOISE:
                X = np.array([features(add_noise(x, nl, rng)) for x in clean])
                for m, a in evaluate(X, y, SEED + s).items():
                    acc[nl].setdefault(m, []).append(a)
            print(f"    seed {s + 1}/{seeds} done", flush=True)
        return acc

    res = {}
    for name, fn, (n, seeds) in [("toy", toy_signal, B_TOY), ("qutip", qutip_signal, B_QUTIP)]:
        t0 = time.time()
        print(f"Running source: {name} (equal energy), {n} per class, {seeds} seeds")
        res[name] = sweep(fn, n, seeds)
        print(f"  done in {time.time() - t0:.0f} s")

    raw = {}   # earlier, un-equalised results
    raw_file = OUT / "mhz_sweep_results.csv"
    if raw_file.exists():
        for r in csv.DictReader(open(raw_file)):
            if r["model"] == "random forest":
                raw[(r["source"], int(float(r["noise_std"])))] = (float(r["mean_accuracy"]), float(r["std_accuracy"]))

    with open(OUT / "mhz_energy_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "noise_std", "model", "mean_accuracy", "std_accuracy"])
        for name, acc in res.items():
            for nl in B_NOISE:
                for m, v in acc[nl].items():
                    w.writerow([name, nl, m, round(np.mean(v), 4), round(np.std(v), 4)])

    print("\nB) random-forest test accuracy: raw (earlier run) vs equal energy")
    print(f"{'noise':>6} | {'toy raw':>8} | {'toy equal':>9} | {'qutip raw':>9} | {'qutip equal':>11} | shuffled (equal, qutip)")
    for nl in B_NOISE:
        g = lambda k: f"{raw[(k, nl)][0]:.3f}" if (k, nl) in raw else "n/a"
        print(f"{nl:>6} | {g('toy'):>8} | {np.mean(res['toy'][nl]['random forest']):>9.3f} | "
              f"{g('qutip'):>9} | {np.mean(res['qutip'][nl]['random forest']):>11.3f} | "
              f"{np.mean(res['qutip'][nl]['shuffled labels']):.3f}")

    colors = {"toy": "#1f6f78", "qutip": "#d98324"}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name in res:
        mean = np.array([np.mean(res[name][nl]["random forest"]) for nl in B_NOISE])
        std = np.array([np.std(res[name][nl]["random forest"]) for nl in B_NOISE])
        ax.plot(B_NOISE, mean, "-", color=colors[name], lw=1.9, label=f"{name}, equal energy")
        ax.fill_between(B_NOISE, mean - std, mean + std, color=colors[name], alpha=0.12, lw=0)
        if all((name, nl) in raw for nl in B_NOISE):
            ax.plot(B_NOISE, [raw[(name, nl)][0] for nl in B_NOISE], "--", color=colors[name],
                    lw=1.4, alpha=0.8, label=f"{name}, raw (earlier run)")
    ax.axhline(0.5, color="#888888", lw=1, ls=":")
    ax.text(B_NOISE[0], 0.505, "chance (coin flip)", color="#666666", fontsize=9, va="bottom")
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("Noise level (std; each peak has height 1)")
    ax.set_ylabel("Test accuracy (random forest)")
    ax.set_title("B) Does equalising signal energy change the toy vs QuTiP gap?", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "mhz_equal_energy.png", dpi=150)
    print(f"\nSaved: {OUT / 'mhz_equal_energy.png'} and mhz_energy_results.csv")


if __name__ == "__main__":
    which = sys.argv[1].upper() if len(sys.argv) > 1 else "AB"
    if "A" in which:
        run_A()
    if "B" in which:
        run_B()
