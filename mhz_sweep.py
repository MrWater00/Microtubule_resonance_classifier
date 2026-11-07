"""
Step 4: accuracy-vs-noise for the MHz band (literature peaks, tau from Q).

Classes:
  ordered   (label 1): Q = Q_ORDERED   (sharp peaks, like the measured MHz band)
  decohered (label 0): Q = Q_DECOHERED (broad peaks, like the measured kHz band)

Two signal sources, same peaks, same features, same classifiers:
  toy    - random-phase sine waves
  qutip  - 8 independent two-level systems with dephasing (slow: about 1 s per signal)

Run:  python mhz_sweep.py          (toy takes ~1 min, qutip adds ~5-8 min)
"""

import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import add_noise, evaluate
from mhz_band import (NOISE_STD, Q_DECOHERED, Q_ORDERED, SEED, features,
                      qutip_signal, toy_signal)

# =====================================================================
# YOUR KNOBS
# =====================================================================
NOISE_LEVELS = [2, 5, 8, 10, 12, 15, 20, 30]
RUN_QUTIP = True
TOY_N_PER_CLASS, TOY_SEEDS = 100, 5
QUTIP_N_PER_CLASS, QUTIP_SEEDS = 40, 3     # small because QuTiP is slow
# =====================================================================

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)


def sweep(make_clean, n_per_class, n_seeds):
    acc = {nl: {} for nl in NOISE_LEVELS}
    for s in range(n_seeds):
        rng = np.random.default_rng(SEED + s)
        clean = [make_clean(Q, rng) for Q in [Q_ORDERED] * n_per_class + [Q_DECOHERED] * n_per_class]
        y = np.array([1] * n_per_class + [0] * n_per_class)
        for nl in NOISE_LEVELS:   # same clean signals, fresh noise each level
            X = np.array([features(add_noise(sig, nl, rng)) for sig in clean])
            for model, a in evaluate(X, y, SEED + s).items():
                acc[nl].setdefault(model, []).append(a)
        print(f"    seed {s + 1}/{n_seeds} done", flush=True)
    return acc


sources = {"toy": (toy_signal, TOY_N_PER_CLASS, TOY_SEEDS)}
if RUN_QUTIP:
    try:
        import qutip  # noqa: F401
        sources["qutip"] = (qutip_signal, QUTIP_N_PER_CLASS, QUTIP_SEEDS)
    except ImportError:
        print("qutip not installed -> toy only\n")

results, n_seeds_used = {}, {}
for name, (fn, n, seeds) in sources.items():
    t0 = time.time()
    print(f"Running source: {name}  ({n} signals per class, {seeds} seeds)")
    results[name] = sweep(fn, n, seeds)
    n_seeds_used[name] = seeds
    print(f"  done in {time.time() - t0:.0f} s")

models = ["logistic regression", "random forest", "shuffled labels"]
with open(OUT / "mhz_sweep_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "noise_std", "model", "mean_accuracy", "std_accuracy"])
    for name, acc in results.items():
        print(f"\n[{name}]  mean test accuracy over {n_seeds_used[name]} seeds")
        print(f"{'noise':>6} | " + " | ".join(f"{m:>19}" for m in models))
        for nl in NOISE_LEVELS:
            row = []
            for m in models:
                mean, std = np.mean(acc[nl][m]), np.std(acc[nl][m])
                w.writerow([name, nl, m, round(mean, 4), round(std, 4)])
                row.append(f"{mean:6.3f} +/- {std:5.3f}  ")
            mark = "  <-- headline noise" if nl == NOISE_STD else ""
            print(f"{nl:>6} | " + " | ".join(row) + mark)

colors = {"toy": "#1f6f78", "qutip": "#d98324"}
styles = {"random forest": "-", "logistic regression": "--"}
fig, ax = plt.subplots(figsize=(8, 4.6))
for name, acc in results.items():
    for m, ls in styles.items():
        mean = np.array([np.mean(acc[nl][m]) for nl in NOISE_LEVELS])
        std = np.array([np.std(acc[nl][m]) for nl in NOISE_LEVELS])
        ax.plot(NOISE_LEVELS, mean, ls, color=colors[name], lw=1.8, label=f"{name} / {m}")
        ax.fill_between(NOISE_LEVELS, mean - std, mean + std, color=colors[name], alpha=0.12, lw=0)
ax.axhline(0.5, color="#888888", lw=1, ls=":")
ax.text(NOISE_LEVELS[0], 0.505, "chance (coin flip)", ha="left", va="bottom", color="#666666", fontsize=9)
ax.axvline(NOISE_STD, color="#bbbbbb", lw=1)
ax.set_ylim(0.3, 1.02)
ax.set_xlabel("Noise level (std; each peak has height 1)")
ax.set_ylabel("Test accuracy")
ax.set_title("MHz band, literature peaks, tau from Q (toy data)", loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "mhz_accuracy_vs_noise.png", dpi=150)
print(f"\nSaved: {OUT / 'mhz_accuracy_vs_noise.png'} and {OUT / 'mhz_sweep_results.csv'}")
