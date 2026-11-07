"""
Step 2: accuracy-vs-noise curve, for BOTH signal sources.

  toy   = random-phase sine waves (Step 1 generator)
  qutip = 3-level quantum system with dephasing (Step 3 generator)

Same peaks, same features, same classifiers - only the signal source differs.
For each noise level, the experiment is repeated with N_SEEDS different seeds,
so the plot shows a mean and a spread, not one lucky number.

Run:  python sweep.py
Needs: pip install numpy scipy scikit-learn matplotlib qutip
"""

import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (NOISE_STD, SEED, TAU_DECOHERED, TAU_ORDERED, add_noise,
                    evaluate, features, toy_clean_signal)

try:
    from qutip_sim import simulate_signal_qutip
    HAVE_QUTIP = True
except ImportError:
    HAVE_QUTIP = False
    print("qutip not installed -> running toy source only (pip install qutip)\n")

# =====================================================================
# YOUR KNOBS
# =====================================================================
NOISE_LEVELS = [1, 2, 3, 4, 6, 8, 10, 12, 15]
N_SEEDS = 5            # repeats per noise level
N_PER_CLASS = 150      # signals per class per repeat
# =====================================================================

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)


def sweep(make_clean):
    """Return {noise: {model: [accuracy per seed]}}."""
    acc = {nl: {} for nl in NOISE_LEVELS}
    for s in range(N_SEEDS):
        rng = np.random.default_rng(SEED + s)
        clean, y = [], []
        for label, tau in [(1, TAU_ORDERED), (0, TAU_DECOHERED)]:
            for _ in range(N_PER_CLASS):
                clean.append(make_clean(tau, rng))
                y.append(label)
        y = np.array(y)
        for nl in NOISE_LEVELS:   # same clean signals, fresh noise each level
            X = np.array([features(add_noise(sig, nl, rng)) for sig in clean])
            for model, a in evaluate(X, y, SEED + s).items():
                acc[nl].setdefault(model, []).append(a)
    return acc


sources = {"toy": lambda tau, rng: toy_clean_signal(tau, rng)}
if HAVE_QUTIP:
    sources["qutip"] = lambda tau, rng: simulate_signal_qutip(tau, rng)[0]

results = {}
for name, fn in sources.items():
    t0 = time.time()
    print(f"Running source: {name} ...")
    results[name] = sweep(fn)
    print(f"  done in {time.time() - t0:.0f} s")

# ---- table + csv
models = ["logistic regression", "random forest", "shuffled labels"]
with open(OUT / "sweep_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "noise_std", "model", "mean_accuracy", "std_accuracy"])
    for name, acc in results.items():
        print(f"\n[{name}]  mean test accuracy over {N_SEEDS} seeds")
        print(f"{'noise':>6} | " + " | ".join(f"{m:>19}" for m in models))
        for nl in NOISE_LEVELS:
            row = []
            for m in models:
                mean, std = np.mean(acc[nl][m]), np.std(acc[nl][m])
                w.writerow([name, nl, m, round(mean, 4), round(std, 4)])
                row.append(f"{mean:6.3f} +/- {std:5.3f}  ")
            mark = "  <-- your chosen noise" if nl == NOISE_STD else ""
            print(f"{nl:>6} | " + " | ".join(row) + mark)

# ---- figure
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
ax.set_ylim(0.4, 1.02)
ax.set_xlabel("Noise level (std, same units as signal peak height = 1)")
ax.set_ylabel("Test accuracy")
ax.set_title("Can the classifier find the resonance as noise rises? (toy data)", loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "accuracy_vs_noise.png", dpi=150)
print(f"\nSaved: {OUT / 'accuracy_vs_noise.png'} and {OUT / 'sweep_results.csv'}")
