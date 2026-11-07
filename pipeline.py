"""
Microtubule Resonance Classifier - Step 1: full pipeline on FAKE data.

TOY SIMULATION. Inspired by Orch-Or / Bandyopadhyay, but it proves nothing
about consciousness. It only tests: can a classifier detect a planted,
sharp resonance signal buried in noise?

Stages:
  1. simulate  -> fake signals (3 resonance peaks, one per band)
  2. noise     -> add noise
  3. features  -> band power + peak sharpness per band
  4. classify  -> ordered (1) vs decohered (0)
  5. evaluate  -> held-out test set + sanity checks

Run:  python pipeline.py
Needs: pip install numpy scipy scikit-learn matplotlib
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# =====================================================================
# YOUR KNOBS (you are the architect - change these and re-run)
# =====================================================================
SEED = 42
N_PER_CLASS = 300          # signals per class
FS = 1000.0                # sampling rate, SCALED units (not real GHz)
DURATION = 2.0             # seconds, SCALED

# One peak per band. Frequencies are PLACEHOLDERS standing in for
# kHz / MHz / GHz. Real values come from the papers later.
BANDS = {
    "low (kHz-ish)":  (5, 15),
    "mid (MHz-ish)":  (40, 60),
    "high (GHz-ish)": (150, 200),
}
PEAKS = [10.0, 50.0, 175.0]   # peak centre inside each band
PEAK_JITTER = 2.0             # random shift per signal, keeps it non-trivial

# Decoherence = how fast the oscillation loses its phase. Medium setting.
# tau = coherence time in seconds. Long tau -> sharp peak. Short tau -> smeared peak.
TAU_ORDERED = 2.0
TAU_DECOHERED = 0.05

AMPLITUDE = 1.0
NOISE_STD = 3.0               # bigger = harder problem
# =====================================================================

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)
t = np.arange(0, DURATION, 1 / FS)


def simulate_signal(tau):
    """Sum of 3 oscillations whose phase randomly drifts at a rate 1/tau."""
    sig = np.zeros_like(t)
    for f0 in PEAKS:
        f = f0 + rng.uniform(-PEAK_JITTER, PEAK_JITTER)
        step_std = np.sqrt(2 * (1 / tau) / FS)       # phase diffusion
        phase = 2 * np.pi * f * t + np.cumsum(rng.normal(0, step_std, t.size))
        sig += AMPLITUDE * np.sin(phase + rng.uniform(0, 2 * np.pi))
    return sig + rng.normal(0, NOISE_STD, t.size)


def features(sig):
    """Per band: log power, and peak sharpness (tallest bin / average bin)."""
    freqs, pxx = welch(sig, fs=FS, nperseg=1000)
    feats = []
    for lo, hi in BANDS.values():
        band = pxx[(freqs >= lo) & (freqs <= hi)]
        feats.append(np.log10(band.sum()))
        feats.append(band.max() / band.mean())
    return feats


# ---- 1-3. simulate + extract features
X, y, raw_examples = [], [], {}
for label, tau in [(1, TAU_ORDERED), (0, TAU_DECOHERED)]:
    for i in range(N_PER_CLASS):
        s = simulate_signal(tau)
        if i == 0:
            raw_examples[label] = s
        X.append(features(s))
        y.append(label)
X, y = np.array(X), np.array(y)
names = [f"{b.split()[0]}_{k}" for b in BANDS for k in ("logpower", "sharpness")]

# ---- 4. classify
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
models = {
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "random forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
}

print(f"Train: {len(ytr)} signals | Test: {len(yte)} signals (never seen in training)\n")
for name, m in models.items():
    m.fit(Xtr, ytr)
    acc = m.score(Xte, yte)
    print(f"{name}: test accuracy = {acc:.3f}")
    print("  confusion matrix [rows=true 0/1, cols=pred 0/1]:")
    print("  ", confusion_matrix(yte, m.predict(Xte)).tolist())

# ---- 5. sanity check: shuffled labels should give ~50% (pure luck)
rf_shuf = RandomForestClassifier(n_estimators=200, random_state=SEED)
rf_shuf.fit(Xtr, rng.permutation(ytr))
print(f"\nSanity check (shuffled labels): {rf_shuf.score(Xte, yte):.3f}  (expect ~0.5)")

rf = models["random forest"]
print("\nWhich features the forest used most:")
for n, imp in sorted(zip(names, rf.feature_importances_), key=lambda p: -p[1]):
    print(f"  {n:20s} {imp:.3f}")

# ---- figure: example spectra
fig, ax = plt.subplots(figsize=(8, 4))
for label, nm, c in [(1, "ordered", "#1f6f78"), (0, "decohered", "#d98324")]:
    fr, p = welch(raw_examples[label], fs=FS, nperseg=1000)
    ax.semilogy(fr, p, color=c, lw=1.2, label=nm)
ax.set_xlim(0, 250)
ax.set_xlabel("Frequency (scaled units)")
ax.set_ylabel("Power")
ax.set_title("Example spectra: ordered vs decohered (toy data)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "spectra.png", dpi=150)
print(f"\nFigure saved to {OUT / 'spectra.png'}")
