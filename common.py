"""
Shared config and helpers for Steps 2 and 3 of the Microtubule Resonance Classifier.

TOY SIMULATION. Inspired by Orch-Or / Bandyopadhyay; proves nothing about
consciousness. It tests whether a classifier can find a planted resonance
signal in noise.

All frequencies and times are SCALED units, not real kHz/MHz/GHz.
"""

import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# =====================================================================
# YOUR KNOBS (you are the architect - change these and re-run)
# =====================================================================
SEED = 42
FS = 1000.0                  # sampling rate (scaled)
DURATION = 2.0               # seconds (scaled)

# Energy levels of the 3-level system (scaled frequency units).
# Level 0 = 0, level 1 = E1, level 2 = E2.
# The three visible peaks are the three transitions between levels:
#   0<->1 at E1,   1<->2 at E2-E1,   0<->2 at E2
# NOTE: in a 3-level system the peaks are linked (highest = sum of the
# other two), so they cannot be placed freely like in Step 1.
E1, E2 = 30.0, 100.0
PEAKS = [E1, E2 - E1, E2]    # -> 30, 70, 100
BAND_HALFWIDTH = 10.0
BANDS = [(p - BAND_HALFWIDTH, p + BAND_HALFWIDTH) for p in PEAKS]
PEAK_JITTER = 2.0            # random shift per signal, keeps it non-trivial

# Coherence time (seconds, scaled). Long = sharp peak, short = smeared peak.
TAU_ORDERED = 2.0
TAU_DECOHERED = 0.05

AMPLITUDE = 1.0              # height of each peak before noise
NOISE_STD = 10.0             # your choice: hard setting
# =====================================================================

TIME = np.arange(0, DURATION, 1 / FS)


def toy_clean_signal(tau, rng):
    """Step 1 style generator: 3 sine waves whose phase randomly drifts at rate 1/tau."""
    sig = np.zeros_like(TIME)
    step_std = np.sqrt(2 / (tau * FS))
    for f0 in PEAKS:
        f = f0 + rng.uniform(-PEAK_JITTER, PEAK_JITTER)
        phase = 2 * np.pi * f * TIME + np.cumsum(rng.normal(0, step_std, TIME.size))
        sig += AMPLITUDE * np.sin(phase + rng.uniform(0, 2 * np.pi))
    return sig


def add_noise(sig, noise_std, rng):
    return sig + rng.normal(0, noise_std, sig.size)


def features(sig):
    """Per band: log power and peak sharpness (tallest bin / average bin)."""
    freqs, pxx = welch(sig, fs=FS, nperseg=int(FS))
    feats = []
    for lo, hi in BANDS:
        band = pxx[(freqs >= lo) & (freqs <= hi)]
        feats.append(np.log10(band.sum()))
        feats.append(band.max() / band.mean())
    return feats


def evaluate(X, y, seed):
    """Train on 70%, test on unseen 30%. Returns accuracy per model + a shuffled-label check."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    models = {
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random forest": RandomForestClassifier(n_estimators=100, random_state=seed),
    }
    out = {name: m.fit(Xtr, ytr).score(Xte, yte) for name, m in models.items()}
    shuffled = RandomForestClassifier(n_estimators=100, random_state=seed)
    shuffled.fit(Xtr, np.random.default_rng(seed).permutation(ytr))
    out["shuffled labels"] = shuffled.score(Xte, yte)   # should be ~0.5
    return out
