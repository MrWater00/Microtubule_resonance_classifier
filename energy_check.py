"""
Energy check: how much total signal power does each source carry (before noise)?

This is the measurement behind experiment B. The QuTiP signal fades, so it
carries less energy than the steady toy signal. A classifier can exploit that
(signal present vs almost nothing) instead of judging peak sharpness.

Run:  python energy_check.py     (about 30 s)
"""

from pathlib import Path

import numpy as np

from mhz_band import Q_DECOHERED, Q_ORDERED, qutip_signal, toy_signal

N = 10
rng = np.random.default_rng(1)
lines = [f"Mean signal power (mean of x^2), before noise, averaged over {N} signals", ""]
for Q, name in [(Q_ORDERED, "ordered"), (Q_DECOHERED, "decohered")]:
    toy = np.mean([np.mean(toy_signal(Q, rng) ** 2) for _ in range(N)])
    qt_ = np.mean([np.mean(qutip_signal(Q, rng) ** 2) for _ in range(N)])
    lines.append(f"{name:10s} Q={Q:<6g} toy = {toy:6.3f}   qutip = {qt_:6.3f}   qutip/toy = {qt_ / toy:.3f}")

text = "\n".join(lines)
print(text)
out = Path(__file__).resolve().parent / "outputs"
out.mkdir(exist_ok=True)
(out / "energy_check_log.txt").write_text(text + "\n")
