# Literature notes: reported microtubule resonance frequencies

Purpose: real numbers to anchor the toy model. Collected 2026-11-06.

**Verification status:** values below were pulled from the papers' web pages
with an automated summarizer. Frequencies marked "from figure axis" were read
off a plot, not stated in the text. Check every number against the original
figures before quoting it in the README or anywhere public.

## Reported by Bandyopadhyay's group

Source A: Fractal and Fractional, 2020 (fractal, scale-free electromagnetic
resonance of a single brain microtubule). Section 3.1, Figures 2-3.

| Band | Reported range | Peak sharpness (Q) | Notes |
|---|---|---|---|
| kHz | 10-100 kHz | about 3-5 | three overlapping peaks; depend on AC bias amplitude |
| MHz | 8-240 MHz | about 100-300 | eight peaks; phase shifts in multiples of pi/4 |
| GHz | 7-13 GHz (also about 19 GHz) | not stated | two Gaussian-like transmissions near 7 GHz pump; peak near 13 GHz |

Source B: atomic water channel paper (single brain microtubule, 2013 as far
as I know). Eight MHz peaks at about 12, 20, 22, 30, 101, 113, 185, 204 MHz
(from figure axis, Fig. 5f). No kHz or GHz peaks reported there.

## Caveats stated by the authors or visible in the papers
- Peaks that also appear in open/short-circuit reference conditions were discarded.
- No background subtraction (authors' choice).
- Low-intensity peaks "blink" (appear and disappear) in more than 88% of devices.
- "Triplet of triplets" is described as not an absolute pattern.
- Source B: sweep range stated as 20 GHz in text but 1.3 GHz in a caption; one
  figure label looks like a unit typo (MOhm vs MHz).
- Neither page shows independent replication.

## Decoherence-time debate (context for the tau knob)
From the review "Quantum effects in the brain: A review" (arXiv 1910.08423):
- Tegmark (2000): about 1e-13 to 1e-20 s for microtubules.
- Hagan et al. (re-modelled): about 1e-5 to 1e-4 s.
- These are quantum-state decoherence estimates. Bandyopadhyay's data are
  classical AC electrical measurements. Linking the two is a hypothesis, not
  an established result. The toy model treats them as the same thing.

## Useful conversion
For a decaying oscillation with coherence time tau, the peak width (FWHM) is
1/(pi*tau), so Q = pi * f * tau, or tau = Q / (pi * f).

## Sources
- https://www.mdpi.com/2504-3110/4/2/11/htm
- https://research.iitj.ac.in/pdf/postprint-atomic-water-channel-controlling-remarkable-properties
- https://arxiv.org/pdf/1910.08423
