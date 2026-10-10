# Self-supervised photon prediction with a 3D-CNN ensemble on simulated quanta videos

## Motivation
"bit2bit: 1-bit quanta video reconstruction via self-supervised photon prediction" (NeurIPS 2024)
reconstructs video from binary single-photon frames by training a network to predict held-out
photon detection events, without ever seeing clean images. This experiment asks whether a small
ensemble of deeper self-supervised 3D CNNs, with temperature scaling, improves on a single
bit2bit-style predictor and also gives useful per-pixel uncertainty.

## Question
Can a small self-supervised 3D-CNN ensemble, which predicts held-out photon bits and gives per-pixel
uncertainty, beat a single bit2bit-style photon predictor in PSNR and calibration on simulated binary
quanta videos of moving digits at several light levels?

## Dataset
MNIST digits (`mnist.npz`, public Google storage URL) turned into short 32x32 videos of digits moving at
constant velocity (slow and fast variants). Binary frames are simulated as Bernoulli(1 - exp(-alpha*I))
at three mean light levels (about 0.05, 0.2 and 1 photon per pixel per frame). Train, validation and test
videos use different digit images, with fixed seeds. This is **not** the paper's dataset; it is a
stand-in simulation because the paper's data and architecture were not available.

## Models
- **Baseline (bit2bit-style):** small 3-layer 3D CNN on a short window of binary frames with random bits
  masked; predicts masked bits with a Bernoulli likelihood; one model per light level; the predicted
  rate is converted from flux to intensity.
- **Improved:** same self-supervised loss, but a deeper U-Net-like 3D CNN with a wider temporal window,
  trained on mixed light levels, as a deep ensemble (spread = uncertainty), with temperature scaling of
  the predicted rate fitted on validation bits.
- **Reference:** temporal average plus Gaussian blur (settings chosen on validation).

Metrics on the same test videos: PSNR (main), SSIM, held-out-bit NLL and ECE, plus uncertainty/error
correlation and interval coverage per light level and motion speed. Everything is implemented in numpy
(hand-written 3D CNN) with a capped training budget so the run stays under 20 minutes on CPU.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/psnr.png`).

---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
