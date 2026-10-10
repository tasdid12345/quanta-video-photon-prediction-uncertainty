# Self-supervised photon prediction on simulated quanta videos

Question: can a small self-supervised 3D-CNN ensemble with per-pixel uncertainty beat a single bit2bit-style photon predictor in PSNR and calibration on simulated binary (single-photon) videos of moving MNIST digits?

- Data: MNIST (downloaded) turned into 16-frame moving-digit videos, Bernoulli photon sampling at 3 light levels (0.05, 0.2, 1.0 photons/pixel/frame). Train/val/test = 240/30/80 videos; seed 42.
- Baseline: small 3-layer 3D CNN (8-frame window), masked-bit Bernoulli loss, one model per light level, no calibration.
- Improved: U-Net-style 3D CNN (16-frame window) trained on mixed light levels and conditioned on the known light level via an extra constant input channel, ensemble of 3 members, temperature scaling fitted on validation bits only. Ensemble spread gives per-pixel uncertainty.
- Metric: PSNR (dB) of the reconstructed intensity versus the clean intensity, averaged over the 3 light levels (higher is better). The test set is never used for tuning.
- Pure numpy implementation (CPU only, wall-clock-capped training, under 20 minutes).

Run: `pip install -r requirements.txt && python analysis.py` -> writes `results.json` and `figures/psnr.png`.

---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
