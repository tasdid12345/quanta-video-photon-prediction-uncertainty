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

## Results (latest run)

On my simulated moving-MNIST binary quanta videos (80 test videos), my re-implementation of your approach reached 19.00 dB PSNR (mean over 3 light levels) versus 19.14 dB for the ensemble, a small 0.14 dB gain, and the ensemble was told the light level.

| Metric | Value |
|---|---|
| baseline_psnr_a0.05_slow | 19.78 |
| improved_psnr_a0.05_slow | 19.97 |
| unc_corr_a0.05_slow | 0.9458 |
| cov90_raw_a0.05_slow | 0.5153 |
| cov90_cal_a0.05_slow | 0.9253 |
| baseline_psnr_a0.05_fast | 17.35 |
| improved_psnr_a0.05_fast | 17.43 |
| unc_corr_a0.05_fast | 0.9515 |
| cov90_raw_a0.05_fast | 0.4578 |
| cov90_cal_a0.05_fast | 0.8904 |
| baseline_psnr_a0.2_slow | 22.02 |
| improved_psnr_a0.2_slow | 22.29 |
| unc_corr_a0.2_slow | 0.9372 |
| cov90_raw_a0.2_slow | 0.6019 |
| cov90_cal_a0.2_slow | 0.9193 |
| baseline_psnr_a0.2_fast | 19.53 |
| improved_psnr_a0.2_fast | 19.69 |
| unc_corr_a0.2_fast | 0.9583 |
| cov90_raw_a0.2_fast | 0.5597 |
| cov90_cal_a0.2_fast | 0.8946 |
| baseline_psnr_a1.0_slow | 18.19 |
| improved_psnr_a1.0_slow | 18.35 |
| unc_corr_a1.0_slow | 0.9655 |
| cov90_raw_a1.0_slow | 0.5822 |
| cov90_cal_a1.0_slow | 0.9141 |
| baseline_psnr_a1.0_fast | 17.13 |
| improved_psnr_a1.0_fast | 17.08 |
| unc_corr_a1.0_fast | 0.9663 |
| cov90_raw_a1.0_fast | 0.5546 |
| cov90_cal_a1.0_fast | 0.9013 |
| baseline_psnr | 19 |
| baseline_psnr_a0.05 | 18.56 |
| baseline_psnr_a0.2 | 20.78 |
| baseline_psnr_a1.0 | 17.66 |
| baseline_ssim | 0.8624 |
| baseline_ssim_a0.05 | 0.8318 |
| baseline_ssim_a0.2 | 0.9075 |
| baseline_ssim_a1.0 | 0.8479 |
| baseline_nll | 0.08784 |
| baseline_nll_a0.05 | 0.09315 |
| baseline_nll_a0.2 | 0.1051 |
| baseline_nll_a1.0 | 0.06525 |
| baseline_ece | 0.001287 |
| baseline_ece_a0.05 | 0.0004058 |
| baseline_ece_a0.2 | 0.001755 |
| baseline_ece_a1.0 | 0.001702 |
| ref_psnr | 18.62 |
| ref_psnr_a0.05 | 18.22 |
| ref_psnr_a0.2 | 19.3 |
| ref_psnr_a1.0 | 18.34 |
| ref_ssim | 0.8321 |
| ref_ssim_a0.05 | 0.7886 |
| ref_ssim_a0.2 | 0.8602 |
| ref_ssim_a1.0 | 0.8475 |
| improved_temperature | 0.9773 |
| improved_temperature_a0.05 | 0.9845 |
| improved_temperature_a0.2 | 0.9915 |
| improved_temperature_a1.0 | 0.9559 |
| improved_psnr | 19.14 |
| improved_psnr_a0.05 | 18.7 |
| improved_psnr_a0.2 | 20.99 |
| improved_psnr_a1.0 | 17.71 |
| improved_ssim | 0.867 |
| improved_ssim_a0.05 | 0.8353 |
| improved_ssim_a0.2 | 0.9122 |
| improved_ssim_a1.0 | 0.8535 |
| improved_nll | 0.08634 |
| improved_nll_a0.05 | 0.09248 |
| improved_nll_a0.2 | 0.1037 |
| improved_nll_a1.0 | 0.06284 |
| improved_ece | 0.001562 |
| improved_ece_a0.05 | 0.0008453 |
| improved_ece_a0.2 | 0.00222 |
| improved_ece_a1.0 | 0.001621 |
| notemp_psnr | 19.05 |
| notemp_psnr_a0.05 | 18.73 |
| notemp_psnr_a0.2 | 20.98 |
| notemp_psnr_a1.0 | 17.43 |
| notemp_ssim | 0.8631 |
| notemp_ssim_a0.05 | 0.8333 |
| notemp_ssim_a0.2 | 0.9118 |
| notemp_ssim_a1.0 | 0.844 |
| notemp_nll | 0.08637 |
| notemp_nll_a0.05 | 0.09249 |
| notemp_nll_a0.2 | 0.1037 |
| notemp_nll_a1.0 | 0.06292 |
| notemp_ece | 0.001636 |
| notemp_ece_a0.05 | 0.0008 |
| notemp_ece_a0.2 | 0.002233 |
| notemp_ece_a1.0 | 0.001874 |
| single_psnr | 19.08 |
| single_psnr_a0.05 | 18.82 |
| single_psnr_a0.2 | 20.74 |
| single_psnr_a1.0 | 17.69 |
| single_ssim | 0.8644 |
| single_ssim_a0.05 | 0.8326 |
| single_ssim_a0.2 | 0.9081 |
| single_ssim_a1.0 | 0.8526 |
| single_nll | 0.08734 |
| single_nll_a0.05 | 0.09262 |
| single_nll_a0.2 | 0.1048 |
| single_nll_a1.0 | 0.0646 |
| single_ece | 0.00161 |
| single_ece_a0.05 | 0.0006775 |
| single_ece_a0.2 | 0.003092 |
| single_ece_a1.0 | 0.001061 |
| unc_corr | 0.9541 |
| cov90_raw | 0.5453 |
| cov90_cal | 0.9075 |
| baseline_score | 19 |
| improved_score | 19.14 |
| improvement | 0.1366 |

Run on 2026-10-10. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in simulation (paper data/architecture unknown); numpy-only CNNs with wall-clock-capped training so step counts depend on hardware. Improved model gets more total training compute than the baseline and is told the (known) light level via an input channel. Motion wraps around the frame; no dark counts; light level = mean flux per pixel-frame. Held-out-bit NLL/ECE use leave-bits-out predictions (K folds) on test bits. Uncertainty is epistemic ensemble spread of reconstructed intensity; cov90_raw uses +-1.645*sd, cov90_cal rescales width (sd+0.05) using a validation quantile. Reference window/sigma and temperature chosen on validation videos only (never test). Few test videos (small sample).
