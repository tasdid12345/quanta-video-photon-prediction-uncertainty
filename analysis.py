"""Self-supervised photon prediction on simulated quanta videos: bit2bit-style baseline vs
a small light-level-conditioned U-Net 3D-CNN ensemble with temperature scaling. Pure numpy (hand-written 3D CNN)."""
import io, json, os, sys, time
import numpy as np, requests
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy import ndimage
from scipy.optimize import minimize_scalar
from scipy.special import expit as sig, logit
from scipy.stats import spearmanr

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
LEVELS, T_LEN, K, NM, EPS = [0.05, 0.2, 1.0], 16, 4, 3, 0.05
NTR, NVA, NTE = 240, 30, 80
BASE_CAP, ENS_CAP, MAX_STEPS = 45, 120, 1500  # seconds per model / step upper bound
np.random.seed(42)
T0 = time.time()
def say(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

def lvc(lv):  # known light level encoded as a constant conditioning channel (log scale, roughly in [-1, 1])
    return float((np.log(lv) - np.log(0.2)) / 1.5)

# ---------------- tiny autodiff for 3D CNNs ----------------
TAPE, TRACK = [], True
class Nd:
    def __init__(s, v, ps=(), bw=None):
        s.v, s.g, s.ps, s.bw = v, 0, ps, bw
        if bw is not None and TRACK: TAPE.append(s)
def backward(root, g):
    root.g = g
    for n in reversed(TAPE):
        for p, gp in zip(n.ps, n.bw(n.g)):
            if gp is not None: p.g = p.g + gp
    TAPE.clear()
def _conv(x, w):  # 'same' 3x3x3 conv via im2col; x (B,C,T,H,W), w (O,C,3,3,3)
    B, C, T, H, W = x.shape; O = w.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1)))
    cols = swv(xp, (3, 3, 3), axis=(2, 3, 4)).transpose(0, 2, 3, 4, 1, 5, 6, 7).reshape(-1, C * 27)
    return (cols @ w.reshape(O, -1).T).reshape(B, T, H, W, O).transpose(0, 4, 1, 2, 3), cols
def conv(x, w, b):
    out, cols = _conv(x.v, w.v)
    out = out + b.v.reshape(1, -1, 1, 1, 1)
    def bw(g):
        gm = g.transpose(0, 2, 3, 4, 1).reshape(-1, w.v.shape[0])
        dx = _conv(g, w.v[:, :, ::-1, ::-1, ::-1].transpose(1, 0, 2, 3, 4))[0] if x.bw else None
        return dx, (cols.T @ gm).T.reshape(w.v.shape), g.sum((0, 2, 3, 4))
    return Nd(out, (x, w, b), bw)
def relu(x): return Nd(np.maximum(x.v, 0), (x,), lambda g: (g * (x.v > 0),))
def pool(x):
    B, C, T, H, W = x.v.shape
    out = x.v.reshape(B, C, T // 2, 2, H // 2, 2, W // 2, 2).mean((3, 5, 7))
    return Nd(out, (x,), lambda g: (np.repeat(np.repeat(np.repeat(g, 2, 2), 2, 3), 2, 4) / 8,))
def up(x):
    B, C, T, H, W = x.v.shape
    out = np.repeat(np.repeat(np.repeat(x.v, 2, 2), 2, 3), 2, 4)
    return Nd(out, (x,), lambda g: (g.reshape(B, C, T, 2, H, 2, W, 2).sum((3, 5, 7)),))
def cat(a, b):
    c = a.v.shape[1]
    return Nd(np.concatenate([a.v, b.v], 1), (a, b), lambda g: (g[:, :c], g[:, c:]))

BASE_SH = [(8, 2), (8, 8), (1, 8)]
UNET_SH = [(8, 3), (12, 8), (16, 12), (12, 28), (8, 20), (1, 8)]  # U-Net input: bits, mask, light-level channel
def fwd_base(P, x):
    h = relu(conv(x, P[0], P[1])); h = relu(conv(h, P[2], P[3])); return conv(h, P[4], P[5])
def fwd_unet(P, x):
    a1 = relu(conv(x, P[0], P[1]))
    a2 = relu(conv(pool(a1), P[2], P[3]))
    a3 = relu(conv(pool(a2), P[4], P[5]))
    u2 = relu(conv(cat(up(a3), a2), P[6], P[7]))
    u1 = relu(conv(cat(up(u2), a1), P[8], P[9]))
    return conv(u1, P[10], P[11])

def init(shapes, rng):
    P = []
    for i, (o, c) in enumerate(shapes):
        w = rng.standard_normal((o, c, 3, 3, 3)) * np.sqrt(2 / (c * 27)) * (0.1 if i == len(shapes) - 1 else 1)
        b = np.full(o, -2.0) if i == len(shapes) - 1 else np.zeros(o)
        P += [Nd(w.astype(np.float32)), Nd(b.astype(np.float32))]
    return P

def train(unet, levels, Ytr, seed, cap):
    """Masked-bit self-supervised training: hide 1/K of the bits, predict them (Bernoulli NLL)."""
    global TRACK
    TRACK = True
    rng = np.random.default_rng([42, seed])
    P = init(UNET_SH if unet else BASE_SH, rng); fwd = fwd_unet if unet else fwd_base
    Tw, B = (16, 4) if unet else (8, 8)
    ms = [np.zeros_like(p.v) for p in P]; vs = [np.zeros_like(p.v) for p in P]
    t0, s, run = time.time(), 0, []
    while True:
        frac = max(s / MAX_STEPS, (time.time() - t0) / cap)
        if frac >= 1: break
        lr = float(3e-3 * 0.5 * (1 + np.cos(np.pi * frac)))
        Yb = np.empty((B, Tw, 16, 16), np.uint8); lvb = []
        for i in range(B):
            lv = levels[rng.integers(len(levels))]; v = rng.integers(len(Ytr[lv])); lvb.append(lv)
            t, y, x = rng.integers(0, T_LEN - Tw + 1), rng.integers(0, 17), rng.integers(0, 17)
            Yb[i] = Ytr[lv][v, t:t + Tw, y:y + 16, x:x + 16]
        if rng.random() < .5: Yb = Yb[:, :, ::-1]
        if rng.random() < .5: Yb = Yb[:, :, :, ::-1]
        if rng.random() < .5: Yb = Yb[:, ::-1]
        mask = rng.random(Yb.shape) < 1 / K
        x = np.stack([Yb * (~mask), mask], 1).astype(np.float32)
        if unet:
            lc = np.array([lvc(l) for l in lvb], np.float32).reshape(B, 1, 1, 1, 1)
            x = np.concatenate([x, np.broadcast_to(lc, (B, 1, Tw, 16, 16))], 1).astype(np.float32)
        out = fwd(P, Nd(x)); p = sig(out.v[:, 0])
        run.append(float(-(Yb * np.log(p + 1e-7) + (1 - Yb) * np.log(1 - p + 1e-7))[mask].mean()))
        backward(out, ((p - Yb) * mask / max(mask.sum(), 1))[:, None].astype(np.float32))
        s += 1
        for i, q in enumerate(P):
            ms[i] = .9 * ms[i] + .1 * q.g; vs[i] = .999 * vs[i] + .001 * q.g ** 2
            q.v -= lr * (ms[i] / (1 - .9 ** s)) / (np.sqrt(vs[i] / (1 - .999 ** s)) + 1e-8); q.g = 0
    say(f"trained {'unet' if unet else 'base'} seed={seed}: {s} steps, masked NLL (last 50) {np.mean(run[-50:]):.4f}")
    return P

def predict(P, unet, Y, seed, lv=None):
    """Leave-bits-out logits: each bit is predicted while masked (K folds), never from itself."""
    global TRACK
    TRACK = False
    fwd, Tw = (fwd_unet, 16) if unet else (fwd_base, 8)
    fold = np.random.default_rng([42, seed]).integers(0, K, Y.shape)
    out = np.zeros(Y.shape, np.float32)
    for k in range(K):
        m = fold == k
        x = np.stack([Y * (~m), m], 1).astype(np.float32)
        if unet:
            x = np.concatenate([x, np.full((x.shape[0], 1) + x.shape[2:], lvc(lv), np.float32)], 1)
        for i in range(0, len(Y), 4):
            for t in range(0, T_LEN, Tw):
                lg = fwd(P, Nd(x[i:i + 4, :, t:t + Tw])).v[:, 0]
                sel = m[i:i + 4, t:t + Tw]; o = out[i:i + 4, t:t + Tw]; o[sel] = lg[sel]
    TRACK = True
    return out

# ---------------- data ----------------
def load_mnist():
    for i in range(5):
        r = requests.get(URL, timeout=60)
        if r.status_code in (429, 500, 502, 503, 504): time.sleep(2 ** i); continue
        r.raise_for_status(); d = np.load(io.BytesIO(r.content)); return d["x_train"], d["x_test"]
    raise RuntimeError("MNIST download failed")
def make_videos(imgs, rng):  # constant-velocity motion; even index slow (0.5 px/f), odd fast (2 px/f)
    V = np.zeros((len(imgs), T_LEN, 32, 32), np.float32)
    for i, im in enumerate(imgs):
        c = np.pad(im / 255., 2); ang = rng.uniform(0, 2 * np.pi); sp = 2.0 if i % 2 else 0.5
        for t in range(T_LEN):
            V[i, t] = ndimage.shift(c, (sp * np.sin(ang) * t, sp * np.cos(ang) * t), order=1, mode="grid-wrap")
    return np.clip(V, 0, 1)

# ---------------- metrics ----------------
def recon(p, a): return np.clip(-np.log1p(-np.clip(p, 0, 1 - 1e-3)) / a, 0, 1)  # flux -> intensity
def psnr(a, b): return 10 * np.log10(1 / np.maximum(((a - b) ** 2).mean((1, 2, 3)), 1e-10))
def ssim(a, b):
    f = lambda x: ndimage.uniform_filter(x, size=(1, 1, 7, 7))
    mx, my = f(a), f(b); vx, vy, c = f(a * a) - mx ** 2, f(b * b) - my ** 2, f(a * b) - mx * my
    return (((2 * mx * my + 1e-4) * (2 * c + 9e-4)) / ((mx ** 2 + my ** 2 + 1e-4) * (vx + vy + 9e-4))).mean((1, 2, 3))
def nll(p, Y):
    p = np.clip(p, 1e-6, 1 - 1e-6); return float(-(Y * np.log(p) + (1 - Y) * np.log(1 - p)).mean())
def ece(p, Y, nb=15):  # equal-mass bins
    p, y = p.ravel(), Y.ravel(); o = np.argsort(p)
    return float(sum(len(c) / len(p) * abs(p[c].mean() - y[c].mean()) for c in np.array_split(o, nb)))

def main():
    xtr, xte = load_mnist(); say("mnist loaded")
    rng = np.random.default_rng(42)
    I = {"tr": make_videos(xtr[:NTR], rng), "va": make_videos(xtr[NTR:NTR + NVA], rng), "te": make_videos(xte[:NTE], rng)}
    fast = {k: np.arange(len(v)) % 2 == 1 for k, v in I.items()}
    mean_i = I["tr"].mean()
    ALPHA = {lv: lv / mean_i for lv in LEVELS}  # mean flux per pixel-frame ~ lv
    Y = {sp: {lv: (rng.random(I[sp].shape) < 1 - np.exp(-ALPHA[lv] * I[sp])).astype(np.uint8) for lv in LEVELS} for sp in I}
    say("videos simulated; mean bit rates", {lv: round(float(Y['tr'][lv].mean()), 3) for lv in LEVELS})

    LB, LM = {}, [{} for _ in range(NM)]
    for j, lv in enumerate(LEVELS):  # baseline: one small 3D CNN per light level
        P = train(False, [lv], Y["tr"], j, BASE_CAP)
        LB[lv] = {sp: predict(P, False, Y[sp][lv], 1 if sp == "va" else 2) for sp in ("va", "te")}
    say("baseline done")
    for m in range(NM):  # improved: light-level-conditioned U-Net ensemble trained on mixed light levels
        P = train(True, LEVELS, Y["tr"], 100 + m, ENS_CAP)
        for lv in LEVELS:
            LM[m][lv] = {sp: predict(P, True, Y[sp][lv], 1 if sp == "va" else 2, lv) for sp in ("va", "te")}
        say(f"member {m} predicted")

    S, M, PV, bars = {}, {}, {}, {}
    def log(name, lv, p, Yt, It, a, bits=True):
        pv, sv = psnr(recon(p, a), It), ssim(recon(p, a), It)
        PV[(name, lv)] = pv; r = {"psnr": pv.mean(), "ssim": sv.mean()}
        if bits: r.update(nll=nll(p, Yt), ece=ece(p, Yt))
        for k, v in r.items(): S.setdefault(f"{name}_{k}", {})[lv] = float(v)
    for lv in LEVELS:
        a, Yt, Yv, It, Iv = ALPHA[lv], Y["te"][lv], Y["va"][lv], I["te"], I["va"]
        log("baseline", lv, sig(LB[lv]["te"]), Yt, It, a)
        # temporal average + Gaussian blur reference; window/sigma picked on validation
        ref = lambda Yy, w, s: ndimage.gaussian_filter(ndimage.uniform_filter1d(Yy.astype(np.float32), w, axis=1), (0, 0, s, s))
        w, s = max(((w, s) for w in (1, 3, 5, 9) for s in (0, 1, 1.5)), key=lambda c: psnr(recon(ref(Yv, *c), a), Iv).mean())
        log("ref", lv, ref(Yt, w, s), Yt, It, a, bits=False)
        pmv = sig(np.stack([LM[m][lv]["va"] for m in range(NM)])); pmt = sig(np.stack([LM[m][lv]["te"] for m in range(NM)]))
        zv, zt = logit(np.clip(pmv.mean(0), 1e-6, 1 - 1e-6)), logit(np.clip(pmt.mean(0), 1e-6, 1 - 1e-6))
        Tm = minimize_scalar(lambda t: nll(sig(zv / t), Yv), bounds=(0.3, 5), method="bounded").x  # val bits only
        S.setdefault("improved_temperature", {})[lv] = float(Tm)
        log("improved", lv, sig(zt / Tm), Yt, It, a)
        log("notemp", lv, pmt.mean(0), Yt, It, a)
        log("single", lv, pmt[0], Yt, It, a)
        # uncertainty = ensemble spread of reconstructed intensity; width scale calibrated on val
        sdv, sdt = recon(pmv, a).std(0), recon(pmt, a).std(0)
        errv, errt = np.abs(recon(sig(zv / Tm), a) - Iv), np.abs(recon(sig(zt / Tm), a) - It)
        sc = np.quantile(errv / (sdv + EPS), 0.9)
        for sp, nm in ((False, "slow"), (True, "fast")):
            ix = fast["te"] == sp
            for name in ("baseline", "improved"): M[f"{name}_psnr_a{lv}_{nm}"] = float(PV[(name, lv)][ix].mean())
            e, sd = errt[ix].ravel(), sdt[ix].ravel()
            sub = np.random.default_rng(42).choice(len(e), min(100000, len(e)), replace=False)
            for k, v in (("unc_corr", spearmanr(sd[sub], e[sub])[0]), ("cov90_raw", np.mean(e <= 1.645 * sd)),
                         ("cov90_cal", np.mean(e <= sc * (sd + EPS)))):
                M[f"{k}_a{lv}_{nm}"] = float(v); S.setdefault(k, {}).setdefault(f"{lv}_{nm}", float(v))
        say(f"level {lv} evaluated; temperature {Tm:.2f}")
    for k, d in S.items():
        M[k] = float(np.mean(list(d.values())))
        if k.split("_")[0] != k and not k.startswith(("unc_", "cov90")):
            for lv, v in d.items(): M[f"{k}_a{lv}"] = v
    M["baseline_score"], M["improved_score"] = M["baseline_psnr"], M["improved_psnr"]
    M["improvement"] = M["improved_score"] - M["baseline_score"]
    if not all(np.isfinite(v) for v in M.values()): sys.exit("non-finite metrics")

    os.makedirs("figures", exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for i, (n, lab) in enumerate((("ref", "avg+blur"), ("baseline", "bit2bit-style"), ("improved", "U-Net ensemble"))):
        plt.bar(np.arange(3) + 0.27 * i, [S[f"{n}_psnr"][lv] for lv in LEVELS], 0.27, label=lab)
    plt.xticks(np.arange(3) + .27, [str(lv) for lv in LEVELS]); plt.xlabel("photons/pixel/frame"); plt.ylabel("test PSNR (dB)")
    plt.legend(); plt.tight_layout(); plt.savefig("figures/psnr.png", dpi=100)

    res = {"question": "Can a small self-supervised 3D-CNN ensemble with per-pixel uncertainty beat a single bit2bit-style photon predictor in PSNR and calibration on simulated quanta videos of moving digits?",
           "method": "Self-supervised masked-bit prediction (Bernoulli NLL) with 3D CNNs on simulated SPAD videos of moving MNIST digits at 3 light levels; leave-bits-out predictions; baseline vs light-level-conditioned U-Net ensemble + temperature scaling.",
           "dataset": {"name": "Simulated moving-MNIST binary quanta videos", "source_url": URL, "n_rows": NTR + NVA + NTE,
                       "n_train": NTR, "n_test": NTE, "same_as_paper": False},
           "metric_name": "PSNR (dB) vs clean intensity, mean over 3 light levels", "higher_is_better": True,
           "baseline": "bit2bit-style small 3-layer 3D CNN (8-frame window), masked-bit Bernoulli loss, one model per light level, no calibration.",
           "improved": "U-Net-like 3D CNN with 16-frame window trained on mixed light levels and conditioned on the known light level (extra constant input channel), ensemble of members, temperature scaling of the rate on validation bits; spread gives uncertainty.",
           "metrics": M,
           "notes": "Stand-in simulation (paper data/architecture unknown); numpy-only CNNs with wall-clock-capped training so step counts depend on hardware. Improved model gets more total training compute than the baseline and is told the (known) light level via an input channel. Motion wraps around the frame; no dark counts; light level = mean flux per pixel-frame. Held-out-bit NLL/ECE use leave-bits-out predictions (K folds) on test bits. Uncertainty is epistemic ensemble spread of reconstructed intensity; cov90_raw uses +-1.645*sd, cov90_cal rescales width (sd+0.05) using a validation quantile. Reference window/sigma and temperature chosen on validation videos only (never test). Few test videos (small sample)."}
    with open("results.json", "w", encoding="utf-8") as f: json.dump(res, f, indent=2)
    say("done", {k: round(M[k], 4) for k in ("baseline_score", "improved_score", "improvement")})

if __name__ == "__main__":
    main()
