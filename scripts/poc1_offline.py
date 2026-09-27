"""PoC 1 - offline decoding of indy_20161005_06 with Wiener and Kalman.

* 5 contiguous temporal folds (no shuffling): each fold is tested once, the
  model is trained on the other 4.
* Metrics per fold and per velocity axis: R^2 and Pearson r.
* Missing-bin sanity check: blank 5 % and 20 % of the test bins (NaN rows);
  Kalman runs prediction-only on them, Wiener sees them as zero counts.
* Outputs: results/poc1_metrics.csv, results/poc1_trace.png

Usage::

    python scripts/poc1_offline.py            # uses data/processed/..._bin20ms.npz (built if missing)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

from icbench.data import load_mat, load_npz  # noqa: E402
from icbench.decoders import KalmanDecoder, WienerDecoder  # noqa: E402

# Configuration fixed after a single exploratory split (see README, "Caveats").
WIENER = dict(K=10, alpha=1e3, smooth_sigma=2.0, causal_smooth=True)
KALMAN = dict(use_acc=True, sqrt_counts=True, smooth_sigma=0.0, lag=3)
MISSING_FRACTIONS = (0.0, 0.05, 0.20)
SEED = 20261005


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out = {}
    for k, ax in enumerate("xy"):
        out[f"r2_{ax}"] = r2_score(y[:, k], p[:, k])
        out[f"r_{ax}"] = pearsonr(y[:, k], p[:, k])[0]
    return out


def blank_rows(X: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    X = X.astype(np.float64).copy()
    if frac > 0:
        idx = rng.choice(len(X), size=int(round(frac * len(X))), replace=False)
        X[idx] = np.nan
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/processed/indy_20161005_06_bin20ms.npz")
    ap.add_argument("--mat", default="data/raw/indy_20161005_06.mat")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    t_start = time.perf_counter()
    if Path(a.npz).exists():
        s = load_npz(a.npz)
    else:
        s = load_mat(a.mat, bin_ms=20.0, include_hash=True)
        s.save(a.npz)
    X, V, P = s.counts.astype(np.float64), s.vel, s.pos
    n = len(X)
    edges = np.linspace(0, n, a.folds + 1).astype(int)
    rng = np.random.default_rng(SEED)

    rows, traces = [], {}
    for f in range(a.folds):
        te = np.arange(edges[f], edges[f + 1])
        tr = np.setdiff1d(np.arange(n), te)
        w = WienerDecoder(**WIENER).fit(X[tr], V[tr])
        k = KalmanDecoder(dt=s.bin_s, **KALMAN).fit(X[tr], V[tr], P[tr])
        for frac in MISSING_FRACTIONS:
            Xte = blank_rows(X[te], frac, rng)
            for name, model in (("wiener", w), ("kalman", k)):
                p = model.predict(Xte)
                rows.append(dict(fold=f + 1, decoder=name, missing_frac=frac, **metrics(V[te], p)))
                if frac == 0.0:
                    traces[(f, name)] = p
        print(f"fold {f + 1}/{a.folds} done ({time.perf_counter() - t_start:.1f} s)")

    df = pd.DataFrame(rows)
    df["r2_mean"] = df[["r2_x", "r2_y"]].mean(axis=1)
    mean = df.groupby(["decoder", "missing_frac"], as_index=False)[["r2_x", "r2_y", "r_x", "r_y", "r2_mean"]].mean()
    mean.insert(0, "fold", "mean")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    full = pd.concat([df, mean], ignore_index=True)
    full.to_csv(out / "poc1_metrics.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 140)
    print("\nPer fold (no missing bins):")
    print(df[df.missing_frac == 0].drop(columns="missing_frac").to_string(index=False, float_format="%.3f"))
    print("\nMean over folds:")
    print(mean.drop(columns="fold").to_string(index=False, float_format="%.3f"))
    base = mean[mean.missing_frac == 0].set_index("decoder")["r2_mean"]
    print("\nDegradation of mean R^2 vs clean (delta):")
    for _, r in mean[mean.missing_frac > 0].iterrows():
        print(f"  {r.decoder:6s} missing={r.missing_frac:.0%}: {r.r2_mean - base[r.decoder]:+.3f}")

    # --- figure: 20 s excerpt of the middle fold
    f = a.folds // 2
    te0 = edges[f]
    start = int(round(10.0 / s.bin_s))  # 10 s into the test fold
    L = int(round(20.0 / s.bin_s))
    sl = slice(start, start + L)
    tt = s.t[te0 + start : te0 + start + L] - s.t[te0 + start]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(tt, V[te0 + start : te0 + start + L, k], color="#52514e", lw=2, label="real")
        ax.plot(tt, traces[(f, "wiener")][sl, k], color="#2a78d6", lw=1.5, label="Wiener")
        ax.plot(tt, traces[(f, "kalman")][sl, k], color="#eb6834", lw=1.5, label="Kalman")
        ax.set_ylabel(f"v{'xy'[k]} (mm/s)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper right", ncol=3, frameon=False)
    axes[-1].set_xlabel(f"tiempo (s) - fold {f + 1}, extracto de 20 s")
    fig.suptitle("indy_20161005_06: velocidad del cursor real vs decodificada (bins de 20 ms)")
    fig.tight_layout()
    fig.savefig(out / "poc1_trace.png", dpi=150)

    print(f"\nsaved -> {out / 'poc1_metrics.csv'}, {out / 'poc1_trace.png'}")
    print(f"runtime: {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
