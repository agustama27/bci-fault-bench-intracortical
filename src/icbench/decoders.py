"""Reference decoders for binned spike counts, implemented from scratch in numpy.

* :class:`WienerDecoder` - ridge regression of velocity on the causal history of
  the last ``K`` bins of counts (Serruya et al., 2002; Hochberg et al., 2006).
* :class:`KalmanDecoder` - linear-Gaussian state-space decoder as in Wu, Gao,
  Bienenstock, Donoghue & Black (2006), Neural Computation 18(1), 80-118.

Missing observations (a whole row of NaN in ``X``) are the hook for the fault
injection line:

* Wiener: the row is zero-filled (0 spikes), the naive behaviour of a stateless
  decoder that receives an empty sample.
* Kalman: the update step is skipped and only the prediction step runs
  (``x = A x``, ``P = A P A' + W``), the natural treatment of a missing
  observation in a state-space model.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- helpers
def gaussian_kernel(sigma: float, causal: bool = True) -> np.ndarray:
    """Normalised Gaussian kernel (sigma in bins). Causal = half-Gaussian over past bins."""
    half = int(np.ceil(4 * sigma))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    if causal:
        k[:half] = 0.0  # keep lag >= 0 only (current + past)
    return k / k.sum()


def smooth_counts(X: np.ndarray, sigma: float, causal: bool = True) -> np.ndarray:
    """Convolve each channel with a Gaussian kernel. ``sigma <= 0`` is a no-op.

    NaN rows are treated as zeros for smoothing and restored as NaN afterwards.
    """
    X = np.asarray(X, dtype=np.float64)
    if sigma <= 0:
        return X.copy()
    nan_rows = np.isnan(X).any(axis=1)
    Xf = np.where(np.isnan(X), 0.0, X)
    k = gaussian_kernel(sigma, causal)
    half = (len(k) - 1) // 2
    out = np.empty_like(Xf)
    for c in range(Xf.shape[1]):
        # 'full' convolution then pick the aligned window: out[t] = sum_j k[j] X[t + half - j]
        out[:, c] = np.convolve(Xf[:, c], k, mode="full")[half : half + Xf.shape[0]]
    out[nan_rows] = np.nan
    return out


def lag_matrix(X: np.ndarray, K: int) -> np.ndarray:
    """Causal history features: row t = [X[t], X[t-1], ..., X[t-K+1]] (zero-padded)."""
    n, c = X.shape
    out = np.zeros((n, c * K), dtype=np.float64)
    for j in range(K):
        out[j:, j * c : (j + 1) * c] = X[: n - j]
    return out


def ridge(Xd: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge with an unpenalised intercept (last row of the returned B)."""
    xm, ym = Xd.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = Xd - xm, Y - ym
    G = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    B = np.linalg.solve(G, Xc.T @ Yc)
    b0 = ym - xm @ B
    return np.vstack([B, b0])


# --------------------------------------------------------------------------- Wiener
class WienerDecoder:
    """Linear (Wiener) filter on the concatenated history of the last ``K`` bins.

    Parameters
    ----------
    K : history length in bins (10 bins x 20 ms = 200 ms).
    alpha : ridge penalty on z-scored features.
    smooth_sigma : Gaussian smoothing of counts before the filter (bins, 0 = off).
    causal_smooth : half-Gaussian (online-compatible) if True.
    """

    def __init__(self, K: int = 10, alpha: float = 1e3, smooth_sigma: float = 0.0, causal_smooth: bool = True):
        self.K, self.alpha = K, alpha
        self.smooth_sigma, self.causal_smooth = smooth_sigma, causal_smooth

    def _features(self, X: np.ndarray) -> np.ndarray:
        X = np.where(np.isnan(X), 0.0, np.asarray(X, dtype=np.float64))  # missing -> 0 spikes
        X = smooth_counts(X, self.smooth_sigma, self.causal_smooth)
        Z = (X - self.mu_) / self.sd_
        return lag_matrix(Z, self.K)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "WienerDecoder":
        Xs = smooth_counts(np.nan_to_num(X_train), self.smooth_sigma, self.causal_smooth)
        self.mu_ = Xs.mean(axis=0)
        self.sd_ = Xs.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        F = self._features(X_train)
        self.B_ = ridge(F, np.asarray(Y_train, dtype=np.float64), self.alpha)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        F = self._features(X_test)
        return F @ self.B_[:-1] + self.B_[-1]


# --------------------------------------------------------------------------- Kalman
class KalmanDecoder:
    """Kalman filter decoder (Wu et al., 2006).

    State ``x_t = [px, py, vx, vy, (ax, ay), 1]``; the trailing constant absorbs
    the observation baseline (intercept of H) so that H, Q are an ordinary
    least-squares fit on uncentred data.

    * dynamics  ``x_{t+1} = A x_t + w``,  ``w ~ N(0, W)``   (least squares on training kinematics)
    * observation ``z_t = H x_t + q``,    ``q ~ N(0, Q)``   (least squares of counts on kinematics)

    Parameters
    ----------
    use_acc : include acceleration in the state.
    sqrt_counts : observe sqrt(counts) (variance stabilisation).
    smooth_sigma : Gaussian smoothing of counts (bins, 0 = off).
    lag : neural lead in bins; the state at t is paired with counts at t - lag.
    dt : bin width in s (only used to integrate velocity to position when
         ``pos`` is not given, and to compute acceleration).
    """

    def __init__(
        self,
        use_acc: bool = True,
        sqrt_counts: bool = True,
        smooth_sigma: float = 0.0,
        lag: int = 3,
        dt: float = 0.02,
        causal_smooth: bool = True,
    ):
        self.use_acc, self.sqrt_counts = use_acc, sqrt_counts
        self.smooth_sigma, self.causal_smooth = smooth_sigma, causal_smooth
        self.lag, self.dt = lag, dt

    # -- data preparation
    def _obs(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.sqrt_counts:
            X = np.sqrt(np.clip(X, 0, None))  # NaN stays NaN
        X = smooth_counts(X, self.smooth_sigma, self.causal_smooth)
        if self.lag > 0:  # z used at state t is the count at t - lag
            Z = np.full_like(X, np.nan)
            Z[self.lag :] = X[: -self.lag]
            X = Z
        return X

    def _state(self, vel: np.ndarray, pos: np.ndarray | None) -> np.ndarray:
        vel = np.asarray(vel, dtype=np.float64)
        if pos is None:
            pos = np.cumsum(vel, axis=0) * self.dt
        cols = [pos, vel]
        if self.use_acc:
            cols.append(np.gradient(vel, self.dt, axis=0))
        cols.append(np.ones((vel.shape[0], 1)))
        return np.hstack(cols)

    # -- API
    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, pos_train: np.ndarray | None = None) -> "KalmanDecoder":
        """Y_train = velocity (n, 2); pos_train optional (else integrated from velocity)."""
        S = self._state(Y_train, pos_train)
        Z = self._obs(X_train)
        ok = ~np.isnan(Z).any(axis=1)
        d = S.shape[1]
        kin = slice(0, d - 1)

        # dynamics: kinematic part by least squares, constant row fixed
        X0, X1 = S[:-1], S[1:]
        Ak = np.linalg.lstsq(X0, X1[:, kin], rcond=None)[0].T  # (d-1, d)
        A = np.zeros((d, d))
        A[kin] = Ak
        A[-1, -1] = 1.0
        R = X1[:, kin] - X0 @ Ak.T
        W = np.zeros((d, d))
        W[kin, kin] = R.T @ R / (len(R) - d)

        # observation model
        H = np.linalg.lstsq(S[ok], Z[ok], rcond=None)[0].T  # (n_ch, d)
        E = Z[ok] - S[ok] @ H.T
        Q = E.T @ E / (ok.sum() - d)
        Q += 1e-6 * np.eye(Q.shape[0])  # silent channels -> keep Q invertible

        self.A_, self.W_, self.H_, self.Q_ = A, W, H, Q
        self.x0_ = S.mean(axis=0)
        P0 = np.zeros((d, d))
        P0[kin, kin] = np.cov(S[:, kin].T)
        self.P0_ = P0
        self.vel_idx_ = slice(2, 4)
        return self

    def filter(self, X_test: np.ndarray) -> np.ndarray:
        """Run predict/update over the sequence; NaN rows -> prediction only. Returns states."""
        Z = self._obs(X_test)
        A, W, H, Q = self.A_, self.W_, self.H_, self.Q_
        x, P = self.x0_.copy(), self.P0_.copy()
        out = np.empty((Z.shape[0], x.size))
        for t in range(Z.shape[0]):
            # predict
            x = A @ x
            P = A @ P @ A.T + W
            z = Z[t]
            if not np.isnan(z).any():
                # update
                S = H @ P @ H.T + Q
                K = np.linalg.solve(S, H @ P).T  # P H' S^-1 (S, P symmetric)
                x = x + K @ (z - H @ x)
                P = P - K @ H @ P
            out[t] = x
        return out

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Decoded velocity (n, 2)."""
        return self.filter(X_test)[:, self.vel_idx_]
