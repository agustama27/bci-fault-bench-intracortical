"""Loader for O'Doherty et al. (2017) sessions (MATLAB v7.3 / HDF5).

Turns one ``indy_*.mat`` / ``loco_*.mat`` file into:

* ``counts``  (n_bins, n_channels) spike counts per bin, one column per electrode
  (all units of the electrode summed);
* ``vel``     (n_bins, 2) cursor velocity in mm/s (average over the bin);
* ``pos``     (n_bins, 2) cursor position in mm at the bin centre;
* ``t``       (n_bins,)  bin-centre times in s (session clock).

Unit convention of the dataset: ``spikes`` is a (5, n_chan) cell. Row 0 is the
unsorted "hash" unit, rows 1..4 are sorted units. Empty cells come back from
h5py as ``uint64 [0, 0]`` with the ``MATLAB_empty`` attribute.

Hash choice: by default hash IS included (``include_hash=True``). Rationale:
real-time intracortical BCIs very often decode from threshold crossings per
electrode without spike sorting, and summing all units per electrode is the
closest offline analogue. ``--no-include-hash`` keeps only sorted units.

Usage::

    python -m icbench.data --mat data/raw/indy_20161005_06.mat --bin-ms 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class Session:
    counts: np.ndarray  # (n_bins, n_channels) int
    vel: np.ndarray  # (n_bins, 2) mm/s
    pos: np.ndarray  # (n_bins, 2) mm
    t: np.ndarray  # (n_bins,) s, bin centres
    chan_names: list[str]
    bin_s: float
    include_hash: bool
    n_units: int  # non-empty units actually summed

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            counts=self.counts,
            vel=self.vel,
            pos=self.pos,
            t=self.t,
            chan_names=np.array(self.chan_names),
            bin_s=self.bin_s,
            include_hash=self.include_hash,
            n_units=self.n_units,
        )
        return path


def load_npz(path: str | Path) -> Session:
    d = np.load(path, allow_pickle=False)
    return Session(
        counts=d["counts"],
        vel=d["vel"],
        pos=d["pos"],
        t=d["t"],
        chan_names=[str(c) for c in d["chan_names"]],
        bin_s=float(d["bin_s"]),
        include_hash=bool(d["include_hash"]),
        n_units=int(d["n_units"]),
    )


def _read_cell(f: h5py.File, ref) -> np.ndarray:
    """Read one MATLAB cell entry; empty cells -> empty float array."""
    ds = f[ref]
    if ds.attrs.get("MATLAB_empty", 0) or ds.dtype == np.uint64:
        return np.empty(0, dtype=np.float64)
    return np.asarray(ds[()], dtype=np.float64).ravel()


def _read_str(f: h5py.File, ref) -> str:
    return "".join(chr(int(c)) for c in f[ref][()].ravel())


def load_mat(path: str | Path, bin_ms: float = 20.0, include_hash: bool = True) -> Session:
    """Bin spikes and compute cursor kinematics on a regular bin grid."""
    bin_s = bin_ms / 1000.0
    with h5py.File(path, "r") as f:
        t_raw = f["t"][()].ravel()
        cursor = f["cursor_pos"][()].T  # (n_samples, 2)
        spk_refs = f["spikes"][()]  # (n_units, n_chan) object refs
        chan_names = [_read_str(f, r) for r in f["chan_names"][()].ravel()]

        t0, t1 = float(t_raw[0]), float(t_raw[-1])
        n_bins = int(np.floor((t1 - t0) / bin_s + 1e-9))
        edges = t0 + bin_s * np.arange(n_bins + 1)

        n_unit_rows, n_chan = spk_refs.shape
        first_unit = 0 if include_hash else 1
        counts = np.zeros((n_bins, n_chan), dtype=np.int32)
        n_units = 0
        for ch in range(n_chan):
            for u in range(first_unit, n_unit_rows):
                st = _read_cell(f, spk_refs[u, ch])
                if st.size == 0:
                    continue
                n_units += 1
                # clip to the kinematics time range (some spikes precede t[0])
                st = st[(st >= edges[0]) & (st < edges[-1])]
                counts[:, ch] += np.histogram(st, bins=edges)[0].astype(np.int32)

    # cursor position resampled at bin edges -> average velocity per bin
    pos_edges = np.column_stack([np.interp(edges, t_raw, cursor[:, k]) for k in range(2)])
    vel = np.diff(pos_edges, axis=0) / bin_s
    centres = edges[:-1] + bin_s / 2
    pos = np.column_stack([np.interp(centres, t_raw, cursor[:, k]) for k in range(2)])

    return Session(
        counts=counts,
        vel=vel,
        pos=pos,
        t=centres,
        chan_names=chan_names,
        bin_s=bin_s,
        include_hash=include_hash,
        n_units=n_units,
    )


def summary(s: Session) -> str:
    rate = s.counts.mean(axis=0) / s.bin_s
    speed = np.linalg.norm(s.vel, axis=1)
    lines = [
        f"n_bins={s.counts.shape[0]}  n_channels={s.counts.shape[1]}  bin={s.bin_s * 1000:.0f} ms  "
        f"duration={s.counts.shape[0] * s.bin_s:.1f} s",
        f"units summed={s.n_units}  include_hash={s.include_hash}  total spikes={int(s.counts.sum())}",
        f"rate per channel (Hz): mean={rate.mean():.2f}  median={np.median(rate):.2f}  "
        f"min={rate.min():.2f}  max={rate.max():.2f}  silent channels={int((rate == 0).sum())}",
        f"velocity (mm/s): mean x={s.vel[:, 0].mean():.2f} y={s.vel[:, 1].mean():.2f}  "
        f"std x={s.vel[:, 0].std():.2f} y={s.vel[:, 1].std():.2f}  "
        f"speed median={np.median(speed):.1f} p95={np.percentile(speed, 95):.1f}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mat", default="data/raw/indy_20161005_06.mat")
    p.add_argument("--bin-ms", type=float, default=20.0)
    p.add_argument(
        "--include-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include unit 0 (unsorted hash) in the per-electrode sum (default: yes)",
    )
    p.add_argument("--out", default=None, help="output .npz (default: data/processed/<stem>_bin<ms>ms.npz)")
    a = p.parse_args(argv)

    s = load_mat(a.mat, bin_ms=a.bin_ms, include_hash=a.include_hash)
    suffix = "" if a.include_hash else "_nohash"
    out = a.out or f"data/processed/{Path(a.mat).stem}_bin{a.bin_ms:g}ms{suffix}.npz"
    s.save(out)
    print(summary(s))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
