"""Construct strictly historical features and 30-day baseline forecasts."""

import os

for k in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[k] = "1"
from pathlib import Path
import json, time, concurrent.futures
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
from etas import fit, compensator, cluster_survival, forecast_from_q, MC, MUP

ROOT = Path(__file__).resolve().parents[1]
EPOCH = pd.Timestamp("1990-01-01", tz="UTC")


def day(date):
    return (pd.Timestamp(date, tz="UTC") - EPOCH).total_seconds() / 86400


BANDS = [("ge35", 3.5, 8.5), ("band35_45", 3.5, 4.5), ("ge45", 4.5, 8.5)]


def bvalue(m):
    return np.log10(np.e) / max(np.mean(m) - (MC - 0.05), 0.08) if len(m) >= 20 else np.nan


def feature_vector(hist, o, cell, par):
    t = hist.t_days.to_numpy()
    m = hist.magnitude.to_numpy()
    d = hist.depth_km.to_numpy()
    lat = hist.latitude.to_numpy()
    lon = hist.longitude.to_numpy()
    f = {}
    idx = {w: np.searchsorted(t, o - w, "left") for w in [7, 30, 90, 365, 1095]}
    for w, j in idx.items():
        f["n" + str(w)] = len(t) - j
    last = min(50, len(t))
    recent = m[idx[30] :]
    rdepth = d[idx[30] :]
    f.update(
        last_latitude=lat[-1],
        last_longitude=lon[-1],
        mean_depth30=(
            float(np.mean(rdepth)) if len(rdepth) else float(np.mean(d[-last:]))
        ),
        magnitude_last=m[-1],
        time_since_last=o - t[-1],
        b100=bvalue(m[-100:]),
        max_magnitude7=float(np.max(m[idx[7] :])) if f["n7"] else MC,
        mag_sd30=float(np.std(recent)) if len(recent) > 1 else 0.0,
        span50=t[-1] - t[-last],
        mean_interval50=float(np.mean(np.diff(t[-last:]))),
        cv_interval50=float(
            np.std(np.diff(t[-last:])) / max(np.mean(np.diff(t[-last:])), 1e-5)
        ),
    )
    for k in range(5):
        s0 = len(m) - 2 * k
        s1 = s0 - 2
        f[f"db{k}"] = bvalue(m[max(0, s0 - 100) : s0]) - bvalue(
            m[max(0, s1 - 100) : s1]
        )
    window = m[-100:]
    bb = bvalue(window)
    xs = np.arange(MC, float(np.max(window)) + 0.001, 0.1)
    ys = np.array([(window >= x - 1e-8).sum() for x in xs])
    valid = ys > 0
    if valid.sum() > 2:
        reg = np.polyfit(xs[valid], np.log10(ys[valid]), 1)
        f["gr_residual"] = float(
            np.mean((np.log10(ys[valid]) - np.polyval(reg, xs[valid])) ** 2)
        )
    else:
        f["gr_residual"] = 0.0
    f["magnitude_excess"] = (
        float(np.max(window) - (MC + np.log10(len(window)) / bb))
        if np.isfinite(bb)
        else 0.0
    )
    f["log_sqrt_energy30"] = (
        float(0.5 * np.log10(np.sum(10 ** (1.5 * recent)))) if len(recent) else 0.0
    )
    xy = np.column_stack(
        [
            (lon[-last:] - cell["center_lon"])
            * 111
            * np.cos(np.deg2rad(cell["center_lat"])),
            (lat[-last:] - cell["center_lat"]) * 111,
        ]
    )
    A = (squareform(pdist(xy)) <= 10).astype(float)
    np.fill_diagonal(A, 0)
    degs = A.sum(1)
    f["clustering50"] = float(np.trace(A @ A @ A) / max(np.sum(degs * (degs - 1)), 1))
    f["spatial_radius50"] = float(
        np.sqrt(np.mean(np.sum((xy - xy.mean(0)) ** 2, axis=1)))
    )
    ent = {}
    for w in [30, 90, 365]:
        j = idx[w]
        gx = np.clip(((lon[j:] - cell["lon_min"]) / (cell.get("grid_degrees", 1) / 4)).astype(int), 0, 3)
        gy = np.clip(((lat[j:] - cell["lat_min"]) / (cell.get("grid_degrees", 1) / 4)).astype(int), 0, 3)
        cnt = np.bincount(gy * 4 + gx, minlength=16).astype(float)
        prob = cnt[cnt > 0] / max(cnt.sum(), 1)
        ent[w] = float(-np.sum(prob * np.log(prob)) / np.log(16))
    f["spatial_entropy90"] = ent[90]
    f["entropy_change"] = ent[30] - ent[365]
    f["depth_change"] = (
        float(np.median(rdepth) - np.median(d[idx[365] :]))
        if len(rdepth) and f["n365"]
        else 0.0
    )
    for w in [30, 90, 365]:
        expected = compensator(par, t, m, o - w, o)
        f[f"expected{w}"] = expected
        f[f"innovation{w}"] = float(
            np.arcsinh((f[f"n{w}"] - expected) / np.sqrt(expected + 1))
        )
    f["residual_persistence"] = (
        float(
            np.sign(f["innovation90"] + f["innovation365"])
            * min(abs(f["innovation90"]), abs(f["innovation365"]))
        )
        if f["innovation90"] * f["innovation365"] > 0
        else 0.0
    )
    f["distributed_residual"] = f["innovation90"] * ent[90]
    f["rate_acceleration"] = float(
        np.log((f["n30"] + 0.5) / (30 * (f["n365"] + 0.5) / 365))
    )
    f["b365"] = bvalue(m[idx[365] :])
    dm = np.diff(m[-200:])
    pos = dm[dm >= 0.1 - 1e-8]
    f["b_positive"] = (
        float(np.log10(np.e) / (pos.mean() - 0.05)) if len(pos) >= 20 else np.nan
    )
    f["size_adjusted_activity"] = float(
        np.log((f["n90"] + 0.5) / 90)
        - np.log(10) * (f["b365"] if np.isfinite(f["b365"]) else bb)
    )
    return f

