"""Marked regional ETAS, likelihood fitting, compensators and branching forecasts."""

import numpy as np
from scipy.optimize import minimize
from numba import njit

MC = 3.0
MUP = 8.5
MEMORY = 365.0


@njit(cache=True)
def gcdf(x, c, p, L=MEMORY):
    x = min(max(x, 0.0), L)
    return (1 - (c / (c + x)) ** (p - 1)) / (1 - (c / (c + L)) ** (p - 1))


@njit(cache=True)
def gdensity(x, c, p, L=MEMORY):
    if x < 0 or x > L:
        return 0.0
    return (p - 1) * c ** (p - 1) / (1 - (c / (c + L)) ** (p - 1)) * (x + c) ** (-p)


def unpack(z, beta):
    mu = np.exp(z[0])
    n = z[1]
    alpha = z[2] * beta
    c = np.exp(z[3])
    p = z[4]
    d = MUP - MC
    em = (
        beta
        / (beta - alpha)
        * (1 - np.exp(-(beta - alpha) * d))
        / (1 - np.exp(-beta * d))
    )
    K = n / em
    return np.array([mu, K, alpha, c, p, n, beta])


@njit(cache=True)
def nll_core(params, t, m, start, end):
    mu, K, alpha, c, p = params[:5]
    total = mu * (end - start)
    left = 0
    for i in range(len(t)):
        if t[i] >= end:
            break
        prod = K * np.exp(alpha * (m[i] - MC))
        total += prod * (gcdf(end - t[i], c, p) - gcdf(start - t[i], c, p))
        if t[i] < start:
            continue
        while t[i] - t[left] > MEMORY:
            left += 1
        lam = mu
        for j in range(left, i):
            lam += K * np.exp(alpha * (m[j] - MC)) * gdensity(t[i] - t[j], c, p)
        total -= np.log(max(lam, 1e-15))
    return total


def fit(t, m, start, end, warm=None):
    sel = (t >= start - MEMORY) & (t < end)
    tt = t[sel]
    mm = m[sel]
    ms = mm[tt >= start]
    if len(ms) == 0:
        # The ordinary likelihood has no magnitude information in a fully quiet window.
        prior = np.asarray(m)[np.asarray(t) < end][-100:]
        beta = float(np.clip(1 / max(float(np.mean(prior) - (MC - 0.05)), .1), .7, 4.0)) if len(prior) else np.log(10.)
        z = np.array([np.log(.5 / (end - start)), .005, .65, np.log(.03), 1.2])
        par = unpack(z, beta)
        return {"params": par.tolist(), "z": z.tolist(), "nll": float(nll_core(par, tt, mm, start, end)), "success": True, "message": "Zero-event fallback: Jeffreys half-event background; minimum branching; historical magnitude prior", "events": 0, "start": float(start), "end": float(end), "iterations": 0}
    beta = 1 / max(float(np.mean(ms) - (MC - 0.05)), 0.1)
    beta = np.clip(beta, 0.7, 4.0)
    rate = len(ms) / (end - start)
    bounds = [
        (np.log(max(rate * 0.001, 1e-6)), np.log(max(rate * 3, 0.01))),
        (0.005, 0.97),
        (0.1, 0.95),
        (np.log(0.001), np.log(3.0)),
        (1.01, 2.2),
    ]
    starts = [
        np.array([np.log(max(rate * 0.5, 1e-5)), 0.5, 0.65, np.log(0.03), 1.2]),
        np.array([np.log(max(rate * 0.3, 1e-5)), 0.7, 0.4, np.log(0.1), 1.35]),
    ]
    if warm is not None:
        starts.insert(0, np.array(warm))
    best = None
    for z in starts:
        z = np.array([np.clip(a, *b) for a, b in zip(z, bounds)])
        r = minimize(
            lambda v: objective_gradient(v, beta, tt, mm, start, end),
            z,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 180, "ftol": 1e-9, "maxls": 30},
        )
        if best is None or r.fun < best.fun:
            best = r
    return {
        "params": unpack(best.x, beta).tolist(),
        "z": best.x.tolist(),
        "nll": float(best.fun),
        "success": bool(best.success),
        "message": str(best.message),
        "events": len(ms),
        "start": float(start),
        "end": float(end),
        "iterations": int(best.nit),
    }


@njit(cache=True)
def compensator(params, t, m, a, b):
    mu, K, alpha, c, p = params[:5]
    v = mu * (b - a)
    for i in range(len(t)):
        if t[i] >= b:
            break
        if t[i] < a - MEMORY:
            continue
        v += (
            K
            * np.exp(alpha * (m[i] - MC))
            * (gcdf(b - t[i], c, p) - gcdf(a - t[i], c, p))
        )
    return v


@njit(cache=True)
def cluster_survival(params, lower, upper, horizon=30.0, dt=0.05, dm=0.025):
    """Probability a fresh random-magnitude cluster avoids the target band.
    Piecewise-constant quadrature includes all offspring generations.
    """
    mu, K, alpha, c, p, n, beta = params
    N = int(round(horizon / dt))
    J = int(round((MUP - MC) / dm))
    weights = np.empty(J)
    kap = np.empty(J)
    keep = np.empty(J)
    norm = 1 - np.exp(-beta * (MUP - MC))
    for j in range(J):
        lo = MC + j * dm
        hi = lo + dm
        center = (lo + hi) / 2
        weights[j] = (np.exp(-beta * (lo - MC)) - np.exp(-beta * (hi - MC))) / norm
        kap[j] = K * np.exp(alpha * (center - MC))
        overlap_lo = max(lo, lower)
        overlap_hi = min(hi, upper)
        inband = 0.0
        if overlap_hi > overlap_lo:
            inband = (
                np.exp(-beta * (overlap_lo - MC)) - np.exp(-beta * (overlap_hi - MC))
            ) / norm
        keep[j] = max(weights[j] - inband, 0.0)
    w = np.empty(N)
    for j in range(N):
        w[j] = gcdf((j + 1) * dt, c, p) - gcdf(j * dt, c, p)
    q = np.empty(N + 1)
    q[0] = np.sum(keep)
    for k in range(1, N + 1):
        h = 0.0
        for j in range(k):
            h += w[j] * (1 - q[k - 1 - j])
        v = 0.0
        for j in range(J):
            v += keep[j] * np.exp(-kap[j] * h)
        q[k] = v
    return q


@njit(cache=True)
def forecast_from_q(params, t, m, origin, q, dt=0.05):
    mu, K, alpha, c, p = params[:5]
    N = len(q) - 1
    z = 0.0
    for j in range(N):
        lo = j * dt
        hi = (j + 1) * dt
        a = mu * dt
        for i in range(len(t)):
            age = origin - t[i]
            if age <= 0 or age >= MEMORY:
                continue
            a += (
                K
                * np.exp(alpha * (m[i] - MC))
                * (gcdf(age + hi, c, p) - gcdf(age + lo, c, p))
            )
        x = 1 - 0.5 * (q[N - j] + q[N - j - 1])
        z += a * x
    return -np.expm1(-z)


@njit(cache=True)
def expected_future(params, t, m, origin, horizon=30.0, dt=0.1):
    """Mean all-magnitude rate using the branching resolvent."""
    mu, K, alpha, c, p, n, beta = params
    N = int(horizon / dt)
    out = np.empty(N)
    w = np.empty(N)
    for j in range(N):
        w[j] = gcdf((j + 1) * dt, c, p) - gcdf(j * dt, c, p)
    for j in range(N):
        a = mu * dt
        for i in range(len(t)):
            age = origin - t[i]
            if 0 < age < MEMORY:
                a += (
                    K
                    * np.exp(alpha * (m[i] - MC))
                    * (gcdf(age + (j + 1) * dt, c, p) - gcdf(age + j * dt, c, p))
                )
        for k in range(j):
            a += n * w[j - 1 - k] * out[k]
        out[j] = a
    return out.sum()


@njit(cache=True)
def cdf_derivatives(x, c, p):
    x = min(max(x, 0.0), MEMORY)
    u = (c / (c + MEMORY)) ** (p - 1)
    v = (c / (c + x)) ** (p - 1)
    Z = 1 - u
    ul = u * (p - 1) * MEMORY / (c + MEMORY)
    vl = v * (p - 1) * x / (c + x)
    up = u * np.log(c / (c + MEMORY))
    vp = v * np.log(c / (c + x))
    return (
        (1 - v) / Z,
        (-vl * Z + (1 - v) * ul) / (Z * Z),
        (-vp * Z + (1 - v) * up) / (Z * Z),
    )


@njit(cache=True)
def objective_gradient(z, beta, t, m, start, end):
    mu = np.exp(z[0])
    n = z[1]
    alpha = z[2] * beta
    c = np.exp(z[3])
    p = z[4]
    D = MUP - MC
    ba = beta - alpha
    E = beta / ba * (1 - np.exp(-ba * D)) / (1 - np.exp(-beta * D))
    K = n / E
    dlogE = 1 / ba - D / np.expm1(ba * D)
    prod = K * np.exp(alpha * (m - MC))
    prod_alpha = prod * (m - MC - dlogE) * beta
    total = mu * (end - start)
    grad = np.zeros(5)
    grad[0] = total
    u = (c / (c + MEMORY)) ** (p - 1)
    Z = 1 - u
    A = (p - 1) * c ** (p - 1) / Z
    dc = (p - 1) + u * (p - 1) * MEMORY / (c + MEMORY) / Z
    dp = 1 / (p - 1) + np.log(c) + u * np.log(c / (c + MEMORY)) / Z
    left = 0
    for i in range(len(t)):
        if t[i] >= end:
            break
        fe, ce, pe = cdf_derivatives(end - t[i], c, p)
        fs, cs, ps = cdf_derivatives(start - t[i], c, p)
        integral = prod[i] * (fe - fs)
        total += integral
        grad[1] += integral / n
        grad[2] += prod_alpha[i] * (fe - fs)
        grad[3] += prod[i] * (ce - cs)
        grad[4] += prod[i] * (pe - ps)
        if t[i] < start:
            continue
        while t[i] - t[left] > MEMORY:
            left += 1
        lam = mu
        dl = np.zeros(5)
        dl[0] = mu
        for j in range(left, i):
            delta = t[i] - t[j]
            if delta <= 0:
                continue
            logtc = np.log(delta + c)
            v = prod[j] * A * np.exp(-p * logtc)
            lam += v
            dl[1] += v / n
            dl[2] += v * (m[j] - MC - dlogE) * beta
            dl[3] += v * (dc - p * c / (delta + c))
            dl[4] += v * (dp - logtc)
        total -= np.log(lam)
        grad -= dl / lam
    return total, grad
