from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm, gamma as gamma_dist, ncx2

def cir_moments(v_t, kappa, theta, xi, dt):
    e = np.exp(-kappa * dt)
    m = theta + (v_t - theta) * e
    s2 = (v_t * xi ** 2 * e / kappa) * (1 - e) + (theta * xi ** 2 / (2 * kappa)) * (1 - e) ** 2
    return m, s2


def cir_stationary_params(kappa, theta, xi):
    return 2 * kappa * theta / xi ** 2, xi ** 2 / (2 * kappa)


def qe_step(v_t, kappa, theta, xi, dt, rng, psi_c=1.5):
    m, s2 = cir_moments(v_t, kappa, theta, xi, dt)
    if m <= 0:
        return 0.0
    psi = s2 / (m * m)
    if psi <= psi_c:                                  
        inv = 1.0 / psi
        b2 = 2 * inv - 1 + np.sqrt(2 * inv) * np.sqrt(max(2 * inv - 1, 0.0))
        a = m / (1 + b2)
        Z = rng.standard_normal()
        return float(a * (np.sqrt(b2) + Z) ** 2)

    p = (psi - 1) / (psi + 1)
    beta = (1 - p) / m
    U = rng.random()
    return 0.0 if U <= p else float(np.log((1 - p) / (1 - U)) / beta)


def exact_cir_step(v_t, kappa, theta, xi, dt, rng):
    if dt <= 0:
        return float(v_t)
    d = 4 * kappa * theta / xi ** 2
    e = np.exp(-kappa * dt)
    c = xi ** 2 * (1 - e) / (4 * kappa)
    lam = v_t * e / c
    return float(c * ncx2.rvs(df=d, nc=lam, random_state=rng))

class HestonState:
    def __init__(self, cfg, rng, v0, S0=None):
        self.cfg = cfg
        self.rng = rng
        self.v = float(v0)
        self.S = float(S0 if S0 is not None else cfg.S0)

    @staticmethod
    def sample_v0(cfg, rng):
        if cfg.v0_mode == "fixed":
            return float(cfg.v0_fixed)
        shape, scale = cir_stationary_params(cfg.kappa, cfg.theta, cfg.xi)
        return float(gamma_dist.rvs(a=shape, scale=scale, random_state=rng))

    def advance_qe(self, dt):
        if dt <= 0:
            return
        cfg = self.cfg
        v_t = self.v
        if cfg.variance_scheme == "exact":
            v_next = exact_cir_step(v_t, cfg.kappa, cfg.theta, cfg.xi, dt, self.rng)
        else:
            v_next = qe_step(v_t, cfg.kappa, cfg.theta, cfg.xi, dt, self.rng, cfg.psi_c)
        int_v = 0.5 * (v_t + v_next) * dt                
        kappa, theta, xi, rho = cfg.kappa, cfg.theta, cfg.xi, cfg.rho
        stoch_v = (v_next - v_t - kappa * theta * dt + kappa * int_v) / xi  
        Zp = self.rng.standard_normal()
        d_logS = ((cfg.r - cfg.q) * dt - 0.5 * int_v
                  + rho * stoch_v
                  + np.sqrt(max(1 - rho ** 2, 0.0)) * np.sqrt(max(int_v, 0.0)) * Zp)
        self.S = float(self.S * np.exp(d_logS))
        self.v = float(max(v_next, 0.0))

class TransientImpact:
    def __init__(self, g, decay):
        self.g = float(g)
        self.decay = float(decay)
        self._acc = 0.0
        self._t = 0.0

    def update(self, t):
        if t > self._t:
            self._acc *= np.exp(-self.decay * (t - self._t))
            self._t = t

    def add_market_order(self, signed_vol, t):
        self.update(t)
        self._acc += signed_vol

    def value(self):
        return self.g * self._acc


def bs_price(S, K, tau, sigma, r=0.0, q=0.0, kind="call"):
    S, K, tau, sigma = map(np.asarray, (S, K, tau, sigma))
    tau = np.maximum(tau, 1e-12); sigma = np.maximum(sigma, 1e-12)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    disc_S = S * np.exp(-q * tau); disc_K = K * np.exp(-r * tau)
    call = disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    return call if kind == "call" else call - disc_S + disc_K


def bs_delta(S, K, tau, sigma, r=0.0, q=0.0, kind="call"):
    S, K, tau, sigma = map(np.asarray, (S, K, tau, sigma))
    tau = np.maximum(tau, 1e-12); sigma = np.maximum(sigma, 1e-12)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    dl = np.exp(-q * tau) * norm.cdf(d1)
    return dl if kind == "call" else dl - np.exp(-q * tau)


def bs_implied_vol(price, S, K, tau, r=0.0, q=0.0, kind="call", tol=1e-7, it=60):
    tau = max(float(tau), 1e-9)
    intrinsic = max((S - K) * np.exp(-q * tau) if kind == "call"
                    else (K - S) * np.exp(-r * tau), 0.0)
    if price <= intrinsic + 1e-12:
        return 1e-6
    lo, hi, sig = 1e-6, 5.0, 0.2
    for _ in range(it):
        p = float(bs_price(S, K, tau, sig, r, q, kind)); diff = p - price
        if abs(diff) < tol:
            return sig
        d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * tau) / (sig * np.sqrt(tau))
        vega = S * np.exp(-q * tau) * norm.pdf(d1) * np.sqrt(tau)
        if vega < 1e-10:
            break
        hi, lo = (sig, lo) if diff > 0 else (hi, sig)
        sig_new = sig - diff / vega
        if not (lo < sig_new < hi):
            sig_new = 0.5 * (lo + hi)
        sig = sig_new
    return sig


class HestonPricer:

    LADDER = ((96, 300.0), (128, 1200.0))
    U_DECAY_C = 8.0
    W_BS = 6.0e-5
    W_CF = 1.6e-4

    def __init__(self, kappa, theta, xi, rho, r=0.0, q=0.0, grid_n=64, grid_u=100.0):
        self.kappa, self.theta, self.xi, self.rho = kappa, theta, xi, rho
        self.r, self.q = r, q
        self.grid_n0, self.grid_u0 = int(grid_n), float(grid_u)
        top_u = max([u for _n, u in self.LADDER] + [float(grid_u)])
        u_req_max = self.U_DECAY_C / np.sqrt(self.W_BS)
        if u_req_max > top_u + 1e-9:
            raise ValueError(
                f"HestonPricer ladder too short for W_BS={self.W_BS:g}: u_req at the "
                f"BS/CF floor is {u_req_max:.0f} > top rung u_max={top_u:.0f}; raise "
                f"the ladder or keep W_BS >= (U_DECAY_C/top_u)^2 = "
                f"{(self.U_DECAY_C / top_u) ** 2:.2e}")
        self._unit = {}
        x, w = self._leggauss(grid_n)
        self.u = 0.5 * grid_u * (x + 1.0)
        self.w = 0.5 * grid_u * w

    def _leggauss(self, n):
        if n not in self._unit:
            self._unit[n] = leggauss(n)
        return self._unit[n]

    def _grid(self, n, u_max):
        x, w = self._leggauss(n)
        return 0.5 * u_max * (x + 1.0), 0.5 * u_max * w

    def wbar(self, tau, v):
        tau = np.asarray(tau, float)
        return self.theta * tau + (float(v) - self.theta) * \
            (1.0 - np.exp(-self.kappa * tau)) / self.kappa

    def _Pj(self, phi, x_logS, lnK, tau, v, j):
        kappa, theta, xi, rho, r, q = (self.kappa, self.theta, self.xi,
                                       self.rho, self.r, self.q)
        i = 1j
        u = 0.5 if j == 1 else -0.5
        b = (kappa - rho * xi) if j == 1 else kappa
        a = kappa * theta
        rspr = rho * xi * phi * i
        d = np.sqrt((rspr - b) ** 2 - xi ** 2 * (2 * u * phi * i - phi ** 2))
        g = (b - rspr - d) / (b - rspr + d)            # Albrecher trap-free form
        exp_dt = np.exp(-d * tau)
        C = ((r - q) * phi * i * tau
             + (a / xi ** 2) * ((b - rspr - d) * tau
                                - 2.0 * np.log((1 - g * exp_dt) / (1 - g))))
        D = ((b - rspr - d) / xi ** 2) * ((1 - exp_dt) / (1 - g * exp_dt))
        f = np.exp(C + D * v + i * phi * x_logS)
        return np.real(np.exp(-i * phi * lnK) * f / (i * phi))

    def _cf_call(self, Sf, Kf, tauf, v, u, w):
        x_logS = np.log(Sf)[None, :]; lnK = np.log(Kf)[None, :]
        tauf2 = tauf[None, :]; phi = u[:, None]; ww = w[:, None]
        P1 = 0.5 + (1.0 / np.pi) * np.sum(ww * self._Pj(phi, x_logS, lnK, tauf2, v, 1), axis=0)
        P2 = 0.5 + (1.0 / np.pi) * np.sum(ww * self._Pj(phi, x_logS, lnK, tauf2, v, 2), axis=0)
        disc_S = Sf * np.exp(-self.q * tauf); disc_K = Kf * np.exp(-self.r * tauf)
        return disc_S * P1 - disc_K * P2

    def price(self, S, K, tau, v, kind="call"):
        v = float(v)
        S = np.asarray(S, float); K = np.asarray(K, float); tau = np.asarray(tau, float)
        Sb, Kb, taub = np.broadcast_arrays(S, K, tau)
        shape = Sb.shape
        Sf, Kf, tauf = (np.atleast_1d(Sb).ravel(), np.atleast_1d(Kb).ravel(),
                        np.atleast_1d(taub).ravel())
        wb = np.maximum(self.wbar(tauf, v), 1e-14)
        blend = np.clip((wb - self.W_BS) / (self.W_CF - self.W_BS), 0.0, 1.0)
        call = np.zeros_like(Sf)

        bs_sel = blend < 1.0
        if bs_sel.any():
            sig_eff = np.sqrt(wb[bs_sel] / np.maximum(tauf[bs_sel], 1e-12))
            bsv = np.asarray(bs_price(Sf[bs_sel], Kf[bs_sel],
                                      np.maximum(tauf[bs_sel], 1e-12),
                                      sig_eff, self.r, self.q, "call"), float)
            call[bs_sel] = (1.0 - blend[bs_sel]) * bsv

        cf_sel = blend > 0.0
        if cf_sel.any():
            idx = np.nonzero(cf_sel)[0]
            u_req = self.U_DECAY_C / np.sqrt(wb[idx])
            rungs = ((self.grid_n0, self.grid_u0),) + self.LADDER
            level = np.full(idx.shape, len(rungs) - 1, int)
            for li in range(len(rungs) - 2, -1, -1):
                level[u_req <= rungs[li][1]] = li
            for li in range(len(rungs)):
                m = level == li
                if not m.any():
                    continue
                j = idx[m]
                u, w = self._grid(*rungs[li])
                call[j] += blend[j] * self._cf_call(Sf[j], Kf[j], tauf[j], v, u, w)

        disc_S = Sf * np.exp(-self.q * tauf); disc_K = Kf * np.exp(-self.r * tauf)
        call = np.maximum(call, np.maximum(disc_S - disc_K, 0.0))   # intrinsic floor
        out = call.reshape(shape)
        if kind == "call":
            return out if shape else float(out)
        put = out - (Sb * np.exp(-self.q * taub)) + (Kb * np.exp(-self.r * taub))
        return put if shape else float(put)

    def _call_chain_greeks(self, S, Kvec, tauvec, v):
        Kvec = np.asarray(Kvec, float); tauvec = np.asarray(tauvec, float)
        hS = max(1e-3 * S, 1e-3)
        sig = np.sqrt(max(v, 1e-12)); hsig = 1e-3
        v_up, v_dn = (sig + hsig) ** 2, (sig - hsig) ** 2
        htau = np.maximum(1e-4, 1e-3 * tauvec)
        P = lambda SS, vv, tt: self.price(SS, Kvec, tt, vv, "call")
        p0 = P(S, v, tauvec)
        pS_up, pS_dn = P(S + hS, v, tauvec), P(S - hS, v, tauvec)
        pv_up, pv_dn = P(S, v_up, tauvec), P(S, v_dn, tauvec)
        pt_up = P(S, v, tauvec + htau); pt_dn = P(S, v, np.maximum(tauvec - htau, 1e-9))
        cuu, cud = P(S + hS, v_up, tauvec), P(S + hS, v_dn, tauvec)
        cdu, cdd = P(S - hS, v_up, tauvec), P(S - hS, v_dn, tauvec)
        return dict(price=p0, delta=(pS_up - pS_dn) / (2 * hS),
                    gamma=(pS_up - 2 * p0 + pS_dn) / (hS ** 2),
                    vega=(pv_up - pv_dn) / (2 * hsig),
                    vanna=(cuu - cud - cdu + cdd) / (4 * hS * hsig),
                    theta=-(pt_up - pt_dn) / (2 * htau))

    def price_chain(self, S_star, v, instruments):
        live = [(nm, kind, K, tau) for (nm, kind, K, tau) in instruments]
        keys = sorted({(K, tau) for (_, _, K, tau) in live})
        if not keys:
            return {}
        Kv = np.array([k for k, _ in keys]); tv = np.array([t for _, t in keys])
        calls = self.price(S_star, Kv, tv, v, "call")
        pos = {kt: i for i, kt in enumerate(keys)}
        eqt = np.exp(-self.q * tv); ert = np.exp(-self.r * tv)
        out = {}
        for nm, kind, K, tau in live:
            i = pos[(K, tau)]
            c = float(calls[i])
            out[nm] = c if kind == "call" else c - S_star * eqt[i] + K * ert[i]
        return out

    def greeks_chain(self, S_star, v, instruments):
        keys = sorted({(K, tau) for (_, _, K, tau) in instruments})
        if not keys:
            return {}
        Kv = np.array([k for k, _ in keys]); tv = np.array([t for _, t in keys])
        ch = self._call_chain_greeks(S_star, Kv, tv, v)
        pos = {kt: i for i, kt in enumerate(keys)}
        eqt = np.exp(-self.q * tv); ert = np.exp(-self.r * tv)
        out = {}
        for nm, kind, K, tau in instruments:
            i = pos[(K, tau)]
            if kind == "call":
                out[nm] = dict(price=float(ch["price"][i]), delta=float(ch["delta"][i]),
                               gamma=float(ch["gamma"][i]), vega=float(ch["vega"][i]),
                               vanna=float(ch["vanna"][i]), theta=float(ch["theta"][i]))
            else:
                out[nm] = dict(
                    price=float(ch["price"][i] - S_star * eqt[i] + K * ert[i]),
                    delta=float(ch["delta"][i] - eqt[i]),
                    gamma=float(ch["gamma"][i]), vega=float(ch["vega"][i]),
                    vanna=float(ch["vanna"][i]),
                    theta=float(ch["theta"][i] - self.q * S_star * eqt[i] + self.r * K * ert[i]))
        return out

    def greeks(self, S, K, tau, v, kind="call"):
        g = self.greeks_chain(S, v, [("x", kind, K, tau)])["x"]
        return {k: g[k] for k in ("delta", "gamma", "vega", "vanna", "theta")}