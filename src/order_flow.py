from __future__ import annotations
import numpy as np

MB, MS, LB, LS = 0, 1, 2, 3


class HawkesProcess:
    def __init__(self, mu, alpha, beta, rng):
        self.mu = np.asarray(mu, float).copy()
        self.alpha = np.asarray(alpha, float).copy()
        self.beta = np.asarray(beta, float).copy()
        self.rng = rng
        self.t_last = 0.0
        self.E = np.zeros(4)
        self.Lam = np.zeros(4)

    def _lambda(self):
        return self.mu + self.alpha @ self.E

    def intensity(self, t):
        self._advance(t - self.t_last)
        return self._lambda()

    def _advance(self, dt):
        if dt <= 0:
            return
        b = self.beta

        integ_E = self.E * (1.0 - np.exp(-b * dt)) / b
        self.Lam += self.mu * dt + self.alpha @ integ_E
        self.E *= np.exp(-b * dt)
        self.t_last += dt

    def next_arrival(self):
        while True:
            lam = self._lambda()
            Lbar = lam.sum()
            if Lbar <= 0:
                Lbar = max(self.mu.sum(), 1e-9)
            w = self.rng.exponential(1.0 / Lbar)
            self._advance(w)
            lam2 = self._lambda()
            L2 = lam2.sum()
            if self.rng.random() * Lbar <= L2:
                d = int(np.searchsorted(np.cumsum(lam2), self.rng.random() * L2))
                d = min(d, 3)
                self.E[d] += 1.0
                return self.t_last, d

    @staticmethod
    def branching_matrix(alpha, beta):
        return np.asarray(alpha, float) / np.asarray(beta, float)[None, :]

    @staticmethod
    def stationary_rate(mu, alpha, beta):
        Gamma = HawkesProcess.branching_matrix(alpha, beta)
        return np.linalg.solve(np.eye(4) - Gamma, np.asarray(mu, float))


class BackgroundLiquidity:
    def __init__(self, cfg):
        self.cfg = cfg

    def make_order(self, side, ref_px, place_rng, size_rng):
        cfg = self.cfg
        depth_ticks = cfg.half_spread_floor_ticks + place_rng.exponential(cfg.bg_depth_ticks_scale)
        depth_ticks = max(1.0, round(depth_ticks))
        dpx = depth_ticks * cfg.tick
        price = ref_px - dpx if side == "bid" else ref_px + dpx
        size = cfg.bg_size_scale * float(
            size_rng.lognormal(cfg.bg_size_lognorm_mean, cfg.bg_size_lognorm_sigma))
        size = max(1.0, round(size))
        return price, size

    def cancel_delay(self, order_price, ref_px, cancel_rng):
        dist_ticks = abs(order_price - ref_px) / self.cfg.tick
        rate = self.cfg.bg_lifetime_rate * (1.0 + self.cfg.bg_distance_bump * dist_ticks)
        return cancel_rng.exponential(1.0 / rate)