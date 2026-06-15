from __future__ import annotations
from src.instruments import UNDERLYING
from src.pricing import bs_delta, bs_implied_vol

import numpy as np


class AvellanedaStoikovAgent:
    def __init__(self, cfg, gamma=0.1, k=10.0, quote_size=5.0,
                 max_inventory=80.0, fallback_iv=0.20, iv_band=(0.05, 0.80),
                 inventory_aware=True):
        self.cfg = cfg
        self.gamma = gamma
        self.k = k
        self.quote_size = quote_size
        self.max_inventory = max_inventory
        self.fallback_iv = fallback_iv
        self.iv_band = iv_band
        self.inventory_aware = inventory_aware
        self.reset()

    def reset(self):
        self.opt_pos = {}
        self.cash = 0.0
        self._last_iv = {}
        self.last_net_delta = 0.0
        self.last_skew = {}

    def _ingest_fills(self, obs):
        for nm, d in obs.items():
            if nm in ("clock", UNDERLYING):
                continue
            for (_oid, px, signed, _t) in d.get("own_fills", []):
                self.opt_pos[nm] = self.opt_pos.get(nm, 0.0) + signed
                self.cash -= signed * px
        live = obs["clock"]["live_mask"]
        for nm in list(self.opt_pos):
            if not live.get(nm, False):
                self.opt_pos.pop(nm, None)

    @staticmethod
    def _obs_mid(entry):
        bp, _, ap, _ = entry["bbo"]
        if np.isfinite(bp) and np.isfinite(ap):
            return 0.5 * (bp + ap)
        if np.isfinite(bp):
            return bp
        if np.isfinite(ap):
            return ap
        return np.nan

    def act(self, obs):
        self._ingest_fills(obs)
        cfg = self.cfg
        t = obs["clock"]["t_now"]
        S_obs = self._obs_mid(obs.get(UNDERLYING, {"bbo": (np.nan,) * 4}))

        quotes = {}
        net_delta = 0.0
        self.last_skew = {}
        for nm, d in obs.items():
            if nm in ("clock", UNDERLYING):
                continue
            ins = cfg_instrument(cfg, nm)
            tau = max(ins["T"] - t, 1e-6)
            m = self._obs_mid(d)

            lo, hi = self.iv_band
            if np.isfinite(m) and np.isfinite(S_obs) and m > 0:
                iv = bs_implied_vol(m, S_obs, ins["K"], tau, cfg.r, cfg.q, ins["kind"])
                if not np.isfinite(iv) or iv <= 1e-4:
                    iv = self._last_iv.get(nm, self.fallback_iv)
                iv = float(np.clip(iv, lo, hi))
                self._last_iv[nm] = iv
            else:
                iv = self._last_iv.get(nm, self.fallback_iv)
            if not np.isfinite(m):
                continue

            q = self.opt_pos.get(nm, 0.0)
            sigma2 = iv * iv

            half = self.gamma * sigma2 * tau + (2.0 / self.gamma) * np.log1p(self.gamma / self.k)
            half = max(half, cfg.half_spread_floor_ticks * cfg.tick)

            raw_skew = (q * self.gamma * sigma2 * tau) if self.inventory_aware else 0.0
            skew = float(np.clip(raw_skew, -0.5 * half, 0.5 * half))
            r_res = m - skew
            self.last_skew[nm] = skew
            bid = r_res - half
            ask = r_res + half

            if q >= self.max_inventory:
                bid = None
            elif q <= -self.max_inventory:
                ask = None
            bid = None if (bid is not None and bid <= 0) else bid
            quotes[nm] = (bid, ask, self.quote_size)

            if np.isfinite(S_obs):
                net_delta += q * float(bs_delta(S_obs, ins["K"], tau, max(iv, 1e-3),
                                                cfg.r, cfg.q, ins["kind"]))
        self.last_net_delta = net_delta
        return {"quotes": quotes}


def cfg_instrument(cfg, nm):
    tag, *rest = nm.split("_")
    kind = "call" if tag == "C" else "put"
    j = int(rest[-1][1:]) - 1
    K = cfg.strike if len(rest) == 1 else float(rest[0])
    return {"kind": kind, "K": K, "T": cfg.expiry_years[j]}