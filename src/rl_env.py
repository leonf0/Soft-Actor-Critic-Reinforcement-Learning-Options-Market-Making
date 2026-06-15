from __future__ import annotations
from src.config import SimConfig
from src.instruments import UNDERLYING
from src.matching_engine import ASK, BID
from src.pricing import bs_delta, bs_implied_vol, bs_price
from src.simulator import OptionsMMSimulator
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm


F_INST = 13
F_GLOB = 33


def _d12(S, K, tau, sig, r=0.0, q=0.0):
    tau = max(tau, 1e-12); sig = max(sig, 1e-12)
    st = sig * np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig * sig) * tau) / st
    return d1, d1 - st, st


def bs_gamma(S, K, tau, sig, r=0.0, q=0.0):
    d1, _, st = _d12(S, K, tau, sig, r, q)
    return float(np.exp(-q * tau) * norm.pdf(d1) / (S * st))


def bs_vega(S, K, tau, sig, r=0.0, q=0.0):
    d1, _, _ = _d12(S, K, tau, sig, r, q)
    return float(S * np.exp(-q * tau) * norm.pdf(d1) * np.sqrt(max(tau, 1e-12)))


def bs_vanna(S, K, tau, sig, r=0.0, q=0.0):
    d1, d2, _ = _d12(S, K, tau, sig, r, q)
    return float(-np.exp(-q * tau) * norm.pdf(d1) * d2 / max(sig, 1e-12))


def bs_volga(S, K, tau, sig, r=0.0, q=0.0):
    d1, d2, _ = _d12(S, K, tau, sig, r, q)
    return float(bs_vega(S, K, tau, sig, r, q) * d1 * d2 / max(sig, 1e-12))


def build_meta(env):
    reg = env.eng.reg
    cfg = env.cfg
    names = list(reg.option_names)
    N = len(names)
    K = np.array([reg.instruments[nm].K for nm in names], float)
    T = np.array([reg.instruments[nm].T for nm in names], float)
    eidx = np.array([reg.instruments[nm].expiry_idx for nm in names], int)
    kind = np.array([1.0 if reg.instruments[nm].kind == "call" else -1.0
                     for nm in names], float)
    pairs = sorted({(float(k), int(j)) for k, j in zip(K, eidx)})
    pair_index = {p: i for i, p in enumerate(pairs)}
    pair_of = np.array([pair_index[(float(k), int(j))] for k, j in zip(K, eidx)], int)
    return dict(N=N, names=names, K=K, T=T, expiry_idx=eidx, kind=kind,
                pairs=pairs, n_pairs=len(pairs), pair_of=pair_of,
                n_expiries=len(cfg.expiry_years))


@dataclass
class EnvConfig:
    max_order_qty: float = 20.0
    round_qty: bool = True
    fallback_iv: float = 0.20
    iv_lo: float = 0.05
    iv_hi: float = 0.80
    inv_scale: float = 50.0
    flow_scale: float = 50.0
    hedge_scale: float = 100.0
    delta_scale: float = 100.0
    gamma_scale: float = 50.0
    vega_scale: float = 2000.0
    vanna_scale: float = 200.0
    volga_scale: float = 5000.0
    intensity_scale: float = 50.0
    delta_limit: float = 150.0
    vega_limit: float = 3000.0
    gross_limit: float = 300.0
    rvol_spans: tuple = (20, 100, 500)
    flow_beta: float = 1200.0
    regime_beta: float = 25.0
    ewma_count_scale: float = 5.0
    regime_rate_scale: float = 5.0e4
    W0: float = 100.0
    max_inventory: float = 60.0
    dt_ref: float = 5.0e-6


@dataclass
class RewardConfig:
    lambda_edge: float = 10.0
    A: float = 1.0
    lambda_vega: float = 0.05
    lambda_delta: float = 0.05
    lambda_gamma: float = 0.0
    risk_cap: float = 100.0
    r_clip: float = 100.0
    mark: str = "true_value"
    greeks: str = "true"
    norm_by: str = "W0"
    discount_rho: float = 100.0
    lambda_hedge: float = 3.0


def _softplus(x):
    x = np.asarray(x, float)
    return np.where(x > 30.0, x, np.log1p(np.exp(np.minimum(x, 30.0))))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def decode_action(pre, R, net_delta_obs, pos, live, meta, cfg, ecfg):
    N = meta["N"]
    pre = np.asarray(pre, float)
    quotes = {}
    skew = {}
    n_ticks = cfg.n_ticks
    for i in range(N):
        nm = meta["names"][i]
        if not live[nm]:
            continue
        raw = pre[4 * i:4 * i + 4]
        bid_off, ask_off, bid_q, ask_q = _softplus(raw)
        Ri = R[i]
        if not np.isfinite(Ri):
            continue
        bid_px = Ri - bid_off
        ask_px = Ri + ask_off
        bid_tick = int(np.floor((bid_px - cfg.price_lo) / cfg.tick))
        ask_tick = int(np.ceil((ask_px - cfg.price_lo) / cfg.tick))
        bid_tick = max(0, min(bid_tick, n_ticks))
        ask_tick = max(0, min(ask_tick, n_ticks))
        if ask_tick <= bid_tick:
            ask_tick = min(bid_tick + 1, n_ticks)
            if ask_tick == bid_tick:
                bid_tick -= 1
        bpx = cfg.tick_to_price(bid_tick) if bid_tick >= 1 else None
        apx = cfg.tick_to_price(ask_tick) if ask_tick <= n_ticks else None
        bq = float(np.clip(bid_q, 0.0, ecfg.max_order_qty))
        aq = float(np.clip(ask_q, 0.0, ecfg.max_order_qty))
        if ecfg.round_qty:
            bq = float(round(bq)); aq = float(round(aq))
        if pos is not None:
            if pos[i] >= ecfg.max_inventory:
                bq = 0.0
            if pos[i] <= -ecfg.max_inventory:
                aq = 0.0
        quotes[nm] = (bpx, apx, bq, aq)
        skew[nm] = 0.5 * (ask_off - bid_off)

    f = float(_sigmoid(pre[-1]))
    hedge_signed = -f * float(net_delta_obs)
    return quotes, hedge_signed, skew


class FeatureEncoder:
    def __init__(self, cfg, meta, ecfg: EnvConfig):
        self.cfg = cfg
        self.meta = meta
        self.ecfg = ecfg
        self.reset()

    def reset(self):
        m, e = self.meta, self.ecfg
        self.pos = np.zeros(m["N"])
        self.h = 0.0
        self.cash = 0.0
        self._iv = np.full(m["n_pairs"], np.nan)
        self._iv_change = np.zeros(m["n_pairs"])
        self._S_obs = np.nan
        self._S_prev = np.nan
        self._t_prev = np.nan
        self._r2_ewma = np.zeros(len(e.rvol_spans))
        self._dt_ewma = np.full(len(e.rvol_spans), np.nan)
        self._warm = np.zeros(len(e.rvol_spans))
        self._flow_ewma = np.zeros(m["N"])
        self._int_ewma = np.zeros(m["N"])
        self._und_flow_ewma = 0.0
        self._und_int_ewma = 0.0
        self._reg_ewma = 0.0
        self._iv_age = np.zeros(m["n_pairs"])
        self._mtm_peak = 0.0
        self.R = np.full(m["N"], np.nan)
        self.R_quote = np.full(m["N"], np.nan)
        self.net_delta_obs = 0.0

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

    def _ex_self_best(self, depth, own_px, own_sz):
        tol = 0.25 * self.cfg.tick
        for lv in np.atleast_2d(np.asarray(depth)):
            if lv.size < 2:
                continue
            px, sz = float(lv[0]), float(lv[1])
            rem = sz - sum(s_ for p_, s_ in zip(own_px, own_sz) if abs(p_ - px) < tol)
            if rem > 1e-9:
                return self.cfg.tick_to_price(self.cfg.price_to_tick(px))
        return np.nan

    def _ex_self_mid(self, d):
        own = d.get("own_orders", [])
        if not own:
            return self._obs_mid(d)
        bpx = [px for (_o, sd, px, _z) in own if sd == BID]
        bsz = [sz for (_o, sd, _p, sz) in own if sd == BID]
        apx = [px for (_o, sd, px, _z) in own if sd == ASK]
        asz = [sz for (_o, sd, _p, sz) in own if sd == ASK]
        bb = self._ex_self_best(d["depth_bid"], bpx, bsz)
        aa = self._ex_self_best(d["depth_ask"], apx, asz)
        if np.isfinite(bb) and np.isfinite(aa):
            return 0.5 * (bb + aa)
        if np.isfinite(bb):
            return bb
        if np.isfinite(aa):
            return aa
        return np.nan

    def _ingest_fills(self, obs):
        names = self.meta["names"]
        for i, nm in enumerate(names):
            d = obs.get(nm)
            if d is None:
                continue
            for (_oid, px, signed, _t) in d.get("own_fills", []):
                self.pos[i] += signed
                self.cash -= signed * px
        u = obs.get(UNDERLYING)
        if u is not None:
            for (_oid, px, signed, _t) in u.get("own_fills", []):
                self.h += signed
                self.cash -= signed * px
        live = obs["clock"]["live_mask"]
        for i, nm in enumerate(names):
            if not live.get(nm, False):
                self.pos[i] = 0.0


    def encode(self, obs):
        self._ingest_fills(obs)
        cfg, m, e = self.cfg, self.meta, self.ecfg
        names, N = m["names"], m["N"]
        clk = obs["clock"]
        t = clk["t_now"]
        live = clk["live_mask"]
        mask = np.array([bool(live.get(nm, False)) for nm in names])

        u = obs.get(UNDERLYING, {"bbo": (np.nan,) * 4})
        ub, ubs, ua, uas = u["bbo"]
        S_obs = self._obs_mid(u)
        if not np.isfinite(S_obs):
            S_obs = self._S_obs if np.isfinite(self._S_obs) else cfg.S0
        und_ret = (np.log(S_obs / self._S_prev)
                   if np.isfinite(self._S_prev) and self._S_prev > 0 and S_obs > 0 else 0.0)
        und_rel_spread = ((ua - ub) / max(S_obs, cfg.tick)
                          if np.isfinite(ua) and np.isfinite(ub) else 0.0)
        und_imb = ((ubs - uas) / (ubs + uas)
                   if np.isfinite(ubs) and np.isfinite(uas) and (ubs + uas) > 0 else 0.0)

        dt = (t - self._t_prev) if np.isfinite(self._t_prev) else np.nan
        rvol = np.zeros(len(e.rvol_spans))
        if np.isfinite(dt) and dt > 0:
            for k, span in enumerate(e.rvol_spans):
                a = 2.0 / (span + 1.0)
                self._r2_ewma[k] = (1 - a) * self._r2_ewma[k] + a * und_ret * und_ret
                self._dt_ewma[k] = (dt if not np.isfinite(self._dt_ewma[k])
                                    else (1 - a) * self._dt_ewma[k] + a * dt)
                self._warm[k] = min(1.0, self._warm[k] + a)
        for k in range(len(e.rvol_spans)):
            if self._warm[k] > 0.5 and np.isfinite(self._dt_ewma[k]) and self._dt_ewma[k] > 0:
                rvol[k] = np.sqrt(max(self._r2_ewma[k], 0.0) / self._dt_ewma[k])

        if np.isfinite(dt) and dt > 0:
            fdec = float(np.exp(-e.flow_beta * dt))
            self._flow_ewma *= fdec
            self._int_ewma *= fdec
            self._und_flow_ewma *= fdec
            self._und_int_ewma *= fdec
            self._reg_ewma *= float(np.exp(-e.regime_beta * dt))
            self._iv_age += dt

        tau_pair = np.array([max(cfg.expiry_years[j] - t, 1e-6) for (_K, j) in m["pairs"]])
        iv_prev = self._iv.copy()
        pair_live = np.zeros(m["n_pairs"], bool)
        pair_members: list[list[int]] = [[] for _ in range(m["n_pairs"])]
        for i in range(N):
            pair_members[m["pair_of"][i]].append(i)
        for p, (K, j) in enumerate(m["pairs"]):
            cands = [i for i in pair_members[p] if mask[i]]
            if not cands:
                continue
            pair_live[p] = True
            otm_kind = 1.0 if K >= S_obs else -1.0
            cands.sort(key=lambda i: 0 if m["kind"][i] == otm_kind else 1)
            iv = np.nan
            for i in cands:
                d = obs.get(names[i])
                mid = self._obs_mid(d) if d is not None else np.nan
                if np.isfinite(mid) and mid > 0:
                    kind = "call" if m["kind"][i] > 0 else "put"
                    iv = bs_implied_vol(mid, S_obs, K, tau_pair[p], cfg.r, cfg.q, kind)
                    break
            if not np.isfinite(iv) or iv <= 1e-4:
                iv = self._iv[p] if np.isfinite(self._iv[p]) else e.fallback_iv
            else:
                self._iv_age[p] = 0.0
            self._iv[p] = float(np.clip(iv, e.iv_lo, e.iv_hi))
        self._iv_change = np.where(np.isfinite(iv_prev), self._iv - iv_prev, 0.0)
        iv_filled = np.where(np.isfinite(self._iv), self._iv, e.fallback_iv)

        tau_i = np.maximum(m["T"] - t, 1e-6)
        iv_i = iv_filled[m["pair_of"]]
        delta_i = np.zeros(N); gamma_i = np.zeros(N); vega_i = np.zeros(N)
        vanna_i = np.zeros(N); volga_i = np.zeros(N)
        for i in range(N):
            if not mask[i] and self.pos[i] == 0.0:
                continue
            kind = "call" if m["kind"][i] > 0 else "put"
            delta_i[i] = float(bs_delta(S_obs, m["K"][i], tau_i[i], max(iv_i[i], 1e-3),
                                        cfg.r, cfg.q, kind))
            gamma_i[i] = bs_gamma(S_obs, m["K"][i], tau_i[i], max(iv_i[i], 1e-3), cfg.r, cfg.q)
            vega_i[i] = bs_vega(S_obs, m["K"][i], tau_i[i], max(iv_i[i], 1e-3), cfg.r, cfg.q)
            vanna_i[i] = bs_vanna(S_obs, m["K"][i], tau_i[i], max(iv_i[i], 1e-3), cfg.r, cfg.q)
            volga_i[i] = bs_volga(S_obs, m["K"][i], tau_i[i], max(iv_i[i], 1e-3), cfg.r, cfg.q)

        X = np.zeros((N, F_INST), np.float32)
        R = np.full(N, np.nan)
        Rq = self.R_quote.copy()
        agg_signed_flow = 0.0
        agg_prints = 0.0
        if u is not None:
            for (px, sz, side, _t) in u.get("tape", []):
                sgn = sz if side == BID else -sz
                agg_signed_flow += sgn
                agg_prints += 1.0
                self._und_flow_ewma += sgn
                self._und_int_ewma += 1.0
                self._reg_ewma += 1.0
        for i, nm in enumerate(names):
            if not mask[i]:
                continue
            d = obs[nm]
            bp, bs_, ap, as_ = d["bbo"]
            mid = self._obs_mid(d)
            R[i] = mid
            rq = self._ex_self_mid(d)
            if np.isfinite(rq):
                Rq[i] = rq
            rel_spread = ((ap - bp) / max(mid, cfg.tick)
                          if np.isfinite(ap) and np.isfinite(bp) and np.isfinite(mid) else 0.0)
            bbo_imb = ((bs_ - as_) / (bs_ + as_)
                       if np.isfinite(bs_) and np.isfinite(as_) and (bs_ + as_) > 0 else 0.0)
            sf = 0.0
            prints_i = 0.0
            for (px, sz, side, _t) in d.get("tape", []):
                sf += sz if side == BID else -sz
                prints_i += 1.0
            agg_prints += prints_i
            agg_signed_flow += sf
            self._flow_ewma[i] += sf
            self._int_ewma[i] += prints_i
            self._reg_ewma += prints_i
            db, da = d["depth_bid"], d["depth_ask"]
            sb = float(db[:, 1].sum()) if db.size else 0.0
            sa = float(da[:, 1].sum()) if da.size else 0.0
            depth_imb = (sb - sa) / (sb + sa) if (sb + sa) > 0 else 0.0
            X[i, 0] = self.pos[i] / e.inv_scale
            X[i, 1] = np.log(m["K"][i] / S_obs)
            X[i, 2] = tau_i[i] / cfg.T_horizon
            X[i, 3] = m["kind"][i]
            X[i, 4] = rel_spread
            X[i, 5] = bbo_imb
            X[i, 6] = iv_i[i]
            X[i, 7] = sf / e.flow_scale
            X[i, 8] = depth_imb
            X[i, 9] = self._iv_change[m["pair_of"][i]]
            X[i, 10] = self._flow_ewma[i] / e.flow_scale
            X[i, 11] = self._int_ewma[i] / e.ewma_count_scale
            X[i, 12] = self._iv_age[m["pair_of"][i]] / cfg.T_horizon

        netD = self.h + float(np.dot(self.pos, delta_i))
        netG = float(np.dot(self.pos, gamma_i))
        netV = float(np.dot(self.pos, vega_i))
        netVa = float(np.dot(self.pos, vanna_i))
        netVo = float(np.dot(self.pos, volga_i))
        per_exp = np.zeros((m["n_expiries"], 3))
        for i in range(N):
            if self.pos[i] != 0.0:
                j = m["expiry_idx"][i]
                per_exp[j, 0] += self.pos[i] * delta_i[i]
                per_exp[j, 1] += self.pos[i] * gamma_i[i]
                per_exp[j, 2] += self.pos[i] * vega_i[i]

        mtm = self.cash + self.h * S_obs
        for i in range(N):
            if self.pos[i] != 0.0:
                mid = R[i]
                if not np.isfinite(mid):
                    kind = "call" if m["kind"][i] > 0 else "put"
                    mid = float(bs_price(S_obs, m["K"][i], tau_i[i], max(iv_i[i], 1e-3),
                                         cfg.r, cfg.q, kind))
                mtm += self.pos[i] * mid
        self._mtm_peak = max(self._mtm_peak, mtm)
        dd = self._mtm_peak - mtm

        live_iv = iv_filled[pair_live] if pair_live.any() else np.array([e.fallback_iv])

        g = np.zeros(F_GLOB, np.float32)
        g[0] = netD / e.delta_scale
        g[1] = netG / e.gamma_scale
        g[2] = netV / e.vega_scale
        g[3] = self.h / e.hedge_scale
        g[4] = und_ret
        g[5] = und_rel_spread
        g[6] = und_imb
        g[7] = clk["t_remaining_episode"] / cfg.T_horizon
        for j in range(min(m["n_expiries"], 3)):
            g[8 + 3 * j + 0] = per_exp[j, 0] / e.delta_scale
            g[8 + 3 * j + 1] = per_exp[j, 1] / e.gamma_scale
            g[8 + 3 * j + 2] = per_exp[j, 2] / e.vega_scale
        g[17] = rvol[0]; g[18] = rvol[1]; g[19] = rvol[2]
        g[20] = rvol[1] - float(np.mean(live_iv))
        g[21] = agg_signed_flow / e.flow_scale
        g[22] = agg_prints / e.intensity_scale
        g[23] = mtm / e.W0
        g[24] = dd / e.W0
        g[25] = abs(netD) / e.delta_limit
        g[26] = abs(netV) / e.vega_limit
        g[27] = float(np.sum(np.abs(self.pos))) / e.gross_limit
        g[28] = netVa / e.vanna_scale
        g[29] = netVo / e.volga_scale
        g[30] = self._und_flow_ewma / e.flow_scale
        g[31] = self._und_int_ewma / e.ewma_count_scale
        g[32] = (self._reg_ewma * e.regime_beta) / e.regime_rate_scale

        self.R = R
        self.R_quote = Rq
        self.net_delta_obs = netD
        self._S_prev = S_obs
        self._S_obs = S_obs
        self._t_prev = t
        return X, g, mask


class OptionsMMEnv:
    PRIV_DIM = 7

    def __init__(self, sim_cfg: SimConfig, ecfg: EnvConfig | None = None,
                 rcfg: RewardConfig | None = None, strikes=None, collect_snaps=False):
        self.env = OptionsMMSimulator(sim_cfg)
        self.cfg = self.env.cfg
        self.ecfg = ecfg or EnvConfig()
        self.rcfg = rcfg or RewardConfig()
        self.strikes = (list(strikes) if strikes is not None
                        else list(self.cfg.fixture_strikes))
        self.collect_snaps = collect_snaps
        self.meta = None
        self.enc = None
        self.adim = None

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed, strikes=self.strikes)
        self.meta = build_meta(self.env)
        self.adim = 4 * self.meta["N"] + 1
        self.enc = FeatureEncoder(self.cfg, self.meta, self.ecfg)
        X, g, mask = self.enc.encode(obs)
        self._raw_obs = obs
        self.last_skew = {}
        self.last_net_delta = float("nan")
        self._D_prev = self._true_book_greeks(info)[0] / self.ecfg.delta_scale
        self._t_prev_r = obs["clock"]["t_now"]
        self.ep_edge = 0.0
        self.ep_risk = 0.0
        self.ep_reward = 0.0
        self.ep_steps = 0
        if self.collect_snaps:
            self._snaps = [self._snap(obs, info)]
        return (X, g, mask, self._priv(info)), info

    def step(self, pre):
        live = self._raw_obs["clock"]["live_mask"]
        quotes, hedge_signed, skew = decode_action(pre, self.enc.R_quote,
                                                   self.enc.net_delta_obs,
                                                   self.enc.pos, live, self.meta,
                                                   self.cfg, self.ecfg)
        self.last_skew = skew
        self.last_net_delta = self.enc.net_delta_obs
        obs, info, term, trunc, gated = self._step_engine(quotes, hedge_signed)
        X, g, mask = self.enc.encode(obs)
        r, rinfo = self._reward(obs, info)
        rinfo["noop"] = gated

        self._raw_obs = obs
        self.ep_reward += r
        self.ep_steps += 1
        if self.collect_snaps:
            self._snaps.append(self._snap(obs, info))
        return (X, g, mask, self._priv(info)), r, term, trunc, rinfo

    def _step_engine(self, quotes, hedge_signed):
        env, eng = self.env, self.env.eng
        eng.tape = {nm: [] for nm in eng.reg.instruments}
        eng.own_fills = {nm: [] for nm in eng.reg.instruments}
        eng.settlement_log = []
        eng.is_terminal_liquidation = False
        eng.agent_steps += 1
        if eng.agent_steps > self.cfg.max_steps:
            eng.truncated = True
        gated = bool(eng.terminated or eng.truncated)
        if not gated:
            self._submit_per_side(eng, quotes, hedge_signed)
            eng.advance_to_next_observable_change()
        return (env._obs_builder.build(eng), env._info_builder.build(eng),
                eng.terminated, eng.truncated, gated)

    @staticmethod
    def _submit_per_side(eng, quotes, hedge_signed):
        for nm in eng.reg.option_names:
            for oid in eng.agent_oids[nm]:
                eng.books[nm].cancel(oid)
            eng.agent_oids[nm] = []
            if nm not in quotes or not eng.reg.live[nm]:
                continue
            bid_px, ask_px, bid_q, ask_q = quotes[nm]
            if bid_px is not None and bid_px > 0 and bid_q > 0:
                oid, fills = eng.books[nm].add_limit(BID, bid_px, bid_q, eng.t, is_agent=True)
                if fills:
                    eng._record_fills(nm, fills, eng.t)
                if oid is not None:
                    eng.agent_oids[nm].append(oid)
            if ask_px is not None and ask_px > 0 and ask_q > 0:
                oid, fills = eng.books[nm].add_limit(ASK, ask_px, ask_q, eng.t, is_agent=True)
                if fills:
                    eng._record_fills(nm, fills, eng.t)
                if oid is not None:
                    eng.agent_oids[nm].append(oid)
        if abs(hedge_signed) > 1e-9 and eng.reg.live[UNDERLYING]:
            taker_side = BID if hedge_signed > 0 else ASK
            fills = eng.books[UNDERLYING].market_order(taker_side, abs(hedge_signed),
                                                       eng.t, taker_is_agent=True)
            eng._record_fills(UNDERLYING, fills, eng.t)
            if eng.cfg.agent_impact:
                signed = sum(f.size for f in fills) * (1 if hedge_signed > 0 else -1)
                eng.impact.add_market_order(signed, eng.t)

    def _true_book_greeks(self, info):
        D = self.enc.h
        G = 0.0
        Y = 0.0
        tg = info["true_greeks"]
        names = self.meta["names"]
        for i in range(self.meta["N"]):
            p = self.enc.pos[i]
            if p != 0.0 and names[i] in tg:
                gr = tg[names[i]]
                D += p * gr["delta"]
                G += p * gr["gamma"]
                Y += p * gr["vega"]
        return D, G, Y

    def _reward(self, obs, info):
        rc, ec = self.rcfg, self.ecfg
        tv = info["true_value"]
        S_star = info["S_star"]
        E = 0.0
        nf = 0
        hedge_qty = 0.0
        for nm, d in obs.items():
            if nm == "clock":
                continue
            for (_oid, px, signed, _t) in d.get("own_fills", []):
                mu = S_star if nm == UNDERLYING else tv.get(nm, px)
                E += signed * (mu - px)
                nf += 1
                if nm == UNDERLYING:
                    hedge_qty += abs(signed)
        D, G, Y = self._true_book_greeks(info)
        dn = D / ec.delta_scale
        yn = Y / ec.vega_scale
        gn = G / ec.gamma_scale

        t = obs["clock"]["t_now"]
        dt = max(t - self._t_prev_r, 0.0)
        self._t_prev_r = t
        risk_rate = (rc.lambda_delta * dn * dn + rc.lambda_vega * yn * yn
                     + rc.lambda_gamma * gn * gn)
        risk = min(risk_rate * (dt / ec.dt_ref), rc.risk_cap)

        gam = float(np.exp(-rc.discount_rho * dt))
        shaping = rc.A * (abs(self._D_prev) - gam * abs(dn))
        r_raw = (rc.lambda_edge * (E / ec.W0) + shaping - risk
                 - rc.lambda_hedge * hedge_qty)
        r = float(np.clip(r_raw, -rc.r_clip, rc.r_clip))
        self._D_prev = dn
        self.ep_edge += E
        self.ep_risk += risk
        return r, dict(edge=float(E), n_fills=nf, D=float(D), G=float(G), Y=float(Y),
                       dn=float(dn), yn=float(yn), shaping=float(shaping),
                       risk=float(risk), dt=float(dt), gam=gam,
                       hedge_qty=float(hedge_qty),
                       r_raw=float(r_raw), r_clipped=bool(r != r_raw))

    def _priv(self, info):
        D, G, Y = self._true_book_greeks(info)
        e = self.ecfg

        return np.array([info["S_star"] / self.cfg.S0 - 1.0, info["v"], info["impact"],
                         D / e.delta_scale, Y / e.vega_scale, G / e.gamma_scale,
                         float(np.log(self.env.eng.regime))], np.float32)

    def _snap(self, obs, info):
        names = self.meta["names"]
        fills = []
        for nm, d in obs.items():
            if nm == "clock":
                continue
            for (_oid, px, signed, _t) in d.get("own_fills", []):
                fills.append((nm, float(signed), float(px)))

        def omid(nm):
            if nm not in obs:
                return np.nan
            bp, _, ap, _ = obs[nm]["bbo"]
            if np.isfinite(bp) and np.isfinite(ap):
                return 0.5 * (bp + ap)
            return bp if np.isfinite(bp) else (ap if np.isfinite(ap) else np.nan)

        return dict(
            t=obs["clock"]["t_now"], S_star=float(info["S_star"]), v=float(info["v"]),
            true_value={k: float(v) for k, v in info["true_value"].items()},
            true_greeks={k: dict(v) for k, v in info["true_greeks"].items()},
            settlement=[(nm, float(cf)) for (nm, cf, _t) in info["settlement"]],
            fills=fills, obs_mid={nm: omid(nm) for nm in names},
            skew={nm: float(self.last_skew.get(nm, np.nan)) for nm in names},
            est_net_delta=float(self.last_net_delta))

    def episode_dict(self, terminated, truncated):
        assert self.collect_snaps, "construct with collect_snaps=True"
        return dict(traj=self._snaps, opt_names=list(self.meta["names"]),
                    terminated=terminated, truncated=truncated,
                    expiry_years=list(self.cfg.expiry_years),
                    T_horizon=self.cfg.T_horizon)