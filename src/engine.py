from __future__ import annotations
from src.instruments import InstrumentRegistry, UNDERLYING
from src.matching_engine import ASK, BID, OrderBook
from src.order_flow import BackgroundLiquidity, HawkesProcess, LB, MB, MS
from src.pricing import HestonPricer, HestonState, TransientImpact
import heapq
import numpy as np


class Event:
    __slots__ = ("t", "seq", "kind", "payload")
    def __init__(self, t, seq, kind, payload):
        self.t, self.seq, self.kind, self.payload = t, seq, kind, payload

class EventQueue:
    def __init__(self):
        self._h = []
        self._seq = 0

    def push(self, t, kind, payload=None):
        heapq.heappush(self._h, (t, self._seq, kind, payload))
        self._seq += 1

    def pop(self):
        t, seq, kind, payload = heapq.heappop(self._h)
        return Event(t, seq, kind, payload)

    def peek_t(self):
        return self._h[0][0] if self._h else None

    def __len__(self):
        return len(self._h)

class DiscreteEventEngine:
    def __init__(self, cfg, rng_streams, registry: InstrumentRegistry):
        self.cfg = cfg
        self.rng = rng_streams
        self.reg = registry
        self.q = EventQueue()
        self.pricer = HestonPricer(cfg.kappa, cfg.theta, cfg.xi, cfg.rho,
                                   cfg.r, cfg.q, cfg.cf_grid_n, cfg.cf_grid_u)
        v0 = HestonState.sample_v0(cfg, self.rng.stream("v0"))
        self.heston = HestonState(cfg, self.rng.stream("heston"), v0=v0, S0=cfg.S0)
        self.impact = TransientImpact(cfg.impact_g, cfg.impact_decay)
        self.books = {nm: OrderBook(cfg, nm) for nm in registry.instruments}
        self.bg = BackgroundLiquidity(cfg)
        regime = float(self.rng.stream("regime").lognormal(0.0, cfg.regime_logmu_sigma))
        mu0, alpha, beta = cfg.hawkes.as_arrays()
        self.regime = regime
        self.hawkes = {nm: HawkesProcess(mu0 * regime, alpha, beta,
                                         self.rng.stream(f"hawkes_{nm}"))
                       for nm in registry.instruments}
        for hp in self.hawkes.values():
            hp.t_last = -cfg.burn_in_years
        self._bg_oid_inst = {}
        self.t = -cfg.burn_in_years
        self._last_reprice = None
        self._mid_cache = {}
        self.cash = 0.0
        self.opt_pos = {nm: 0.0 for nm in registry.option_names}
        self.und_pos = 0.0
        self.agent_oids = {nm: [] for nm in registry.instruments}
        self.tape = {nm: [] for nm in registry.instruments}
        self.own_fills = {nm: [] for nm in registry.instruments}
        self.settlement_log = []
        self.is_terminal_liquidation = False
        self.terminated = False
        self.truncated = False
        self.agent_steps = 0
        self.event_count = 0
        self.self_trades = 0

    def S_star(self):
        return self.heston.S + self.impact.value()

    def _true_mid(self, ins):
        if ins.name == UNDERLYING:
            return self.S_star()
        return float(self.pricer.price(self.S_star(), ins.K, max(ins.T - self.t, 1e-9),
                                       self.heston.v, ins.kind))

    def _reprice(self, force=False):
        cfg = self.cfg
        S = self.S_star(); v = self.heston.v
        if not force and self._last_reprice is not None:
            S0, v0, t0 = self._last_reprice
            if (abs(S - S0) / max(S0, 1e-9) < cfg.reprice_dS_frac
                    and abs(v - v0) / max(v0, 1e-9) < cfg.reprice_dv_frac
                    and abs(self.t - t0) < cfg.reprice_dt):
                return
        spec = self.reg.live_chain_spec(self.t)
        mids = self.pricer.price_chain(S, v, spec)
        mids[UNDERLYING] = S
        self._mid_cache = mids
        self._last_reprice = (S, v, self.t)

    def _ref_mid(self, nm):
        return self._mid_cache.get(nm, self._true_mid(self.reg.instruments[nm]))

    def schedule_first_arrivals(self):
        for nm in self.reg.instruments:
            t, d = self.hawkes[nm].next_arrival()
            self.q.push(t, "hawkes", (nm, d))

    def _schedule_next_arrival(self, nm):
        if not self.reg.live[nm] and nm != UNDERLYING:
            return
        t, d = self.hawkes[nm].next_arrival()
        self.q.push(t, "hawkes", (nm, d))

    def _process(self, ev):
        dt = ev.t - self.t
        if dt > 0:
            self.heston.advance_qe(dt)
        self.t = ev.t
        self.impact.update(ev.t)
        self._reprice()

        if ev.kind == "settle":
            return ("settle", ev.payload), False
        if ev.kind == "terminal":
            return ("terminal", None), False

        traded = False
        if ev.kind == "hawkes":
            nm, d = ev.payload
            traded = self._apply_flow(nm, d, ev.t)
            self._schedule_next_arrival(nm)
        elif ev.kind == "bg_cancel":
            nm, oid = ev.payload
            self.books[nm].cancel(oid)
            self._bg_oid_inst.pop(oid, None)
        return None, traded

    def _apply_flow(self, nm, d, t):
        if not self.reg.live[nm] and nm != UNDERLYING:
            return False
        bk = self.books[nm]
        traded = False
        if d in (MB, MS):
            size = max(1.0, round(self.cfg.bg_size_scale *
                       self.rng.stream("bg_size").lognormal(
                           self.cfg.bg_size_lognorm_mean, self.cfg.bg_size_lognorm_sigma)))
            taker_side = BID if d == MB else ASK
            fills = bk.market_order(taker_side, size, t, taker_is_agent=False)
            if fills:
                traded = True
                self._record_fills(nm, fills, t)
                if nm == UNDERLYING:
                    signed = sum(f.size for f in fills) * (1 if d == MB else -1)
                    self.impact.add_market_order(signed, t)
        else:
            side = "bid" if d == LB else "ask"
            ref = self._ref_mid(nm)
            if ref == ref and ref > 0:
                px, sz = self.bg.make_order(side, ref, self.rng.stream("bg_place"),
                                            self.rng.stream("bg_size"))
                oid, fills = bk.add_limit(BID if side == "bid" else ASK, px, sz, t,
                                          is_agent=False)
                if fills:
                    traded = True
                    self._record_fills(nm, fills, t)
                if oid is not None:
                    self._bg_oid_inst[oid] = nm
                    delay = self.bg.cancel_delay(px, ref, self.rng.stream("bg_cancel"))
                    self.q.push(t + delay, "bg_cancel", (nm, oid))
        return traded

    def _record_fills(self, nm, fills, t):
        for f in fills:
            self.tape[nm].append((f.price, f.size, f.aggressor_side, t))
            if f.maker_is_agent or f.taker_is_agent:
                self._apply_agent_fill(nm, f, t)

    def _apply_agent_fill(self, nm, f, t):
        if f.maker_is_agent and f.taker_is_agent:
            self.self_trades += 1
            return
        if f.maker_is_agent:
            agent_buys = (f.aggressor_side == ASK)
        else:
            agent_buys = (f.aggressor_side == BID)
        signed = f.size if agent_buys else -f.size
        self.cash -= signed * f.price
        if nm == UNDERLYING:
            self.und_pos += signed
        else:
            self.opt_pos[nm] += signed
        self.own_fills[nm].append((f.maker_oid, f.price, signed, t))

    def submit_action(self, action):
        if not action:
            return
        quotes = action.get("quotes", {})
        for nm in self.reg.option_names:
            for oid in self.agent_oids[nm]:
                self.books[nm].cancel(oid)
            self.agent_oids[nm] = []
            if nm not in quotes or not self.reg.live[nm]:
                continue
            bid_px, ask_px, size = quotes[nm]
            if size <= 0:
                continue
            if bid_px is not None and bid_px > 0:
                oid, fills = self.books[nm].add_limit(BID, bid_px, size, self.t, is_agent=True)
                if fills: self._record_fills(nm, fills, self.t)
                if oid is not None: self.agent_oids[nm].append(oid)
            if ask_px is not None and ask_px > 0:
                oid, fills = self.books[nm].add_limit(ASK, ask_px, size, self.t, is_agent=True)
                if fills: self._record_fills(nm, fills, self.t)
                if oid is not None: self.agent_oids[nm].append(oid)
        hedge = action.get("hedge", 0.0)
        if abs(hedge) > 1e-9 and self.reg.live[UNDERLYING]:
            taker_side = BID if hedge > 0 else ASK
            fills = self.books[UNDERLYING].market_order(taker_side, abs(hedge),
                                                        self.t, taker_is_agent=True)
            self._record_fills(UNDERLYING, fills, self.t)
            if self.cfg.agent_impact:
                signed = sum(f.size for f in fills) * (1 if hedge > 0 else -1)
                self.impact.add_market_order(signed, self.t)

    def _bbo_snapshot(self):
        snap = {}
        for nm in self.reg.instruments:
            if nm != UNDERLYING and not self.reg.live[nm]:
                continue
            bk = self.books[nm]
            bt, at = bk.best_bid_tick, bk.best_ask_tick
            snap[nm] = (bt, bk.levels[bt].total_size if bt is not None else None,
                        at, bk.levels[at].total_size if at is not None else None)
        return snap

    def advance_to_next_observable_change(self):
        cfg = self.cfg
        before = self._bbo_snapshot()
        guard = 0
        while len(self.q):
            self.event_count += 1
            guard += 1
            if guard > cfg.max_events_per_step or self.event_count > cfg.event_budget:
                self.truncated = True; return
            ev = self.q.pop()
            marker, traded = self._process(ev)
            if marker is not None:
                kind, j = marker
                if kind == "settle":
                    self._settle_expiry(j)
                    return
                if kind == "terminal":
                    self._terminal_liquidation()
                    self.terminated = True
                    return
            if not np.isfinite(self.heston.S) or not np.isfinite(self.heston.v) \
                    or self.S_star() <= 0:
                self.truncated = True; return
            after = self._bbo_snapshot()
            if traded or after != before:
                return
        self.truncated = True

    def _settle_expiry(self, j):
        S = self.S_star()
        settled = self.reg.settle_expiry(j, S)
        for nm, (kind, K, intrinsic) in settled.items():
            qty = self.opt_pos[nm]
            cashflow = qty * intrinsic
            self.cash += cashflow
            self.opt_pos[nm] = 0.0
            self.settlement_log.append((nm, cashflow, self.t))
            self.books[nm] = OrderBook(self.cfg, nm)
            self.agent_oids[nm] = []
        if j == len(self.cfg.expiry_years) - 1:
            self.q.push(self.t, "terminal", None)

    def _terminal_liquidation(self):
        if abs(self.und_pos) > 1e-12:
            taker_side = ASK if self.und_pos > 0 else BID
            fills = self.books[UNDERLYING].market_order(
                taker_side, abs(self.und_pos), self.t, taker_is_agent=True)
            self._record_fills(UNDERLYING, fills, self.t)
            if abs(self.und_pos) > 1e-9:
                self.cash += self.und_pos * self.S_star()
                self.und_pos = 0.0
        self.is_terminal_liquidation = True

    def run_burn_in(self):
        self._reprice(force=True)
        while len(self.q):
            t = self.q.peek_t()
            if t > 0.0:
                dt = 0.0 - self.t
                if dt > 0:
                    self.heston.advance_qe(dt); self.t = 0.0
                    self.impact.update(0.0); self._reprice()
                break
            ev = self.q.pop()
            self._process(ev)
            self.event_count += 1
        self.t = 0.0