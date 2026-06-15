from __future__ import annotations
from src.config import RNGStreams
from src.engine import DiscreteEventEngine
from src.instruments import InstrumentRegistry, UNDERLYING
from src.matching_engine import ASK, BID
import numpy as np


class ObservationBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, eng: DiscreteEventEngine):
        cfg = self.cfg
        obs = {}
        for nm, ins in eng.reg.instruments.items():
            if nm != UNDERLYING and not eng.reg.live[nm]:
                continue
            bk = eng.books[nm]
            db = bk.depth(BID, cfg.obs_depth_N)
            da = bk.depth(ASK, cfg.obs_depth_N)
            own = []
            for oid in eng.agent_oids[nm]:
                o = bk.registry.get(oid)
                if o is not None:
                    own.append((oid, o.side, cfg.tick_to_price(o.tick), o.size))
            obs[nm] = dict(
                bbo=bk.bbo(),
                depth_bid=np.array(db, dtype=np.float32).reshape(-1, 2),
                depth_ask=np.array(da, dtype=np.float32).reshape(-1, 2),
                tape=list(eng.tape[nm]),
                own_orders=own,
                own_fills=list(eng.own_fills[nm]),
            )
        live_ttm = {f"T{j+1}": max(T - eng.t, 0.0)
                    for j, T in enumerate(cfg.expiry_years)
                    if j in eng.reg.live_expiries()}
        obs["clock"] = dict(
            t_now=eng.t,
            ttm=live_ttm,
            t_remaining_episode=max(cfg.T_horizon - eng.t, 0.0),
            live_mask={nm: eng.reg.live[nm] for nm in eng.reg.instruments},
        )
        return obs


class PrivilegedInfoBuilder:
    def build(self, eng: DiscreteEventEngine):
        S = eng.S_star(); v = eng.heston.v
        spec = eng.reg.live_chain_spec(eng.t)
        gch = eng.pricer.greeks_chain(S, v, spec)
        true_value = {nm: gch[nm]["price"] for nm in gch}
        true_greeks = {nm: dict(delta=gch[nm]["delta"], gamma=gch[nm]["gamma"],
                                vega=gch[nm]["vega"], vanna=gch[nm]["vanna"],
                                theta=gch[nm]["theta"]) for nm in gch}
        return dict(
            S_heston=eng.heston.S, v=v, impact=eng.impact.value(), S_star=S,
            true_value=true_value, true_greeks=true_greeks,
            settlement=list(eng.settlement_log),
            is_terminal_liquidation=eng.is_terminal_liquidation,
        )


class OptionsMMSimulator:
    def __init__(self, cfg):
        self.cfg = cfg.validate()
        self._obs_builder = ObservationBuilder(self.cfg)
        self._info_builder = PrivilegedInfoBuilder()
        self.eng = None

    def reset(self, seed=None, strikes=None):
        cfg = self.cfg
        rng = RNGStreams(cfg.seed if seed is None else seed)
        reg = InstrumentRegistry(cfg, strikes=strikes)
        self.eng = DiscreteEventEngine(cfg, rng, reg)
        for j, T in enumerate(cfg.expiry_years):
            self.eng.q.push(T, "settle", j)
        self.eng.schedule_first_arrivals()
        self.eng.run_burn_in()
        self.eng.tape = {nm: [] for nm in reg.instruments}
        self.eng.own_fills = {nm: [] for nm in reg.instruments}
        self.eng.settlement_log = []
        return self._obs_builder.build(self.eng), self._info_builder.build(self.eng)

    def step(self, action):
        eng = self.eng
        eng.tape = {nm: [] for nm in eng.reg.instruments}
        eng.own_fills = {nm: [] for nm in eng.reg.instruments}
        eng.settlement_log = []
        eng.is_terminal_liquidation = False
        eng.agent_steps += 1
        if eng.agent_steps > self.cfg.max_steps:
            eng.truncated = True
        eng.submit_action(action)
        if not (eng.terminated or eng.truncated):
            eng.advance_to_next_observable_change()
        return (self._obs_builder.build(eng), self._info_builder.build(eng),
                eng.terminated, eng.truncated)

    def _terminate_check(self):
        return self.eng.terminated