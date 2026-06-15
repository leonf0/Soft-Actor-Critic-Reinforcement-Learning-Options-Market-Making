from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

UNDERLYING = "UND"


@dataclass(frozen=True)
class Instrument:
    name: str
    kind: str
    K: Optional[float]
    T: Optional[float]
    expiry_idx: Optional[int]


class InstrumentRegistry:
    def __init__(self, cfg, strikes=None):
        self.cfg = cfg
        strikes = [cfg.strike] if strikes is None else list(strikes)
        self.instruments: dict[str, Instrument] = {
            UNDERLYING: Instrument(UNDERLYING, "underlying", None, None, None)}
        self.option_names: list[str] = []
        for j, T in enumerate(cfg.expiry_years):
            for K in strikes:
                for kind in ("call", "put"):
                    tag = "C" if kind == "call" else "P"
                    nm = f"{tag}_T{j+1}" if len(strikes) == 1 else f"{tag}_{int(K)}_T{j+1}"
                    self.instruments[nm] = Instrument(nm, kind, K, T, j)
                    self.option_names.append(nm)
        self.live = {nm: True for nm in self.instruments}

    def options(self):
        return [self.instruments[nm] for nm in self.option_names]

    def live_options(self):
        return [self.instruments[nm] for nm in self.option_names if self.live[nm]]

    def live_chain_spec(self, t):
        out = []
        for nm in self.option_names:
            if not self.live[nm]:
                continue
            ins = self.instruments[nm]
            out.append((nm, ins.kind, ins.K, max(ins.T - t, 1e-9)))
        return out

    def live_expiries(self):
        return sorted({self.instruments[nm].expiry_idx
                       for nm in self.option_names if self.live[nm]})

    def settle_expiry(self, j, S_star):
        settled = {}
        for nm in self.option_names:
            ins = self.instruments[nm]
            if ins.expiry_idx != j or not self.live[nm]:
                continue
            intrinsic = (max(S_star - ins.K, 0.0) if ins.kind == "call"
                         else max(ins.K - S_star, 0.0))
            settled[nm] = (ins.kind, ins.K, intrinsic)
            self.live[nm] = False
        return settled