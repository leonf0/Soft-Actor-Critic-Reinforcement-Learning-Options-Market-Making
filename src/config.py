from __future__ import annotations
from dataclasses import dataclass, field
import zlib
import numpy as np

TRADING_DAYS = 252.0

class RNGStreams:
    def __init__(self, seed: int):
        self.seed = int(seed)
        self._root = np.random.SeedSequence(self.seed)
        self._cache: dict[str, np.random.Generator] = {}

    def stream(self, name: str) -> np.random.Generator:
        if name not in self._cache:
            tag = zlib.crc32(name.encode()) % (2 ** 31)
            child = np.random.SeedSequence(entropy=self.seed, spawn_key=(tag,))
            self._cache[name] = np.random.default_rng(child)
        return self._cache[name]

HAWKES_DIMS = ("MB", "MS", "LB", "LS")

@dataclass
class HawkesParams:
    mu: tuple = (1500.0, 1500.0, 9000.0, 9000.0)
    alpha: tuple = (
        (180.0,  60.0,  20.0,  10.0),
        (60.0, 180.0,  10.0,  20.0),
        (40.0,  20.0, 300.0,  80.0),
        (20.0,  40.0,  80.0, 300.0),
    )
    beta: tuple = (1200.0, 1200.0, 2000.0, 2000.0)

    def as_arrays(self):
        return (np.array(self.mu, float),
                np.array(self.alpha, float),
                np.array(self.beta, float))

@dataclass
class SimConfig:
    seed: int = 0

    r: float = 0.0
    q: float = 0.0
    kappa: float = 6.0
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7
    S0: float = 100.0
    v0_mode: str = "stationary"
    v0_fixed: float = 0.04
    psi_c: float = 1.5
    variance_scheme: str = "qe"

    impact_g: float = 2.0e-4
    impact_decay: float = 200.0
    agent_impact: bool = False

    strike: float = 100.0
    expiry_days: tuple = (10.0, 20.0, 30.0)

    fixture_strikes: tuple = (90.0, 95.0, 100.0, 105.0, 110.0)

    tick: float = 0.1
    price_lo: float = 0.0
    price_hi: float = 1000.0

    hawkes: HawkesParams = field(default_factory=HawkesParams)
    regime_logmu_sigma: float = 0.25
    bg_depth_ticks_scale: float = 3.0
    bg_size_lognorm_mean: float = 0.0
    bg_size_lognorm_sigma: float = 0.5
    bg_size_scale: float = 5.0
    bg_lifetime_rate: float = 700.0
    bg_distance_bump: float = 0.25
    half_spread_floor_ticks: float = 1.0
    reprice_dS_frac: float = 5e-4
    reprice_dv_frac: float = 1e-2
    reprice_dt: float = 2e-4
    obs_depth_N: int = 5
    burn_in_years: float = 1.0e-2
    max_steps: int = 100000
    max_events_per_step: int = 500000
    event_budget: int = 5000000
    cf_grid_n: int = 64
    cf_grid_u: float = 100.0

    def __post_init__(self):
        self.expiry_years = tuple(d / TRADING_DAYS for d in self.expiry_days)
        self.T_horizon = self.expiry_years[-1]

    @property
    def n_ticks(self) -> int:
        return int(round((self.price_hi - self.price_lo) / self.tick))

    def price_to_tick(self, px: float) -> int:
        return int(round((px - self.price_lo) / self.tick))

    def tick_to_price(self, t: int) -> float:
        return self.price_lo + t * self.tick

    def validate(self) -> "SimConfig":
        def _req(cond, msg):
            if not cond:
                raise ValueError(msg)
        kT = self.kappa * self.T_horizon
        _req(0.2 <= kT <= 3.0,
             f"kappa*T3={kT:.3f} outside ~[0.2,3.0]: variance would be near-frozen "
             "or over-reverting within an episode (SS10.3).")
        _req(self.price_lo < self.S0 < self.price_hi,
             "S0 outside the price grid")

        for K in tuple(self.fixture_strikes) + (self.strike,):
            _req(self.price_lo < K < self.price_hi,
                 f"strike {K} outside price grid [{self.price_lo}, {self.price_hi}]: "
                 "quotes around it would be silently dropped (matching_engine.py).")

        sig_S = self.S0 * np.sqrt(self.theta * self.T_horizon)
        _req(min(self.S0 - self.price_lo, self.price_hi - self.S0) > 6 * sig_S,
             "price grid bounds too tight vs expected S range (SS6.3)")

        mu, alpha, beta = self.hawkes.as_arrays()
        Gamma = alpha / beta[None, :]
        eta = max(abs(np.linalg.eigvals(Gamma)))
        _req(eta < 1.0, f"Hawkes branching ratio eta={eta:.3f} >= 1 (non-stationary)")
        _req(self.variance_scheme in ("qe", "exact"), "unknown variance_scheme")
        _req(self.v0_mode in ("stationary", "fixed"), "unknown v0_mode")
        return self

def headline_regime(seed: int = 0) -> SimConfig:
    return SimConfig(seed=seed).validate()


def demo_regime(seed: int = 0) -> SimConfig:
    hk = HawkesParams(mu=(400.0, 400.0, 2400.0, 2400.0))
    return SimConfig(seed=seed, hawkes=hk).validate()