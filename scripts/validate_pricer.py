import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from src.config import SimConfig
from src.pricing import HestonPricer

import time as _time
_cfgP = SimConfig()
_P = HestonPricer(_cfgP.kappa, _cfgP.theta, _cfgP.xi, _cfgP.rho, 0.0, 0.0,
                  _cfgP.cf_grid_n, _cfgP.cf_grid_u)

def _ref_price(S, K, tau, v):
    wb = max(float(_P.wbar(tau, v)), 1e-14)
    u_max = max(14.0 / np.sqrt(wb), 800.0)
    n = int(min(max(512, u_max * 0.35), 6000))
    a = _P._cf_call(np.array([S]), np.array([K]), np.array([tau]), v,
                    *_P._grid(n, u_max))[0]
    b = _P._cf_call(np.array([S]), np.array([K]), np.array([tau]), v,
                    *_P._grid(int(n * 1.6), u_max * 1.4))[0]
    assert abs(a - b) < 5e-6, "reference not converged"
    return max(b, max(S - K, 0.0))

_t0 = _time.time(); _worst = (0.0, None)
for _v in (0.005, 0.04, 0.12, 0.30):
    for _tau in (1e-4, 5e-4, 2e-3, 4e-3, 1/252, 5/252, 10/252, 30/252):
        for _K in (80.0, 90.0, 100.0, 110.0, 120.0):
            _e = abs(float(_P.price(100.0, _K, _tau, _v, "call"))
                     - _ref_price(100.0, _K, _tau, _v))
            if _e > _worst[0]:
                _worst = (_e, (_tau, _v, _K))
print(f"E3a: {4*8*5} cases in {_time.time()-_t0:.0f}s; "
      f"max |err| = {_worst[0]:.2e} at (tau={_worst[1][0]:.1e}, "
      f"v={_worst[1][1]}, K={_worst[1][2]})  [full 455-case record: 3.31e-3]")
assert _worst[0] < 5e-3

for _v in (0.005, 0.04, 0.3):
    _tg = np.linspace(2e-4, 8e-3, 1201)
    _pr = np.array([float(_P.price(100.0, 100.0, t, _v, "call")) for t in _tg])
    _d = np.abs(np.diff(_pr))
    _slope = np.abs(np.gradient(_pr, _tg))
    _exc = (_d / (0.5 * (_slope[:-1] + _slope[1:]) * (_tg[1] - _tg[0]) + 1e-30)).max()
    print(f"E3b: v={_v}: max jump / (|theta|*dtau) = {_exc:.3f}")
    assert _exc < 1.05, "branch discontinuity at the seam"

class _BadW(HestonPricer):
    W_BS = 3.0e-5
try:
    _BadW(_cfgP.kappa, _cfgP.theta, _cfgP.xi, _cfgP.rho, 0.0, 0.0, 64, 100.0)
    raise AssertionError("ladder invariant failed to fire")
except ValueError as _e:
    print(f"E3c: W_BS<->ladder invariant raises OK ({str(_e)[:58]}...)")