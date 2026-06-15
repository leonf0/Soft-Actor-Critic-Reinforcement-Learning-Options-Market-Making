import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from scipy.stats import norm as _normE5
_sp = lambda x: np.log1p(np.exp(-abs(x))) + max(x, 0.0)
print(f"a=0  -> off {_sp(0):.4f} px | qty {_sp(0):.4f} (~1 lot) | f 0.5   "
      f"== PPO init behaviour (parity)")
print(f"a=+1 -> off {_sp(3):.3f} px | qty {_sp(20):.4f} = max_order_qty")
print(f"a=-1 -> off {_sp(-3):.4f} px (touch reachable) | qty {_sp(-20):.1e}")
_a = np.linspace(-1, 1, 400001)
_off = np.array([_sp(3.0 * x) for x in _a])
_q = np.array([_sp(20.0 * x) for x in _a])
_p_sub = float(np.mean(_off < 0.1)); _p_zero = float(np.mean(_q < 0.5))
_thr = np.log(np.exp(0.5) - 1) / 20.0
_p_zero_init = float(_normE5.cdf(np.arctanh(_thr) / 0.61))
print(f"plateaus: P(off < 1 tick) = {_p_sub:.3f}   "
      f"P(qty -> 0 lots | uniform a) = {_p_zero:.3f}   "
      f"P(qty -> 0 | init policy) = {_p_zero_init:.3f}")
assert abs(_p_sub - 0.125) < 0.01 and abs(_p_zero_init - 0.486) < 0.01