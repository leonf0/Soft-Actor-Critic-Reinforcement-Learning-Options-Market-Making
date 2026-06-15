import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from src.config import HawkesParams, SimConfig, demo_regime
from src.rl_env import OptionsMMEnv

import time as _time
_rngE2 = np.random.default_rng(0)

def _measure(env, n_cap, label):
    env.reset(seed=2)
    A_ = 4 * env.meta["N"] + 1
    eng = env.env.eng
    dts, evs = [], []
    term = trunc = False
    t0 = _time.time()
    for _ in range(n_cap):
        e0, tt0 = eng.event_count, eng.t
        _, r_, term, trunc, _ri = env.step(_rngE2.normal(0.0, 0.61, A_))
        dts.append(eng.t - tt0); evs.append(eng.event_count - e0)
        if term or trunc:
            break
    dts, evs = np.array(dts), np.array(evs)
    print(f"{label}: steps={len(dts)} term={term} ({_time.time()-t0:.0f}s)  "
          f"dt mean={dts.mean():.2e} median={np.median(dts):.2e}  "
          f"events/step={evs.mean():.2f}  sim-time={dts.sum()*252:.2f}d")
    return len(dts), term

_measure(OptionsMMEnv(demo_regime(seed=2)), 600, "demo_regime (600-step window)")
_TRAIN_E2 = dict(kappa=25.0, expiry_days=(2.0, 4.0, 6.0),
                 hawkes=HawkesParams(mu=(400.0, 400.0, 2400.0, 2400.0)),
                 max_steps=12000)
_nsteps, _term = _measure(OptionsMMEnv(SimConfig(seed=2, **_TRAIN_E2)), 12000,
                          "train_regime (FULL episode)")
assert _term, "train regime must terminate naturally within the cap"
print(f"-> main-run sizing: full episode = {_nsteps} steps; max_steps=6000 is a "
      f"pure safety cap; per env one episode every ~{int(np.ceil(_nsteps/128))} "
      f"iterations at 128 steps/iter")