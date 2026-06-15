import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.config import HawkesParams, SimConfig
from src.evaluation import decompose
from src.instruments import UNDERLYING
from src.matching_engine import ASK, BID
from src.pricing import HestonPricer, bs_price
from src.rl_env import F_GLOB, F_INST, FeatureEncoder, OptionsMMEnv, RewardConfig, decode_action

import sys, time
sys.path.insert(0, ".")
import numpy as np


CFG = dict(kappa=25.0, expiry_days=(1.0, 2.0, 3.0),
           hawkes=HawkesParams(mu=(200.0, 200.0, 1200.0, 1200.0)))


def make_env(seed, collect=False, rcfg=None):
    return OptionsMMEnv(SimConfig(seed=seed, **CFG), rcfg=rcfg, collect_snaps=collect)


def random_pre(rng, N):
    pre = np.empty(4 * N + 1)
    for i in range(N):
        pre[4 * i + 0] = rng.normal(-2.0, 0.7)
        pre[4 * i + 1] = rng.normal(-2.0, 0.7)
        pre[4 * i + 2] = rng.normal(1.5, 0.5)
        pre[4 * i + 3] = rng.normal(1.5, 0.5)
    pre[-1] = rng.normal(0.0, 1.0)
    return pre


env = make_env(11)
(X, g, mask, priv), info = env.reset(seed=11)
N = env.meta["N"]
adim = env.adim
assert (N, adim) == (30, 121), (N, adim)
assert X.shape == (30, F_INST) and g.shape == (F_GLOB,) and priv.shape == (env.PRIV_DIM,)
assert (F_INST, F_GLOB, env.PRIV_DIM) == (13, 33, 7)

fakeR = np.full(N, 5.0)
live = env._raw_obs["clock"]["live_mask"]
pos0 = np.zeros(N)
nd = 37.0
q, h, sk = decode_action(np.zeros(adim), fakeR, nd, pos0, live, env.meta, env.cfg, env.ecfg)
assert len(q) == 30
for nm, (bpx, apx, bq, aq) in q.items():
    assert bpx is not None and apx is not None and bpx < apx, (nm, bpx, apx)
assert abs(h - (-0.5 * nd)) < 1e-9, h
pre = np.zeros(adim)
for i in range(N):
    pre[4 * i + 3] = -50.0
q, h, sk = decode_action(pre, fakeR, 0.0, pos0, live, env.meta, env.cfg, env.ecfg)
assert all(aq == 0.0 for (_b, _a, _bq, aq) in q.values())
assert all(bq > 0.0 for (_b, _a, bq, _aq) in q.values())

q, _, _ = decode_action(np.full(adim, -50.0), fakeR, 0.0, pos0, live, env.meta, env.cfg,
                        env.ecfg)
for nm, (bpx, apx, bq, aq) in q.items():
    assert bpx is not None and apx is not None and (apx - bpx) >= env.cfg.tick - 1e-12

nm0 = env.meta["names"][0]
posv = np.zeros(N); posv[0] = env.ecfg.max_inventory
q, _, _ = decode_action(np.zeros(adim), fakeR, 0.0, posv, live, env.meta, env.cfg, env.ecfg)
assert q[nm0][2] == 0.0 and q[nm0][3] > 0.0, "long cap must kill the bid only"
posv[0] = -env.ecfg.max_inventory
q, _, _ = decode_action(np.zeros(adim), fakeR, 0.0, posv, live, env.meta, env.cfg, env.ecfg)
assert q[nm0][3] == 0.0 and q[nm0][2] > 0.0, "short cap must kill the ask only"
assert q[env.meta["names"][1]][2] > 0.0 and q[env.meta["names"][1]][3] > 0.0
print("T2 decode invariants (+ inventory gate)       PASS")

rcfg = RewardConfig()
env = make_env(7, collect=True, rcfg=rcfg)
(X, g, mask, priv), info = env.reset(seed=7)
rng = np.random.default_rng(0)
t0 = time.time()
term = trunc = False
n = 0
finite_ok = True
mask_path = [int(mask.sum())]
n_fills_opt = 0
n_hedge_mo = 0
rew_seq = []
risk_seq = []
dn_seq, dt_seq, gam_seq, shp_seq = [], [], [], []
max_abs_pos = 0.0
while not (term or trunc) and n < 60000:
    pre = random_pre(rng, N)
    (X, g, mask, priv), r, term, trunc, ri = env.step(pre)
    rew_seq.append(r)
    risk_seq.append(ri["risk"])
    dn_seq.append(ri["dn"]); dt_seq.append(ri["dt"])
    gam_seq.append(ri["gam"]); shp_seq.append(ri["shaping"])
    max_abs_pos = max(max_abs_pos, float(np.max(np.abs(env.enc.pos))))
    finite_ok &= bool(np.isfinite(X).all() and np.isfinite(g).all()
                      and np.isfinite(priv).all() and np.isfinite(r))
    mask_path.append(int(mask.sum()))
    n += 1
wall = time.time() - t0
assert term and not trunc, (term, trunc)
assert finite_ok, "non-finite feature/reward encountered"
mp = np.array(mask_path)
drops = sorted(set(mp.tolist()), reverse=True)
assert drops == [30, 20, 10, 0], drops
assert np.all(env.enc.pos[~mask] == 0.0) if (~mask).any() else True
assert abs(priv[-1] - np.log(env.env.eng.regime)) < 1e-6
print(f"T1 features finite over {n} steps ({wall:.0f}s)      PASS")
print(f"T4 live-mask cascade {drops}                PASS")

ep = env.episode_dict(term, trunc)
dc = decompose(ep)
edge_dc = dc["edge_opt"][-1] + dc["edge_hedge"][-1]
diff = abs(env.ep_edge - edge_dc)
n_fills_opt = int(dc["fills_opt"][-1])
n_hedge = int(dc["fills_hedge"][-1])
print(f"   option fills={n_fills_opt}  hedge fills={n_hedge}  "
      f"edge(wrapper)={env.ep_edge:+.6f}  edge(decompose)={edge_dc:+.6f}")
assert n_fills_opt > 50, "too few option fills to trust the reconciliation"
assert n_hedge > 5, "hedge channel not exercised"
assert diff <= 1e-9, f"SS8 reconciliation FAILED: |diff|={diff:.3e}"
print(f"T3 SS8 reconciliation |diff|={diff:.2e}          PASS")

rs = np.array(rew_seq)
ks = np.array(risk_seq)
print(f"   reward/step: mean={rs.mean():+.5f}  p1={np.percentile(rs, 1):+.4f}  "
      f"p99={np.percentile(rs, 99):+.4f}  max|r|={np.abs(rs).max():.3f}")
print(f"   risk/step:   mean={ks.mean():.6f}  max={ks.max():.4f}  "
      f"ep_risk={env.ep_risk:.4f}  max|pos|={max_abs_pos:.1f}")
assert np.abs(rs).max() <= rcfg.r_clip + 1e-9, "reward escaped the r_clip bound"
assert ks.max() <= rcfg.risk_cap + 1e-9, "per-step risk charge escaped risk_cap"
assert ks.min() >= 0.0
assert max_abs_pos <= env.ecfg.max_inventory + env.ecfg.max_order_qty + 1e-9, \
    (max_abs_pos, env.ecfg.max_inventory)
print("T7 reward v2 bounds / inventory cap            PASS")

def run_short(seed, steps=400):
    e = make_env(seed)
    e.reset(seed=seed)
    r = np.random.default_rng(123)
    out = []
    for _ in range(steps):
        (Xa, ga, ma, pa), rr, tm, tr, _ = e.step(random_pre(r, e.meta["N"]))
        out.append(rr)
        if tm or tr:
            break
    return np.array(out)

a, b = run_short(5), run_short(5)
assert a.shape == b.shape and np.array_equal(a, b), "same-seed runs diverged"
print("T5 same-seed determinism                      PASS")

env6 = make_env(3)
env6.reset(seed=3)
meta = env6.meta
names = meta["names"]
ecfg6 = env6.ecfg
cfg6 = env6.cfg
enc = FeatureEncoder(cfg6, meta, ecfg6)

def frame(t, tapes=None, mids=None):
    tapes = tapes or {}
    mids = mids or {}
    e0 = lambda: np.zeros((0, 2), np.float32)
    obs = {}
    for nm in names:
        mm = mids.get(nm)
        bbo = (mm - 0.05, 5.0, mm + 0.05, 5.0) if mm is not None else (np.nan,) * 4
        obs[nm] = dict(bbo=bbo, depth_bid=e0(), depth_ask=e0(),
                       tape=list(tapes.get(nm, [])), own_fills=[])
    obs[UNDERLYING] = dict(bbo=(99.95, 5.0, 100.05, 5.0), depth_bid=e0(), depth_ask=e0(),
                           tape=list(tapes.get(UNDERLYING, [])), own_fills=[])
    obs["clock"] = dict(t_now=t, ttm={},
                        t_remaining_episode=max(cfg6.T_horizon - t, 0.0),
                        live_mask={**{nm: True for nm in names}, UNDERLYING: True})
    return obs

t0_, dt1, dt2 = 1.0e-5, 4.0e-5, 2.0e-5
nm0 = names[0]
X6, g6, m6 = enc.encode(frame(t0_, tapes={
    nm0: [(1.0, 3.0, BID, t0_), (1.0, 2.0, BID, t0_)],
    UNDERLYING: [(100.0, 4.0, ASK, t0_)]}))
assert enc._flow_ewma[0] == 5.0 and enc._int_ewma[0] == 2.0
assert enc._und_flow_ewma == -4.0 and enc._und_int_ewma == 1.0 and enc._reg_ewma == 3.0
assert np.all(enc._iv_age == 0.0)

fb, rb = ecfg6.flow_beta, ecfg6.regime_beta
X6, g6, m6 = enc.encode(frame(t0_ + dt1, tapes={nm0: [(1.0, 1.0, ASK, t0_ + dt1)]}))
f_exp = 5.0 * np.exp(-fb * dt1) - 1.0
i_exp = 2.0 * np.exp(-fb * dt1) + 1.0
assert abs(enc._flow_ewma[0] - f_exp) < 1e-12, (enc._flow_ewma[0], f_exp)
assert abs(enc._int_ewma[0] - i_exp) < 1e-12
assert abs(enc._und_flow_ewma - (-4.0) * np.exp(-fb * dt1)) < 1e-12
assert abs(enc._und_int_ewma - 1.0 * np.exp(-fb * dt1)) < 1e-12
assert abs(enc._reg_ewma - (3.0 * np.exp(-rb * dt1) + 1.0)) < 1e-12
assert abs(X6[0, 10] - enc._flow_ewma[0] / ecfg6.flow_scale) < 1e-6
assert abs(X6[0, 11] - enc._int_ewma[0] / ecfg6.ewma_count_scale) < 1e-6
assert abs(g6[30] - enc._und_flow_ewma / ecfg6.flow_scale) < 1e-6
assert abs(g6[32] - enc._reg_ewma * rb / ecfg6.regime_rate_scale) < 1e-6
assert np.allclose(enc._iv_age, dt1), "all pairs stale by dt (no finite option mid yet)"
assert np.allclose(X6[:, 12], dt1 / cfg6.T_horizon, atol=1e-6)

ia = next(i for i in range(meta["N"])
          if meta["K"][i] == 100.0 and meta["expiry_idx"][i] == 0 and meta["kind"][i] > 0)
X6, g6, m6 = enc.encode(frame(t0_ + dt1 + dt2, mids={names[ia]: 0.30}))
p_atm = int(meta["pair_of"][ia])
assert enc._iv_age[p_atm] == 0.0, "successful inversion must reset staleness"
assert np.allclose(np.delete(enc._iv_age, p_atm), dt1 + dt2)
assert ecfg6.iv_lo <= enc._iv[p_atm] <= ecfg6.iv_hi
assert abs(enc._flow_ewma[0] - f_exp * np.exp(-fb * dt2)) < 1e-12
assert abs(X6[ia, 12]) < 1e-12
print("T6 flow-EWMA / staleness closed form          PASS")

dts = np.array(dt_seq); gams = np.array(gam_seq)
dns = np.array(dn_seq); shps = np.array(shp_seq)
assert (dts >= 0.0).all()
assert np.max(np.abs(gams - np.exp(-rcfg.discount_rho * dts))) < 1e-12
dn_prev = np.concatenate([[0.0], dns[:-1]])
tele = rcfg.A * (np.abs(dn_prev) - gams * np.abs(dns))
assert np.max(np.abs(shps - tele)) < 1e-12, "shaping is not the exact telescope"
print(f"T8 time-discount + telescope (rho={rcfg.discount_rho:.0f}/yr) exact    PASS")

env9 = make_env(13)
env9.reset(seed=13)
r9 = np.random.default_rng(9)
for _ in range(10):
    env9.step(random_pre(r9, env9.meta["N"]))
enc9, raw9 = env9.enc, env9._raw_obs
nm9 = next(nm for nm in env9.meta["names"]
           if nm in raw9 and np.isfinite(raw9[nm]["bbo"][0])
           and np.isfinite(raw9[nm]["bbo"][2])
           and raw9[nm]["bbo"][2] - raw9[nm]["bbo"][0] > 2.1 * env9.cfg.tick)

d0 = dict(raw9[nm9], own_orders=[])
m0 = enc9._obs_mid(d0)
bp9, _, ap9, asz9 = d0["bbo"]
own_px = env9.cfg.tick_to_price(env9.cfg.price_to_tick(bp9) + 1)
d1 = dict(d0, own_orders=[(999, BID, own_px, 3.0)],
          depth_bid=np.vstack([[own_px, 3.0], d0["depth_bid"]]).astype(np.float32),
          bbo=(own_px, 3.0, ap9, asz9))
assert abs(enc9._obs_mid(d1) - m0) > 1e-9, "test setup: BBO must actually move"
assert abs(enc9._ex_self_mid(d1) - m0) < 1e-9, "ex-self must remove the own order"
d2 = dict(d0, own_orders=[(999, BID, float(d0["depth_bid"][0, 0]), 2.0)])
db2 = d0["depth_bid"].copy(); db2[0, 1] += 2.0
d2["depth_bid"] = db2
assert abs(enc9._ex_self_mid(d2) - m0) < 1e-9, "own-alongside-external must be invisible"
assert abs(enc9._ex_self_mid(d0) - m0) < 1e-12, "no own orders -> as-is mid"

assert np.isfinite(enc9.R_quote[env9.meta["names"].index(nm9)])
print("T9 ex-self quote anchor reconstruction        PASS")

env10 = OptionsMMEnv(SimConfig(seed=3, max_steps=3, **CFG))
env10.reset(seed=3)
r10 = np.random.default_rng(2)
for _ in range(3):
    *_, ri_n = env10.step(random_pre(r10, env10.meta["N"]))
assert ri_n["noop"] is False, "normal steps must not be flagged"
t_frozen = env10.env.eng.t
(_, _, _, _), r_tr, tm10, tr10, ri10 = env10.step(random_pre(r10, env10.meta["N"]))
assert tr10 and not tm10
assert r_tr == 0.0 and ri10["dt"] == 0.0 and ri10["n_fills"] == 0, (r_tr, ri10)
assert ri10["noop"] is True, "the gated cap step must be flagged for the buffer skip"
assert env10.env.eng.t == t_frozen, "engine advanced on the truncation step"
print("T10 truncation no-op (gated submit + flag)     PASS")

cfgP = SimConfig(seed=0, **CFG)
P = HestonPricer(cfgP.kappa, cfgP.theta, cfgP.xi, cfgP.rho, cfgP.r, cfgP.q,
                 cfgP.cf_grid_n, cfgP.cf_grid_u)
S11 = 100.0
Ks11 = np.array([80., 90., 95., 100., 105., 110., 120.])
taus11 = sorted([1e-4, 5e-4, 2e-3, 4e-3, 1 / 252, 5 / 252, 10 / 252, 30 / 252])
for v11 in (0.005, 0.04, 0.3):
    prev = None
    for tau in taus11:
        c = P.price(S11, Ks11, tau, v11, "call")
        q_ = P.price(S11, Ks11, tau, v11, "put")
        assert np.max(np.abs((c - q_) - (S11 - Ks11))) < 1e-9, "parity"
        assert (c[:-2] - 2 * c[1:-1] + c[2:] > -1e-6).all(), "butterfly"
        if prev is not None:
            assert ((c - prev) > -2e-5).all(), ("calendar", v11, tau)
        prev = c
    wb11 = float(P.wbar(5e-4, v11))
    if wb11 < P.W_CF:
        sig_eff = np.sqrt(wb11 / 5e-4)
        gap = abs(float(P.price(S11, 100.0, 5e-4, v11, "call"))
                  - float(np.asarray(bs_price(S11, 100.0, 5e-4, sig_eff))))
        assert gap < 2e-3, ("BS_eff handoff gap", v11, gap)
    c0 = float(P.price(S11, 100.0, 1e-9, v11, "call"))
    assert abs(c0) < 1e-2, "tau->0 must collapse to intrinsic"
print("T11 pricer: parity/butterfly/calendar/handoff   PASS")


rc12 = RewardConfig(lambda_hedge=0.5)
env12 = OptionsMMEnv(SimConfig(seed=11, max_steps=400, **CFG), rcfg=rc12)
env12.reset(seed=11)
r12 = np.random.default_rng(5)
saw_hedge = False
for _k in range(250):
    pre12 = random_pre(r12, env12.meta["N"])
    pre12[-1] = 6.0
    (_, _, _, _), _r, tm12, tr12, ri12 = env12.step(pre12)
    und_fills = env12._raw_obs[UNDERLYING].get("own_fills", [])
    assert abs(ri12["hedge_qty"] - sum(abs(sg) for (_o, _p, sg, _t) in und_fills)) < 1e-12
    expect12 = (rc12.lambda_edge * ri12["edge"] / env12.ecfg.W0 + ri12["shaping"]
                - ri12["risk"] - rc12.lambda_hedge * ri12["hedge_qty"])
    assert abs(ri12["r_raw"] - expect12) < 1e-12, (_k, ri12["r_raw"], expect12)
    saw_hedge |= ri12["hedge_qty"] > 0
    if tm12 or tr12:
        break
assert saw_hedge, "T12 never hedged -- raise steps or quoting aggression"
print("T12 hedging penalty: executed-qty identity     PASS")

print("\nALL INTEGRATION TESTS PASSED")