import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.config import HawkesParams, SimConfig
from src.rl_env import F_GLOB, F_INST, OptionsMMEnv
from src.sac import ActionBounds, ObsNormalizer, ReplayBuffer, SACConfig, SACLearner, SyncVectorEnv, build_sac_nets, save_checkpoint

import copy, numpy as np, torch

TINY = dict(seed=3, kappa=25.0, expiry_days=(1.0, 2.0, 3.0),
            hawkes=HawkesParams(mu=(200.0, 200.0, 1200.0, 1200.0)), max_steps=40)
make_env = lambda: OptionsMMEnv(SimConfig(**TINY))
rng = np.random.default_rng(0)

venv = SyncVectorEnv([make_env], base_seed=7)
obs = venv.reset_all()
N = venv.envs[0].meta["N"]; A = 4 * N + 1; P = venv.envs[0].PRIV_DIM
buf = ReplayBuffer(2_000, N, F_INST, F_GLOB, P, A)
lo_t, hi_t = ActionBounds().per_type()
dim_type = np.concatenate([np.tile(np.arange(4), N), [4]])
mid = 0.5 * (lo_t.numpy() + hi_t.numpy())[dim_type]
half = 0.5 * (hi_t.numpy() - lo_t.numpy())[dim_type]
seen_trunc = False
stored = 0
chain_breaks = set()

for t in range(90):
    X, g, m, p = obs
    live = np.concatenate([np.repeat(m.astype(np.float32), 4, axis=1),
                           np.ones((1, 1), np.float32)], axis=1)
    a = (rng.uniform(-1, 1, size=(1, A)).astype(np.float32) * live)
    pre = mid[None, :] + half[None, :] * a
    obs2, (r, term, trunc, finals, infos) = venv.step(pre)
    dt1 = np.array([infos[0]["dt"]], np.float32)
    X2e, g2e, m2e, p2e = (obs2[0].copy(), obs2[1].copy(), obs2[2].copy(), obs2[3].copy())
    if finals[0] is not None:
        fX, fg, fm, fp = finals[0]
        X2e[0], g2e[0], m2e[0], p2e[0] = fX, fg, fm, fp
        seen_trunc = True

        assert not (np.allclose(fg, obs2[1][0]) and np.allclose(fX, obs2[0][0])), \
            "finals passthrough returned reset obs"
        assert bool(trunc[0]) and not bool(term[0]), "40-step cut must be trunc, not term"
    if infos[0].get("noop", False):
        assert bool(trunc[0]) and float(r[0]) == 0.0 and infos[0]["dt"] == 0.0
        chain_breaks.add(stored - 1)
    else:
        buf.add_batch(X, g, m, p, a, r, dt1, X2e, g2e, m2e, p2e,
                      term.astype(np.float32), trunc.astype(np.float32))
        stored += 1
    obs = obs2
assert seen_trunc and len(chain_breaks) >= 2 and buf.size >= 80

for t in range(buf.size - 1):
    if t in chain_breaks:
        continue
    if buf.term[t] == 0 and buf.trunc[t] == 0:
        assert torch.equal(buf.X2[t], buf.X[t + 1]) and torch.equal(buf.g2[t], buf.g[t + 1])
        assert torch.equal(buf.m2[t], buf.m[t + 1])
print(f"[A] buffer semantics OK: {buf.size} stored ({len(chain_breaks)} no-ops "
      f"skipped), s'-chaining exact off-boundary, finals!=reset, flags correct")

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
actor, q1, q2 = build_sac_nets(N, priv_dim=P)
cfg_t = SACConfig(critic_lr=1e-3, discount_rho=100.0)
learner = SACLearner(actor, q1, q2, cfg_t, dev)
b = buf.sample(64, dev)
with torch.no_grad():
    gam_b = torch.exp(-cfg_t.discount_rho * b["dt"])
    y_fix = b["r"] + gam_b * (1 - b["term"]) * torch.randn_like(b["r"]) * 0.1
l0 = None
for k in range(300):
    q1v = learner.q1(b["X"], b["g"], b["m"], b["p"], b["a"])
    q2v = learner.q2(b["X"], b["g"], b["m"], b["p"], b["a"])
    loss = ((q1v - y_fix) ** 2).mean() + ((q2v - y_fix) ** 2).mean()
    learner.critic_opt.zero_grad(set_to_none=True)
    loss.backward()
    learner.critic_opt.step()
    if k == 0:
        l0 = float(loss)
lr_ratio = float(loss) / l0
print(f"[B] critic overfit: loss {l0:.4f} -> {float(loss):.5f} (ratio {lr_ratio:.4f})")
assert lr_ratio < 0.05, "critic cannot memorize one batch — wiring/gradient bug"

def alpha_step_direction(lp_val, n_live_val=121):
    la = torch.tensor(float(np.log(0.2)), requires_grad=True)
    opt = torch.optim.Adam([la], lr=3e-4)
    lp = torch.full((32,), float(lp_val))
    H_tgt = torch.full((32,), -1.0 * n_live_val)
    (-(la * (lp + H_tgt)).mean()).backward()
    opt.step()
    return float(la) - float(np.log(0.2))
d_up = alpha_step_direction(lp_val=+200.0)
d_dn = alpha_step_direction(lp_val=-300.0)
assert d_up > 0 and d_dn < 0, (d_up, d_dn)
print(f"[C] alpha direction OK: dlog_alpha={d_up:+.2e} when under-exploring, "
      f"{d_dn:+.2e} when over-exploring")

tw0 = learner.q1_t.q_head[-1].weight.detach().clone()
la0 = float(learner.log_alpha)
for _ in range(10):
    mets = learner.update(buf.sample(64, dev), stats=None)
    assert all(np.isfinite(v) for v in mets.values()), mets
assert not torch.allclose(tw0, learner.q1_t.q_head[-1].weight), "Polyak frozen"

assert float(learner.log_alpha) < la0, (float(learner.log_alpha), la0)

norm_m = ObsNormalizer(priv_dim=P)
save_checkpoint("sac_mech_ckpt.pt", learner, norm_m, cfg_t, it=10, total_steps=buf.size)
actor2, q12, q22 = build_sac_nets(N, priv_dim=P)
learner2 = SACLearner(actor2, q12, q22, cfg_t, dev)
ck = torch.load("sac_mech_ckpt.pt", map_location=dev, weights_only=True)
learner2.load_state_dict(ck["learner"])
for pa, pb in zip(learner.actor.parameters(), learner2.actor.parameters()):
    assert torch.allclose(pa, pb)
for pa, pb in zip(learner.q1_t.parameters(), learner2.q1_t.parameters()):
    assert torch.allclose(pa, pb)
assert abs(float(learner.log_alpha) - float(learner2.log_alpha)) < 1e-7
print(f"[D] 10 real-batch updates finite (last qL={mets['q_loss']:.3f}, "
      f"a={mets['alpha']:.3f}, ent/d={mets['ent_pd']:+.2f}); checkpoint roundtrip exact")
print("\nSAC MECHANICS TESTS PASSED")