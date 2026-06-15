import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.baselines import AvellanedaStoikovAgent
from src.config import HawkesParams, SimConfig
from src.evaluation import decompose, run_episode
from src.rl_env import EnvConfig, OptionsMMEnv, RewardConfig
from src.sac import ObsNormalizer, SACConfig, SACLearner, build_sac_nets, run_sac_episode
from src.simulator import OptionsMMSimulator

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats as _st

N_SEEDS          = 50
EVAL_SEEDS       = list(range(10_000, 10_000 + N_SEEDS))

EVAL_REGIME      = dict(kappa=25.0, expiry_days=(2.0, 4.0, 6.0),
                        hawkes=HawkesParams(mu=(400.0, 400.0, 2400.0, 2400.0)),
                        max_steps=12000)
AS_GAMMA         = 0.10
SAC_DETERMINISTIC = True

MAX_STEPS_EVAL   = None
BOOTSTRAP_N      = 10_000
SAVE_PREFIX      = "mc_sac_vs_as"
RNG_BOOT         = np.random.default_rng(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


try:
    _actor = learner.actor.to(DEVICE)
    _norm  = normalizer
    print("using in-scope `learner.actor` / `normalizer`")
except NameError:
    _ckpt_path = SACConfig().checkpoint_path
    print(f"`learner`/`normalizer` not found -> loading checkpoint {_ckpt_path!r}")
    _probe = OptionsMMEnv(SimConfig(seed=0, **EVAL_REGIME))
    _probe.reset(seed=0)
    _N, _P = _probe.meta["N"], _probe.PRIV_DIM
    _a, _q1, _q2 = build_sac_nets(_N, priv_dim=_P)
    _ln = SACLearner(_a, _q1, _q2, SACConfig(discount_rho=RewardConfig().discount_rho),
                     DEVICE)
    _norm = ObsNormalizer(priv_dim=_P)
    _ck = torch.load(_ckpt_path, map_location=DEVICE, weights_only=True)
    _ln.load_state_dict(_ck["learner"]); _norm.load_state_dict(_ck["normalizer"])
    _actor = _ln.actor.to(DEVICE)
_actor.eval()


class _FixtureSimMC(OptionsMMSimulator):
    def reset(self, seed=None, strikes=None):
        return super().reset(seed=seed, strikes=self.cfg.fixture_strikes)


def _summarize(dc, ep):
    d, vega = dc["true_delta"], dc["true_vega"]
    return dict(
        W            = float(dc["W"][-1]),
        edge_opt     = float(dc["edge_opt"][-1]),
        edge_hedge   = float(dc["edge_hedge"][-1]),
        settle       = float(dc["settle"][-1]),
        reval        = float(dc["reval"][-1]),
        fills_opt    = float(dc["fills_opt"][-1]),
        fills_hedge  = float(dc["fills_hedge"][-1]),
        delta_max    = float(np.nanmax(np.abs(d))),
        delta_mean   = float(np.nanmean(np.abs(d))),
        vega_max     = float(np.nanmax(np.abs(vega))),
        vega_mean    = float(np.nanmean(np.abs(vega))),
        terminated   = bool(ep["terminated"]),
        truncated    = bool(ep["truncated"]),
        resid        = float(dc["max_abs_residual"]),
    )


def eval_sac(seed):
    env = OptionsMMEnv(SimConfig(seed=seed, **EVAL_REGIME),
                       rcfg=RewardConfig(), collect_snaps=True)
    out = run_sac_episode(env, _actor, _norm, seed=seed,
                          deterministic=SAC_DETERMINISTIC, device=DEVICE,
                          max_steps=MAX_STEPS_EVAL)
    return _summarize(decompose(out["episode"]), out["episode"])


def eval_as(seed):
    cfg = SimConfig(seed=seed, **EVAL_REGIME).validate()
    env = _FixtureSimMC(cfg)
    agent = AvellanedaStoikovAgent(cfg, gamma=AS_GAMMA)
    ep = run_episode(env, agent, seed=seed, max_steps=MAX_STEPS_EVAL)
    return _summarize(decompose(ep), ep)


print(f"\nMC: {N_SEEDS} seeds x {{SAC(det={SAC_DETERMINISTIC}), A-S(gamma={AS_GAMMA})}} "
      f"on regime {EVAL_REGIME['expiry_days']}  device={DEVICE}")
rows = {"sac": [], "as": []}
t0 = time.time()
for i, seed in enumerate(EVAL_SEEDS):
    for arm, fn in (("sac", eval_sac), ("as", eval_as)):
        try:
            rows[arm].append(fn(seed))
        except Exception as e:
            print(f"  ! seed {seed} arm {arm} failed: {type(e).__name__}: {e}")
            rows[arm].append({k: np.nan for k in
                              ("W", "edge_opt", "edge_hedge", "settle", "reval",
                               "fills_opt", "fills_hedge", "delta_max", "delta_mean",
                               "vega_max", "vega_mean", "resid")}
                             | {"terminated": False, "truncated": True})
    if (i + 1) % 5 == 0 or i + 1 == N_SEEDS:
        el = time.time() - t0
        print(f"  [{i+1:>3}/{N_SEEDS}] {el:6.0f}s  ({el/(i+1):.1f}s/seed)")


def _col(arm, key):
    return np.array([r[key] for r in rows[arm]], float)


W_sac, W_as = _col("sac", "W"), _col("as", "W")
ok = np.isfinite(W_sac) & np.isfinite(W_as)
n_ok = int(ok.sum())
delta = W_sac[ok] - W_as[ok]

n_trunc = int((~_col("sac", "terminated").astype(bool)).sum()
              + (~_col("as", "terminated").astype(bool)).sum())
max_resid = float(np.nanmax([_col("sac", "resid"), _col("as", "resid")]))
if n_trunc:
    print(f"\n  WARNING: {n_trunc} episode(s) did not terminate naturally; their "
          f"W is a mark-to-model value, not a settled one.")
if max_resid > 1e-3:
    print(f"  WARNING: max P&L-decomposition residual {max_resid:.2e} (expected ~0) "
          f"-> check wealth accounting.")

def _ci(x):
    bs = np.array([RNG_BOOT.choice(x, size=x.size, replace=True).mean()
                   for _ in range(BOOTSTRAP_N)])
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


mean_sac, mean_as = float(W_sac[ok].mean()), float(W_as[ok].mean())
md_lo, md_hi = _ci(delta)
win = float(np.mean(delta > 0))
dz = float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) > 0 else float("nan")
p_w = (_st.wilcoxon(W_sac[ok], W_as[ok], zero_method="wilcox").pvalue
       if np.any(delta != 0) else float("nan"))
p_t = (_st.ttest_rel(W_sac[ok], W_as[ok]).pvalue
       if delta.std(ddof=1) > 0 else float("nan"))

print("\n" + "=" * 78)
print(f"TERMINAL P&L over {n_ok} paired seeds      (regime = "
      f"{'IN-DIST / train' if EVAL_REGIME['expiry_days'] == (2.0,4.0,6.0) else 'custom'})")
print("-" * 78)
print(f"  SAC : mean {mean_sac:+8.2f}   median {np.median(W_sac[ok]):+8.2f}   "
      f"sd {W_sac[ok].std(ddof=1):7.2f}   [{W_sac[ok].min():+.1f}, {W_sac[ok].max():+.1f}]")
print(f"  A-S : mean {mean_as:+8.2f}   median {np.median(W_as[ok]):+8.2f}   "
      f"sd {W_as[ok].std(ddof=1):7.2f}   [{W_as[ok].min():+.1f}, {W_as[ok].max():+.1f}]")
print(f"  delta (SAC - A-S): mean {delta.mean():+8.2f}  95% CI [{md_lo:+.2f}, {md_hi:+.2f}]  "
      f"median {np.median(delta):+.2f}")
print(f"  win-rate {win:.0%}   Cohen's dz {dz:+.2f}   "
      f"Wilcoxon p={p_w:.2e}   paired-t p={p_t:.2e}")
print("-" * 78)
print("  mean P&L decomposition           edge_opt   edge_hdg     settle      reval")
for arm, lbl in (("sac", "SAC"), ("as", "A-S")):
    print(f"    {lbl}: {_col(arm,'edge_opt').mean():+10.2f} "
          f"{_col(arm,'edge_hedge').mean():+10.2f} {_col(arm,'settle').mean():+10.2f} "
          f"{_col(arm,'reval').mean():+10.2f}")
print("  realized risk (book greeks)     max|delta| mean|delta|  max|vega| mean|vega|  hedges")
for arm, lbl in (("sac", "SAC"), ("as", "A-S")):
    print(f"    {lbl}: {_col(arm,'delta_max').mean():10.1f} "
          f"{_col(arm,'delta_mean').mean():11.1f} {_col(arm,'vega_max').mean():10.1f} "
          f"{_col(arm,'vega_mean').mean():10.1f} {_col(arm,'fills_hedge').mean():8.0f}")
print("=" * 78)

C_SAC, C_AS = "#1f77b4", "#d62728"
fig, ax = plt.subplots(2, 3, figsize=(15.5, 9))

lim = [min(W_sac[ok].min(), W_as[ok].min()), max(W_sac[ok].max(), W_as[ok].max())]
pad = 0.05 * (lim[1] - lim[0] + 1e-9); lim = [lim[0] - pad, lim[1] + pad]
ax[0, 0].plot(lim, lim, "k--", lw=1, zorder=1)
ax[0, 0].scatter(W_as[ok], W_sac[ok], s=28, c=np.where(delta > 0, C_SAC, C_AS),
                 alpha=0.8, zorder=2, edgecolor="none")
ax[0, 0].set(xlim=lim, ylim=lim, xlabel="A-S terminal P&L", ylabel="SAC terminal P&L",
             title=f"Paired by seed (above line = SAC wins, {win:.0%})")
ax[0, 0].set_aspect("equal", "box")

order = np.argsort(delta)
ax[0, 1].bar(range(n_ok), delta[order],
             color=np.where(delta[order] > 0, C_SAC, C_AS))
ax[0, 1].axhline(0, color="k", lw=0.8)
ax[0, 1].axhline(delta.mean(), color="k", ls="--", lw=1,
                 label=f"mean {delta.mean():+.1f}")
ax[0, 1].set(xlabel="seed (sorted)", ylabel="SAC - A-S terminal P&L",
             title="Per-seed difference"); ax[0, 1].legend()

bp = ax[0, 2].boxplot([W_sac[ok], W_as[ok]], widths=0.6, showmeans=True,
                      patch_artist=True)
for patch, c in zip(bp["boxes"], (C_SAC, C_AS)):
    patch.set_facecolor(c); patch.set_alpha(0.35)
for j, (vals, c) in enumerate(((W_sac[ok], C_SAC), (W_as[ok], C_AS)), start=1):
    ax[0, 2].scatter(np.full_like(vals, j) + RNG_BOOT.uniform(-0.08, 0.08, vals.size),
                     vals, s=14, c=c, alpha=0.6, zorder=3, edgecolor="none")
ax[0, 2].set_xticks([1, 2]); ax[0, 2].set_xticklabels(["SAC", "A-S"])
ax[0, 2].axhline(0, color="grey", lw=0.8, ls=":")
ax[0, 2].set(ylabel="terminal P&L", title="Terminal P&L distribution")

comps = ["edge_opt", "edge_hedge", "settle", "reval"]
x = np.arange(len(comps)); w = 0.38
ax[1, 0].bar(x - w/2, [_col("sac", c).mean() for c in comps], w, label="SAC",
             color=C_SAC, alpha=0.85)
ax[1, 0].bar(x + w/2, [_col("as", c).mean() for c in comps], w, label="A-S",
             color=C_AS, alpha=0.85)
ax[1, 0].axhline(0, color="k", lw=0.8)
ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(comps, rotation=20)
ax[1, 0].set(ylabel="mean contribution to P&L", title="P&L decomposition (the 'why')")
ax[1, 0].legend()

ax[1, 1].boxplot([_col("sac", "delta_max"), _col("as", "delta_max")], widths=0.6,
                 showmeans=True)
ax[1, 1].axhline(EnvConfig().delta_limit, color="grey", ls="--", lw=1,
                 label=f"env delta_limit {EnvConfig().delta_limit:.0f}")
ax[1, 1].set_xticks([1, 2]); ax[1, 1].set_xticklabels(["SAC", "A-S"])
ax[1, 1].set(ylabel="max |net delta| per episode",
             title="Hedgeable risk (A-S cannot hedge)"); ax[1, 1].legend()

ax[1, 2].boxplot([_col("sac", "vega_max"), _col("as", "vega_max")], widths=0.6,
                 showmeans=True)
ax[1, 2].set_xticks([1, 2]); ax[1, 2].set_xticklabels(["SAC", "A-S"])
ax[1, 2].set(ylabel="max |net vega| per episode",
             title="Un-hedgeable risk (managed via quoting only)")

fig.suptitle(f"SAC vs Avellaneda-Stoikov  -  {n_ok} paired seeds, "
             f"{'in-distribution' if EVAL_REGIME['expiry_days']==(2.0,4.0,6.0) else 'custom regime'}",
             fontsize=13, y=1.00)
fig.tight_layout()
plt.show()

if SAVE_PREFIX:
    keys = ["W", "edge_opt", "edge_hedge", "settle", "reval", "fills_opt",
            "fills_hedge", "delta_max", "delta_mean", "vega_max", "vega_mean",
            "terminated", "truncated", "resid"]
    np.savez(f"{SAVE_PREFIX}.npz", seeds=np.array(EVAL_SEEDS),
             **{f"sac_{k}": _col("sac", k) for k in keys},
             **{f"as_{k}": _col("as", k) for k in keys})
    import csv
    with open(f"{SAVE_PREFIX}.csv", "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["seed"] + [f"sac_{k}" for k in keys] + [f"as_{k}" for k in keys])
        for j, s in enumerate(EVAL_SEEDS):
            wtr.writerow([s] + [rows["sac"][j][k] for k in keys]
                         + [rows["as"][j][k] for k in keys])
    fig.savefig(f"{SAVE_PREFIX}.png", dpi=120, bbox_inches="tight")
    print(f"\nsaved {SAVE_PREFIX}.npz / .csv / .png")