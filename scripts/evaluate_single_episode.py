import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.baselines import AvellanedaStoikovAgent
from src.config import HawkesParams, SimConfig
from src.evaluation import decompose, plot_dashboard, run_episode
from src.rl_env import OptionsMMEnv, RewardConfig
from src.sac import run_sac_episode
from src.simulator import OptionsMMSimulator

from train import learner, normalizer

import numpy as np, torch

SEED = 123
device = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN = dict(kappa=25.0, expiry_days=(2.0, 4.0, 6.0),
             hawkes=HawkesParams(mu=(400.0, 400.0, 2400.0, 2400.0)),
             max_steps=12000)

eval_env = OptionsMMEnv(SimConfig(seed=SEED, **TRAIN), rcfg=RewardConfig(),
                        collect_snaps=True)
out = run_sac_episode(eval_env, learner.actor.to(device), normalizer, seed=SEED,
                      device=device)
print({k: out[k] for k in ("terminated", "truncated", "steps", "reward", "edge", "risk")})
ep_sac = out["episode"]; dc_sac = decompose(ep_sac)
print(f"SAC   W={dc_sac['W'][-1]:+.2f}  edge_opt={dc_sac['edge_opt'][-1]:+.2f}  "
      f"edge_hedge={dc_sac['edge_hedge'][-1]:+.2f}  "
      f"max|netD|={float(np.nanmax(np.abs(dc_sac['true_delta']))):.1f}")
plot_dashboard(ep_sac, dc_sac, title="SAC ")

class _FixtureEnv(OptionsMMSimulator):
    def reset(self, seed=None, strikes=None):
        return super().reset(seed=seed, strikes=self.cfg.fixture_strikes)

bl_cfg = SimConfig(seed=SEED, **TRAIN).validate()
bl_env = _FixtureEnv(bl_cfg)
agent = AvellanedaStoikovAgent(bl_cfg, gamma=0.1)
ep_bl = run_episode(bl_env, agent, seed=SEED)
dc_bl = decompose(ep_bl)
print(f"A-S   W={dc_bl['W'][-1]:+.2f}  edge_opt={dc_bl['edge_opt'][-1]:+.2f}  "
      f"max|netD|={float(np.nanmax(np.abs(dc_bl['true_delta']))):.1f}")
plot_dashboard(ep_bl, dc_bl, title="Avellaneda-Stoikov baseline")