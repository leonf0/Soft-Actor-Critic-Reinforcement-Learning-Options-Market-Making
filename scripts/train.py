import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.config import HawkesParams, SimConfig
from src.rl_env import OptionsMMEnv, RewardConfig
from src.sac import SACConfig, train_sac

import torch

TRAIN = dict(kappa=25.0, expiry_days=(2.0, 4.0, 6.0),
             hawkes=HawkesParams(mu=(400.0, 400.0, 2400.0, 2400.0)),
             max_steps=6000)

RCFG = RewardConfig()

make_env = lambda: OptionsMMEnv(SimConfig(seed=0, **TRAIN), rcfg=RCFG)

cfg = SACConfig()

learner, normalizer, hist = train_sac(make_env, cfg)