from __future__ import annotations
from src.rl_env import EnvConfig, F_GLOB, F_INST, OptionsMMEnv
from dataclasses import dataclass
import copy
import math
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOG2 = math.log(2.0)
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
LOG_STD_INIT = -0.5


class SAB(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, x, pad):
        h = self.ln1(x)
        a, _ = self.mha(h, h, h, key_padding_mask=pad, need_weights=False)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x


class SetEncoder(nn.Module):
    def __init__(self, f_inst=F_INST, d=128, heads=4, ffn=512, n_sab=2):
        super().__init__()
        self.d = d
        self.phi = nn.Sequential(nn.Linear(f_inst, d), nn.GELU(),
                                 nn.Linear(d, d), nn.GELU())
        self.sabs = nn.ModuleList([SAB(d, heads, ffn) for _ in range(n_sab)])
        self.pma_ln = nn.LayerNorm(d)
        self.seed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pma = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, X, mask):
        B = X.size(0)
        pad = ~mask
        all_dead = pad.all(dim=-1)
        pad_safe = pad.clone()
        pad_safe[all_dead, 0] = False
        e = self.phi(X) * mask.unsqueeze(-1).float()
        for sab in self.sabs:
            e = sab(e, pad_safe)
        e = e * mask.unsqueeze(-1).float()
        kv = self.pma_ln(e)
        z, _ = self.pma(self.seed.expand(B, -1, -1), kv, kv,
                        key_padding_mask=pad_safe, need_weights=False)
        z = z.squeeze(1)
        z = torch.where(all_dead.unsqueeze(-1), torch.zeros_like(z), z)
        return e, z


class FeedforwardCore(nn.Module):
    def __init__(self, in_dim, H=256, n_blocks=2):
        super().__init__()
        self.inp = nn.Linear(in_dim, H)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(H), nn.Linear(H, H), nn.GELU(),
                          nn.Linear(H, H))
            for _ in range(n_blocks)])
        self.out_ln = nn.LayerNorm(H)

    def forward(self, zg):
        h = self.inp(zg)
        for b in self.blocks:
            h = h + b(h)
        return self.out_ln(h)


class ObsNormalizer:
    def __init__(self, f_inst=F_INST, f_glob=F_GLOB, priv_dim=7, clip=10.0):
        self.clip = clip
        self._stats = {k: [np.zeros(d), np.ones(d), 1e-4] for k, d in
                       (("x", f_inst), ("g", f_glob), ("p", priv_dim))}

    def _upd(self, key, rows):
        mean, m2, n = self._stats[key]
        for r in np.atleast_2d(rows):
            n += 1.0
            dlt = r - mean
            mean += dlt / n
            m2 += dlt * (r - mean)
        self._stats[key] = [mean, m2, n]

    def update(self, X, mask, g, priv):
        live = X[mask]
        if live.size:
            self._upd("x", live)
        self._upd("g", g)
        self._upd("p", priv)

    def _norm(self, key, arr):
        mean, m2, n = self._stats[key]
        var = m2 / max(n - 1.0, 1.0)
        out = (arr - mean) / np.sqrt(var + 1e-8)
        return np.clip(out, -self.clip, self.clip).astype(np.float32)

    def apply(self, X, g, priv):
        return self._norm("x", X), self._norm("g", g), self._norm("p", priv)

    def torch_stats(self, device):
        out = {}
        for k, (mean, m2, n) in self._stats.items():
            var = m2 / max(n - 1.0, 1.0)
            out[k] = (torch.as_tensor(mean, dtype=torch.float32, device=device),
                      torch.as_tensor(np.sqrt(var + 1e-8), dtype=torch.float32,
                                      device=device))
        return out

    def state_dict(self):
        return {k: [v[0].tolist(), v[1].tolist(), float(v[2])]
                for k, v in self._stats.items()}

    def load_state_dict(self, sd):
        self._stats = {k: [np.asarray(v[0], float).copy(),
                           np.asarray(v[1], float).copy(), float(v[2])]
                       for k, v in sd.items()}


def apply_norm_torch(stats, clip, X, g, p):
    if stats is None:
        return X, g, p
    mx, sx = stats["x"]; mg, sg = stats["g"]; mp, sp_ = stats["p"]
    Xn = torch.clamp((X - mx) / sx, -clip, clip)
    gn = torch.clamp((g - mg) / sg, -clip, clip)
    pn = torch.clamp((p - mp) / sp_, -clip, clip)
    return Xn, gn, pn


class SyncVectorEnv:
    def __init__(self, env_fns, base_seed=0):
        self.envs = [fn() for fn in env_fns]
        self.E = len(self.envs)
        self._seed = base_seed

    def _next_seed(self):
        self._seed += 1
        return self._seed

    @staticmethod
    def _stack(obs_list):
        X = np.stack([o[0] for o in obs_list])
        g = np.stack([o[1] for o in obs_list])
        m = np.stack([o[2] for o in obs_list])
        p = np.stack([o[3] for o in obs_list])
        return X, g, m, p

    def reset_all(self):
        return self._stack([e.reset(seed=self._next_seed())[0] for e in self.envs])

    def step(self, acts):
        obs_list, rs, terms, truncs, finals, infos = [], [], [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, tm, tr, ri = env.step(np.asarray(acts[i]))
            fin = None
            if tm or tr:
                ri = dict(ri, episode_return=env.ep_reward, episode_edge=env.ep_edge,
                          episode_risk=getattr(env, "ep_risk", float("nan")),
                          episode_steps=env.ep_steps)
                fin = o
                o, _ = env.reset(seed=self._next_seed())
            obs_list.append(o)
            rs.append(r)
            terms.append(tm)
            truncs.append(tr)
            finals.append(fin)
            infos.append(ri)
        X, g, m, p = self._stack(obs_list)
        return (X, g, m, p), (np.array(rs, np.float32), np.array(terms), np.array(truncs),
                              finals, infos)


def _grad_norm(params):
    tot = 0.0
    for p in params:
        if p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum())
    return math.sqrt(tot)


def _orth_blocks(w, n_blocks, gain):
    rows = w.size(0) // n_blocks
    for b in range(n_blocks):
        nn.init.orthogonal_(w.data[b * rows:(b + 1) * rows], gain=gain)


def live_dims(mask):
    opt = mask.repeat_interleave(4, dim=1).float()
    one = torch.ones(mask.size(0), 1, device=mask.device)
    return torch.cat([opt, one], dim=-1)


@dataclass
class ActionBounds:
    off_lo: float = -3.0
    off_hi: float = 3.0
    qty_lo: float = -20.0
    qty_hi: float = 20.0
    hedge_lo: float = -6.0
    hedge_hi: float = 6.0

    def per_type(self):
        lo = [self.off_lo, self.off_lo, self.qty_lo, self.qty_lo, self.hedge_lo]
        hi = [self.off_hi, self.off_hi, self.qty_hi, self.qty_hi, self.hedge_hi]
        return (torch.tensor(lo, dtype=torch.float32),
                torch.tensor(hi, dtype=torch.float32))


class SquashedGaussianActor(nn.Module):
    def __init__(self, N, f_inst=F_INST, f_glob=F_GLOB, d=128, core_hidden=256,
                 heads=4, ffn=512, n_core_blocks=2, bounds: ActionBounds | None = None):
        super().__init__()
        self.N = N
        self.d = d
        self.H = core_hidden
        self.adim = 4 * N + 1
        self.enc = SetEncoder(f_inst, d, heads, ffn)
        self.core = FeedforwardCore(d + f_glob, core_hidden, n_core_blocks)
        self.opt_head = nn.Sequential(nn.Linear(core_hidden + d, 128), nn.GELU(),
                                      nn.Linear(128, 8))
        self.hedge_head = nn.Sequential(nn.Linear(core_hidden, 128), nn.GELU(),
                                        nn.Linear(128, 2))
        dim_type = torch.cat([torch.arange(4).repeat(N), torch.tensor([4])])
        self.register_buffer("dim_type", dim_type)
        lo_t, hi_t = (bounds or ActionBounds()).per_type()
        pre_lo = lo_t[dim_type]
        pre_hi = hi_t[dim_type]
        self.register_buffer("pre_lo", pre_lo)
        self.register_buffer("pre_hi", pre_hi)
        self.register_buffer("pre_mid", 0.5 * (pre_lo + pre_hi))
        self.register_buffer("pre_half", 0.5 * (pre_hi - pre_lo))

    def forward(self, X, g, mask):
        eA, zA = self.enc(X, mask)
        c = self.core(torch.cat([zA, g], dim=-1))
        B = c.size(0)
        cx = c.unsqueeze(1).expand(B, self.N, self.H)
        opt = self.opt_head(torch.cat([cx, eA], dim=-1))
        mu_o = opt[..., :4] * mask.unsqueeze(-1).float()
        ls_o = opt[..., 4:]
        hg = self.hedge_head(c)
        mu = torch.cat([mu_o.reshape(B, 4 * self.N), hg[:, :1]], dim=-1)
        ls = torch.cat([ls_o.reshape(B, 4 * self.N), hg[:, 1:]], dim=-1)
        return mu, ls.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, X, g, mask, deterministic=False, with_logp=True):
        mu, ls = self(X, g, mask)
        live = live_dims(mask)
        if deterministic:
            u = mu
        else:
            u = mu + ls.exp() * torch.randn_like(mu)
        a = torch.tanh(u) * live
        if not with_logp:
            return a, None
        std = ls.exp()
        logp_u = -0.5 * ((u - mu) / std) ** 2 - ls - 0.5 * math.log(2 * math.pi)
        corr = 2.0 * (LOG2 - u - F.softplus(-2.0 * u))
        logp = ((logp_u - corr) * live).sum(-1)
        return a, logp

    def to_pre(self, a):
        return self.pre_mid + self.pre_half * a


class QNet(nn.Module):
    def __init__(self, N, f_inst=F_INST, f_glob=F_GLOB, d=128, core_hidden=256,
                 priv_dim=7, heads=4, ffn=512, n_core_blocks=2):
        super().__init__()
        self.N = N
        self.enc = SetEncoder(f_inst + 4, d, heads, ffn)
        self.core = FeedforwardCore(d + f_glob + 1, core_hidden, n_core_blocks)
        self.q_head = nn.Sequential(nn.Linear(core_hidden + priv_dim, 128), nn.GELU(),
                                    nn.Linear(128, 1))

    def forward(self, X, g, mask, priv, a):
        B, N = X.size(0), self.N
        a_inst = a[:, :4 * N].reshape(B, N, 4)
        a_h = a[:, -1:]
        e, z = self.enc(torch.cat([X, a_inst], dim=-1), mask)
        c = self.core(torch.cat([z, g, a_h], dim=-1))
        return self.q_head(torch.cat([c, priv], dim=-1)).squeeze(-1)


def init_sac_(actor: SquashedGaussianActor, *qnets: QNet):
    for model in (actor,) + tuple(qnets):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in model.modules():
            if isinstance(m, nn.MultiheadAttention):
                _orth_blocks(m.in_proj_weight, 3, 1.0)
                nn.init.zeros_(m.in_proj_bias)
                nn.init.orthogonal_(m.out_proj.weight, gain=1.0)
                nn.init.zeros_(m.out_proj.bias)
    nn.init.orthogonal_(actor.opt_head[-1].weight, gain=0.01)
    nn.init.zeros_(actor.opt_head[-1].bias)
    actor.opt_head[-1].bias.data[4:].fill_(LOG_STD_INIT)
    nn.init.orthogonal_(actor.hedge_head[-1].weight, gain=0.01)
    nn.init.zeros_(actor.hedge_head[-1].bias)
    actor.hedge_head[-1].bias.data[1] = LOG_STD_INIT
    for q in qnets:
        nn.init.orthogonal_(q.q_head[-1].weight, gain=1.0)
        nn.init.zeros_(q.q_head[-1].bias)


def build_sac_nets(N, priv_dim=7, bounds: ActionBounds | None = None,
                   d=128, core_hidden=256, heads=4, ffn=512, n_core_blocks=2):
    actor = SquashedGaussianActor(N, d=d, core_hidden=core_hidden, heads=heads,
                                  ffn=ffn, n_core_blocks=n_core_blocks, bounds=bounds)
    q1 = QNet(N, d=d, core_hidden=core_hidden, priv_dim=priv_dim, heads=heads,
              ffn=ffn, n_core_blocks=n_core_blocks)
    q2 = QNet(N, d=d, core_hidden=core_hidden, priv_dim=priv_dim, heads=heads,
              ffn=ffn, n_core_blocks=n_core_blocks)
    init_sac_(actor, q1, q2)
    return actor, q1, q2


class ReplayBuffer:
    def __init__(self, capacity, N, f_inst=F_INST, f_glob=F_GLOB, priv_dim=7,
                 adim=None):
        adim = adim if adim is not None else 4 * N + 1
        c = int(capacity)
        self.cap = c
        self.X = torch.zeros(c, N, f_inst)
        self.g = torch.zeros(c, f_glob)
        self.p = torch.zeros(c, priv_dim)
        self.m = torch.zeros(c, N, dtype=torch.bool)
        self.a = torch.zeros(c, adim)
        self.r = torch.zeros(c)
        self.dt = torch.zeros(c)
        self.X2 = torch.zeros(c, N, f_inst)
        self.g2 = torch.zeros(c, f_glob)
        self.p2 = torch.zeros(c, priv_dim)
        self.m2 = torch.zeros(c, N, dtype=torch.bool)
        self.term = torch.zeros(c)
        self.trunc = torch.zeros(c)
        self.ptr = 0
        self.size = 0

    def add_batch(self, X, g, m, p, a, r, dt, X2, g2, m2, p2, term, trunc):
        E = len(r)
        idx = torch.as_tensor((self.ptr + np.arange(E)) % self.cap)

        for buf, val, dty in ((self.X, X, torch.float32), (self.g, g, torch.float32),
                              (self.p, p, torch.float32), (self.m, m, torch.bool),
                              (self.a, a, torch.float32), (self.r, r, torch.float32),
                              (self.dt, dt, torch.float32),
                              (self.X2, X2, torch.float32), (self.g2, g2, torch.float32),
                              (self.p2, p2, torch.float32), (self.m2, m2, torch.bool),
                              (self.term, term, torch.float32),
                              (self.trunc, trunc, torch.float32)):
            buf[idx] = torch.as_tensor(np.asarray(val), dtype=dty)
        self.ptr = (self.ptr + E) % self.cap
        self.size = min(self.size + E, self.cap)

    def sample(self, batch_size, device):
        i = torch.randint(0, self.size, (batch_size,))
        to = lambda t: t[i].to(device)
        return dict(X=to(self.X), g=to(self.g), m=to(self.m), p=to(self.p),
                    a=to(self.a), r=to(self.r), dt=to(self.dt), X2=to(self.X2),
                    g2=to(self.g2), m2=to(self.m2), p2=to(self.p2),
                    term=to(self.term))


@dataclass
class SACConfig:
    num_envs: int = 8
    steps_per_iter: int = 128
    iterations: int = 100
    start_steps: int = 5_000
    update_after: int = 2_000
    utd: float = 1.0
    batch_size: int = 256
    discount_rho: float | None = None
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.2
    target_entropy_per_dim: float = -1.0
    reward_scale: float = 1.0
    huber_beta: float = 10.0
    max_grad_norm: float = 10.0
    norm_obs: bool = True
    buffer_capacity: int = 500_000
    seed: int = 0
    log_every: int = 1
    checkpoint_every: int = 20
    checkpoint_path: str = "sac_ff_checkpoint.pt"


class SACLearner:
    def __init__(self, actor, q1, q2, cfg: SACConfig, device):
        self.cfg = cfg
        self.device = device
        self.actor = actor.to(device)
        self.q1 = q1.to(device)
        self.q2 = q2.to(device)
        self.q1_t = copy.deepcopy(self.q1)
        self.q2_t = copy.deepcopy(self.q2)
        for p_ in list(self.q1_t.parameters()) + list(self.q2_t.parameters()):
            p_.requires_grad_(False)
        self.log_alpha = torch.tensor(math.log(cfg.init_alpha), device=device,
                                      requires_grad=True)
        self.actor_params = list(self.actor.parameters())
        self.q_params = list(self.q1.parameters()) + list(self.q2.parameters())
        self.actor_opt = torch.optim.Adam(self.actor_params, lr=cfg.actor_lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.q_params, lr=cfg.critic_lr, eps=1e-5)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.updates = 0

    @property
    def alpha(self):
        return float(self.log_alpha.exp())

    @torch.no_grad()
    def _polyak(self):
        tau = self.cfg.tau
        for tgt, src in ((self.q1_t, self.q1), (self.q2_t, self.q2)):
            for pt, ps in zip(tgt.parameters(), src.parameters()):
                pt.mul_(1.0 - tau).add_(ps, alpha=tau)

    def update(self, b, stats=None, norm_clip=10.0):
        cfg = self.cfg
        Xn, gn, pn = apply_norm_torch(stats, norm_clip, b["X"], b["g"], b["p"])
        X2n, g2n, p2n = apply_norm_torch(stats, norm_clip, b["X2"], b["g2"], b["p2"])

        with torch.no_grad():
            a2, lp2 = self.actor.sample(X2n, g2n, b["m2"])
            alpha = self.log_alpha.exp()
            qt = torch.min(self.q1_t(X2n, g2n, b["m2"], p2n, a2),
                           self.q2_t(X2n, g2n, b["m2"], p2n, a2))
            gam = torch.exp(-cfg.discount_rho * b["dt"])
            y = cfg.reward_scale * b["r"] + gam * (1.0 - b["term"]) * (qt - alpha * lp2)
        q1v = self.q1(Xn, gn, b["m"], pn, b["a"])
        q2v = self.q2(Xn, gn, b["m"], pn, b["a"])
        q_loss = (F.smooth_l1_loss(q1v, y, beta=cfg.huber_beta)
                  + F.smooth_l1_loss(q2v, y, beta=cfg.huber_beta))
        self.critic_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        gC = float(nn.utils.clip_grad_norm_(self.q_params, cfg.max_grad_norm))
        self.critic_opt.step()

        for p_ in self.q_params:
            p_.requires_grad_(False)
        an, lpn = self.actor.sample(Xn, gn, b["m"])
        q_pi = torch.min(self.q1(Xn, gn, b["m"], pn, an),
                         self.q2(Xn, gn, b["m"], pn, an))
        alpha = self.log_alpha.exp().detach()
        pi_loss = (alpha * lpn - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        gA = float(nn.utils.clip_grad_norm_(self.actor_params, cfg.max_grad_norm))
        self.actor_opt.step()
        for p_ in self.q_params:
            p_.requires_grad_(True)

        live = live_dims(b["m"])
        n_live = live.sum(-1).clamp(min=1.0)
        H_tgt = cfg.target_entropy_per_dim * n_live
        alpha_loss = -(self.log_alpha * (lpn.detach() + H_tgt)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(-10.0, 3.0)

        self._polyak()
        self.updates += 1
        with torch.no_grad():
            ent = -lpn.mean()
            ent_pd = (-lpn / n_live).mean()
        return dict(q_loss=float(q_loss), pi_loss=float(pi_loss),
                    alpha_loss=float(alpha_loss), alpha=self.alpha,
                    q1_mu=float(q1v.mean()), y_mu=float(y.mean()),
                    y_sd=float(y.std()), y_mx=float(y.abs().max()),
                    ent=float(ent), ent_pd=float(ent_pd), gA=gA, gC=gC)

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), q1=self.q1.state_dict(),
                    q2=self.q2.state_dict(), q1_t=self.q1_t.state_dict(),
                    q2_t=self.q2_t.state_dict(),
                    log_alpha=float(self.log_alpha.detach().cpu()),
                    actor_opt=self.actor_opt.state_dict(),
                    critic_opt=self.critic_opt.state_dict(),
                    alpha_opt=self.alpha_opt.state_dict(), updates=self.updates)

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.q1.load_state_dict(sd["q1"])
        self.q2.load_state_dict(sd["q2"])
        self.q1_t.load_state_dict(sd["q1_t"])
        self.q2_t.load_state_dict(sd["q2_t"])
        with torch.no_grad():
            self.log_alpha.fill_(sd["log_alpha"])
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        self.alpha_opt.load_state_dict(sd["alpha_opt"])
        self.updates = sd.get("updates", 0)


def save_checkpoint(path, learner, normalizer, cfg, it, total_steps):
    torch.save(dict(learner=learner.state_dict(),
                    normalizer=normalizer.state_dict(),
                    sac_cfg=vars(cfg), it=it, total_steps=total_steps), path)


@torch.no_grad()
def _sigma_by_type(actor, Xn, gn, m):
    _, ls = actor(Xn, gn, m)
    std = ls.exp()
    live = live_dims(m)
    out = []
    for t in range(5):
        sel = (actor.dim_type == t).float().unsqueeze(0) * live
        out.append(float((std * sel).sum() / sel.sum().clamp(min=1.0)))
    return out


def train_sac(make_env_fn, cfg: SACConfig, device=None, learner=None,
              normalizer=None, callback=None, resume=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed + 12345)

    venv = SyncVectorEnv([make_env_fn for _ in range(cfg.num_envs)],
                         base_seed=1000 * cfg.seed)

    rho_env = float(venv.envs[0].rcfg.discount_rho)
    if cfg.discount_rho is None:
        cfg.discount_rho = rho_env
    elif abs(cfg.discount_rho - rho_env) > 1e-12:
        raise ValueError(
            f"SACConfig.discount_rho={cfg.discount_rho} != env RewardConfig."
            f"discount_rho={rho_env}: set it in RewardConfig (single source, v3)")
    obs = venv.reset_all()
    N = venv.envs[0].meta["N"]
    A = 4 * N + 1
    P = venv.envs[0].PRIV_DIM
    E = cfg.num_envs
    assert cfg.update_after >= cfg.batch_size

    if learner is None:
        actor, q1, q2 = build_sac_nets(N, priv_dim=P)
        learner = SACLearner(actor, q1, q2, cfg, device)
    if normalizer is None:
        normalizer = ObsNormalizer(priv_dim=P)
    buffer = ReplayBuffer(cfg.buffer_capacity, N, F_INST, F_GLOB, P, A)
    learner.buffer = buffer

    it0, total_steps = 1, 0
    if resume is not None:
        ck = torch.load(resume, map_location=device, weights_only=True)
        learner.load_state_dict(ck["learner"])
        normalizer.load_state_dict(ck["normalizer"])
        it0 = ck["it"] + 1
        total_steps = ck["total_steps"]
        print(f"resumed from {resume} @ iteration {ck['it']} "
              f"({total_steps} transitions; buffer restarts empty)")

    pre_mid = learner.actor.pre_mid.detach().cpu().numpy()
    pre_half = learner.actor.pre_half.detach().cpu().numpy()

    def prep(o):
        X, g, m, p = o
        if cfg.norm_obs:
            outs = [normalizer.apply(X[i], g[i], p[i]) for i in range(len(X))]
            Xn = np.stack([o_[0] for o_ in outs])
            gn = np.stack([o_[1] for o_ in outs])
            pn = np.stack([o_[2] for o_ in outs])
        else:
            Xn, gn, pn = X.astype(np.float32), g.astype(np.float32), p.astype(np.float32)
        return (torch.as_tensor(Xn, device=device),
                torch.as_tensor(gn, device=device),
                torch.as_tensor(m, device=device),
                torch.as_tensor(pn, device=device))

    ep_returns = deque(maxlen=100)
    ep_edges = deque(maxlen=100)
    ep_risks = deque(maxlen=100)
    history = []

    for it in range(it0, cfg.iterations + 1):
        t0 = time.time()
        raw_r = np.zeros((cfg.steps_per_iter, E), np.float32)
        roll_edge = roll_risk = roll_fills = 0.0
        roll_dn = roll_yn = 0.0
        n_done = 0

        learner.actor.eval()
        for t in range(cfg.steps_per_iter):
            X, g, m, p = obs
            if cfg.norm_obs:
                for i in range(E):
                    normalizer.update(X[i], m[i], g[i], p[i])
            live_np = np.concatenate(
                [np.repeat(m.astype(np.float32), 4, axis=1), np.ones((E, 1), np.float32)],
                axis=1)
            if total_steps < cfg.start_steps:
                a_np = (rng.uniform(-1.0, 1.0, size=(E, A)).astype(np.float32) * live_np)
            else:
                with torch.no_grad():
                    Xt, gt, mt, pt = prep(obs)
                    a_t, _ = learner.actor.sample(Xt, gt, mt, with_logp=False)
                a_np = a_t.cpu().numpy()
            pre = pre_mid[None, :] + pre_half[None, :] * a_np
            obs2, (r_np, term, trunc, finals, infos) = venv.step(pre)
            raw_r[t] = r_np
            dt_np = np.array([infos[i]["dt"] for i in range(E)], np.float32)
            X2, g2, m2, p2 = obs2
            X2e, g2e, m2e, p2e = X2.copy(), g2.copy(), m2.copy(), p2.copy()
            for i in range(E):
                ri = infos[i]
                roll_edge += ri["edge"]
                roll_risk += ri["risk"]
                roll_fills += ri["n_fills"]
                roll_dn = max(roll_dn, abs(ri["dn"]))
                roll_yn = max(roll_yn, abs(ri["yn"]))
                if finals[i] is not None:
                    fX, fg, fm, fp = finals[i]
                    X2e[i], g2e[i], m2e[i], p2e[i] = fX, fg, fm, fp
                if term[i] or trunc[i]:
                    n_done += 1
                    ep_returns.append(ri["episode_return"])
                    ep_edges.append(ri["episode_edge"])
                    ep_risks.append(ri.get("episode_risk", float("nan")))

            keep = np.array([not infos[i].get("noop", False) for i in range(E)], bool)
            if keep.any():
                k = np.nonzero(keep)[0]
                buffer.add_batch(X[k], g[k], m[k], p[k], a_np[k], r_np[k], dt_np[k],
                                 X2e[k], g2e[k], m2e[k], p2e[k],
                                 term[k].astype(np.float32),
                                 trunc[k].astype(np.float32))
            total_steps += E
            obs = obs2
        t_collect = time.time() - t0

        t1 = time.time()
        mets = []
        last_batch = None
        if buffer.size >= cfg.update_after:
            stats = normalizer.torch_stats(device) if cfg.norm_obs else None

            nominal = int(round(cfg.utd * cfg.steps_per_iter * E))
            allowed = int(cfg.utd * max(0, total_steps - cfg.update_after))
            n_upd = max(0, min(nominal, allowed - learner.updates))
            learner.actor.train()
            for _ in range(n_upd):
                b = buffer.sample(cfg.batch_size, device)
                mets.append(learner.update(b, stats, norm_clip=normalizer.clip))
                last_batch = b
        t_update = time.time() - t1

        TE = cfg.steps_per_iter * E
        if mets:
            mk = lambda k: float(np.mean([m_[k] for m_ in mets]))
            stats_now = normalizer.torch_stats(device) if cfg.norm_obs else None
            Xn, gn, _ = apply_norm_torch(stats_now, normalizer.clip,
                                         last_batch["X"], last_batch["g"],
                                         last_batch["p"])
            sig = _sigma_by_type(learner.actor, Xn, gn, last_batch["m"])
        else:
            mk = lambda k: float("nan")
            sig = [float("nan")] * 5
        rec = dict(it=it, total_steps=total_steps, buffer=buffer.size,
                   ep_return=(float(np.mean(ep_returns)) if ep_returns else float("nan")),
                   ep_edge=(float(np.mean(ep_edges)) if ep_edges else float("nan")),
                   ep_risk=(float(np.mean(ep_risks)) if ep_risks else float("nan")),
                   n_done=n_done,
                   r_step=float(raw_r.mean()),
                   r_p1=float(np.percentile(raw_r, 1)),
                   r_p99=float(np.percentile(raw_r, 99)),
                   edge_step=roll_edge / TE, risk_step=roll_risk / TE,
                   fills_step=roll_fills / TE, dn_max=roll_dn, yn_max=roll_yn,
                   n_updates=len(mets), q_loss=mk("q_loss"), pi_loss=mk("pi_loss"),
                   alpha=mk("alpha"), alpha_loss=mk("alpha_loss"),
                   q1_mu=mk("q1_mu"), y_mu=mk("y_mu"), y_sd=mk("y_sd"),
                   y_mx=(float(np.max([m_["y_mx"] for m_ in mets])) if mets
                         else float("nan")),
                   ent=mk("ent"), ent_pd=mk("ent_pd"),
                   gA=(mets[-1]["gA"] if mets else float("nan")),
                   gC=(mets[-1]["gC"] if mets else float("nan")),
                   sig_bid_off=sig[0], sig_ask_off=sig[1], sig_bid_qty=sig[2],
                   sig_ask_qty=sig[3], sig_hedge=sig[4],
                   t_collect=t_collect, t_update=t_update)
        history.append(rec)
        if it % cfg.log_every == 0:
            print(f"[{it:4d}] R={rec['ep_return']:+.3f} edge={rec['ep_edge']:+.2f} "
                  f"r/st={rec['r_step']:+.4f}[{rec['r_p1']:+.2f},{rec['r_p99']:+.2f}] "
                  f"edge/st={rec['edge_step']:+.4f} risk/st={rec['risk_step']:.4f} "
                  f"fl/st={rec['fills_step']:.2f} |dn|x={rec['dn_max']:.2f} "
                  f"|yn|x={rec['yn_max']:.2f} buf={rec['buffer']}")
            print(f"       qL={rec['q_loss']:.4f} piL={rec['pi_loss']:+.4f} "
                  f"q1={rec['q1_mu']:+.2f} y(mu={rec['y_mu']:+.2f},sd={rec['y_sd']:.2f},"
                  f"mx={rec['y_mx']:.1f}) a={rec['alpha']:.4f} ent/d={rec['ent_pd']:+.3f} "
                  f"|gA|={rec['gA']:.2f} |gC|={rec['gC']:.2f} "
                  f"sig=({rec['sig_bid_off']:.2f},{rec['sig_ask_off']:.2f},"
                  f"{rec['sig_bid_qty']:.2f},{rec['sig_ask_qty']:.2f},"
                  f"{rec['sig_hedge']:.2f}) upd={rec['n_updates']} "
                  f"{rec['t_collect']:.0f}s/{rec['t_update']:.0f}s")
        if cfg.checkpoint_every and it % cfg.checkpoint_every == 0:
            save_checkpoint(cfg.checkpoint_path, learner, normalizer, cfg, it,
                            total_steps)
            print(f"       checkpoint -> {cfg.checkpoint_path}")
        if callback is not None:
            callback(it, learner, normalizer, history)

    save_checkpoint(cfg.checkpoint_path, learner, normalizer, cfg,
                    cfg.iterations, total_steps)
    print(f"final checkpoint -> {cfg.checkpoint_path}")
    return learner, normalizer, history


@torch.no_grad()
def run_sac_episode(env: OptionsMMEnv, actor, normalizer, seed=0, deterministic=True,
                    device="cpu", max_steps=None):
    actor.eval()
    obs, info = env.reset(seed=seed)
    term = trunc = False
    n = 0
    cap = max_steps if max_steps is not None else env.cfg.max_steps + 5
    pre_mid = actor.pre_mid.detach().cpu().numpy()
    pre_half = actor.pre_half.detach().cpu().numpy()
    while not (term or trunc) and n < cap:
        X, g, m, p = obs
        if normalizer is not None:
            X, g, p = normalizer.apply(X, g, p)
        a, _ = actor.sample(
            torch.as_tensor(X[None], device=device, dtype=torch.float32),
            torch.as_tensor(g[None], device=device, dtype=torch.float32),
            torch.as_tensor(m[None], device=device),
            deterministic=deterministic, with_logp=False)
        pre = pre_mid + pre_half * a.squeeze(0).cpu().numpy()
        obs, r, term, trunc, ri = env.step(pre)
        n += 1
    out = dict(terminated=term, truncated=trunc, steps=env.ep_steps,
               reward=env.ep_reward, edge=env.ep_edge,
               risk=getattr(env, "ep_risk", float("nan")))
    if env.collect_snaps:
        out["episode"] = env.episode_dict(term, trunc)
    return out


def _ortho_err(W, gain):
    W = W.detach()
    if W.size(0) >= W.size(1):
        gram = W.t() @ W
    else:
        gram = W @ W.t()
    I = torch.eye(gram.size(0), device=W.device)
    return float((gram - gain * gain * I).abs().max())


def _fake_batch(N, P, A, B=6, seed=0):
    g_ = torch.Generator().manual_seed(seed)
    m = torch.ones(B, N, dtype=torch.bool)
    m[1, 15:] = False
    m[2, :] = False
    live = live_dims(m)
    a = (torch.rand(B, A, generator=g_) * 2 - 1) * live
    term = torch.zeros(B)
    term[2] = 1.0
    return dict(X=torch.randn(B, N, F_INST, generator=g_),
                g=torch.randn(B, F_GLOB, generator=g_), m=m,
                p=torch.randn(B, P, generator=g_), a=a,
                r=torch.randn(B, generator=g_) * 0.1,
                dt=torch.rand(B, generator=g_) * 1e-5,
                X2=torch.randn(B, N, F_INST, generator=g_),
                g2=torch.randn(B, F_GLOB, generator=g_), m2=m.clone(),
                p2=torch.randn(B, P, generator=g_), term=term,
                trunc=torch.zeros(B))


def sac_shape_self_test(device="cpu", N=30, seed=0):
    torch.manual_seed(seed)
    actor, q1, q2 = build_sac_nets(N)
    actor, q1, q2 = actor.to(device), q1.to(device), q2.to(device)
    A = 4 * N + 1
    n_a = sum(p.numel() for p in actor.parameters())
    n_q = sum(p.numel() for p in q1.parameters())
    print(f"params: actor {n_a/1e6:.2f}M | per Q-net {n_q/1e6:.2f}M "
          f"(x2 online, x2 target)")

    assert _ortho_err(actor.opt_head[-1].weight, 0.01) < 1e-4
    assert _ortho_err(actor.hedge_head[-1].weight, 0.01) < 1e-4
    assert _ortho_err(q1.q_head[-1].weight, 1.0) < 1e-4
    assert _ortho_err(actor.enc.phi[0].weight, math.sqrt(2)) < 1e-4
    assert _ortho_err(q1.enc.phi[0].weight, math.sqrt(2)) < 1e-4
    for blk in actor.core.blocks:
        assert _ortho_err(blk[1].weight, math.sqrt(2)) < 1e-4
    for sab in q1.enc.sabs:
        d = sab.mha.embed_dim
        for b_ in range(3):
            assert _ortho_err(sab.mha.in_proj_weight[b_ * d:(b_ + 1) * d], 1.0) < 1e-4
    assert torch.allclose(actor.opt_head[-1].bias[4:],
                          torch.full((4,), LOG_STD_INIT, device=device))
    assert float(actor.hedge_head[-1].bias[1]) == LOG_STD_INIT
    print("orthogonal init (Linear sqrt2 | MHA 1.0 | heads .01/1.0) + "
          f"log_std bias {LOG_STD_INIT} (sigma_0={math.exp(LOG_STD_INIT):.2f}) OK")

    ones = torch.ones(A, device=device)
    hi = actor.to_pre(ones)
    lo = actor.to_pre(-ones)
    assert torch.allclose(hi, actor.pre_hi) and torch.allclose(lo, actor.pre_lo)

    assert torch.allclose(actor.pre_mid, torch.zeros_like(actor.pre_mid)), \
        'ActionBounds must be symmetric: a=0 must map to pre=0'
    qty_hi = float(actor.pre_hi[2])
    sp_hi = math.log1p(math.exp(min(qty_hi, 30.0))) if qty_hi < 30 else qty_hi
    e_cap = EnvConfig().max_order_qty
    assert abs(sp_hi - e_cap) < 0.5, (sp_hi, e_cap)
    off_hi = float(actor.pre_hi[0])
    off_lo_v = float(actor.pre_lo[0])
    sp_lo = math.log1p(math.exp(off_lo_v))
    assert sp_lo < 0.1, (sp_lo, "tightest offset must stay inside one tick (v3)")
    assert sp_lo > 0.02, (sp_lo, "off_lo too low: sub-tick plateau returns")
    print(f"bounds OK: qty softplus({qty_hi:.0f})={sp_hi:.2f} ~ max_order_qty={e_cap}; "
          f"offsets up to softplus({off_hi:.0f})={math.log1p(math.exp(off_hi)):.2f} px; "
          f"hedge f in [{1/(1+math.exp(-float(actor.pre_lo[-1]))):.4f}, "
          f"{1/(1+math.exp(-float(actor.pre_hi[-1]))):.4f}]")

    B = 4
    X = torch.randn(B, N, F_INST, device=device)
    g = torch.randn(B, F_GLOB, device=device)
    m = torch.ones(B, N, dtype=torch.bool, device=device)
    m[1, 15:] = False
    m[2, :] = False
    p = torch.randn(B, 7, device=device)
    mu, ls = actor(X, g, m)
    assert mu.shape == (B, A) and ls.shape == (B, A)
    assert torch.isfinite(mu).all() and torch.isfinite(ls).all()
    assert (ls >= LOG_STD_MIN - 1e-6).all() and (ls <= LOG_STD_MAX + 1e-6).all()
    assert (mu[2, :4 * N].abs() < 1e-6).all(), "dead rows must emit zero option means"
    a, lp = actor.sample(X, g, m)
    live = live_dims(m)
    assert torch.isfinite(lp).all()
    assert (a.abs() <= 1.0).all()
    assert (a[2, :4 * N].abs() < 1e-9).all(), "dead action dims must be exactly 0"
    a_det1, _ = actor.sample(X, g, m, deterministic=True, with_logp=False)
    a_det2, _ = actor.sample(X, g, m, deterministic=True, with_logp=False)
    assert torch.allclose(a_det1, a_det2), "deterministic eval must be repeatable"

    torch.manual_seed(seed + 1)
    mu_s = torch.randn(5, 7) * 0.5
    ls_s = torch.randn(5, 7) * 0.3 - 0.7
    std_s = ls_s.exp()
    u = mu_s + std_s * torch.randn(5, 7)
    a_s = torch.tanh(u)
    lp_ours = (-0.5 * ((u - mu_s) / std_s) ** 2 - ls_s - 0.5 * math.log(2 * math.pi)
               - 2.0 * (LOG2 - u - F.softplus(-2.0 * u))).sum(-1)
    base = torch.distributions.Normal(mu_s, std_s)
    dist = torch.distributions.TransformedDistribution(
        base, [torch.distributions.TanhTransform(cache_size=0)])
    lp_ref = dist.log_prob(a_s).sum(-1)
    assert torch.allclose(lp_ours, lp_ref, atol=1e-4, rtol=1e-4), \
        float((lp_ours - lp_ref).abs().max())
    print("masked sampling / squashed log-prob (vs TransformedDistribution) OK")

    qv = q1(X, g, m, p, a)
    assert qv.shape == (B,) and torch.isfinite(qv).all()
    a_pert = a + (1.0 - live) * 0.5
    assert torch.allclose(qv, q1(X, g, m, p, a_pert), atol=1e-6), \
        "Q must be invariant to dead-instrument action dims"
    perm = torch.randperm(N)
    Xp = X[:, perm]
    mp_ = m[:, perm]
    ap = a.clone()
    ap[:, :4 * N] = a[:, :4 * N].reshape(B, N, 4)[:, perm].reshape(B, 4 * N)
    assert torch.allclose(qv, q1(Xp, g, mp_, p, ap), atol=1e-4, rtol=1e-4), \
        "Q must be invariant under joint instrument/action permutation"
    assert torch.allclose(qv, q1(X, g, m, p, a)), "Q must be stateless"
    print("Q(s,a): per-token injection, dead-dim + permutation invariance OK")

    cfg = SACConfig(discount_rho=100.0)
    learner = SACLearner(actor, q1, q2, cfg, device)
    for pt, ps in zip(learner.q1_t.parameters(), learner.q1.parameters()):
        assert torch.allclose(pt, ps)
    b = {k: (v.to(device) if torch.is_tensor(v) else v)
         for k, v in _fake_batch(N, 7, A, seed=seed).items()}
    p_before = learner.q1_t.q_head[-1].weight.detach().clone()
    la_before = float(learner.log_alpha)
    mets = learner.update(b, stats=None)
    for k, v in mets.items():
        assert np.isfinite(v), (k, v)
    mets2 = learner.update(b, stats=None)
    assert all(np.isfinite(v) for v in mets2.values())
    assert not torch.allclose(p_before, learner.q1_t.q_head[-1].weight), \
        "Polyak must move the targets"
    assert float(learner.log_alpha) != la_before, "alpha must update"
    print(f"learner update OK: qL={mets['q_loss']:.3f} piL={mets['pi_loss']:+.3f} "
          f"alpha {la_before and math.exp(la_before):.3f}->{learner.alpha:.3f} "
          f"ent/d={mets['ent_pd']:+.3f}")
    print("SAC SHAPE SELF-TEST PASSED")
    return learner