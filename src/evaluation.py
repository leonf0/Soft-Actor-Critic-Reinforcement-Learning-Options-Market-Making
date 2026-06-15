from __future__ import annotations
from src.instruments import UNDERLYING
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def run_episode(env, agent, seed=0, max_steps=None):
    obs, info = env.reset(seed=seed)
    agent.reset()
    cfg = env.cfg
    opt_names = list(env.eng.reg.option_names)

    def snap(obs, info, agent):
        fills = []
        for nm, d in obs.items():
            if nm == "clock":
                continue
            for (_oid, px, signed, _t) in d.get("own_fills", []):
                fills.append((nm, float(signed), float(px)))
        def omid(nm):
            if nm not in obs:
                return np.nan
            bp, _, ap, _ = obs[nm]["bbo"]
            if np.isfinite(bp) and np.isfinite(ap): return 0.5 * (bp + ap)
            return bp if np.isfinite(bp) else (ap if np.isfinite(ap) else np.nan)
        return dict(
            t=obs["clock"]["t_now"], S_star=float(info["S_star"]), v=float(info["v"]),
            true_value={k: float(v) for k, v in info["true_value"].items()},
            true_greeks={k: dict(v) for k, v in info["true_greeks"].items()},
            settlement=[(nm, float(cf)) for (nm, cf, _t) in info["settlement"]],
            fills=fills, obs_mid={nm: omid(nm) for nm in opt_names},
            skew={nm: float(getattr(agent, "last_skew", {}).get(nm, np.nan)) for nm in opt_names},
            est_net_delta=float(getattr(agent, "last_net_delta", np.nan)))

    traj = [snap(obs, info, agent)]
    term = trunc = False; n = 0
    cap = max_steps if max_steps is not None else cfg.max_steps + 5
    while not (term or trunc) and n < cap:
        obs, info, term, trunc = env.step(agent.act(obs))
        traj.append(snap(obs, info, agent)); n += 1
    return dict(traj=traj, opt_names=opt_names, terminated=term, truncated=trunc,
                expiry_years=list(cfg.expiry_years), T_horizon=cfg.T_horizon)

def decompose(ep):
    traj = ep["traj"]; opt_names = ep["opt_names"]
    nstep = len(traj)
    cash = 0.0; pos = {nm: 0.0 for nm in opt_names}; und = 0.0

    t = np.array([s["t"] for s in traj]); S_star = np.array([s["S_star"] for s in traj])
    v = np.array([s["v"] for s in traj])
    W = np.zeros(nstep)
    reval = np.zeros(nstep); edge_opt = np.zeros(nstep)
    edge_hedge = np.zeros(nstep); settle = np.zeros(nstep)
    fills_opt = np.zeros(nstep); fills_hedge = np.zeros(nstep)
    und_path = np.zeros(nstep)
    inv = {nm: np.zeros(nstep) for nm in opt_names}
    true_delta = np.zeros(nstep); true_vega = np.zeros(nstep); true_gamma = np.zeros(nstep)
    est_delta = np.array([s["est_net_delta"] for s in traj])
    obs_mid = {nm: np.array([s["obs_mid"].get(nm, np.nan) for s in traj]) for nm in opt_names}
    true_mid = {nm: np.array([s["true_value"].get(nm, np.nan) for s in traj]) for nm in opt_names}
    skew = {nm: np.array([s["skew"].get(nm, np.nan) for s in traj]) for nm in opt_names}

    def wealth(cash, pos, und, V, S):
        return cash + und * S + sum(pos.get(nm, 0.0) * V.get(nm, 0.0) for nm in V)

    W[0] = wealth(cash, pos, und, traj[0]["true_value"], S_star[0])
    for nm in opt_names: inv[nm][0] = pos.get(nm, 0.0)
    und_path[0] = und
    true_delta[0] = und

    for k in range(1, nstep):
        prev, cur = traj[k - 1], traj[k]
        Vp, Vc = prev["true_value"], cur["true_value"]

        rv = und * (cur["S_star"] - prev["S_star"])
        for nm in Vp:
            if nm in Vc:
                rv += pos.get(nm, 0.0) * (Vc[nm] - Vp[nm])
        reval[k] = rv

        settle_intrinsic = {nm: None for nm, _ in cur["settlement"]}
        for (nm, signed, px) in cur["fills"]:
            cash -= signed * px
            if nm == UNDERLYING:
                edge_hedge[k] += signed * (cur["S_star"] - px); fills_hedge[k] += 1
                und += signed
            else:
                mark = Vc.get(nm, px)
                edge_opt[k] += signed * (mark - px); fills_opt[k] += 1
                pos[nm] = pos.get(nm, 0.0) + signed

        for (nm, cashflow) in cur["settlement"]:
            settle[k] += cashflow - pos.get(nm, 0.0) * Vp.get(nm, 0.0)
            cash += cashflow
            pos[nm] = 0.0

        W[k] = wealth(cash, pos, und, Vc, cur["S_star"])
        und_path[k] = und
        for nm in opt_names: inv[nm][k] = pos.get(nm, 0.0)
        td = und; tv = 0.0; tg = 0.0
        for nm, gr in cur["true_greeks"].items():
            p = pos.get(nm, 0.0); td += p * gr["delta"]; tv += p * gr["vega"]; tg += p * gr["gamma"]
        true_delta[k] = td; true_vega[k] = tv; true_gamma[k] = tg

    dW = np.diff(W, prepend=W[0])
    residual = dW - (reval + edge_opt + edge_hedge + settle)
    return dict(t=t, W=W, S_star=S_star, v=v, est_delta=est_delta,
                true_delta=true_delta, true_vega=true_vega, true_gamma=true_gamma,
                und_pos=und_path, inv=inv, obs_mid=obs_mid, true_mid=true_mid, skew=skew,
                reval=np.cumsum(reval), edge_opt=np.cumsum(edge_opt),
                edge_hedge=np.cumsum(edge_hedge), settle=np.cumsum(settle),
                residual=np.cumsum(residual),
                fills_opt=np.cumsum(fills_opt), fills_hedge=np.cumsum(fills_hedge),
                max_abs_residual=float(np.max(np.abs(residual))))


def plot_dashboard(ep, dc, title="Avellaneda-Stoikov MM", save_path=None):
    t = dc["t"]; Ts = ep["expiry_years"]
    rep = "C_T3" if "C_T3" in ep["opt_names"] else ep["opt_names"][0]

    def vlines(ax):
        for j, T in enumerate(Ts):
            ax.axvline(T, color="0.6", ls=":", lw=0.9)

    fig, ax = plt.subplots(5, 2, figsize=(14, 19))
    fig.suptitle(f"{title}   (terminated={ep['terminated']}, truncated={ep['truncated']})",
                 fontsize=13, y=0.997)

    a = ax[0, 0]; a.plot(t, dc["W"], color="#1b4965", lw=1.4); a.axhline(0, color="0.7", lw=0.8)
    a.set_title(f"Wealth path  W_t   (terminal = {dc['W'][-1]:.1f})"); a.set_ylabel("wealth"); vlines(a)

    a = ax[0, 1]
    a.plot(t, dc["edge_opt"], label="option edge (spread vs fair)", color="#2a9d8f")
    a.plot(t, dc["reval"], label="inventory reval (MtM)", color="#264653")
    a.plot(t, dc["settle"], label="settlement gap", color="#8a5a44")
    a.plot(t, dc["W"] - dc["W"][0], label="ΔW (total)", color="k", ls="--", lw=1.2)
    a.set_title("Cumulative PnL attribution"); a.legend(fontsize=7); a.axhline(0, color="0.7", lw=0.8); vlines(a)

    a = ax[1, 0]; a.plot(t, dc["residual"], color="#9b2226")
    a.set_title(f"Reconciliation residual ΔW−Σcomponents (max|step|={dc['max_abs_residual']:.1e})")
    a.set_ylabel("cum residual"); vlines(a)

    a = ax[1, 1]; a.plot(t, dc["S_star"], color="#005f73", label="S*")
    a.set_ylabel("S*", color="#005f73"); a.tick_params(axis="y", labelcolor="#005f73")
    a2 = a.twinx(); a2.plot(t, dc["v"], color="#bb3e03", lw=0.9)
    a2.set_ylabel("variance v", color="#bb3e03"); a2.tick_params(axis="y", labelcolor="#bb3e03")
    a.set_title("Underlying S* and variance v"); vlines(a)

    a = ax[2, 0]
    a.plot(t, dc["est_delta"], label="agent est net Δ (BS lens)", color="#457b9d", lw=0.9)
    a.plot(t, dc["true_delta"], label="true net Δ (Heston)", color="#e63946", lw=0.9)
    a.axhline(0, color="0.7", lw=0.8)
    a.set_title("Net delta carried")
    a.legend(fontsize=7); vlines(a)

    a = ax[2, 1]; a.plot(t, dc["true_vega"], color="#5f0f40", label="net vega")
    a.axhline(0, color="0.7", lw=0.8); a.set_ylabel("net vega", color="#5f0f40")
    a.tick_params(axis="y", labelcolor="#5f0f40")
    a3 = a.twinx(); a3.plot(t, dc["true_gamma"], color="#0f6f40", lw=0.9)
    a3.set_ylabel("net gamma", color="#0f6f40"); a3.tick_params(axis="y", labelcolor="#0f6f40")
    a.set_title("Un-hedged vega / gamma carried"); vlines(a)

    a = ax[3, 0]
    for nm in ep["opt_names"]:
        a.plot(t, dc["inv"][nm], lw=0.9, label=nm)
    a.axhline(0, color="0.7", lw=0.8); a.set_title("Per-option inventory")
    a.legend(fontsize=7, ncol=3); vlines(a)

    a = ax[3, 1]
    a.plot(t, dc["skew"][rep], color="#7b2cbf", lw=0.9)
    a.axhline(0, color="0.7", lw=0.8)
    a.set_ylabel(f"{rep} reservation skew (r−m)", color="#7b2cbf")
    a.tick_params(axis="y", labelcolor="#7b2cbf")
    ai = a.twinx(); ai.plot(t, dc["inv"][rep], color="#1d3557", lw=0.9)
    ai.set_ylabel(f"{rep} inventory", color="#1d3557"); ai.tick_params(axis="y", labelcolor="#1d3557")
    a.set_title(f"{rep}: inventory skew (quotes pushed away from inventory)"); vlines(a)

    a = ax[4, 0]
    a.plot(t, dc["obs_mid"][rep], label="observed BBO mid", color="#ee9b00", lw=0.9)
    a.plot(t, dc["true_mid"][rep], label="true fair (Heston)", color="#005f73", lw=0.9)
    a.set_title(f"{rep}: observed book mid vs true fair value"); a.set_xlabel("t (yr)")
    a.legend(fontsize=7); vlines(a)

    a = ax[4, 1]
    a.plot(t, dc["fills_opt"], label="option fills", color="#2a9d8f")
    a.set_title("Cumulative own option fills"); a.set_xlabel("t (yr)"); a.legend(fontsize=7); vlines(a)

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()
    return fig