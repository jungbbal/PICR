"""PCA of the context latent z, the mass axis in it, and one episode's LIVE z track.

Three questions, one script:

  1. What does z look like?  PCA over every phase-3 window of the contract's `val`
     episodes. The axes come from z's own covariance -- mass is never used to fit
     them, only to colour afterwards. (This checkpoint has ``latent_dim = 2``, so the
     PCA is a rotation of the raw latent, not a projection: nothing is discarded and
     PC1 + PC2 explain 100%. The rotation still earns its place -- it puts the
     direction z actually varies along on the x axis, which is where the mass
     ordering becomes readable.)

  2. Light vs heavy.  The lattice draws mass from 8 levels, 0.1 ... 2.0 kg. Each level
     gets a centroid and a covariance ellipse from the val cloud; those are the
     "regions" the next question refers to.

  3. Where does the running z sit inside one episode?  The encoder is re-run at every
     timestep on the trailing ``history_len`` window -- exactly what a controller would
     get online -- and the resulting path is drawn over the map. Windows that straddle
     the lift (phase < 3) are drawn dashed: the model never saw them in training, and
     the point of showing them is that z only settles into its mass band once the
     payload is actually on the arm.

Only the arm channels reach the encoder for this checkpoint (``context_input_arm_only``:
joint_pos, joint_vel, applied_torque), so nothing here reads the object's pose.

Run::

    source env.sh && PICR_DATA_ROOT=... python3 -m cdelan.analysis.z_pca_mass \
        --ckpt runs/ours/armpay_lattice_dq713_s0/model.pt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")

from . import _plotstyle

_plotstyle.apply(0.82)

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse

from cdelan import paths
from ._common import DEVICE, build_dataset, load_model

DEFAULT_CKPT = "runs/ours/armpay_lattice_dq713_s0/model.pt"
DEFAULT_CONTRACT = "contracts/protocol_lattice_fitcarve_v1.json"
MASS_CMAP = "viridis"


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #
def encode(model, history_states: np.ndarray, history_torques: np.ndarray,
           batch: int = 8192) -> np.ndarray:
    """z = mu(history), the deterministic latent an evaluation run would use."""
    out = []
    with torch.no_grad():
        for i in range(0, len(history_states), batch):
            hs = torch.from_numpy(history_states[i : i + batch]).to(DEVICE)
            ht = torch.from_numpy(history_torques[i : i + batch]).to(DEVICE)
            out.append(model.encode_context(hs, ht, deterministic_context=True).mu.cpu().numpy())
    return np.concatenate(out, axis=0)


def episode_meta(hdf5_path: str, names: list[str]) -> dict[str, dict]:
    """Per-episode (mass, CoM, phase) straight from the recording's randomization group."""
    meta: dict[str, dict] = {}
    with h5py.File(hdf5_path, "r") as fh:
        data = fh["data"]
        for name in names:
            grid = np.asarray(data[name]["episode_physics_randomization"]["object_masscom_grid"][0])
            phase = np.asarray(data[name]["obs"]["phase"]).reshape(-1)
            meta[name] = {
                "mass": float(grid[1]),
                "com": grid[2:5].astype(float).tolist(),
                "n_phase3": int((phase == 3).sum()),
            }
    return meta


def episode_track(model, cfg: dict, hdf5_path: str, name: str) -> dict:
    """The z a controller would compute at EVERY timestep of one episode.

    ``phase_filter`` is lifted and the horizon shortened to 1 so the windows run from the
    first full history to the end of the recording rather than only over phase 3. The
    history content of a window is unaffected by either knob -- it is
    ``state[t-k:t+1]`` regardless -- so these z's are the same numbers the phase-3 cloud
    holds, plus the pre-lift ones the cloud filters out.
    """
    live_cfg = dict(cfg)
    live_cfg["phase_filter"] = None
    live_cfg["rollout_horizon"] = 1
    ds = build_dataset([(hdf5_path, name)], live_cfg)

    order = np.argsort(ds.window_start)
    z = encode(model, ds.history_states[order], ds.history_torques[order])
    # Index of the window's LAST history frame -- the "now" the encoder is standing at.
    now = ds.window_start[order] + int(cfg["history_len"]) - 1

    with h5py.File(hdf5_path, "r") as fh:
        phase = np.asarray(fh["data"][name]["obs"]["phase"]).reshape(-1)
    if cfg.get("align_object_clock"):
        phase = phase[1:]  # the obs clock is shifted by one; see the dataset's note
    # The lift is where phase 3 begins AND HOLDS to the end of the recording, not the first
    # frame labelled 3. Some episodes carry a one-frame phase-3 blip during the grasp --
    # demo_31500 reads `1 1 1 1 3 2 2 2 2 3 ...` -- and taking the first 3 puts t = 0 five
    # frames early, which then marks genuinely pre-lift windows as held.
    not_three = np.flatnonzero(phase != 3)
    lift = int(not_three[-1]) + 1 if len(not_three) else 0
    if lift >= len(phase):  # never settles into phase 3; fall back to the first 3 there is
        lift = int(np.argmax(phase == 3)) if (phase == 3).any() else 0
    return {
        "z": z,
        "now": now,
        "phase": phase[now],
        # The payload is on the arm from `lift` onward. Consumers want this rather than
        # `phase == 3`, which would also pick up the blip above as a lone held frame.
        "held": now >= lift,
        "t_rel": (now - lift) * float(cfg["dt"]),  # seconds relative to the lift
        "mass": float(ds.object_context[0, 0]),
    }


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def pca_fit(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, top-two components, explained-variance ratio of every component) from z alone.

    Only PC1 and PC2 are returned, so a latent wider than 2 (the InfoNCE runs are 4-D) projects
    to the same 2-D map every figure draws; the ratios cover all components, so the axis labels
    say how much of z the map shows. For a 2-D latent this is the full rotation, as before.
    """
    mean = z.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(z - mean, full_matrices=False)
    return mean, vt[:2], (s**2) / (s**2).sum()


def cov_ellipse(ax, pts: np.ndarray, n_std: float = 2.0, **kw) -> None:
    """2-sigma ellipse of a point set -- the 'region' a mass level occupies."""
    if len(pts) < 3:
        return
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse(pts.mean(0), width, height, angle=angle, **kw))


def time_cmap():
    """magma, truncated before it turns near-white -- a track has to stay visible on paper."""
    base = plt.get_cmap("magma")
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "magma_cut", base(np.linspace(0.02, 0.82, 256))
    )


def nearest_level(p1: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Which mass level's PC1 centroid each window sits closest to."""
    return np.argmin(np.abs(p1[:, None] - centroids[None, :]), axis=1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) by rank -- how separable light from heavy is along one axis."""
    both = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(both)).astype(float) + 1.0
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# --------------------------------------------------------------------------- #
def build_cloud(model, cfg: dict, hdf5_path: str, names: list[str],
                meta: dict[str, dict]) -> dict:
    """Every phase-3 window of the chosen val episodes, encoded."""
    ds = build_dataset([(hdf5_path, n) for n in names], cfg)
    z = encode(model, ds.history_states, ds.history_torques)
    ep_name = np.array([m[1] for m in ds.sample_meta])
    return {
        "z": z,
        "mass": ds.object_context[:, 0].astype(float),
        "com_x": np.array([meta[n]["com"][0] for n in ep_name]),
        "episode": ep_name,
        "n_windows": len(z),
    }


def pick_demo_episodes(meta: dict[str, dict], masses: list[float],
                       min_phase3: int = 70) -> list[str]:
    """One episode per requested mass level, all sharing a CoM cell where possible.

    Holding CoM fixed across the light/heavy pair means the only thing that differs
    between the two tracks is the payload mass, which is the comparison being drawn.
    """
    by_com: dict[tuple, dict[float, list[str]]] = defaultdict(lambda: defaultdict(list))
    for name, m in meta.items():
        if m["n_phase3"] >= min_phase3:
            by_com[tuple(np.round(m["com"], 4))][round(m["mass"], 3)].append(name)

    wanted = [round(m, 3) for m in masses]
    best = max(by_com.values(), key=lambda cell: sum(1 for w in wanted if cell.get(w)))
    picked = []
    for w in wanted:
        pool = best.get(w) or [n for n, m in meta.items()
                               if round(m["mass"], 3) == w and m["n_phase3"] >= min_phase3]
        picked.append(sorted(pool)[0])
    return picked


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figure_map(cloud: dict, mean: np.ndarray, comps: np.ndarray, evr: np.ndarray,
               levels: np.ndarray, out: Path) -> dict:
    p = (cloud["z"] - mean) @ comps.T
    mass = cloud["mass"]
    norm = plt.Normalize(levels.min(), levels.max())
    cmap = plt.get_cmap(MASS_CMAP)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 13.0))
    (ax_map, ax_lh), (ax_pc1, ax_com) = axes

    # (a) the whole cloud, colour = true mass
    sc = ax_map.scatter(p[:, 0], p[:, 1], s=4, c=mass, cmap=cmap, norm=norm,
                        alpha=0.45, linewidths=0, rasterized=True)
    for lv in levels:
        pts = p[np.isclose(mass, lv, atol=1e-3)]
        cov_ellipse(ax_map, pts, 2.0, facecolor="none", edgecolor=cmap(norm(lv)), lw=2.0, zorder=4)
        ax_map.scatter(*pts.mean(0), s=110, marker="o", facecolor="w",
                       edgecolor=cmap(norm(lv)), lw=2.4, zorder=5)
    fig.colorbar(sc, ax=ax_map, label="true payload mass (kg)")
    ax_map.set_xlabel(f"PC1  ({evr[0] * 100:.1f}% of z variance)")
    ax_map.set_ylabel(f"PC2  ({evr[1] * 100:.1f}%)")
    ax_map.set_title("(a) PCA of z — axes from z alone, colour added after")

    # (b) light vs heavy only
    light, heavy = levels[0], levels[-1]
    p_l, p_h = p[np.isclose(mass, light, atol=1e-3)], p[np.isclose(mass, heavy, atol=1e-3)]
    ax_lh.scatter(p[:, 0], p[:, 1], s=4, c="0.86", linewidths=0, rasterized=True, zorder=1)
    for pts, lv, col in ((p_l, light, cmap(0.08)), (p_h, heavy, cmap(0.92))):
        ax_lh.scatter(pts[:, 0], pts[:, 1], s=7, color=col, alpha=0.6, linewidths=0,
                      rasterized=True, zorder=2, label=f"{lv:.2f} kg   (n={len(pts)})")
        cov_ellipse(ax_lh, pts, 2.0, facecolor="none", edgecolor=col, lw=2.6, zorder=4)
    sep = auroc(p_h[:, 0], p_l[:, 0])
    ax_lh.set_xlabel("PC1")
    ax_lh.set_ylabel("PC2")
    ax_lh.set_title(f"(b) light vs heavy — PC1 separates them, AUC = {sep:.3f}")
    ax_lh.legend(loc="best", framealpha=0.92)

    # (c) PC1 marginal per mass level
    parts = [p[np.isclose(mass, lv, atol=1e-3), 0] for lv in levels]
    vp = ax_pc1.violinplot(parts, positions=levels, widths=0.20, showmedians=True)
    for body, lv in zip(vp["bodies"], levels):
        body.set_facecolor(cmap(norm(lv)))
        body.set_alpha(0.85)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        vp[key].set_color("0.25")
    rho = spearman(mass, p[:, 0])
    ax_pc1.set_xlabel("true payload mass (kg)")
    ax_pc1.set_ylabel("PC1")
    ax_pc1.set_title(f"(c) PC1 is monotone in mass — Spearman ρ = {rho:.3f}")

    # (d) same cloud, colour = CoM offset. Mass was the only supervised axis; this asks
    #     whether the second latent direction picked the other randomized parameter up.
    sc2 = ax_com.scatter(p[:, 0], p[:, 1], s=4, c=cloud["com_x"] * 100.0, cmap="coolwarm",
                         alpha=0.5, linewidths=0, rasterized=True)
    fig.colorbar(sc2, ax=ax_com, label="CoM offset, body-x (cm)")
    rho_com = spearman(cloud["com_x"], p[:, 1])
    ax_com.set_xlabel("PC1")
    ax_com.set_ylabel("PC2")
    ax_com.set_title(f"(d) unsupervised axis: CoM vs PC2, ρ = {rho_com:.3f}")

    fig.suptitle("Context latent z of the trained model — phase-3 windows, contract val split",
                 fontsize=21, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out.with_suffix(".png"), dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    # How well does the position of z alone name the mass? Nearest PC1 centroid, over the
    # same cloud the centroids come from -- descriptive, not a held-out classifier.
    centroids = np.array([p[np.isclose(mass, lv, atol=1e-3), 0].mean() for lv in levels])
    truth = np.argmin(np.abs(mass[:, None] - levels[None, :]), axis=1)
    guess = nearest_level(p[:, 0], centroids)
    per_ep = []
    for name in np.unique(cloud["episode"]):
        sel = cloud["episode"] == name
        vote = np.bincount(guess[sel], minlength=len(levels)).argmax()
        per_ep.append(vote == truth[sel][0])

    return {
        "auc_light_vs_heavy": sep,
        "spearman_pc1_mass": rho,
        "spearman_pc2_com": rho_com,
        "window_level_accuracy": round(float((guess == truth).mean()), 4),
        "window_within_one_level": round(float((np.abs(guess - truth) <= 1).mean()), 4),
        "episode_vote_accuracy": round(float(np.mean(per_ep)), 4),
    }


def figure_tracks(cloud: dict, tracks: list[dict], mean: np.ndarray, comps: np.ndarray,
                  evr: np.ndarray, levels: np.ndarray, out: Path) -> dict:
    p = (cloud["z"] - mean) @ comps.T
    mass = cloud["mass"]
    norm = plt.Normalize(levels.min(), levels.max())
    cmap = plt.get_cmap(MASS_CMAP)

    tcmap = time_cmap()
    settle_after = 0.4  # s. The first ~0.3 s after the lift is the encoder's transient.

    fig = plt.figure(figsize=(17.5, 12.6))
    gs = fig.add_gridspec(2, len(tracks), height_ratios=[1.35, 1.0], hspace=0.30, wspace=0.20)
    ax_map = fig.add_subplot(gs[0, :])

    ax_map.scatter(p[:, 0], p[:, 1], s=4, c="0.88", linewidths=0, rasterized=True, zorder=1)
    for i, lv in enumerate(levels):
        pts = p[np.isclose(mass, lv, atol=1e-3)]
        cov_ellipse(ax_map, pts, 2.0, facecolor=cmap(norm(lv)), alpha=0.18, edgecolor="none", zorder=2)
        cov_ellipse(ax_map, pts, 2.0, facecolor="none", edgecolor=cmap(norm(lv)), lw=1.6, zorder=3)
        # Staggered, above and below the arc in turn, so eight labels fit without overlapping.
        cen = pts.mean(0)
        ax_map.annotate(f"{lv:.2f}", (cen[0], cen[1] + (2.15 if i % 2 else -2.15)),
                        fontsize=12.5, fontweight="bold", ha="center", va="center",
                        color=cmap(norm(lv)), zorder=6,
                        bbox=dict(boxstyle="round,pad=0.18", fc="w", ec=cmap(norm(lv)),
                                  lw=1.4, alpha=0.95))

    stats, lc = {}, None
    for tr in tracks:
        q = (tr["z"] - mean) @ comps.T
        held = tr["held"]
        col = cmap(norm(tr["mass"]))
        qh, th = q[held], tr["t_rel"][held]

        # pre-lift half: outside the training distribution, so a thin dashed lead-in only.
        if (~held).any():
            ax_map.plot(q[~held, 0], q[~held, 1], "--", color="0.5", lw=1.4, alpha=0.8, zorder=7)
        # In phase 3 the line is coloured by time, so the direction of travel is visible. The
        # first `settle_after` seconds are drawn thin: over that stretch the trailing window
        # still contains pre-lift frames, so z is answering a question about a payload the arm
        # was not yet carrying. It is kept rather than cut -- that transient is the cost of
        # a window-based estimator -- but it is not what the figure is about.
        tnorm = plt.Normalize(0.0, max(t["t_rel"].max() for t in tracks))
        for keep, width, alpha, zo in ((th[:-1] < settle_after, 1.1, 0.45, 6),
                                       (th[:-1] >= settle_after, 2.8, 1.0, 8)):
            seg = np.stack([qh[:-1][keep], qh[1:][keep]], axis=1)
            lc_part = LineCollection(seg, cmap=tcmap, array=th[:-1][keep], lw=width,
                                     alpha=alpha, zorder=zo, norm=tnorm)
            ax_map.add_collection(lc_part)
            if width > 2:
                lc = lc_part
        settled_mask = th >= settle_after
        ax_map.scatter(qh[settled_mask, 0], qh[settled_mask, 1], s=22, color=col,
                       edgecolor="k", lw=0.35, zorder=9)
        ax_map.scatter(*qh[0], s=210, marker="X", color="w", edgecolor="k", lw=1.8, zorder=11)

        settled = qh[settled_mask]
        ax_map.scatter(*settled.mean(0), s=330, marker="o", facecolor="none", edgecolor="k",
                       lw=2.6, zorder=12)
        ax_map.annotate(f"{tr['mass']:.2f} kg", (settled.mean(0)[0], settled.mean(0)[1] + 0.95),
                        fontsize=15, fontweight="bold", color="k", ha="center", zorder=13,
                        bbox=dict(boxstyle="round,pad=0.22", fc="w", ec=col, lw=2.2, alpha=0.97))

        own = p[np.isclose(mass, tr["mass"], atol=1e-3)]
        rad = 2.0 * np.sqrt(np.mean(np.sum((own - own.mean(0)) ** 2, axis=1)) / 2.0)
        centroids = np.array([p[np.isclose(mass, lv, atol=1e-3), 0].mean() for lv in levels])
        decoded = levels[nearest_level(qh[th >= settle_after][:, 0], centroids)]
        stats[f"{tr['mass']:.2f}kg"] = {
            "episode": tr["name"],
            "n_windows_phase3": int(held.sum()),
            "frac_inside_own_2sigma_after_settle": round(
                float(np.mean(np.linalg.norm(settled - own.mean(0), axis=1) <= rad)), 3),
            "settled_decoded_level_kg": round(float(np.median(decoded)), 3),
            "settled_level_correct_frac": round(float(np.mean(np.isclose(decoded, tr["mass"], atol=1e-3))), 3),
            "pc1_std_after_settle": round(float(settled[:, 0].std()), 3),
        }

    ax_map.plot([], [], "--", color="0.5", lw=1.4, label="before the lift (phase < 3, unseen in training)")
    ax_map.plot([], [], "-", color="0.4", lw=1.1, label=f"live z, transient (0 < t < {settle_after:g} s)")
    ax_map.plot([], [], "-", color="0.15", lw=2.8, label=f"live z, settled (t > {settle_after:g} s)")
    ax_map.scatter([], [], s=150, marker="X", color="w", edgecolor="k", lw=1.6, label="lift moment (t = 0)")
    ax_map.scatter([], [], s=190, marker="o", facecolor="none", edgecolor="k", lw=2.2,
                   label="settled mean")
    ax_map.legend(loc="upper left", fontsize=12.5, framealpha=0.94, ncol=2)
    # Clipped to the cloud's bulk: the pre-lift lead-in reaches far outside it and a handful of
    # windows reach far above it, and letting either set the limits squashes the region the
    # figure is actually about.
    lo, hi = np.percentile(p, [0.2, 99.8], axis=0)
    pad = 0.09 * (hi - lo)
    ax_map.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax_map.set_ylim(lo[1] - pad[1], hi[1] + 2.2 * pad[1])
    ax_map.set_xlabel(f"PC1  ({evr[0] * 100:.1f}%)")
    ax_map.set_ylabel(f"PC2  ({evr[1] * 100:.1f}%)")
    ax_map.set_title("Live z inside one episode, over the mass regions of the val cloud "
                     "(shaded: 2σ per mass level)")
    fig.colorbar(lc, ax=ax_map, label="time since lift (s)", fraction=0.028, pad=0.01)

    for j, tr in enumerate(tracks):
        ax = fig.add_subplot(gs[1, j])
        q = (tr["z"] - mean) @ comps.T
        held = tr["held"]
        for lv in levels:
            band = p[np.isclose(mass, lv, atol=1e-3), 0]
            own = np.isclose(lv, tr["mass"], atol=1e-3)
            ax.axhspan(band.mean() - 2 * band.std(), band.mean() + 2 * band.std(),
                       color=cmap(norm(lv)), alpha=0.45 if own else 0.16, lw=0, zorder=1)
            ax.axhline(band.mean(), color=cmap(norm(lv)), lw=2.2 if own else 1.0,
                       alpha=0.95 if own else 0.7, zorder=2)
        ax.plot(tr["t_rel"][~held], q[~held, 0], "--", color="0.5", lw=1.8, zorder=4)
        ax.plot(tr["t_rel"][held], q[held, 0], "-", color="k", lw=2.4, zorder=5)
        ax.axvline(0.0, color="0.3", ls=":", lw=1.6, zorder=3)
        ax.axvspan(tr["t_rel"].min(), settle_after, color="0.5", alpha=0.10, lw=0, zorder=0)
        ax.set_xlabel("time since lift (s)")
        if j == 0:
            ax.set_ylabel("PC1 of the live z")
        ax.set_title(f"{tr['mass']:.2f} kg — {tr['name']}", fontsize=15)
        ax.set_xlim(-0.45, max(t["t_rel"].max() for t in tracks))
        ax.set_ylim(p[:, 0].min() - 0.4, p[:, 0].max() + 0.4)

    fig.suptitle("Where the online z sits — one episode, encoder re-run every timestep",
                 fontsize=21, fontweight="bold")
    # subplots_adjust, not tight_layout: the map's colorbar is attached to one axes of a
    # gridspec that also spans the row below it, which tight_layout cannot solve (and says so).
    fig.subplots_adjust(top=0.915, bottom=0.065, left=0.062, right=0.945)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return stats


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--contract", default=DEFAULT_CONTRACT)
    ap.add_argument("--episodes-per-mass", type=int, default=70,
                    help="val episodes per mass level in the background cloud")
    ap.add_argument("--track-mass", type=float, nargs="+", default=[0.1, 1.186, 2.0],
                    help="mass levels to draw a live z track for")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    # plot/, not results/: these are figures, and plot/ is where this repo keeps the ones the
    # paper draws from. `results/` holds metric dumps.
    out_dir = Path(args.out_dir or (paths.REPO_ROOT.parent / "plot" / "latent"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(args.ckpt)
    contract = json.loads(Path(args.contract).read_text())
    hdf5_path = str(paths.dataset_file(contract["dataset_id"]))
    val_names = contract["splits"]["val"]

    print(f"device      {DEVICE}")
    print(f"checkpoint  {args.ckpt}  (latent_dim={model.latent_dim}, "
          f"context_input_arm_only={model.context_input_arm_only})")
    print(f"dataset     {hdf5_path}")
    print(f"split       contract val, {len(val_names)} episodes\n")

    meta = episode_meta(hdf5_path, val_names)
    levels = np.array(sorted({round(m["mass"], 3) for m in meta.values()}))

    rng = np.random.default_rng(0)
    chosen: list[str] = []
    for lv in levels:
        pool = sorted(n for n, m in meta.items() if round(m["mass"], 3) == lv)
        take = min(args.episodes_per_mass, len(pool))
        chosen += [pool[i] for i in rng.choice(len(pool), size=take, replace=False)]
    cloud = build_cloud(model, cfg, hdf5_path, chosen, meta)
    print(f"cloud       {cloud['n_windows']} phase-3 windows from {len(chosen)} episodes, "
          f"{len(levels)} mass levels {levels.tolist()}")

    mean, comps, evr = pca_fit(cloud["z"])
    # Sign only: orient PC1 so heavier is to the right, and keep the frame right-handed.
    if spearman(cloud["mass"], (cloud["z"] - mean) @ comps[0]) < 0:
        comps = comps * np.array([[-1.0], [-1.0]])
    print(f"PCA         explained variance {np.round(evr, 4).tolist()}  "
          f"(latent_dim={cloud['z'].shape[1]}, so this is a rotation)")

    demo = pick_demo_episodes(meta, args.track_mass)
    tracks = []
    for name in demo:
        tr = episode_track(model, cfg, hdf5_path, name)
        tr["name"] = name
        tr["com"] = meta[name]["com"]
        tracks.append(tr)
        print(f"track       {name}  mass={tr['mass']:.3f} kg  com={np.round(tr['com'], 4).tolist()}  "
              f"{len(tr['z'])} windows ({int(tr['held'].sum())} held)")
    print()

    map_stats = figure_map(cloud, mean, comps, evr, levels, out_dir / "z_pca_mass")
    track_stats = figure_tracks(cloud, tracks, mean, comps, evr, levels, out_dir / "z_live_track")

    summary = {
        "checkpoint": args.ckpt,
        "n_windows": cloud["n_windows"],
        "n_episodes": len(chosen),
        "mass_levels": levels.tolist(),
        "explained_variance_ratio": evr.tolist(),
        **map_stats,
        "tracks": track_stats,
    }
    (out_dir / "z_pca_mass.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out_dir}/z_pca_mass.{{png,pdf}}, {out_dir}/z_live_track.{{png,pdf}}")


if __name__ == "__main__":
    main()
