"""Swap the payload mid-window: does z walk over to the NEW mass's region?

z is not a recurrent state. It is a function of the last ``history_len`` frames of
(q, dq, tau), so "the mass changed at time t" has exactly one meaning here: the window
holds a DIFFERENT payload's frames, and the new payload's frames then arrive one per
step. At step 0 the window is pure donor; at step ``history_len`` it is pure recipient
and z is, by construction, a real window of a real episode of the new mass. The question
is not whether it arrives -- it must -- but **how many of those steps it needs**, and
whether it lands in the new mass's region rather than somewhere between.

Both halves of the splice are in-distribution phase-3 windows. That matters: a window
straddling a real grasp is input the model never saw under ``phase_filter: 3``, and
encoding one gives nonsense for light and heavy alike (see `z_adaptation_video`), so a
real lift cannot serve as the "wrong belief" starting point.

The seam. The two halves come from different episodes, so the arm jumps where they meet.
The donor is chosen to minimise that jump in [q, dq] -- velocity included, because a dq
discontinuity is what a spurious acceleration, i.e. a spurious mass, would look like --
and the residual seam is reported so it can be weighed against the crossing time.

Outputs
    z_mass_swap.png/.pdf   the swap drawn on the same PCA map as `z_pca_mass`
    z_mass_swap.json       per-pair medians and the full from x to matrix

Run::

    source env.sh && PICR_DATA_ROOT=... python3 -m cdelan.analysis.z_mass_swap \
        --ckpt runs/ours/armpay_lattice_dq713_s0/model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

from . import _plotstyle

_plotstyle.apply(0.82)

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from cdelan import paths
from ._common import DEVICE, build_dataset, load_model
from .z_pca_mass import (
    DEFAULT_CKPT,
    DEFAULT_CONTRACT,
    MASS_CMAP,
    cov_ellipse,
    encode,
    episode_meta,
    pca_fit,
    spearman,
    time_cmap,
)


def level_stats(p: np.ndarray, mass: np.ndarray, levels: np.ndarray) -> dict:
    """Centroid and covariance of each mass level in PCA space -- the target 'regions'."""
    mu, cov = {}, {}
    for lv in levels:
        pts = p[np.isclose(mass, lv, atol=1e-3)]
        mu[lv] = pts.mean(0)
        cov[lv] = np.cov(pts.T)
    return {"mu": mu, "cov": cov}


def mahalanobis(pts: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = pts - mu
    return np.sqrt(np.maximum(np.einsum("...i,ij,...j->...", d, np.linalg.inv(cov), d), 0.0))


def splice_traces(model, ds, z_all, src_rows: np.ndarray, dst_rows: np.ndarray,
                  chrono: dict[int, np.ndarray], history_len: int, settle: int,
                  n_pairs: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Encode every step of ``n_pairs`` swaps. Returns (z per step, seam size per pair).

    Step s < history_len holds ``donor[s:] + recipient[:s]``; from history_len on the window
    is pure recipient and simply keeps sliding through its own episode, which is what shows
    whether z stays put once it has arrived.
    """
    pos = {int(r): i for rows in chrono.values() for i, r in enumerate(rows)}
    ep_of = ds.episode_id
    room = np.array([len(chrono[int(ep_of[r])]) - pos[int(r)] - 1 for r in dst_rows])
    keep = dst_rows[room >= settle]
    if len(keep) < 4:
        keep = dst_rows[np.argsort(-room)[: max(4, min(8, len(dst_rows)))]]
    if len(keep) > n_pairs:
        keep = keep[rng.choice(len(keep), size=n_pairs, replace=False)]

    # Nearest donor by the newest [q, dq] the encoder would see, standardised per channel.
    f_src = ds.history_states[src_rows][:, -1, :18]
    f_dst = ds.history_states[keep][:, 0, :18]
    scale = np.maximum(np.concatenate([f_src, f_dst]).std(0), 1e-6)
    dist = np.linalg.norm((f_src[:, None] - f_dst[None]) / scale, axis=2)
    # The same-mass control splices two episodes of ONE level, so the donor must be barred
    # from the recipient's own episode -- otherwise it can pick the recipient's own
    # neighbouring window and the "splice" is not a splice at all.
    same = ds.episode_id[src_rows][:, None] == ds.episode_id[keep][None, :]
    dist = np.where(same, np.inf, dist)
    donor = src_rows[dist.argmin(0)]
    seam = np.linalg.norm(ds.history_states[donor][:, -1, :18]
                          - ds.history_states[keep][:, 0, :18], axis=1)

    states, torques = [], []
    for i, j in zip(donor, keep):
        for s in range(history_len + 1):
            states.append(np.concatenate([ds.history_states[i][s:], ds.history_states[j][:s]], 0))
            torques.append(np.concatenate([ds.history_torques[i][s:], ds.history_torques[j][:s]], 0))
        rows = chrono[int(ep_of[j])]
        p0 = pos[int(j)]
        for k in range(1, settle + 1):
            r = rows[p0 + k]
            states.append(ds.history_states[r])
            torques.append(ds.history_torques[r])
    z = encode(model, np.stack(states), np.stack(torques))
    return z.reshape(len(keep), history_len + 1 + settle, -1), seam


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--contract", default=DEFAULT_CONTRACT)
    ap.add_argument("--episodes-per-mass", type=int, default=45)
    ap.add_argument("--pairs", type=int, default=120, help="swaps averaged per (from, to)")
    ap.add_argument("--settle", type=int, default=8, help="steps to keep sliding after the window is pure")
    ap.add_argument("--show", type=float, nargs="+", default=[0.1, 2.0],
                    help="the two levels drawn on the map, in both directions")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or (paths.REPO_ROOT.parent / "plot" / "latent"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(args.ckpt)
    contract = json.loads(Path(args.contract).read_text())
    hdf5_path = str(paths.dataset_file(contract["dataset_id"]))
    H = int(cfg["history_len"])

    meta = episode_meta(hdf5_path, contract["splits"]["val"])
    levels = np.array(sorted({round(m["mass"], 3) for m in meta.values()}))

    rng = np.random.default_rng(0)
    chosen: list[str] = []
    for lv in levels:
        pool = sorted(n for n, m in meta.items() if round(m["mass"], 3) == lv)
        chosen += [pool[i] for i in rng.choice(len(pool),
                                               size=min(args.episodes_per_mass, len(pool)),
                                               replace=False)]
    ds = build_dataset([(hdf5_path, n) for n in chosen], cfg)
    z_all = encode(model, ds.history_states, ds.history_torques)
    mass = ds.object_context[:, 0].astype(float)

    mean, comps, evr = pca_fit(z_all)
    if spearman(mass, (z_all - mean) @ comps[0]) < 0:
        comps = comps * np.array([[-1.0], [-1.0]])
    p = (z_all - mean) @ comps.T
    stats = level_stats(p, mass, levels)
    centroids = np.array([stats["mu"][lv][0] for lv in levels])

    print(f"device      {DEVICE}\ncheckpoint  {args.ckpt}")
    print(f"cloud       {len(p)} windows, {len(chosen)} episodes, PC1 {evr[0] * 100:.1f}%")
    print(f"window      history_len = {H} steps, so step {H} is a pure recipient window\n")

    rows_of = {lv: np.where(np.isclose(mass, lv, atol=1e-3))[0] for lv in levels}
    chrono = {e: rows for e, rows in enumerate(ds.episode_window_indices)}

    def project(z):
        return (z - mean) @ comps.T

    # ---- the full from x to matrix ----
    steps_med = np.full((len(levels), len(levels)), np.nan)
    final_kg = np.full((len(levels), len(levels)), np.nan)
    traces: dict[tuple, dict] = {}
    for a, src in enumerate(levels):
        for b, dst in enumerate(levels):
            zt, seam = splice_traces(model, ds, z_all, rows_of[src], rows_of[dst], chrono,
                                     H, args.settle, args.pairs, rng)
            q = project(zt.reshape(-1, zt.shape[-1])).reshape(zt.shape[0], zt.shape[1], 2)
            # "arrived" = inside the target level's own 2-sigma region, the same ellipse the
            # figures draw. First step at which that is true, per pair.
            d = mahalanobis(q, stats["mu"][dst], stats["cov"][dst])
            inside = d <= 2.0
            first = np.where(inside.any(1), inside.argmax(1), H + args.settle + 1)
            steps_med[a, b] = float(np.median(first))
            kg = np.interp(q[:, -1, 0], centroids, levels)
            final_kg[a, b] = float(np.median(kg))
            traces[(round(src, 3), round(dst, 3))] = {
                "q": q, "first": first, "seam": seam, "inside": inside,
            }
        print(f"from {src:.2f} kg -> " + "  ".join(
            f"{levels[b]:.2f}:{steps_med[a, b]:.0f}" for b in range(len(levels)) if b != a))

    # ---- control: splice two episodes of the SAME mass -------------------------------
    # The two halves always come from different episodes, so the arm jumps at the seam
    # whether or not the mass changed. If that jump alone moved z out of its region, the
    # crossing times above would be measuring the seam. On the diagonal the mass does not
    # change, so anything that moves here is seam and not payload.
    ctrl_first, ctrl_inside = [], []
    for lv in levels:
        tr = traces[(round(lv, 3), round(lv, 3))]
        ctrl_first.append(np.median(tr["first"]))
        ctrl_inside.append(tr["inside"][:, : H + 1].mean())
    print(f"\ncontrol (same mass, different episode): median steps to enter own region "
          f"{np.median(ctrl_first):.0f}, windows inside own 2σ across the splice "
          f"{np.mean(ctrl_inside) * 100:.1f}%")

    show = [round(v, 3) for v in args.show]
    pairs = [(show[0], show[-1]), (show[-1], show[0])]

    print()
    summary = {"checkpoint": args.ckpt, "history_len": H, "settle_steps": args.settle,
               "n_windows_cloud": int(len(p)), "levels": levels.tolist(),
               "median_steps_to_enter_target": steps_med.tolist(),
               "median_final_readout_kg": final_kg.tolist(),
               "control_same_mass": {
                   "median_steps_to_enter_own_region": float(np.median(ctrl_first)),
                   "frac_windows_inside_own_2sigma": float(np.mean(ctrl_inside)),
               },
               "pairs": {}}
    for key in pairs:
        tr = traces[key]
        first = tr["first"]
        never = int((first > H).sum())
        summary["pairs"][f"{key[0]:.2f}->{key[1]:.2f}"] = {
            "n_pairs": int(len(first)),
            "median_steps_to_enter_target": float(np.median(first)),
            "iqr": [float(np.percentile(first, 25)), float(np.percentile(first, 75))],
            "needed_full_window": f"{never}/{len(first)}",
            "seam_dq_median": float(np.median(tr["seam"])),
            # If the seam were driving the crossing, a bigger jump would move z sooner.
            "spearman_seam_vs_steps": round(spearman(tr["seam"], first.astype(float)), 3),
            "final_readout_kg_median": float(np.median(
                np.interp(tr["q"][:, -1, 0], centroids, levels))),
        }
        s = summary["pairs"][f"{key[0]:.2f}->{key[1]:.2f}"]
        print(f"[{key[0]:.2f} -> {key[1]:.2f} kg]  {s['n_pairs']} swaps  "
              f"median {s['median_steps_to_enter_target']:.0f} steps "
              f"(IQR {s['iqr'][0]:.0f}-{s['iqr'][1]:.0f}) of {H}  "
              f"needed the full window: {s['needed_full_window']}  "
              f"seam |d| {s['seam_dq_median']:.3f} (rho vs steps {s['spearman_seam_vs_steps']:+.2f})  "
              f"final {s['final_readout_kg_median']:.2f} kg")

    figure(p, mass, levels, stats, centroids, evr, traces, pairs, H, args.settle,
           out_dir / "z_mass_swap")
    (out_dir / "z_mass_swap.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out_dir}/z_mass_swap.{{png,pdf,json}}")


def figure(p, mass, levels, stats, centroids, evr, traces, pairs, H, settle, out: Path):
    norm = plt.Normalize(levels.min(), levels.max())
    cmap = plt.get_cmap(MASS_CMAP)
    scmap = time_cmap()

    fig = plt.figure(figsize=(17.0, 12.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.30, 1.0], hspace=0.30, wspace=0.22)
    ax_map = fig.add_subplot(gs[0, :])
    ax_kg = fig.add_subplot(gs[1, 0])
    ax_mat = fig.add_subplot(gs[1, 1])

    # ---- map ----
    ax_map.scatter(p[:, 0], p[:, 1], s=3, c="0.89", linewidths=0, rasterized=True, zorder=1)
    for i, lv in enumerate(levels):
        pts = p[np.isclose(mass, lv, atol=1e-3)]
        cov_ellipse(ax_map, pts, 2.0, facecolor=cmap(norm(lv)), alpha=0.18, edgecolor="none", zorder=2)
        cov_ellipse(ax_map, pts, 2.0, facecolor="none", edgecolor=cmap(norm(lv)), lw=1.5, zorder=3)
        ax_map.annotate(f"{lv:.2f}", (pts.mean(0)[0], pts.mean(0)[1] + (2.15 if i % 2 else -2.15)),
                        fontsize=12, fontweight="bold", ha="center", va="center",
                        color=cmap(norm(lv)), zorder=6,
                        bbox=dict(boxstyle="round,pad=0.18", fc="w", ec=cmap(norm(lv)),
                                  lw=1.3, alpha=0.95))

    lc = None
    for pi, (src, dst) in enumerate(pairs):
        q = traces[(src, dst)]["q"].mean(0)  # the mean swap, not one cherry-picked pair
        seg = np.stack([q[:-1], q[1:]], axis=1)
        lc = LineCollection(seg, cmap=scmap, norm=plt.Normalize(0, H), lw=4.0,
                            array=np.arange(len(seg)), zorder=8)
        ax_map.add_collection(lc)
        ax_map.scatter(q[:, 0], q[:, 1], s=30, c="w", edgecolor="k", lw=0.9, zorder=9)
        ax_map.scatter(*q[0], s=290, marker="X", color=cmap(norm(src)), edgecolor="k",
                       lw=1.8, zorder=11)
        ax_map.scatter(*q[H], s=330, marker="*", color=cmap(norm(dst)), edgecolor="k",
                       lw=1.4, zorder=11)
        # One label above the arc and one below: the two directions run over the same
        # ground, so a shared offset puts them on top of each other.
        mid = q[max(1, H // 3)]
        ax_map.annotate(f"{src:.2f} → {dst:.2f} kg", (mid[0], mid[1] + (1.15 if pi % 2 == 0 else -1.30)),
                        fontsize=15, fontweight="bold", ha="center", va="center", zorder=12,
                        bbox=dict(boxstyle="round,pad=0.24", fc="w", ec=cmap(norm(dst)),
                                  lw=2.0, alpha=0.96))
    ax_map.scatter([], [], s=170, marker="X", color="0.5", edgecolor="k", label="step 0 — window all donor")
    ax_map.scatter([], [], s=190, marker="*", color="0.5", edgecolor="k",
                   label=f"step {H} — window all recipient")
    ax_map.legend(loc="upper left", fontsize=13, framealpha=0.94)
    lo, hi = np.percentile(p, [0.2, 99.8], axis=0)
    pad = 0.09 * (hi - lo)
    ax_map.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax_map.set_ylim(lo[1] - pad[1], hi[1] + 2.0 * pad[1])
    ax_map.set_xlabel(f"PC1  ({evr[0] * 100:.1f}%)")
    ax_map.set_ylabel(f"PC2  ({evr[1] * 100:.1f}%)")
    ax_map.set_title("Payload swapped inside the window — the mean z path, step by step")
    fig.colorbar(lc, ax=ax_map, label="steps of the new payload in the window",
                 fraction=0.028, pad=0.01)

    # ---- readout vs step ----
    steps = np.arange(H + 1 + settle)
    for src, dst in pairs:
        q = traces[(src, dst)]["q"]
        kg = np.interp(q[:, :, 0], centroids, levels)
        med = np.median(kg, 0)
        ax_kg.fill_between(steps, np.percentile(kg, 25, axis=0), np.percentile(kg, 75, axis=0),
                           color=cmap(norm(dst)), alpha=0.22, lw=0)
        ax_kg.plot(steps, med, "-o", ms=4, color=cmap(norm(dst)), lw=2.6,
                   label=f"{src:.2f} → {dst:.2f} kg")
        ax_kg.axhline(dst, color=cmap(norm(dst)), ls="--", lw=1.6, alpha=0.9)
        ax_kg.axhline(src, color=cmap(norm(src)), ls=":", lw=1.6, alpha=0.9)
    ax_kg.axvline(H, color="0.3", ls=":", lw=1.8)
    ax_kg.annotate("window is pure\nfrom here on", (H, ax_kg.get_ylim()[1]),
                   xytext=(-8, -6), textcoords="offset points", fontsize=12,
                   ha="right", va="top", color="0.3")
    ax_kg.set_xlabel("steps of the new payload in the window")
    ax_kg.set_ylabel("mass read off PC1 (kg)")
    ax_kg.set_title("median readout, IQR shaded")
    ax_kg.legend(loc="center right", fontsize=12.5, framealpha=0.93)

    # ---- from x to matrix ----
    mat = np.full((len(levels), len(levels)), np.nan)
    for (src, dst), tr in traces.items():
        a = int(np.argmin(np.abs(levels - src)))
        b = int(np.argmin(np.abs(levels - dst)))
        mat[a, b] = np.median(tr["first"])
    im = ax_mat.imshow(mat, cmap="viridis_r", vmin=0, vmax=H, origin="lower")
    for a in range(len(levels)):
        for b in range(len(levels)):
            if a == b:  # the control: same mass, different episode
                ax_mat.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, facecolor="none",
                                               edgecolor="k", lw=2.0, zorder=3))
            ax_mat.text(b, a, f"{mat[a, b]:.0f}", ha="center", va="center", fontsize=12,
                        color="w" if mat[a, b] < H * 0.55 else "k", zorder=4)
    ax_mat.set_xticks(range(len(levels)))
    ax_mat.set_yticks(range(len(levels)))
    ax_mat.set_xticklabels([f"{v:.2f}" for v in levels], rotation=45, fontsize=11)
    ax_mat.set_yticklabels([f"{v:.2f}" for v in levels], fontsize=11)
    ax_mat.set_xlabel("swapped TO (kg)")
    ax_mat.set_ylabel("swapped FROM (kg)")
    ax_mat.set_title(f"median steps to enter the target 2σ region  (of {H})\n"
                     "boxed diagonal = same-mass control", fontsize=15)
    fig.colorbar(im, ax=ax_mat, label="steps", fraction=0.046, pad=0.02)

    fig.suptitle("Mid-window payload swap — z leaves the old mass region and enters the new one",
                 fontsize=20, fontweight="bold")
    fig.subplots_adjust(top=0.925, bottom=0.065, left=0.058, right=0.965)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
