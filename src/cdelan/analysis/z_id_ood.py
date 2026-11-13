"""Three trained payload masses and one unseen, heavier one, in the context latent.

One figure, one message: the ID masses form distinct regions, the online z converges into
its own region after the payload changes, and an unseen heavier payload is also carried to
a sensible place -- past the heaviest trained region along the mass axis.

  top     PC1-PC2 map. The three ID levels' phase-3 clouds and 2-sigma regions, the median
          live z path of each (encoder re-run every step from the lift), and the mean path of
          a mid-window swap from an ID donor to the OOD payload (as in `z_mass_swap`).
  bottom  PC1 against time since the payload changed, mean +/- std over many episodes
          (ID: t = 0 at the lift) or many swaps (OOD: t = 0 at the swap).

OOD source. The lattice stops at 2.0 kg. The heavier windows come from
``lift_randmass_phase3_9590ep`` (0.1-3.5 kg, same arm, joint damping 4.0), which the
checkpoint never saw. A different recording can shift z for reasons other than mass, so the
script first encodes that recording's ID-range episodes and reports where they read out;
the OOD result is only as good as that check.

Run::

    source env.sh && python3 -m cdelan.analysis.z_id_ood --ckpt checkpoints/ours_s0.pt \
        --contract data/contracts/official_train.json --path-contract data/contracts/official_carry.json \\
        --ood-dataset official_ood_2to4kg --ood-mass 3.0 --out <out>
    (as called by 3_latent/run.sh)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")

from . import _plotstyle

_plotstyle.apply(0.82)

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from cdelan import paths
from ._common import build_dataset, load_model
from .z_mass_swap import splice_traces
from .z_pca_mass import (
    DEFAULT_CKPT,
    DEFAULT_CONTRACT,
    cov_ellipse,
    encode,
    episode_meta,
    episode_track,
    pca_fit,
    spearman,
)

OOD_DATASET = "lift_randmass_phase3_9590ep"
#: ID = one hue, light -> dark with mass. OOD = a different hue entirely.
ID_COLORS = ["#6BAED6", "#2171B5", "#08306B"]
OOD_COLOR = "#E6550D"
HALO = [pe.Stroke(linewidth=5.2, foreground="w"), pe.Normal()]


def randmass_meta(path: str) -> dict[str, dict]:
    with h5py.File(path, "r") as fh:
        d = fh["data"]
        return {n: {"mass": float(np.asarray(d[n]["episode_physics_randomization/object_mass"]).reshape(-1)[0]),
                    "n_phase3": int((np.asarray(d[n]["obs/phase"]) == 3).sum())}
                for n in d.keys()}


def fan_apex(p: np.ndarray, mass: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Least-squares meeting point of every level's long axis -- the apex the regions fan out from."""
    a, b = np.zeros((2, 2)), np.zeros(2)
    for lv in levels:
        pts = p[np.isclose(mass, lv, atol=1e-3)]
        vals, vecs = np.linalg.eigh(np.cov(pts.T))
        u = vecs[:, np.argmax(vals)]
        perp = np.eye(2) - np.outer(u, u)
        a += perp
        b += perp @ pts.mean(0)
    return np.linalg.solve(a, b)


def on_grid(tracks: list[dict], key: str, dt: float, t_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-episode live tracks on one lift-aligned clock, NaN where an episode has no frame."""
    t = np.arange(0.0, t_max + 0.5 * dt, dt)
    out = np.full((len(tracks), len(t)), np.nan) if key == "p1" else np.full((len(tracks), len(t), 2), np.nan)
    for i, tr in enumerate(tracks):
        k = np.round(tr["t_rel"] / dt).astype(int)
        ok = (k >= 0) & (k < len(t))
        out[i, k[ok]] = tr["p"][ok, 0] if key == "p1" else tr["p"][ok]
    return t, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--contract", default=DEFAULT_CONTRACT)
    ap.add_argument("--id-mass", type=float, nargs="+", default=[0.1, 1.186, 2.0])
    ap.add_argument("--ood-dataset", default=OOD_DATASET,
                    help="dataset id of the unseen-payload recording (e.g. ppo_lat8orig_meantau_ood3to4_nodrop "
                         "for the standard-set checkpoints)")
    ap.add_argument("--ood-mass", type=float, default=3.0)
    ap.add_argument("--ood-tol", type=float, default=0.1)
    ap.add_argument("--donor-mass", type=float, default=0.1, help="ID level the OOD swap starts from")
    ap.add_argument("--episodes-per-mass", type=int, default=70, help="val episodes per level for the cloud")
    ap.add_argument("--track-episodes", type=int, default=50, help="live-tracked episodes per ID level")
    ap.add_argument("--ood-episodes", type=int, default=150)
    ap.add_argument("--pairs", type=int, default=120)
    ap.add_argument("--settle", type=int, default=24, help="steps the OOD window keeps sliding once pure")
    ap.add_argument("--reloc-mass", type=float, default=2.0, help="trained mass the ID relocation swaps into")
    ap.add_argument("--t-max", type=float, default=0.72)
    ap.add_argument("--id-skip", type=int, default=10, help="ID steps after the lift left off the map (panel a)")
    ap.add_argument("--example", default="",
                    help="window from z_ood_predict's json to mark (illustrative|typical|accurate; '' = none)")
    ap.add_argument("--example-json", default=None, help="defaults to plot/latent/z_ood_predict.json")
    ap.add_argument("--path-contract", default=None,
                    help="contract whose val episodes give the donors, live tracks and relocation targets "
                         "(the ID test set; default: --contract). The cloud and its PCA frame stay --contract's.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or (paths.REPO_ROOT.parent / "plot" / "latent" / "z_id_ood"))
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    model, cfg = load_model(args.ckpt)
    H, dt = int(cfg["history_len"]), float(cfg["dt"])
    contract = json.loads(Path(args.contract).read_text())
    lat_path = str(paths.dataset_file(contract["dataset_id"]))
    ood_path = str(paths.dataset_file(args.ood_dataset))
    meta = episode_meta(lat_path, contract["splits"]["val"])
    levels = np.array(sorted({round(m["mass"], 3) for m in meta.values()}))
    id_lv = [float(levels[np.argmin(np.abs(levels - m))]) for m in args.id_mass]
    # this script's own figure: the three-colour ramp for three levels, a sampled one for any other count
    id_cols = ID_COLORS if len(id_lv) == len(ID_COLORS) else [
        matplotlib.colors.to_hex(c) for c in plt.get_cmap("Blues")(np.linspace(0.45, 0.95, len(id_lv)))]
    donor_lv = float(levels[np.argmin(np.abs(levels - args.donor_mass))])

    # ---- the cloud and its PCA frame: all eight trained levels, as in z_pca_mass ----
    chosen = []
    for lv in levels:
        pool = sorted(n for n, m in meta.items() if round(m["mass"], 3) == lv)
        chosen += [pool[i] for i in rng.choice(len(pool), size=min(args.episodes_per_mass, len(pool)), replace=False)]
    ds = build_dataset([(lat_path, n) for n in chosen], cfg)
    z = encode(model, ds.history_states, ds.history_torques)
    mass = ds.object_context[:, 0].astype(float)
    mean, comps, evr = pca_fit(z)
    if spearman(mass, (z - mean) @ comps[0]) < 0:
        comps = comps * np.array([[-1.0], [1.0]])
    p = (z - mean) @ comps.T
    # PC2's sign is arbitrary; pin it so the arc opens upward in every figure drawn from this frame.
    if np.corrcoef(np.abs(p[:, 0] - np.median(p[:, 0])), p[:, 1])[0, 1] < 0:
        comps = comps * np.array([[1.0], [-1.0]])
        p[:, 1] *= -1

    def proj(zz):
        return (zz - mean) @ comps.T

    mu = {lv: p[np.isclose(mass, lv, atol=1e-3)].mean(0) for lv in levels}
    sd1 = {lv: p[np.isclose(mass, lv, atol=1e-3), 0].std() for lv in levels}
    cen = np.array([mu[lv][0] for lv in levels])
    slope, icpt = np.polyfit(levels, cen, 1)  # linear PC1-vs-mass, for reading past 2.0 kg

    def readout(p1):
        return (np.asarray(p1) - icpt) / slope

    print(f"cloud  {len(p)} windows, PCA {np.round(evr, 3).tolist()}")
    if args.path_contract:
        pc = json.loads(Path(args.path_contract).read_text())
        id_path = str(paths.dataset_file(pc["dataset_id"]))
        id_meta = episode_meta(id_path, pc["splits"]["val"])
        print(f"paths  donors / tracks / relocation from {pc['dataset_id']} val ({len(id_meta)} eps)")
    else:
        id_path, id_meta = lat_path, meta

    # ---- the OOD recording: calibrate on its ID-range episodes first ----
    rmeta = randmass_meta(ood_path)
    calib = {}
    # calibrated on a FIXED three levels, whatever --id-mass asks for: the draws here feed the OOD
    # episode draw below, so tying them to --id-mass would change the OOD sample with it
    calib_lv = [float(levels[np.argmin(np.abs(levels - m))]) for m in (0.1, 1.186, 2.0)]
    for lv in calib_lv:
        pool = [n for n, m in rmeta.items() if abs(m["mass"] - lv) <= 0.05 and m["n_phase3"] >= 30]
        if not pool:  # an OOD-only recording (e.g. 3-4 kg) has no ID-range episodes to calibrate on
            print(f"calib  no ~{lv:.2f} kg episodes in {args.ood_dataset}; skipped")
            continue
        pool = [pool[i] for i in rng.choice(len(pool), size=min(40, len(pool)), replace=False)]
        dsc = build_dataset([(ood_path, n) for n in pool], cfg)
        pc = proj(encode(model, dsc.history_states, dsc.history_torques))
        calib[f"{lv:.3f}"] = {"n_episodes": len(pool), "n_windows": len(pc),
                              "pc1_median": float(np.median(pc[:, 0])), "lattice_pc1_centroid": float(mu[lv][0]),
                              "readout_kg_median": float(np.median(readout(pc[:, 0])))}
        print(f"calib  randmass ~{lv:.2f} kg -> PC1 {np.median(pc[:, 0]):+.2f} (lattice {mu[lv][0]:+.2f}), "
              f"reads {np.median(readout(pc[:, 0])):.2f} kg")

    pool = [n for n, m in rmeta.items() if abs(m["mass"] - args.ood_mass) <= args.ood_tol and m["n_phase3"] >= 30]
    ood_names = [pool[i] for i in rng.choice(len(pool), size=min(args.ood_episodes, len(pool)), replace=False)]

    # ---- OOD swap: lattice donor windows spliced into the unseen payload's windows ----
    donor_pool = sorted(n for n, m in id_meta.items() if round(m["mass"], 3) == donor_lv)
    donor_names = [donor_pool[i] for i in rng.choice(len(donor_pool), size=min(45, len(donor_pool)), replace=False)]
    dss = build_dataset([(id_path, n) for n in donor_names] + [(ood_path, n) for n in ood_names], cfg)
    ms = dss.object_context[:, 0].astype(float)
    src = np.where(np.isclose(ms, donor_lv, atol=1e-3))[0]
    dst = np.where(np.abs(ms - args.ood_mass) <= args.ood_tol)[0]
    chrono = {e: rows for e, rows in enumerate(dss.episode_window_indices)}
    zt, seam = splice_traces(model, dss, None, src, dst, chrono, H, args.settle, args.pairs, rng)
    q_ood = proj(zt.reshape(-1, zt.shape[-1])).reshape(zt.shape[0], zt.shape[1], 2)
    p_ood_cloud = proj(encode(model, dss.history_states[dst], dss.history_torques[dst]))
    t_ood = np.arange(q_ood.shape[1]) * dt
    print(f"ood    {len(ood_names)} eps ~{args.ood_mass} kg, {len(dst)} windows; {len(q_ood)} swaps from {donor_lv} kg")

    # ---- ID relocation: the same 0.10 kg donors spliced into windows of a TRAINED mass (same
    # recording), so the map can show where a wrong z goes when the observed payload is in range.
    # Its own rng, so adding it leaves every draw above and below unchanged. ----
    rng_r = np.random.default_rng(7)
    reloc_lv = float(levels[np.argmin(np.abs(levels - args.reloc_mass))])
    tgt_pool = sorted(n for n, m in id_meta.items() if round(m["mass"], 3) == reloc_lv)
    tgt_names = [tgt_pool[i] for i in rng_r.choice(len(tgt_pool), size=min(45, len(tgt_pool)), replace=False)]
    dsr = build_dataset([(id_path, n) for n in donor_names] + [(id_path, n) for n in tgt_names], cfg)
    mr = dsr.object_context[:, 0].astype(float)
    src_r = np.where(np.isclose(mr, donor_lv, atol=1e-3))[0]
    dst_r = np.where(np.isclose(mr, reloc_lv, atol=1e-3))[0]
    chrono_r = {e: rows for e, rows in enumerate(dsr.episode_window_indices)}
    zr, _ = splice_traces(model, dsr, None, src_r, dst_r, chrono_r, H, args.settle, args.pairs, rng_r)
    q_reloc = proj(zr.reshape(-1, zr.shape[-1])).reshape(zr.shape[0], zr.shape[1], 2)
    print(f"reloc  {len(q_reloc)} swaps {donor_lv} -> {reloc_lv} kg (trained), same donors")

    # ---- one window from `z_ood_predict` (the cube figure's), rebuilt: the stale z and the
    # adapted z its rollouts ran with, and the path z takes when THAT window is swapped in from a
    # light donor -- the path's last step is the adapted z by construction, and is checked to be ----
    ex = None
    if args.example:
        exj = json.loads(Path(args.example_json or out.with_name("z_ood_predict.json")).read_text())
        e = exj["example_windows"][args.example]
        dse = build_dataset([(e["file"], e["episode"])], cfg)
        row = next(i for i, m in enumerate(dse.sample_meta) if int(m[2]) == e["window_start"])
        rec_s, rec_t = np.asarray(dse.history_states[row]), np.asarray(dse.history_torques[row])
        f_src = np.asarray(dss.history_states[src])[:, -1, :18]
        scale = np.maximum(np.concatenate([f_src, rec_s[None, 0, :18]]).std(0), 1e-6)
        d = src[np.argmin(np.linalg.norm((f_src - rec_s[0, :18]) / scale, axis=1))]
        st = np.stack([np.concatenate([dss.history_states[d][s:], rec_s[:s]]) for s in range(H + 1)])
        tq = np.stack([np.concatenate([dss.history_torques[d][s:], rec_t[:s]]) for s in range(H + 1)])
        z_path = encode(model, st, tq)
        assert np.allclose(z_path[-1], e["z_adapted"], atol=1e-3), "rebuilt window is not the example's"
        ex = {"path": proj(z_path), "stale": proj(np.asarray(exj["z_stale"])[None])[0],
              "err": e["h8_err_cm"], "episode": e["episode"], "window_start": e["window_start"]}
        print(f"example {args.example}: {e['episode']} @ {e['window_start']}, adapted z at PC "
              f"{np.round(ex['path'][-1], 2).tolist()}, stale z at PC {np.round(ex['stale'], 2).tolist()}")

    # ---- ID live tracks, many episodes per level ----
    tracks = {}
    for lv in id_lv:
        pool = sorted(n for n, m in id_meta.items() if round(m["mass"], 3) == lv and m["n_phase3"] >= 70)
        names = [pool[i] for i in rng.choice(len(pool), size=min(args.track_episodes, len(pool)), replace=False)]
        trs = []
        for n in names:
            tr = episode_track(model, cfg, id_path, n)
            tr["p"] = proj(tr["z"])
            trs.append(tr)
        tracks[lv] = trs
        print(f"track  {lv:.2f} kg: {len(trs)} episodes")

    # ---- OOD region, drawn to continue the fan of the trained regions ----
    # Every trained level's 2-sigma ellipse points its long axis at a common apex (z's norm is
    # free under the cosine loss, so a level spreads radially). The OOD region takes the
    # direction from that apex to the OOD centre, the OOD windows' own spread along it, and the
    # heaviest trained level's aspect ratio. Its empirical 2-sigma is rounder than this: the
    # drawn region is schematic, the points under it are not.
    # Centred where the swapped z settles (the star), not on the OOD windows' mean: that mean
    # sits high, pulled up by a recording whose windows spread wider in PC2 than the lattice's.
    ood_c = q_ood[:, -1].mean(0)
    try:
        apex = fan_apex(p, mass, levels)
        u = (ood_c - apex) / np.linalg.norm(ood_c - apex)
    except np.linalg.LinAlgError:
        # A latent whose levels do not fan out (near-parallel long axes) has no apex; the
        # heaviest trained level's own long axis is then the continuation.
        apex = np.full(2, np.nan)
        vals, vecs = np.linalg.eigh(np.cov(p[np.isclose(mass, levels[-1], atol=1e-3)].T))
        u = vecs[:, np.argmax(vals)]
        print("fan    no apex (levels do not fan out); OOD region follows the heaviest level's axis")
    ref_vals = np.sort(np.linalg.eigvalsh(np.cov(p[np.isclose(mass, levels[-1], atol=1e-3)].T)))[::-1]
    ood_w = 2 * 2.0 * float(np.sqrt(u @ np.cov(p_ood_cloud.T) @ u))
    ood_h = ood_w * float(np.sqrt(ref_vals[1] / ref_vals[0]))
    ood_ang = float(np.degrees(np.arctan2(u[1], u[0])))
    print(f"fan    apex {np.round(apex, 2).tolist()}, OOD region axis {ood_ang:.0f} deg")

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(12.5, 11.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.26)
    ax, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # Every trained window in grey (all eight levels: the whole ID support). The unseen payload
    # has no training windows, so it gets no points -- only its region and the swap path.
    ax.scatter(p[:, 0], p[:, 1], s=4, color="0.55", alpha=0.35, linewidths=0, rasterized=True, zorder=1)
    for lv, col in zip(id_lv, id_cols):
        pts = p[np.isclose(mass, lv, atol=1e-3)]
        cov_ellipse(ax, pts, 2.0, facecolor=col, alpha=0.16, edgecolor="none", zorder=2)
        cov_ellipse(ax, pts, 2.0, facecolor="none", edgecolor=col, lw=2.2, zorder=3)
    ax.add_patch(Ellipse(ood_c, ood_w, ood_h, angle=ood_ang, facecolor=OOD_COLOR, alpha=0.10,
                         edgecolor="none", zorder=2))
    ax.add_patch(Ellipse(ood_c, ood_w, ood_h, angle=ood_ang, facecolor="none", edgecolor=OOD_COLOR,
                         lw=2.2, ls="--", zorder=3))

    stats_id = {}
    for lv, col in zip(id_lv, id_cols):
        t, P = on_grid(tracks[lv], "p", dt, args.t_max)
        # The first `id_skip` steps after the lift are dropped: the window is still mostly
        # pre-lift frames there, input the model never trained on.
        path = np.nanmedian(P, axis=0)[args.id_skip:]
        ax.plot(path[:, 0], path[:, 1], "-", color=col, lw=2.6, zorder=8, path_effects=HALO)
        ax.scatter(path[:, 0], path[:, 1], s=20, color=col, edgecolor="w", lw=0.6, zorder=9)
        ax.scatter(*path[0], s=95, marker="o", facecolor="w", edgecolor=col, lw=2.2, zorder=10)
        ax.scatter(*path[-1], s=140, marker="o", color=col, edgecolor="w", lw=1.4, zorder=10)
        ax.annotate(f"{lv:.2f} kg", mu[lv], xytext=(0, -62), textcoords="offset points", ha="center",
                    fontsize=14, fontweight="bold", color="#222", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.22", fc="w", ec=col, lw=2.0, alpha=0.95))
        # convergence: first time PC1 enters its own +/-2 sigma band and stays for 0.2 s
        _, P1 = on_grid(tracks[lv], "p1", dt, args.t_max)
        inside = np.abs(P1 - mu[lv][0]) <= 2 * sd1[lv]
        w = int(round(0.2 / dt))
        conv = []
        for row in inside:
            ok = [k for k in range(len(row) - w) if row[k : k + w].all()]
            conv.append(t[ok[0]] if ok else np.nan)
        stats_id[f"{lv:.3f}"] = {"n_episodes": len(tracks[lv]),
                                "t_converge_s_median": float(np.nanmedian(conv)),
                                "t_converge_s_iqr": np.nanpercentile(conv, [25, 75]).tolist(),
                                "never_converged": int(np.isnan(conv).sum())}

    path = q_ood.mean(0)
    ax.plot(path[:, 0], path[:, 1], "-", color=OOD_COLOR, lw=2.8, zorder=8, path_effects=HALO)
    ax.scatter(path[:, 0], path[:, 1], s=46, color=OOD_COLOR, edgecolor="w", lw=0.9, zorder=9)
    ax.scatter(*path[0], s=150, marker="X", color="w", edgecolor=OOD_COLOR, lw=2.0, zorder=10)
    ax.scatter(*path[-1], s=260, marker="*", color=OOD_COLOR, edgecolor="w", lw=1.2, zorder=10)
    ax.annotate(f"{args.ood_mass:.1f} kg (unseen)", path[-1], xytext=(16, -26), textcoords="offset points",
                ha="left", fontsize=14, fontweight="bold", color="#222", zorder=10,
                bbox=dict(boxstyle="round,pad=0.22", fc="w", ec=OOD_COLOR, lw=2.0, alpha=0.95))

    if ex is not None:
        # Only the two z's on the map: one window's own swap path is a jagged single trace next
        # to the 120-swap mean, and reads as noise there. Its PC1 over time is in panel (b).
        pth = ex["path"]
        ax.scatter(*ex["stale"], s=190, marker="D", color=ID_COLORS[0], edgecolor="k", lw=1.5, zorder=12)
        ax.scatter(*pth[-1], s=210, marker="D", color=OOD_COLOR, edgecolor="k", lw=1.5, zorder=12)
        ax.annotate("adapted z", pth[-1], xytext=(12, 8), textcoords="offset points", fontsize=12.5,
                    fontweight="bold", color="#222", zorder=13)
        ax.annotate("stale z", ex["stale"], xytext=(-12, 12), textcoords="offset points", ha="right",
                    fontsize=12.5, fontweight="bold", color="#222", zorder=13)

    handles = [plt.Line2D([], [], color=c, lw=3.0, label=f"{lv:.2f} kg  (ID)") for lv, c in zip(id_lv, id_cols)]
    handles += [plt.Line2D([], [], color=OOD_COLOR, lw=3.0, label=f"{args.ood_mass:.1f} kg  (OOD, unseen)"),
                plt.Line2D([], [], color="0.55", marker="o", ls="none", ms=6, label="trained windows (all 8 levels)"),
                plt.Line2D([], [], color=OOD_COLOR, lw=2.0, ls="--", label="OOD region")]
    if ex is not None:
        handles += [plt.Line2D([], [], color=ID_COLORS[0], marker="D", mec="k", ls="none", ms=9,
                               label="stale z  (cube example)"),
                    plt.Line2D([], [], color=OOD_COLOR, marker="D", mec="k", ls="--", ms=9,
                               label="adapted z  (cube example)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=12,
              frameon=False, handlelength=1.8, columnspacing=1.2)
    allp = np.concatenate([p, p_ood_cloud])
    lo, hi = np.percentile(allp, [0.4, 99.6], axis=0)
    if ex is not None:  # the example's z is one window, so the percentiles cannot be trusted to hold it
        exp = np.vstack([ex["path"], ex["stale"][None]])
        lo, hi = np.minimum(lo, exp.min(0) - 0.3), np.maximum(hi, exp.max(0) + 0.3)
    pad = 0.08 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + 1.2 * pad[1])
    ax.set_xlabel(f"PC1  ({evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2  ({evr[1] * 100:.1f}%)")
    ax.set_title("(a)", loc="left", fontsize=15, pad=34)

    # ---- bottom: PC1 vs time ----
    for lv, col in zip(id_lv, id_cols):
        axb.axhspan(mu[lv][0] - 2 * sd1[lv], mu[lv][0] + 2 * sd1[lv], color=col, alpha=0.07, lw=0, zorder=0)
        axb.axhline(mu[lv][0], color=col, ls="--", lw=1.2, alpha=0.9, zorder=1)
        t, P1 = on_grid(tracks[lv], "p1", dt, args.t_max)
        m, s = np.nanmean(P1, 0), np.nanstd(P1, 0)
        axb.fill_between(t, m - s, m + s, color=col, alpha=0.22, lw=0, zorder=2)
        axb.plot(t, m, "-", color=col, lw=2.4, zorder=4, label=f"{lv:.2f} kg")
    ood_mu = float(np.mean(p_ood_cloud[:, 0]))
    axb.axhline(ood_mu, color=OOD_COLOR, ls="--", lw=1.2, alpha=0.9, zorder=1)
    m, s = q_ood[:, :, 0].mean(0), q_ood[:, :, 0].std(0)
    axb.fill_between(t_ood, m - s, m + s, color=OOD_COLOR, alpha=0.22, lw=0, zorder=2)
    axb.plot(t_ood, m, "-", color=OOD_COLOR, lw=2.6, zorder=5,
             label=f"{args.ood_mass:.1f} kg")
    axb.axvline(H * dt, color="0.35", ls=":", lw=1.5, zorder=3)
    axb.annotate(f"window fully refreshed ({H} steps)", (H * dt, 1.0), xycoords=("data", "axes fraction"),
                 xytext=(6, -6), textcoords="offset points", fontsize=11.5, color="0.35", va="top")
    ylo = min(mu[id_lv[0]][0] - 2.6 * sd1[id_lv[0]], np.percentile(p_ood_cloud[:, 0], 1))
    yhi = max(mu[id_lv[-1]][0] + 2.6 * sd1[id_lv[-1]], np.percentile(p_ood_cloud[:, 0], 99))
    if ex is not None:
        tt = np.arange(H + 1) * dt
        axb.plot(tt, ex["path"][:, 0], "--", color=OOD_COLOR, lw=1.8, zorder=6)
        axb.scatter(tt[-1], ex["path"][-1, 0], s=130, marker="D", color=OOD_COLOR, edgecolor="k", lw=1.3, zorder=8)
        axb.scatter(0.0, ex["stale"][0], s=120, marker="D", color=ID_COLORS[0], edgecolor="k", lw=1.3,
                    zorder=8, clip_on=False)
        ylo, yhi = min(ylo, ex["path"][:, 0].min() - 0.5), max(yhi, ex["path"][:, 0].max() + 0.5)
    axb.set_ylim(ylo, yhi)
    axb.set_xlim(0, args.t_max)
    axb.set_xlabel("time since payload change (s)      [ID: t = 0 at lift,  OOD: t = 0 at swap]")
    axb.set_ylabel("PC1 of the online z")
    axb.set_title("(b)", loc="left", fontsize=15)
    axb.annotate(f"mean ± std   ·   ID: {len(tracks[id_lv[0]])} episodes / mass   ·   OOD: {len(q_ood)} swaps "
                 f"from {donor_lv:.2f} kg   ·   dashed = region centre",
                 (1.0, 1.01), xycoords="axes fraction", ha="right", va="bottom", fontsize=11.5, color="0.3")

    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.08, right=0.97)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    top = levels[-1]
    summary = {
        "checkpoint": args.ckpt, "pca_evr": evr.tolist(), "id_levels": id_lv, "donor_level": donor_lv,
        "ood": {"dataset": args.ood_dataset, "mass_kg": args.ood_mass, "tol_kg": args.ood_tol,
                "n_episodes": len(ood_names), "n_windows": int(len(dst)), "n_swaps": int(len(q_ood)),
                "pc1_mean": ood_mu, "pc1_std": float(p_ood_cloud[:, 0].std()),
                "pc1_beyond_2kg_centroid_in_2kg_sigma": float((ood_mu - mu[top][0]) / sd1[top]),
                "frac_windows_right_of_2kg_centroid": float(np.mean(p_ood_cloud[:, 0] > mu[top][0])),
                "linear_readout_kg_median": float(np.median(readout(p_ood_cloud[:, 0]))),
                "swap_final_pc1_mean": float(q_ood[:, -1, 0].mean()),
                "swap_seam_median": float(np.median(seam))},
        "lattice_pc1_centroids": dict(zip([f"{v:.3f}" for v in levels], cen.tolist())),
        "calibration_randmass_id_range": calib,
        "id_tracks": stats_id,
        "example": None if ex is None else {
            "which": args.example, "episode": ex["episode"], "window_start": ex["window_start"],
            "adapted_z_pc": ex["path"][-1].tolist(), "stale_z_pc": ex["stale"].tolist(),
            "adapted_z_pc1_in_2kg_sigma": float((ex["path"][-1, 0] - mu[top][0]) / sd1[top]),
            "h8_err_cm": ex["err"]},
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))

    # What the official figure draws, so it can be drawn at print size without re-encoding.
    sub = np.random.default_rng(1).choice(len(p), size=min(20000, len(p)), replace=False)
    t_grid = on_grid(tracks[id_lv[0]], "p1", dt, args.t_max)[0]
    P = [on_grid(tracks[lv], "p", dt, args.t_max)[1] for lv in id_lv]
    P1 = [on_grid(tracks[lv], "p1", dt, args.t_max)[1] for lv in id_lv]
    np.savez_compressed(
        out.with_name(out.name + "_data.npz"),
        evr=evr, H=H, dt=dt, id_skip=args.id_skip, id_levels=np.array(id_lv), ood_mass=args.ood_mass,
        donor_level=donor_lv, cloud=p[sub].astype(np.float32),
        id_mu=np.array([p[np.isclose(mass, lv, atol=1e-3)].mean(0) for lv in id_lv]),
        id_cov=np.array([np.cov(p[np.isclose(mass, lv, atol=1e-3)].T) for lv in id_lv]),
        id_sd1=np.array([sd1[lv] for lv in id_lv]),
        id_path=np.array([np.nanmedian(x, 0) for x in P]),
        t=t_grid, id_mean=np.array([np.nanmean(x, 0) for x in P1]), id_std=np.array([np.nanstd(x, 0) for x in P1]),
        ood_path=q_ood.mean(0), t_ood=t_ood, ood_mean=q_ood[:, :, 0].mean(0), ood_std=q_ood[:, :, 0].std(0),
        ood_pc1_centre=float(np.mean(p_ood_cloud[:, 0])),
        # the per-episode / per-swap PC tracks themselves, so a figure can re-project them onto any
        # direction in the PC1-PC2 plane (the mass-aligned coordinate) and take its own mean/std
        id_tracks=np.array(P, dtype=np.float32), ood_traces=q_ood.astype(np.float32),
        ood_cloud_mean=p_ood_cloud.mean(0), ood_cloud_cov=np.cov(p_ood_cloud.T),
        ood_points=p_ood_cloud.astype(np.float32),                 # every OOD window's z, for raw scatter
        reloc_traces=q_reloc.astype(np.float32), reloc_level=reloc_lv,
        # each ID level's own points, for density contours drawn from the data rather than a Gaussian
        **{f"id_points_{i}": p[np.isclose(mass, lv, atol=1e-3)].astype(np.float32) for i, lv in enumerate(id_lv)},
        ood_region=np.array([*ood_c, ood_w, ood_h, ood_ang]),
    )
    print(json.dumps(summary, indent=2))
    print(f"saved -> {out}.{{png,pdf,json}}")


if __name__ == "__main__":
    main()
