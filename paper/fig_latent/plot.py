"""The paper's latent figure (four panels), drawn from the intermediates run.sh writes to <work>.

  (a) PCA of PICR's context z (official_train val + official_ood_2to4kg windows), coloured by payload mass, with the
      mass direction and the two z's of panel (b)
  (b) one 3 kg window rolled out 12 steps open loop with z_0.1kg (stale) and with the window's own z_3kg (adapted),
      on the Isaac Sim replay of the arm
  (c) carry: PC1 of z against time from the lift; every payload starts at the same (pre-lift) reading and settles on
      the PC1 level of its own payload
  (d) payload swept 0.1 -> 4 kg and 4 -> 0.1 kg during a figure-eight: PC1 follows it in real time

    python3 plot.py --work <work> --frames assets/frames --out figures/fig_latent
-> <out>.png, <out>.pdf, <work>/fig_numbers.json ((c)'s centroid distance and mass read-out error, pre-lift vs carry)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.legend_handler import HandlerTuple

import arm_frames as A

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--frames", required=True)
ap.add_argument("--out", required=True, help="output path without extension")
args = ap.parse_args()
W, FRAMES, OUT = Path(args.work), Path(args.frames), Path(args.out)

# ---------------------------------------------------------------- style
NORM = Normalize(0.1, 4.0)
_g = np.linspace(0.1, 4.0, 512)                           # trained 0.1-2 kg: viridis to green; unseen 2-4 kg: orange ramp
CMAP = ListedColormap([plt.cm.viridis(0.82 * min(m - 0.1, 1.9) / 1.9) if m <= 2.01 else plt.cm.Oranges(0.35 + 0.6 * (m - 2.0) / 2.0)
                       for m in _g])
cm = lambda m: CMAP(NORM(m))
INK, INK2, INK3 = "#1a1a1a", "#555555", "#9a9a9a"
C_STALE, C_ADAPT = A.TINT["stale"], A.TINT["ours"]
FS = 1.25                                                 # scale of the bottom row's hard-coded sizes
WS = [pe.withStroke(linewidth=2.0, foreground="white")]
FONT_DIR = Path("/usr/share/fonts/truetype/msttcorefonts")   # Arial if installed (else DejaVu Sans)
for f in ("Arial.ttf", "Arial_Bold.ttf", "Arial_Italic.ttf"):
    if (FONT_DIR / f).exists(): font_manager.fontManager.addfont(str(FONT_DIR / f))
plt.rcParams.update({"font.family": "Arial" if (FONT_DIR / "Arial.ttf").exists() else "DejaVu Sans", "mathtext.fontset": "custom",
                     "mathtext.it": "Arial:italic", "mathtext.rm": "Arial", "font.size": 11.2, "axes.labelsize": 11.2, "axes.titlesize": 11.2,
                     "xtick.labelsize": 10.6, "ytick.labelsize": 10.6, "legend.fontsize": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
                     "xtick.minor.width": 0.4, "xtick.minor.size": 1.2,
                     "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2.0,
                     "pdf.fonttype": 42, "savefig.dpi": 300})

# ---------------------------------------------------------------- inputs
z4 = np.load(W / "z" / "pca.npz", allow_pickle=True)         # PCA frame (pca_frame.py)
win = np.load(W / "arm_window_h12.npz", allow_pickle=True)   # panel (b) window (arm_window.py)
HIST, DT = 12, 0.02

# ---------------------------------------------------------------- layout (inches)
SX = 1.30                                                 # whole-figure zoom of the 7.0 in column layout
FW_ROW, FH_ROW = 7.0, 2.62                                # the bottom row's own canvas
ROWH = FH_ROW * SX * 0.785                                # bottom row height (tight gap)
FW, FH = 7.0 * SX, 2.68 * SX + ROWH
fig = plt.figure(figsize=(FW, FH))


def box(x, y, w, h):                                      # top-row inches -> figure fraction
    return fig.add_axes([x * SX / FW, (y * SX + ROWH) / FH, w * SX / FW, h * SX / FH])


# ================================================================ top row: (a) PCA, (b) arm panels
def proj_top(z):
    return (np.atleast_2d(z) - z4["zmean"]) @ z4["V"].T


AB_Y = 0.68                                               # bottom of every panel (room for axis labels / legend)
H12 = int(win["horizon"])
rows = ["stale", "ours"]
panels = [A.build(FRAMES, r, H12) for r in rows]
cover = np.maximum.reduce([c for _, c in panels])
project, _ = A.camera(FRAMES)
track = {k: project(np.vstack([win["anchor_obj"][None, :3], win[f"{k}_obj"][:, :3]])) for k in rows + ["gt"]}
pts = np.vstack(list(track.values()))
ys, xs = np.where(cover > 0.3)
pad = 28
y0, y1 = int(max(min(ys.min(), pts[:, 1].min()) - pad, 0)), int(min(max(ys.max(), pts[:, 1].max()) + pad, cover.shape[0]))
x0, x1 = int(max(min(xs.min(), pts[:, 0].min()) - pad, 0)), int(min(max(xs.max(), pts[:, 0].max()) + pad, cover.shape[1]))
CW = 1.81                                                 # the same crop and size for both panels
CH = CW * (y1 - y0) / (x1 - x0)
CY = AB_Y
CX = 3.31
ax_c1, ax_c2 = box(CX, CY, CW, CH), box(CX + CW + 0.06, CY, CW, CH)
ax_a, cax = box(0.52, CY, 1.95, CH), box(2.63, CY, 0.08, CH)   # (a) as tall as the arm panels

# ---- (a) PCA
Pc = np.vstack([z4["P"][:, :2], z4["P_ood"][:, :2]])      # every window of both sets, in random order
mcl = np.concatenate([z4["mass"], z4["mass_ood"]])
o = np.random.default_rng(0).permutation(len(Pc))
ax_a.scatter(*Pc[o].T, c=mcl[o], cmap=CMAP, norm=NORM, s=0.5, alpha=0.22, lw=0, rasterized=True)
lo, hi = np.quantile(np.vstack([z4["P"][:, :2], z4["P_ood"][:, :2]]), [0.005, 0.995], axis=0)
ax_a.set_xlim(lo[0] - 0.3, hi[0] + 0.3); ax_a.set_ylim(lo[1] - 0.3, hi[1] + 0.95)
ax_a.set_xlabel("PC1"); ax_a.set_ylabel("PC2")
# fixed ticks when the panel spans are within x1.5 of the reference spans and every tick falls inside; otherwise
# matplotlib's own. The data-unit label offsets below scale with the spans.
REF_XS, REF_YS = 15.763428878784177, 16.295139923095697
_xl, _yl = ax_a.get_xlim(), ax_a.get_ylim()
SXA, SYA = (_xl[1] - _xl[0]) / REF_XS, (_yl[1] - _yl[0]) / REF_YS
if abs(SXA - 1) < 1e-9: SXA = 1.0
if abs(SYA - 1) < 1e-9: SYA = 1.0
for _set, _lim, _sc, _tk in ((ax_a.set_xticks, _xl, SXA, [-4, 0, 4, 8]), (ax_a.set_yticks, _yl, SYA, [-3, 0, 3])):
    if 1 / 1.5 <= _sc <= 1.5 and _lim[0] <= min(_tk) and max(_tk) <= _lim[1]: _set(_tk)
# the mass direction: parallel to z_0.1kg -> z_3kg, anchored to the axes limits
zs_, za_ = proj_top(win["z_stale"])[0, :2], proj_top(win["z_adapted"])[0, :2]
k_ = (za_[1] - zs_[1]) / (za_[0] - zs_[0])
_xl, _yl = ax_a.get_xlim(), ax_a.get_ylim(); _L = 0.42 * (_xl[1] - _xl[0])
A0 = np.array([_xl[0] + 0.26 * (_xl[1] - _xl[0]), _yl[1] - 0.34 * (_yl[1] - _yl[0])])
A1 = A0 + [_L, _L * k_]
ax_a.annotate("", xy=A1, xytext=A0, arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=7), zorder=6)
d0, d1 = ax_a.transData.transform(A0), ax_a.transData.transform(A1)
ax_a.text(*((A0 + A1) / 2 + [0, 1.25]), "increasing payload mass", ha="center", va="bottom", fontsize=10.0, color=INK,
          rotation=np.degrees(np.arctan2(d1[1] - d0[1], d1[0] - d0[0])), rotation_mode="anchor", zorder=6)
# the two z's (b) is rolled out with
zs, za = proj_top(win["z_stale"])[0, :2], proj_top(win["z_adapted"])[0, :2]
for p, col, txt, dxy, ha in ((zs, C_STALE, "$z_{0.1\,\mathrm{kg}}$", (0.15 * SXA, 1.35 * SYA), "center"),
                             (za, C_ADAPT, "$z_{3\,\mathrm{kg}}$", (-1.6 * SXA, 1.45 * SYA), "center")):
    ax_a.scatter(*p, s=46, facecolor="white", edgecolor=col, lw=1.3, zorder=7)
    ax_a.scatter(*p, s=9, color=col, lw=0, zorder=8)
    ax_a.annotate(txt, p, xytext=(p[0] + dxy[0], p[1] + dxy[1]), ha=ha, va="bottom", fontsize=10.5, color=col,
                  fontweight="bold", linespacing=0.95, zorder=9,
                  path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
                  arrowprops=dict(arrowstyle="-", color=col, lw=0.6, shrinkA=0, shrinkB=4))
cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, ticks=[0.1, 1, 2, 3, 4])
cb.ax.set_yticklabels(["0.1", "1", "2", "3", "4"]); cb.ax.tick_params(labelsize=10, length=1.5, width=0.4, pad=1)
cb.outline.set_linewidth(0.4)
cb.ax.axhline(2.0, color="white", lw=1.0)
cb.set_label("payload mass (kg)", fontsize=10.5, labelpad=2)
for y, txt in ((1.05, "trained"), (3.0, "unseen")):
    cb.ax.text(-0.9, y, txt, rotation=90, transform=cb.ax.get_yaxis_transform(), ha="center", va="center",
               fontsize=9.2, color=INK2)

# ---- (b) arm panels: same state, same recorded torques, two z's
C_REC = "0.42"                                            # recorded = grey
TITLE = {"stale": ("$z_{0.1\,\mathrm{kg}}$", C_STALE), "ours": ("$z_{3\,\mathrm{kg}}$", C_ADAPT)}
for ax, key, (img, _) in zip((ax_c1, ax_c2), rows, panels):
    ax.imshow(img[y0:y1, x0:x1], interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("0.78"); s.set_linewidth(0.5)
    col = TITLE[key][1]
    gt_t, pr_t = track["gt"] - [x0, y0], track[key] - [x0, y0]
    ax.plot(*gt_t.T, "-", color=C_REC, lw=1.2, zorder=5)
    ax.plot(*gt_t[1:].T, "o", color=C_REC, ms=2.0, zorder=5)
    ax.plot(*pr_t.T, "-", color=col, lw=1.2, zorder=6)
    ax.plot(*pr_t[1:].T, "o", color=col, ms=2.0, zorder=6)
    ax.plot(*gt_t[0], "o", ms=5.0, mfc="white", mec=C_REC, mew=1.0, zorder=7)       # shared start
    ax.plot([gt_t[-1, 0], pr_t[-1, 0]], [gt_t[-1, 1], pr_t[-1, 1]], color=col, lw=1.3, ls=(0, (2.5, 1.2)), zorder=7)
    ax.plot(*gt_t[-1], "o", ms=6.5, mfc=C_REC, mec="white", mew=0.9, zorder=8)
    ax.plot(*pr_t[-1], "*", ms=12, mfc=col, mec="white", mew=0.7, zorder=8)
    # the end-point error under the panel, off the picture
    ax.text(0.5, -0.025, f"end-point error  {float(win[f'err_{key}'][-1]):.1f} cm", transform=ax.transAxes, ha="center",
            va="top", fontsize=10, color=col, fontweight="bold")
    # the payload: one tag on the table, pointing at the cube where the prediction starts
    ax.annotate("3 kg payload", xy=gt_t[0] + [0, 14], xytext=(gt_t[0][0] + 12, (y1 - y0) - 18), ha="right", va="bottom",
                fontsize=8.5, color=INK, zorder=9, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.55", lw=0.5),
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkA=1, shrinkB=0))
    ax.text(0.97, 0.975, TITLE[key][0], transform=ax.transAxes, ha="right", va="top", fontsize=12.5, color=col,
            fontweight="bold", path_effects=[pe.withStroke(linewidth=2.0, foreground="w")])
legc = [plt.Line2D([], [], color=C_REC, lw=1.2, marker="o", ms=2.2),
        (plt.Line2D([], [], color=C_STALE, lw=1.2, marker="o", ms=2.2), plt.Line2D([], [], color=C_ADAPT, lw=1.2, marker="o", ms=2.2)),
        plt.Line2D([], [], ls="none", marker="o", ms=4.5, mfc="white", mec=C_REC, mew=0.9),
        plt.Line2D([], [], ls="none", marker="o", ms=5.0, mfc=C_REC, mec="white"),
        plt.Line2D([], [], ls="none", marker="*", ms=8.0, mfc=INK2, mec="white", mew=0.4)]
p1, p2 = ax_c1.get_position(), ax_c2.get_position()
fig.legend(legc, ["recorded", "predicted", "start", "recorded end", "predicted end"],
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)}, loc="upper center",
           bbox_to_anchor=((p1.x0 + p2.x1) / 2, p1.y0 - 0.30 / FH), ncol=5, frameon=False, fontsize=9.2,
           handlelength=1.3, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.0)
for ax, lab in ((ax_a, "a"), (ax_c1, "b")):
    ax.text(-0.02, 1.03, f"({lab})", transform=ax.transAxes, fontweight="bold", fontsize=10 * FS, ha="right", va="bottom")


# ================================================================ bottom row: (c) carry, (d) payload sweep
MU, V = z4["zmean"], z4["V"]
proj = lambda z: (np.atleast_2d(np.asarray(z, float)) - MU) @ V.T[:, :2]
ex = np.load(W / "z" / "official_train.npz", allow_pickle=True)       # the PCA set (official_train val)
zx, mx = ex["z"].astype(float), np.round(ex["mass"].astype(float), 3)
assert np.allclose(zx.mean(0), MU, atol=1e-5), "official_train z is not the frame of pca.npz"
Wr = np.linalg.lstsq(np.c_[zx, np.ones(len(zx))], mx, rcond=None)[0]   # linear mass read-out of z, this set only
readout = lambda z: np.c_[np.asarray(z, float), np.ones(len(z))] @ Wr
LEVELS = np.unique(mx)
PX = proj(zx)
CEN4 = {v: zx[mx == v].mean(0) for v in LEVELS}

Wf, Hf = fig.get_size_inches(); sx = Wf / FW_ROW
bbox = lambda x, y, w, h: fig.add_axes([x / Wf, y / Hf, w / Wf, h / Hf])
PY_, PH, HFAC = 0.44, 1.86, 0.90
axa = bbox(0.52 * sx, PY_ * sx, 2.90 * sx, PH * sx * HFAC)
axb = bbox(3.90 * sx, PY_ * sx, 2.90 * sx, PH * sx * HFAC)

# ---- (c) carry: PC1 against time. Encoded from 25 frames BEFORE the lift (make_prelift.py): those first windows see
# only the approach and the grasp, no payload is being carried yet, so z cannot know the mass.
d = np.load(W / "z" / "official_carry_prelift.npz", allow_pickle=True)
lift = json.loads((W / "prelift" / "lift.json").read_text())         # first carry frame of the original recording
e, ws = d["episode"], d["window_start"].astype(int)
m, zc = np.round(d["mass"].astype(float), 3), d["z"].astype(float)
rel = ws - np.array([lift[n] for n in e])                     # window start relative to the lift
pc = proj(zc)
KMIN, KMAX = -25, 57                                          # -0.5 s (pre-lift) .. 1.14 s of carry
KK = np.arange(KMIN, KMAX + 1); tk = KK * DT
SHOWL = [0.1, 1.186, 2.0]                                     # three trained levels > 1 sd apart
LVL = {v: PX[mx == v][:, 0].mean() for v in LEVELS}
SIG = {v: PX[mx == v][:, 0].std() for v in LEVELS}
yc = pc[:, 0]
YL, YT = (-6.0, 13.4), [-4, 0, 4, 8, 12]
for v in SHOWL:                                               # +-1 sd of that level in the PCA set = its region
    axa.axhspan(LVL[v] - SIG[v], LVL[v] + SIG[v], color=cm(v), alpha=.15, lw=0, zorder=0)
    axa.axhline(LVL[v], color=cm(v), lw=.7, ls=(0, (3.5, 2.2)), alpha=.85, zorder=1)
for v in SHOWL:
    y = np.array([np.median(yc[(rel == k) & (m == v)]) for k in KK])
    axa.plot(tk, y, color="white", lw=3.2, zorder=3, solid_capstyle="round")
    axa.plot(tk, y, color=cm(v), lw=2.2, zorder=4, solid_capstyle="round")
    axa.scatter(tk[0], y[0], s=16, facecolor="white", edgecolor=cm(v), lw=1.0, zorder=5)
for v, x, ha in ((0.1, 0.012, "left"), (1.186, 0.60, "center"), (2.0, 0.988, "right")):
    axa.text(x, LVL[v], f"{v:.2f} kg" if v < 1.5 and v > 1 else f"{v:g} kg",
             transform=axa.get_yaxis_transform(), color=cm(v), fontsize=7.6 * FS, fontweight="bold", va="top",
             ha=ha, path_effects=WS, zorder=9)
dist = np.array([np.linalg.norm(zc[j] - CEN4[m[j]]) for j in range(len(zc))])
err = np.abs(readout(zc) - m)
c_d0, c_d1 = np.median(dist[rel <= -15]), np.median(dist[rel >= 40])
c_e0, c_e1 = np.median(err[rel <= -15]), np.median(err[rel >= 40])
axa.axvline(0, color=INK, lw=.6, ls=(0, (2.5, 1.8)), zorder=2)
axa.text(0.02, YL[1] - 0.35, "lift", color=INK, fontsize=7.6 * FS, ha="left", va="top", path_effects=WS)
axa.set_xlim(KMIN * DT - 0.03, 1.17); axa.set_ylim(*YL)
axa.set_yticks(YT)
axa.set_xlabel("time from the lift (s)")
axa.set_ylabel("PC1 of $z$")


# ---- (d) the two payload sweeps (figure-eight, 0.1 -> 4 kg and 4 -> 0.1 kg): PC1 against time
def ramp(name):
    dd = np.load(W / "z" / f"{name}.npz", allow_pickle=True)
    pay = np.load(W / "z" / f"{name}_payload.npy")              # payload at each window's last frame (sweep_payload.py)
    ee, w2 = dd["episode"], dd["window_start"].astype(int); out = []
    for n in np.unique(ee):
        i = np.where(ee == n)[0]; i = i[np.argsort(w2[i])]
        out.append(dict(p=proj(dd["z"][i]), m=pay[i], t=(w2[i] - w2[i].min()) * DT))
    return out


UP, DN = ramp("ramp8_up"), ramp("ramp8_down")


def grad_line(ax, x, y, c, lw, zorder, ls="-"):
    seg = np.stack([np.c_[x, y][:-1], np.c_[x, y][1:]], 1)
    lc = LineCollection(seg, cmap=CMAP, norm=NORM, lw=lw, zorder=zorder, linestyle=ls, capstyle="round")
    lc.set_array(np.asarray(c)[:-1]); ax.add_collection(lc); return lc


def smooth(x, k=11):
    from numpy.lib.stride_tricks import sliding_window_view
    mm = np.median(sliding_window_view(x, k), axis=1)
    return np.concatenate([np.full(k - 1, mm[0]), mm])


T = np.arange(0, 31.5, 0.1)
for v in SHOWL:                                               # the same reference levels as (c)
    axb.axhspan(LVL[v] - SIG[v], LVL[v] + SIG[v], color=cm(v), alpha=.15, lw=0, zorder=0)
    axb.axhline(LVL[v], color=cm(v), lw=.7, ls=(0, (3.5, 2.2)), alpha=.85, zorder=1)
for v, va in ((0.1, "top"), (2.0, "bottom")):
    axb.text(0.995, LVL[v], f"{v:g} kg ", transform=axb.get_yaxis_transform(), color=cm(v),
             fontsize=7.6 * FS, fontweight="bold", va=va, ha="right", path_effects=WS, zorder=9)
for data, lab, xy, va in ((UP, "0.1 $\\rightarrow$ 4 kg", (27.5, 12.0), "bottom"),
                          (DN, "4 $\\rightarrow$ 0.1 kg", (6.2, 12.0), "bottom")):
    P1 = np.stack([np.interp(T, dd["t"], dd["p"][:, 0]) for dd in data])
    M = np.stack([np.interp(T, dd["t"], dd["m"]) for dd in data])
    mm, yy = np.median(M, 0), smooth(np.median(P1, 0))
    axb.fill_between(T, smooth(np.percentile(P1, 25, axis=0)), smooth(np.percentile(P1, 75, axis=0)),
                     color="0.55", alpha=.18, lw=0, zorder=3)
    grad_line(axb, T, yy, mm, 2.2, 4)
    axb.annotate(lab, xy, fontsize=7.8 * FS, color=INK, ha="center", va=va, path_effects=WS)
    if lab.startswith("0.1"):                                 # when the payload is actually being ramped
        mv = np.abs(np.gradient(mm, T)) > 1e-3
        axb.axvspan(T[mv][0], T[mv][-1], color="0.5", alpha=.07, lw=0, zorder=0)
axb.set_xlim(0, 31.4); axb.set_ylim(*YL); axb.set_yticks(YT)
axb.set_xlabel("time (s)")
axb.set_ylabel("PC1 of $z$")
for ax, lab in ((axa, "c"), (axb, "d")):
    ax.text(-0.02, 1.012, f"({lab})", transform=ax.transAxes, fontweight="bold", fontsize=10 * FS, ha="right",
            va="bottom", color=INK)

OUT.parent.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(OUT.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.02)
print("->", OUT.with_suffix(".png"))
print(f"(b) {win['episode']} @ {int(win['window'])}: stale {float(win['err_stale'][-1]):.2f} / adapted "
      f"{float(win['err_ours'][-1]):.2f} cm")
print(f"(c) |z-centroid| {c_d0:.2f} -> {c_d1:.2f}, mass error {c_e0:.3f} -> {c_e1:.3f} kg")
(W / "fig_numbers.json").write_text(json.dumps(dict(c_centroid=[float(c_d0), float(c_d1)],
                                                    c_mass_err=[float(c_e0), float(c_e1)]), indent=1))
