"""The numbers the paper text quotes from the latent figure.

    python3 summary_numbers.py <work>   -> stdout (run.sh writes it to $PICR_RESULTS_ROOT/fig_latent_numbers.txt)

PCA frame = z/pca.npz (official_train val). Spearman rho of PC1 vs payload mass over every window and over
per-episode means (PC2 window rho in brackets); (b) = end-point errors of the example window (h = 12); the unseen
3 kg rollout check of stale_z.py; (c) = the pre-lift vs carry numbers plot.py writes to fig_numbers.json.
(File not named numbers.py: that name shadows the stdlib module numpy imports.)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

W = Path(sys.argv[1])
z4 = np.load(W / "z" / "pca.npz")
win = np.load(W / "arm_window_h12.npz", allow_pickle=True)
st = json.loads((W / "stale_z.json").read_text())
fn = json.loads((W / "fig_numbers.json").read_text())
print("PCA of z (official_train val), explained variance %: " + " / ".join(f"{v * 100:.1f}" for v in z4["evr"])
      + f"   -> PC1 share {z4['evr'][0] * 100:.1f}%")
print("Spearman rho of PC1 vs payload mass, window / episode mean (PC2 window rho in brackets):")
for lab, f in (("excitation (official_train val)", "official_train"), ("carry (official_carry val)", "official_carry"),
               ("OOD 2-4 kg (official_ood_2to4kg)", "official_ood_2to4kg"),
               ("figure-eight carry (track8ppo3d_carry)", "track8ppo3d_carry"),
               ("figure-eight OOD (track8ppo3d_ood_2to4kg)", "track8ppo3d_ood_2to4kg")):
    d = np.load(W / "z" / f"{f}.npz", allow_pickle=True)
    P = (d["z"] - z4["zmean"]) @ z4["V"].T
    eps = np.unique(d["episode"])
    pm = [P[d["episode"] == e, 0].mean() for e in eps]; mm = [d["mass"][d["episode"] == e][0] for e in eps]
    print(f"  {lab:42s} {spearmanr(P[:, 0], d['mass']).correlation:+.2f} / {spearmanr(pm, mm).correlation:+.2f}"
          f"   ({spearmanr(P[:, 1], d['mass']).correlation:+.2f})   {len(eps)} eps, {len(P)} windows")
print(f"(b) {win['episode']} @ {int(win['window'])}, h={int(win['horizon'])}: end-point error z_0.1kg (stale) "
      f"{float(win['err_stale'][-1]):.1f} cm, z_3kg (adapted) {float(win['err_ours'][-1]):.1f} cm")
o = st["ood"]
print(f"unseen {o['mass_kg']:g} kg ({o['dataset']}), {o['n_episodes']} eps / {o['n_windows']} windows, h={o['horizon']} "
      f"mean final-step cube error: adapted z {o['mean_cm_adapted']:.2f} / stale z {o['mean_cm_stale']:.2f} cm")
print(f"(c) |z - own-mass centroid| {fn['c_centroid'][0]:.2f} -> {fn['c_centroid'][1]:.2f}, mass read-out error "
      f"{fn['c_mass_err'][0]:.3f} -> {fn['c_mass_err'][1]:.3f} kg (pre-lift windows <= -15 frames vs carry >= 40 frames)")
