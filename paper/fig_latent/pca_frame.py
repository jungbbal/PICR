"""PCA frame of the latent figure: PCA of z over every official_train val window, signs set so that PC1 and PC2
both correlate positively (Spearman) with the payload mass. The OOD 2-4 kg windows are projected into the same frame.

    python3 pca_frame.py <work>/z      (reads official_train.npz, official_ood_2to4kg.npz; writes pca.npz)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

Z = Path(sys.argv[1])
tr, od = np.load(Z / "official_train.npz"), np.load(Z / "official_ood_2to4kg.npz")
z, zo = tr["z"], od["z"]
zmean = z.mean(0)
U, Sg, Vt = np.linalg.svd(z - zmean, full_matrices=False)
evr = Sg ** 2 / (Sg ** 2).sum()
for i in (0, 1):
    r = spearmanr((z - zmean) @ Vt[i], tr["mass"]).correlation
    if r < 0:
        Vt[i] *= -1
        print(f"PC{i + 1} sign flipped (rho was {r:+.3f})")
np.savez(Z / "pca.npz", z=z, P=(z - zmean) @ Vt.T, V=Vt, zmean=zmean, evr=evr, mass=tr["mass"],
         com=tr["com"], episode=tr["episode"], z_ood=zo, P_ood=(zo - zmean) @ Vt.T,
         mass_ood=od["mass"], com_ood=od["com"], levels=np.unique(tr["mass"].round(3)))
print("explained variance", np.round(evr, 3))
