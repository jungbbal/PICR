"""Payload mass of every window of a payload-sweep encoding, read from the recording (it changes within an episode).

    python3 sweep_payload.py <work>/z ramp8_up ramp8_down
-> <work>/z/<set>_payload.npy, aligned with the rows of <work>/z/<set>.npz: the mass at the window's last frame.
"""
import sys
from pathlib import Path

import h5py
import numpy as np

from cdelan import paths

HIST = 12
Z = Path(sys.argv[1])
for name in sys.argv[2:]:
    d = np.load(Z / f"{name}.npz", allow_pickle=True)
    ep, ws = d["episode"], d["window_start"].astype(int)
    out = np.full(len(ep), np.nan)
    with h5py.File(paths.dataset_file(name), "r") as f:
        for n in np.unique(ep):
            i = np.where(ep == n)[0]
            mass = np.asarray(f["data"][str(n)]["object_dynamics/mass"]).reshape(-1)
            out[i] = mass[np.clip(ws[i] + HIST - 1, 0, len(mass) - 1)]
    np.save(Z / f"{name}_payload.npy", out)
    print(f"{name}: {len(out)} windows, payload {np.nanmin(out):.2f}..{np.nanmax(out):.2f} kg")
