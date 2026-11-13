"""Copy the frames plot.py reads from a replay_arm.py output folder into assets/frames (depth as compressed npz).

    python3 pack_frames.py <replay dir>/frames ../assets/frames [--last 12]
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--last", type=int, default=12, help="the frame panel (b) compares (= the horizon)")
a = ap.parse_args()
src, dst = Path(a.src), Path(a.dst)
dst.mkdir(parents=True, exist_ok=True)
shutil.copy(src / "cam.json", dst / "cam.json")
L = f"{a.last:03d}"
need = {"gt": ["bg.png", f"f{L}.png", "bg_depth", "d000", f"d{L}"],
        "stale": [f"f{L}.png", "bg_depth", f"d{L}"], "ours": [f"f{L}.png", "bg_depth", f"d{L}"]}
for k, files in need.items():
    (dst / k).mkdir(exist_ok=True)
    for f in files:
        if f.endswith(".png"):
            shutil.copy(src / k / f, dst / k / f)
        else:
            np.savez_compressed(dst / k / f"{f}.npz", depth=np.load(src / k / f"{f}.npy"))
print("->", dst)
