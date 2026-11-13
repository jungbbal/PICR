"""Cut panel (b)'s example window into replay npz files for render/replay_arm.py.

One file per rollout (gt, ours, stale): the arm's joint angles and the payload pose, frame 0 being the pose the
prediction starts from. The fingers are not part of the 7-DOF model's state, so they are held at the width the
recording grips the 4.8 cm cube with.

    python3 replay_prep.py --npz <work>/fig_latent/arm_window_h12.npz --out <replay dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GRIP = 0.024          # m per finger


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="gt,ours,stale")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"episode": str(d["episode"]), "window": int(d["window"]), "dt": float(d["dt"]), "grip": GRIP,
            "err_cm": {k: float(d[f"err_{k}"][-1]) for k in ("ours", "stale")}}
    for key in args.rows.split(","):
        q = np.vstack([d["anchor_q"][None], d[f"{key}_q"]])             # frame 0 = prediction start
        obj = np.vstack([d["anchor_obj"][None], d[f"{key}_obj"]])
        q9 = np.c_[q[:, :7], np.full((len(q), 2), GRIP)]
        np.savez(out / f"{key}.npz", q=q9.astype(np.float32), obj_pos=obj[:, :3].astype(np.float32),
                 obj_quat=obj[:, 3:7].astype(np.float32))
        meta[key] = {"frames": int(len(q9))}
        print(f"{key:6s} {len(q9)} frames -> {out / (key + '.npz')}")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
