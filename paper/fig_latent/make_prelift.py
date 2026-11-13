"""Derived set for panel (c): 200 official_carry val episodes whose carry phase starts LEAD frames before the lift.

The encoder only sees carry-phase (obs/phase == 3) windows. To show z *before* the payload is carried, this copies
200 official_carry val episodes and relabels obs/phase so the carry phase begins LEAD frames (0.5 s) earlier, during
the late approach and the grasp; nothing else in the recording changes. The 200 episodes are the ones `encode_z.py
--max-eps 200` (seed 0) draws from the official_carry val split.

    python3 make_prelift.py <out dir> [LEAD=25]
-> <out dir>/official_carry_prelift.hdf5, official_carry_prelift.json (contract, all episodes in "val"),
   lift.json (episode -> first carry frame of the original recording = the lift)
"""
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from cdelan import paths

OUT = Path(sys.argv[1])
LEAD = int(sys.argv[2]) if len(sys.argv) > 2 else 25
SRC, DST = "official_carry", "official_carry_prelift"
src_contract = json.loads((Path(os.environ["PICR_CONTRACTS"]) / f"{SRC}.json").read_text())
val = list(src_contract["splits"]["val"])
rng = np.random.default_rng(0)                                   # = encode_z.py --max-eps 200 --seed 0
names = sorted(np.array(val)[np.sort(rng.choice(len(val), 200, replace=False))].tolist())

OUT.mkdir(parents=True, exist_ok=True)
tmp = OUT / f"{DST}.hdf5.part"
kept, lift = [], {}
with h5py.File(paths.dataset_file(SRC), "r") as f, h5py.File(tmp, "w") as g:
    for k, v in f.attrs.items():
        g.attrs[k] = v
    d = g.create_group("data")
    for k, v in f["data"].attrs.items():
        d.attrs[k] = v
    for n in names:
        f.copy(f["data"][n], d, name=n)
        ph = np.asarray(d[n]["obs/phase"]).reshape(-1)
        i3 = np.flatnonzero(ph == 3)
        if not len(i3):
            continue
        new = ph.copy()
        new[max(0, i3.min() - LEAD):i3.max() + 1] = 3
        d[n]["obs/phase"][...] = new.reshape(d[n]["obs/phase"].shape)
        kept.append(n)
        lift[n] = int(i3.min())
tmp.rename(OUT / f"{DST}.hdf5")
c = dict(src_contract, dataset_id=DST, dataset_file=f"{DST}.hdf5", protocol_version=f"{DST}-v1",
         splits={"train": [], "val": kept},
         note=f"copy of {SRC} val episodes; obs/phase == 3 extended {LEAD} frames before the lift (figure only)")
(OUT / f"{DST}.json").write_text(json.dumps(c, indent=1))
(OUT / "lift.json").write_text(json.dumps(lift))
print(f"{len(kept)} episodes -> {OUT / f'{DST}.hdf5'}")
