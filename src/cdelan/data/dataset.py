from __future__ import annotations

import glob
import json
import os
import random
from typing import List

try:
    import h5py
except ImportError:  # pragma: no cover - depends on local environment.
    h5py = None

from cdelan import paths

from .hdf5_schema import EpisodeRef, Hdf5Groups

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

#: Redirects a "val" split request to another key of the external split (see resolve_split_refs).
EVAL_SPLIT_KEY_ENV = "PICR_EVAL_SPLIT_KEY"


def h5_open(path: str):
    if h5py is None:
        raise RuntimeError("h5py is required to load robot-object world-model datasets.") from None
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


def discover_hdf5_files(dataset_dir: str) -> List[str]:
    """Find the ``*.hdf5`` files under ``dataset_dir``.

    Falls back to one level of nesting when the directory holds no data files
    directly, so that pointing at the data lake's ``raw/`` root finds the
    per-dataset directories inside it.
    """
    root = os.path.abspath(dataset_dir)
    direct = sorted(glob.glob(os.path.join(root, "*.hdf5")))
    if direct:
        return direct
    return sorted(glob.glob(os.path.join(root, "*", "*.hdf5")))


def load_episode_names(hdf5_path: str) -> List[str]:
    with h5_open(hdf5_path) as file:
        return list(file[Hdf5Groups.DATA].keys())


def load_all_episode_refs(hdf5_paths: List[str]) -> List[EpisodeRef]:
    refs: List[EpisodeRef] = []
    for path in hdf5_paths:
        for name in load_episode_names(path):
            refs.append((path, name))
    return refs


def episode_masses(episode_refs: List[EpisodeRef]) -> List[float]:
    """Payload mass (kg) of each episode, in the order the refs are given."""
    masses: List[float] = []
    by_path: dict[str, List[int]] = {}
    for i, (path, _name) in enumerate(episode_refs):
        by_path.setdefault(path, []).append(i)
    out = [0.0] * len(episode_refs)
    for path, idx in by_path.items():
        with h5_open(path) as file:
            data = file[Hdf5Groups.DATA]
            for i in idx:
                grp = data[episode_refs[i][1]]["object_dynamics"]["mass"]
                out[i] = float(grp[()].reshape(-1)[0])
    masses = out
    return masses


def filter_refs_by_mass(
    episode_refs: List[EpisodeRef],
    mass_min: float,
    mass_max: float,
) -> List[EpisodeRef]:
    """Keep only episodes whose payload mass is in [mass_min, mass_max)."""
    masses = episode_masses(episode_refs)
    return [ref for ref, m in zip(episode_refs, masses) if mass_min <= m < mass_max]


def load_external_split(path: str) -> dict[str, List[str]]:
    """Read an externally frozen episode-name partition from JSON.

    Two layouts are accepted: a bare ``{"<split>": [names...]}`` mapping, or a
    larger contract document carrying that mapping under a ``"splits"`` key
    (``data/contracts/*.json``). Only lists of
    strings are kept, so sibling metadata in the contract is ignored rather than
    mistaken for a split.

    ``${PICR_DATA_ROOT}`` expands, like every other path a config carries, so a
    contract can travel next to the dataset it partitions instead of as an
    absolute machine path.
    """
    with open(paths.expand(path)) as fh:
        doc = json.load(fh)
    table = doc.get("splits", doc) if isinstance(doc, dict) else None
    if not isinstance(table, dict):
        raise ValueError(f"External split {path!r} is not a mapping of split name -> episodes.")
    out = {
        key: [str(v) for v in value]
        for key, value in table.items()
        if isinstance(value, list) and all(isinstance(v, str) for v in value)
    }
    if not out:
        raise ValueError(f"External split {path!r} carries no episode-name lists.")
    return out


def refs_for_external_split(
    episode_refs: List[EpisodeRef],
    split_table: dict[str, List[str]],
    key: str,
) -> List[EpisodeRef]:
    """Select the refs named by ``split_table[key]``, in dataset order.

    Every named episode must be present: a silently shrinking split is the exact
    failure that makes two runs look comparable when they are not, so a missing
    name raises instead of being dropped.
    """
    if key not in split_table:
        raise KeyError(f"External split has no '{key}'; available: {sorted(split_table)}")
    wanted = set(split_table[key])
    refs = [ref for ref in episode_refs if ref[1] in wanted]
    found = {name for _path, name in refs}
    missing = wanted - found
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise RuntimeError(
            f"External split '{key}' names {len(missing)} episodes absent from the dataset "
            f"(e.g. {sample}). The split was frozen against a different file."
        )
    return refs


def _cfg_get(cfg, name: str, default=None):
    """Read a field from either a TrainConfig-like object or a plain dict."""
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def resolve_split_refs(episode_refs: List[EpisodeRef], cfg, split: str) -> List[EpisodeRef]:
    """Refs for one split, honouring ``external_split_json`` when it is set.

    Without an external split this is the historical seeded shuffle, so existing
    checkpoints resolve to exactly the episodes they trained on. With one, the
    partition comes from the file and ``seed``/``train_split`` no longer affect
    it -- which is the point: an external contract is only worth anything if the
    local knobs cannot move it. ``split`` may name any key in that file, so an
    externally defined third band (e.g. a withheld mass range) is addressable
    the same way as train and val.
    """
    path = _cfg_get(cfg, "external_split_json", "") or ""
    if path:
        table = load_external_split(path)
        key = split
        if split == "val":
            # The figure and diagnostic scripts ask for "val" by name and take their config
            # from a checkpoint, so there is no argument to thread a different band through.
            # $PICR_EVAL_SPLIT_KEY redirects that one request -- which is what lets the whole
            # latent suite be re-run against a withheld band without touching ten entry
            # points. It follows the Z_CKPT / Z_OUT_DIR convention those scripts already use.
            key = (
                os.environ.get(EVAL_SPLIT_KEY_ENV)
                or _cfg_get(cfg, "external_split_val_key", "id_test")
                or "id_test"
            )
        return refs_for_external_split(episode_refs, table, key)
    if split not in ("train", "val"):
        raise ValueError(
            f"split '{split}' needs external_split_json; the seeded shuffle defines "
            "only 'train' and 'val'."
        )
    train_refs, val_refs = split_episode_refs(
        episode_refs, _cfg_get(cfg, "train_split"), _cfg_get(cfg, "seed")
    )
    return train_refs if split == "train" else val_refs


def split_episode_refs(
    episode_refs: List[EpisodeRef],
    train_split: float,
    seed: int,
) -> tuple[List[EpisodeRef], List[EpisodeRef]]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be in (0, 1).")
    refs = episode_refs.copy()
    random.Random(seed).shuffle(refs)
    n_train = int(len(refs) * train_split)
    train_refs = refs[:n_train]
    val_refs = refs[n_train:]
    if not train_refs or not val_refs:
        raise RuntimeError("Episode split produced an empty train or validation set.")
    return train_refs, val_refs
