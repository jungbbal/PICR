"""Resolution of data-lake paths.

Code lives in git; data does not. Configs and scripts refer to data through
``${PICR_DATA_ROOT}`` / ``${PICR_RUNS_ROOT}`` instead of absolute paths, so the
lake can be moved to another disk without touching a single config file.

Layout under the data root::

    raw/<dataset_id>/       one recorded dataset + its manifest.yaml
    policies/<policy_id>/   RL policy checkpoints + manifest.yaml
    runs/<run_id>/          training outputs
    media/                  videos, figure dumps

Both variables fall back to the current on-disk locations when unset, so
existing commands keep working without any environment setup.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

DEFAULT_DATA_ROOT = REPO_ROOT.parent / "data"
DEFAULT_RUNS_SUBDIR = "runs"

#: Environment variables understood by :func:`expand`.
DATA_ROOT_ENV = "PICR_DATA_ROOT"
RUNS_ROOT_ENV = "PICR_RUNS_ROOT"
RESULTS_ROOT_ENV = "PICR_RESULTS_ROOT"
PROTOCOL_PATH_ENV = "PICR_PROTOCOL_PATH"


def data_root() -> Path:
    """Root of the data lake. Override with ``$PICR_DATA_ROOT``."""
    value = os.environ.get(DATA_ROOT_ENV)
    return Path(value).expanduser() if value else DEFAULT_DATA_ROOT


def runs_root() -> Path:
    """Where training runs are written. Override with ``$PICR_RUNS_ROOT``."""
    value = os.environ.get(RUNS_ROOT_ENV)
    return Path(value).expanduser() if value else data_root() / DEFAULT_RUNS_SUBDIR


def results_root() -> Path:
    """Where analysis writes its tables, figures and metric dumps. Override with ``$PICR_RESULTS_ROOT``.

    `REPO_ROOT` here is ``src/``, hence ``REPO_ROOT.parent``.
    """
    value = os.environ.get(RESULTS_ROOT_ENV)
    return Path(value).expanduser() if value else REPO_ROOT.parent / "results"


def protocol_path() -> Path:
    """The contract split, protocol v1.1. Read-only. Override ``$PICR_PROTOCOL_PATH``."""
    value = os.environ.get(PROTOCOL_PATH_ENV)
    return Path(value).expanduser() if value else REPO_ROOT.parent / "protocol" / "protocol.json"


def raw_root() -> Path:
    return data_root() / "raw"


def media_root() -> Path:
    return data_root() / "media"


def policies_root() -> Path:
    return data_root() / "policies"


def eval_root() -> Path:
    """Standalone evaluation/diagnostic artifacts (not tied to one training run)."""
    return data_root() / "eval"


def vision_root() -> Path:
    """Extracted frames and encoder latents (``Images``, ``Images_latents``)."""
    return data_root() / "vision"


def expand(value: str) -> str:
    """Expand ``${PICR_DATA_ROOT}``, ``${PICR_RUNS_ROOT}``, ``~`` and other env vars.

    The two PICR variables are substituted from :func:`data_root` /
    :func:`runs_root` so they resolve to the defaults above even when unset in
    the environment. Anything else is left to :func:`os.path.expandvars`.
    """
    if not isinstance(value, str) or not value:
        return value

    for name, resolved in ((DATA_ROOT_ENV, data_root()), (RUNS_ROOT_ENV, runs_root())):
        for token in (f"${{{name}}}", f"${name}"):
            if token in value:
                value = value.replace(token, str(resolved))

    return os.path.expanduser(os.path.expandvars(value))


def dataset_dir(dataset_id: str) -> Path:
    """Directory holding one recorded dataset."""
    return raw_root() / dataset_id


def dataset_file(dataset_id: str) -> Path:
    """The data file of ``dataset_id``, as named by its manifest.

    Falls back to the single ``*.hdf5`` in the directory when the manifest is
    missing or unreadable, so a dataset dropped in without a manifest still
    resolves. Raises when the directory holds no or several candidates and the
    manifest cannot disambiguate.
    """
    directory = dataset_dir(dataset_id)
    manifest = directory / "manifest.yaml"

    if manifest.is_file():
        name = _manifest_file_field(manifest)
        if name:
            path = directory / name
            if path.is_file():
                return path
            raise FileNotFoundError(f"{manifest} names {name!r}, which is not in {directory}")

    candidates = sorted(directory.glob("*.hdf5"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No *.hdf5 found in {directory}")
    raise ValueError(
        f"{directory} holds {len(candidates)} *.hdf5 files; add a 'file:' field to {manifest}"
    )


def _manifest_file_field(manifest: Path) -> str | None:
    """Read the top-level ``file:`` field without requiring PyYAML."""
    try:
        import yaml  # noqa: PLC0415 - optional dependency

        with manifest.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        value = loaded.get("file") if isinstance(loaded, dict) else None
        return str(value) if value else None
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - a broken manifest must not block loading
        return None

    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("file:"):
                return line.split(":", 1)[1].strip() or None
    return None
