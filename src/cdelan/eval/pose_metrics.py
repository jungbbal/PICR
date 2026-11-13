"""6D pose evaluation metrics for state prediction (rollout) settings.

Everything here is in **metres** and **radians-in, degrees-out**, and everything is batched:
a pose argument carries arbitrary leading dimensions, and the returned error carries the same
ones. For a rollout the natural layout is ``(..., T)`` with the horizon last, which is what the
time-axis helpers in the bottom section reduce over.

Three layers:

* **frame metrics** -- :func:`translation_error`, :func:`rotation_error_deg`, :func:`add`,
  :func:`add_s`, and :func:`add_auto`, which picks ADD or ADD-S from the object's symmetry
  flag. ADD is Hinterstoisser et al., ACCV 2012.
* **set aggregates** -- :func:`acc_2cm_5deg`, :func:`acc_threshold`, :func:`add_auc`, plus
  :func:`summarize` which reports every one of them together with a mean *and* a median,
  because a mean alone cannot tell a uniformly mediocre model from a mostly-good one with a
  few broken frames.
* **time-axis aggregates** -- :func:`ade`, :func:`fde`, :func:`horizon_curve`, and
  :func:`rollout_metrics`, for prediction sequences.

The model point set is sampled uniformly over the object's *surface*, area-proportional, and
cached per object. Mesh vertices are deliberately not used: they cluster wherever the mesh is
finely tessellated -- a cube's eight vertices are its eight maximum-radius points, so an ADD
built on them reports the worst case as if it were the average.

**ADD-S has a sampling floor.** Its nearest-neighbour search can never return less than the
mean spacing of the point set, ~``0.5 / sqrt(n_points / surface_area)``: 1.3 mm for the 2000
points this module defaults to on a 4.8 cm cube, 0.65 mm at 8000. A symmetric object in a
perfectly-explained-away pose therefore scores that spacing, not zero, and an ADD-S below it
is not measurable. Raise `n_points` if the errors under study approach it.

Poses are accepted in three interchangeable forms, see :func:`to_Rt`:

    to_Rt(T)             # (..., 4, 4) homogeneous
    to_Rt((R, t))        # (..., 3, 3) rotation matrix + (..., 3) translation
    to_Rt((q, t))        # (..., 4) quaternion, Hamilton **wxyz** + (..., 3) translation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import itertools

import numpy as np
from scipy.spatial import cKDTree

# A translation this large is not metres. Catches a millimetre-unit input, which would
# otherwise pass silently and report every threshold metric as 0.
_MAX_PLAUSIBLE_METRES = 1.0e3


# --------------------------------------------------------------------------- #
# pose input handling
# --------------------------------------------------------------------------- #
def quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """(..., 4) Hamilton wxyz -> (..., 3, 3). Normalises; never loops."""
    q = np.asarray(quat, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion must have last dimension 4, got {q.shape}")
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def to_Rt(pose) -> tuple[np.ndarray, np.ndarray]:
    """Normalise any accepted pose form to ``(R, t)`` with shapes (..., 3, 3) and (..., 3)."""
    if isinstance(pose, (tuple, list)):
        if len(pose) != 2:
            raise ValueError("a pose pair must be (rotation, translation)")
        rot, t = (np.asarray(a, dtype=np.float64) for a in pose)
        if rot.shape[-2:] == (3, 3):
            R = rot
        elif rot.shape[-1] == 4:
            R = quat_wxyz_to_rotmat(rot)
        else:
            raise ValueError(f"rotation must be (...,3,3) or (...,4) wxyz, got {rot.shape}")
    else:
        T = np.asarray(pose, dtype=np.float64)
        if T.shape[-2:] != (4, 4):
            raise ValueError(f"a single-array pose must be (...,4,4) homogeneous, got {T.shape}")
        R, t = T[..., :3, :3], T[..., :3, 3]
    if t.shape[-1] != 3:
        raise ValueError(f"translation must be (...,3), got {t.shape}")
    if R.shape[:-2] != t.shape[:-1]:
        raise ValueError(f"rotation batch {R.shape[:-2]} does not match translation batch {t.shape[:-1]}")
    return R, t


def _check_pair(gt, pred) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Both poses in (R, t) form, in the same frame, in metres."""
    R_gt, t_gt = to_Rt(gt)
    R_pred, t_pred = to_Rt(pred)
    if R_gt.shape != R_pred.shape or t_gt.shape != t_pred.shape:
        raise ValueError(
            f"gt and pred must have the same batch shape, got {t_gt.shape} and {t_pred.shape}"
        )
    finite = np.isfinite(t_gt).all() and np.isfinite(t_pred).all()
    if not finite:
        raise ValueError("translations contain non-finite values")
    scale = max(np.abs(t_gt).max(initial=0.0), np.abs(t_pred).max(initial=0.0))
    if scale > _MAX_PLAUSIBLE_METRES:
        raise ValueError(
            f"largest |translation| is {scale:.3g}; this module works in metres, so the inputs "
            "are probably in millimetres or in a different frame"
        )
    return R_gt, t_gt, R_pred, t_pred


# --------------------------------------------------------------------------- #
# symmetry groups
# --------------------------------------------------------------------------- #
IDENTITY_GROUP = np.eye(3)[None]


@lru_cache(maxsize=None)
def cube_symmetry_group() -> np.ndarray:
    """The 24 proper rotations of a cube (the octahedral group), as (24, 3, 3).

    Every signed permutation matrix is a symmetry of the cube; the 24 with determinant +1 are
    the rotations, the other 24 involve a reflection and are not physically reachable.
    """
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            M = np.zeros((3, 3))
            for row, col in enumerate(perm):
                M[row, col] = signs[row]
            if np.linalg.det(M) > 0:
                mats.append(M)
    out = np.asarray(mats)
    assert out.shape == (24, 3, 3)
    return out


# --------------------------------------------------------------------------- #
# object models: surface sampling, cached per object
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObjectModel:
    """A rigid object's point set, its symmetry, and the KD-tree ADD-S needs.

    `points` are in the object's own frame, in metres, sampled uniformly over the surface.
    """

    name: str
    points: np.ndarray
    symmetric: bool = False
    sym_group: np.ndarray = field(default_factory=lambda: IDENTITY_GROUP)
    _tree: list = field(default_factory=list, repr=False, compare=False)

    @property
    def tree(self) -> cKDTree:
        """KD-tree over the *object-frame* points, built once.

        ADD-S never needs a per-frame tree. Because R_pred is orthogonal,

            ‖(R_gt x₁ + t_gt) − (R_pred x₂ + t_pred)‖ = ‖R_predᵀ(R_gt x₁ + t_gt − t_pred) − x₂‖

        so the nearest-neighbour search is always against the same fixed point cloud, in the
        object frame. One tree serves every frame of every trajectory.
        """
        if not self._tree:
            self._tree.append(cKDTree(self.points))
        return self._tree[0]

    @property
    def diameter(self) -> float:
        """Largest distance between any two model points -- the `d` in the ADD-0.1d rule."""
        hull = self.points
        return float(np.linalg.norm(hull[:, None, :] - hull[None, :, :], axis=-1).max())


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def sample_box_surface(extents: tuple[float, float, float], n_points: int, seed: int = 0) -> np.ndarray:
    """`n_points` uniform over the surface of a box centred on the object origin.

    Area-proportional: each of the six faces gets a share of the points equal to its share of
    the total surface area, and the points are uniform within each face. For a cube that
    reduces to an equal count per face.
    """
    a, b, c = (float(e) for e in extents)
    if min(a, b, c) <= 0:
        raise ValueError(f"box extents must be positive, got {extents}")
    if n_points < 6:
        raise ValueError(f"n_points={n_points} cannot cover six faces")
    half = np.array([a, b, c]) / 2.0
    # face k of axis i has area = product of the other two extents; two faces per axis
    areas = np.array([b * c, b * c, a * c, a * c, a * b, a * b])
    counts = np.floor(n_points * areas / areas.sum()).astype(int)
    counts[: n_points - counts.sum()] += 1  # hand the rounding remainder to the largest faces
    rng = _rng(seed)
    faces = []
    for face, count in enumerate(counts):
        axis, sign = face // 2, (-1.0, 1.0)[face % 2]
        pts = rng.uniform(-half, half, size=(count, 3))
        pts[:, axis] = sign * half[axis]
        faces.append(pts)
    return np.concatenate(faces, axis=0)


def sample_mesh_surface(vertices: np.ndarray, faces: np.ndarray, n_points: int, seed: int = 0) -> np.ndarray:
    """`n_points` uniform over a triangle mesh's surface, area-weighted.

    Triangles are drawn with probability proportional to their area and a point is drawn
    uniformly inside each with the square-root barycentric trick. This is the general form of
    the rule the box sampler implements exactly.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]                                            # (F, 3, 3)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=-1)
    if not area.sum() > 0:
        raise ValueError("mesh has zero total surface area")
    rng = _rng(seed)
    idx = rng.choice(len(f), size=n_points, p=area / area.sum())
    r1, r2 = rng.random((2, n_points, 1))
    su = np.sqrt(r1)
    w = np.concatenate([1 - su, su * (1 - r2), su * r2], axis=1)   # (n, 3) barycentric
    return np.einsum("nk,nkj->nj", w, tri[idx])


@lru_cache(maxsize=None)
def box_model(
    name: str,
    extents: tuple[float, float, float],
    n_points: int = 2000,
    symmetric: bool = False,
    seed: int = 0,
) -> ObjectModel:
    """A cached box :class:`ObjectModel`. Cached per (name, extents, count, symmetry, seed)."""
    points = sample_box_surface(extents, n_points, seed)
    a, b, c = extents
    if symmetric and not (np.isclose(a, b) and np.isclose(b, c)):
        raise ValueError(f"symmetric=True on a non-cube box {extents}: its group is not the cube's")
    group = cube_symmetry_group() if symmetric else IDENTITY_GROUP
    return ObjectModel(name=name, points=points, symmetric=symmetric, sym_group=group)


def cube_model(name: str, side: float, n_points: int = 2000, symmetric: bool = False, seed: int = 0) -> ObjectModel:
    """A cached cube :class:`ObjectModel` of edge length `side` metres."""
    return box_model(name, (side, side, side), n_points, symmetric, seed)


# --------------------------------------------------------------------------- #
# frame metrics
# --------------------------------------------------------------------------- #
def translation_error(gt, pred) -> np.ndarray:
    """‖t_gt − t_pred‖ in metres, shaped like the pose batch."""
    _R_gt, t_gt, _R_pred, t_pred = _check_pair(gt, pred)
    return np.linalg.norm(t_gt - t_pred, axis=-1)


def rotation_error_deg(gt, pred, sym_group: np.ndarray | None = None) -> np.ndarray:
    """arccos((tr(R_gtᵀ R_pred) − 1) / 2) in degrees, shaped like the pose batch.

    With a symmetry group, the minimum over ``R_pred · S`` for S in the group -- the smallest
    rotation that maps one pose onto an indistinguishable copy of the other. The arccos
    argument is clipped to [-1, 1]: a valid rotation can land a few ulp outside it and produce
    a NaN that then poisons every downstream mean.
    """
    R_gt, _t_gt, R_pred, _t_pred = _check_pair(gt, pred)
    M = np.einsum("...ki,...kj->...ij", R_gt, R_pred)          # R_gtᵀ R_pred
    if sym_group is None or len(sym_group) == 1:
        trace = np.trace(M, axis1=-2, axis2=-1)
    else:
        # tr(M S) summed without materialising the product; max trace == min angle
        trace = np.einsum("...ij,sji->...s", M, np.asarray(sym_group, dtype=np.float64)).max(axis=-1)
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(M,3) object points under (..., 3, 3) x (..., 3) -> (..., M, 3). Vectorised."""
    return points @ np.swapaxes(R, -1, -2) + t[..., None, :]


def add(gt, pred, model: ObjectModel) -> np.ndarray:
    """ADD: mean over x of ‖(R_gt x + t_gt) − (R_pred x + t_pred)‖, metres.

    Same-index points on both sides -- no nearest-neighbour matching. Use this whenever the
    object's orientation is a real quantity rather than one inferred up to a symmetry.
    """
    R_gt, t_gt, R_pred, t_pred = _check_pair(gt, pred)
    diff = _transform(model.points, R_gt, t_gt) - _transform(model.points, R_pred, t_pred)
    return np.linalg.norm(diff, axis=-1).mean(axis=-1)


def add_s(gt, pred, model: ObjectModel) -> np.ndarray:
    """ADD-S: mean over x₁ of the nearest-point distance to the predicted point cloud, metres.

    Uses the object-frame identity documented on :attr:`ObjectModel.tree`, so the KD-tree is
    built once and every frame is answered by a single batched query -- no per-frame loop.
    """
    R_gt, t_gt, R_pred, t_pred = _check_pair(gt, pred)
    # y = R_predᵀ (R_gt x + t_gt − t_pred), i.e. the GT points expressed in the predicted
    # object frame; their distance to the model cloud is exactly the ADD-S summand.
    world = _transform(model.points, R_gt, t_gt) - t_pred[..., None, :]
    y = world @ R_pred                                          # R_predᵀ w for row vectors
    dist, _ = model.tree.query(y.reshape(-1, 3), workers=-1)
    return dist.reshape(y.shape[:-1]).mean(axis=-1)


def add_auto(gt, pred, model: ObjectModel) -> tuple[np.ndarray, str]:
    """ADD-S if the object is flagged symmetric, ADD otherwise. Returns (values, metric name)."""
    if model.symmetric:
        return add_s(gt, pred, model), "add_s"
    return add(gt, pred, model), "add"


def pose_errors(
    gt, pred, model: ObjectModel | None = None, chunk_size: int | None = None
) -> dict[str, np.ndarray]:
    """Every frame metric at once: ``t_err_m``, ``r_err_deg``, and ADD(-S) if a model is given.

    The rotation error is symmetry-aware exactly when the model is flagged symmetric, so the
    rotation column and the ADD column always make the same assumption.

    ADD materialises an (n_frames, n_points, 3) array, which is 12 GB at half a million frames
    and 2000 points. `chunk_size` caps how many frames are transformed at once; the loop is
    over chunks, never over frames or points, so the vectorisation is untouched.
    """
    R_gt, t_gt = to_Rt(gt)
    R_pred, t_pred = to_Rt(pred)
    batch = t_gt.shape[:-1]
    n_frames = int(np.prod(batch)) if batch else 1

    if chunk_size is None or model is None or n_frames <= chunk_size:
        out = {"t_err_m": translation_error(gt, pred)}
        sym = model.sym_group if (model is not None and model.symmetric) else None
        out["r_err_deg"] = rotation_error_deg(gt, pred, sym)
        if model is not None:
            values, name = add_auto(gt, pred, model)
            out[f"{name}_m"] = values
        return out

    flat = (R_gt.reshape(-1, 3, 3), t_gt.reshape(-1, 3)), (R_pred.reshape(-1, 3, 3), t_pred.reshape(-1, 3))
    pieces: list[dict[str, np.ndarray]] = []
    for start in range(0, n_frames, chunk_size):
        sl = slice(start, start + chunk_size)
        g = (flat[0][0][sl], flat[0][1][sl])
        p = (flat[1][0][sl], flat[1][1][sl])
        pieces.append(pose_errors(g, p, model))
    return {k: np.concatenate([p[k] for p in pieces]).reshape(batch) for k in pieces[0]}


def add_key(errs: dict[str, np.ndarray]) -> str | None:
    """Whichever of ``add_m`` / ``add_s_m`` is present in a :func:`pose_errors` result."""
    for key in ("add_m", "add_s_m"):
        if key in errs:
            return key
    return None


# --------------------------------------------------------------------------- #
# set aggregates
# --------------------------------------------------------------------------- #
def acc_2cm_5deg(t_err_m, r_err_deg, t_thresh_m: float = 0.02, r_thresh_deg: float = 5.0) -> float:
    """Fraction of frames with translation error < 2 cm **and** rotation error < 5°."""
    t = np.asarray(t_err_m).ravel()
    r = np.asarray(r_err_deg).ravel()
    if t.shape != r.shape:
        raise ValueError(f"translation and rotation errors differ in size: {t.shape} vs {r.shape}")
    return float(((t < t_thresh_m) & (r < r_thresh_deg)).mean())


def acc_threshold(add_m, thresh_m: float = 0.02) -> float:
    """Fraction of frames with ADD(-S) below `thresh_m` metres (2 cm by default)."""
    a = np.asarray(add_m).ravel()
    return float((a < thresh_m).mean())


def add_auc(add_m, max_threshold_m: float = 0.1, n_grid: int = 1001) -> float:
    """Area under the ADD(-S) accuracy-threshold curve over [0, `max_threshold_m`], scaled 0–100.

    Sweeps the threshold on a dense grid, records the pass rate at each, and integrates by the
    trapezoidal rule, normalised by the threshold range so a model that is perfect everywhere
    scores 100 and one that never passes scores 0. Computed by sorting once and
    ``searchsorted``-ing the grid, rather than materialising a (grid × frames) mask.
    """
    a = np.sort(np.asarray(add_m).ravel())
    if a.size == 0:
        return float("nan")
    grid = np.linspace(0.0, max_threshold_m, n_grid)
    rate = np.searchsorted(a, grid, side="right") / a.size    # inclusive: ADD <= threshold
    return float(np.trapz(rate, grid) / max_threshold_m * 100.0)


def summarize(errs: dict[str, np.ndarray], max_threshold_m: float = 0.1) -> dict[str, float]:
    """Mean and median of every error channel, plus the three accuracy aggregates.

    Both a mean and a median, always: they separate a model that is uniformly mediocre from
    one that is usually right and occasionally broken, and only the second is worth debugging
    frame by frame.
    """
    out: dict[str, float] = {}
    for key, values in errs.items():
        v = np.asarray(values).ravel()
        out[f"{key}_mean"] = float(v.mean())
        out[f"{key}_median"] = float(np.median(v))
    if "t_err_m" in errs and "r_err_deg" in errs:
        out["acc_2cm_5deg"] = acc_2cm_5deg(errs["t_err_m"], errs["r_err_deg"])
    key = add_key(errs)
    if key is not None:
        out["acc_2cm"] = acc_threshold(errs[key])
        out["add_s_auc"] = add_auc(errs[key], max_threshold_m)
        out["add_metric"] = key[: -len("_m")]      # "add" or "add_s"
    out["n_frames"] = int(np.asarray(next(iter(errs.values()))).size)
    return out


# --------------------------------------------------------------------------- #
# time-axis aggregates (rollout)
# --------------------------------------------------------------------------- #
def ade(err: np.ndarray, k: int | None = None) -> float:
    """ADE@k: the mean error over steps 1..k. The horizon is the **last** axis."""
    e = np.asarray(err)
    horizon = e.shape[-1]
    k = horizon if k is None else int(k)
    if not 1 <= k <= horizon:
        raise ValueError(f"k={k} outside the available horizon 1..{horizon}")
    return float(e[..., :k].mean())


def fde(err: np.ndarray, k: int | None = None) -> float:
    """FDE@k: the mean error *at* step k alone. The horizon is the **last** axis."""
    e = np.asarray(err)
    horizon = e.shape[-1]
    k = horizon if k is None else int(k)
    if not 1 <= k <= horizon:
        raise ValueError(f"k={k} outside the available horizon 1..{horizon}")
    return float(e[..., k - 1].mean())


def horizon_curve(errs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-step mean and median of each channel: arrays of length T, ready to plot against step.

    Every leading dimension is treated as a trajectory index and reduced away; the horizon is
    the last axis and is the one that survives.
    """
    out: dict[str, np.ndarray] = {}
    for key, values in errs.items():
        v = np.asarray(values)
        flat = v.reshape(-1, v.shape[-1])
        out[f"{key}_mean"] = flat.mean(axis=0)
        out[f"{key}_median"] = np.median(flat, axis=0)
    out["step"] = np.arange(1, np.asarray(next(iter(errs.values()))).shape[-1] + 1)
    return out


def rollout_metrics(
    gt_seq,
    pred_seq,
    model: ObjectModel | None = None,
    ks: tuple[int, ...] | None = None,
    max_threshold_m: float = 0.1,
) -> dict:
    """Every metric for a set of predicted pose sequences shaped ``(..., T, ...)``.

    `gt_seq` / `pred_seq` are pose arguments in any form :func:`to_Rt` accepts, whose batch
    shape ends in the horizon T. Returns the raw per-frame errors, the pooled summary, the
    per-step curves, and ADE/FDE at each k in `ks` (default: every step).
    """
    errs = pose_errors(gt_seq, pred_seq, model)
    horizon = int(np.asarray(errs["t_err_m"]).shape[-1])
    ks = tuple(range(1, horizon + 1)) if ks is None else tuple(int(k) for k in ks)

    result = {
        "horizon": horizon,
        "errors": errs,
        "summary": summarize(errs, max_threshold_m),
        "curve": horizon_curve(errs),
        "ade": {}, "fde": {},
    }
    for key, values in errs.items():
        result["ade"][key] = {f"k{k}": ade(values, k) for k in ks}
        result["fde"][key] = {f"k{k}": fde(values, k) for k in ks}
    return result
