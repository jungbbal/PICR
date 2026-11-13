"""Panel (b)'s arm images, composed from the Isaac Sim kinematic replay (render/replay_arm.py) of the example window.

Frames layout (assets/frames/): cam.json (the replay camera) and one folder per rollout (gt = recorded, stale, ours)
holding f<k>.png (RGB of frame k, 0 = where the prediction starts) and d<k>.npz / bg_depth.npz (depth to the image
plane of frame k / of the empty scene, float16). Only the frames the figure reads are shipped: gt bg/f012/d000/d012,
stale and ours f012/d012/bg_depth.

Each robot is cut out of its frame against the empty scene's depth; a panel = the empty scene, the start pose as a
faint outline, the recorded end pose in neutral grey, the prediction's end pose tinted in the panel's colour.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

#: the figure's colours: viridis 0.0 for the 0.10 kg belief, the unseen payload's orange
TINT = {"gt": (0.35, 0.35, 0.35), "ours": (0.902, 0.333, 0.051), "stale": (0.267, 0.005, 0.329)}
GT_GREY, GT_EDGE = 0.62, (0.12, 0.12, 0.12)
START_EDGE = (0.45, 0.45, 0.45)


def rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def depth(path: Path) -> np.ndarray:
    return np.load(path)["depth"]


def cut(root: Path, name: str, k: int | None, near: float = 0.01, soft: float = 0.012) -> np.ndarray:
    """How much of each pixel the robot (and its payload) covers, from the depth step over the
    empty scene. `k` None is the empty scene itself."""
    d = depth(root / name / ("bg_depth.npz" if k is None else f"d{k:03d}.npz")).astype(np.float32)
    bg = depth(root / name / "bg_depth.npz").astype(np.float32)
    return np.clip((bg - d - near) / soft, 0.0, 1.0)


def edge(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """The silhouette of a mask, `width` px wide."""
    m = mask > 0.5
    return (m & ~binary_erosion(m, iterations=max(width, 1))).astype(np.float32)


def tinted(frame: np.ndarray, tint, strength: float) -> np.ndarray:
    """Keep the render's shading, push its hue to `tint`."""
    lum = frame.mean(2, keepdims=True)
    return (1 - strength) * frame + strength * (0.42 + 0.78 * lum) * np.asarray(tint)[None, None]


def grey(frame: np.ndarray, level: float) -> np.ndarray:
    """The render as neutral grey: the recorded robot reads as the reference."""
    lum = frame.mean(2, keepdims=True)
    return np.clip(level * (0.45 + 0.9 * lum), 0.0, 1.0).repeat(3, axis=2)


def camera(path: Path):
    """-> (world point -> pixel) for the camera the frames were rendered with."""
    c = json.loads((path / "cam.json").read_text())
    eye, tgt = np.array(c["eye"]), np.array(c["tgt"])
    fwd = tgt - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    fx = c["width"] * c["focal"] / c["aperture"]

    def project(p):
        v = np.atleast_2d(p) - eye
        z = v @ fwd
        return np.c_[c["width"] / 2 + fx * (v @ right) / z, c["height"] / 2 - fx * (v @ up) / z]

    return project, c


def background(root: Path) -> np.ndarray:
    """The empty scene (gt/bg.png) with the table's mounting post (behind the arm) removed.

    Every background pixel is lifted to the world with its depth; the post = what stands above the table top
    (z > 1 cm) plus everything at that post's footprint, filled along the table's back edge from the pixels either
    side; its soft shadow on the top is lifted by a brightness gain (texture kept)."""
    path = root / "gt" / "bg.png"
    img = rgb(path)
    cam = json.loads((root / "cam.json").read_text())
    d = depth(root / "gt" / "bg_depth.npz").astype(np.float32)
    eye, tgt = np.array(cam["eye"]), np.array(cam["tgt"])
    f = (tgt - eye) / np.linalg.norm(tgt - eye)
    r = np.cross(f, [0.0, 0.0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    fx = cam["width"] * cam["focal"] / cam["aperture"]
    vv, uu = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    ray = f + ((uu - d.shape[1] / 2) / fx)[..., None] * r - ((vv - d.shape[0] / 2) / fx)[..., None] * u
    X = eye + d[..., None] * ray
    sky = d >= 999.0
    up = (~sky) & (X[..., 2] > 0.01)
    (x0_, y0_), (x1_, y1_) = X[up][:, :2].min(0) - 0.05, X[up][:, :2].max(0) + 0.05
    foot = ((~sky) & (np.abs(X[..., 2]) > 0.004)            # not the table top itself
            & (X[..., 0] > x0_) & (X[..., 0] < x1_) & (X[..., 1] > y0_ - 0.25) & (X[..., 1] < y1_))   # + its bracket
    post = binary_dilation(up | foot, iterations=4)
    fill = post
    img = img.copy()
    # fill along the table's back edge, not along image rows: the edge is slightly slanted in the image
    c0, c1 = int(np.flatnonzero(fill.any(0)).min()) - 1, int(np.flatnonzero(fill.any(0)).max()) + 1
    nb = [c for c in list(range(c0 - 40, c0 + 1)) + list(range(c1, c1 + 41)) if 0 <= c < img.shape[1]]
    top = np.array([np.flatnonzero(~sky[:, c]).min() for c in nb])   # first table row per neighbouring column
    slope, icpt = np.polyfit(nb, top, 1)                  # the back edge as a line: row = slope * col + icpt
    vv_, cc_ = np.mgrid[0:img.shape[0], 0:img.shape[1]]  # + the post's foot where it meets the top (z ~ 0 there)
    fill = fill | ((cc_ >= c0) & (cc_ <= c1) & (vv_ < slope * cc_ + icpt + 6))
    sky_col = np.median(img[sky & ~fill], axis=0)
    src = img.copy()
    for v, c in zip(*np.nonzero(fill)):
        cl, cr = c, c                                      # nearest kept columns in this row
        while cl > 0 and fill[v, cl]: cl -= 1
        while cr < img.shape[1] - 1 and fill[v, cr]: cr += 1
        w = (c - cl) / max(cr - cl, 1)
        side = []
        for cs in (np.arange(cl - 20, cl + 1), np.arange(cr, cr + 21)):  # median of 21 columns: a screw hole won't smear
            cs = np.clip(cs, 0, img.shape[1] - 1)
            vs = np.clip(np.round(v + slope * (cs - c)).astype(int), 0, img.shape[0] - 1)
            side.append(np.median(src[vs, cs], axis=0))
        t = np.clip(v - (slope * c + icpt) + 0.5, 0.0, 1.0)  # 0 above the edge line, 1 below, one-pixel ramp
        img[v, c] = (1 - t) * sky_col + t * ((1 - w) * side[0] + w * side[1])
    # the post's soft shadow on the table top: brightness only (texture kept). Expected brightness per row = linear
    # between the smoothed top brightness 60 px either side of the post; gain = expected / smoothed actual
    tbl = (~sky) & (np.abs(X[..., 2]) < 0.004) & ~fill
    lum = img.mean(2)
    Ls = gaussian_filter(lum * tbl, 10) / np.maximum(gaussian_filter(tbl.astype(np.float32), 10), 1e-3)
    cl, cr = c0 - 60, c1 + 60
    cc = np.arange(cl, cr + 1)
    E = Ls[:, [cl]] + (Ls[:, [cr]] - Ls[:, [cl]]) * (cc - cl) / (cr - cl)
    g = np.where(tbl[:, cl:cr + 1], E / np.maximum(Ls[:, cl:cr + 1], 1e-3), 1.0)
    img[:, cl:cr + 1] = np.clip(img[:, cl:cr + 1] * g[..., None], 0, 1)
    return img


def build(root: Path, key: str, last: int, final_alpha=0.80, over_gt=0.14, tint=0.78, outline=3):
    """One panel's image and the coverage used to crop every panel the same way. `last` = the frame compared."""
    out = background(root)
    cover = np.zeros(out.shape[:2], np.float32)

    # 1. where the prediction starts: outline only, identical in both panels
    m0 = cut(root, "gt", 0)
    e0 = edge(m0, outline)[:, :, None]
    out = out * (1 - 0.12 * m0[:, :, None]) + np.float32([0.75, 0.75, 0.75]) * (0.12 * m0)[:, :, None]
    out = out * (1 - e0) + np.asarray(START_EDGE)[None, None] * e0
    cover = np.maximum(cover, m0)

    # 2. the recorded pose at the end, in neutral grey with a dark outline
    f, m = rgb(root / "gt" / f"f{last:03d}.png"), cut(root, "gt", last)
    out = out * (1 - m[:, :, None]) + grey(f, GT_GREY) * m[:, :, None]
    cover = np.maximum(cover, m)
    gt_m = m

    # 3. the prediction's last pose in colour, over the grey one
    keep = 1.0 - (1.0 - over_gt) * gt_m
    fk, mk = rgb(root / key / f"f{last:03d}.png"), cut(root, key, last)
    w = (mk * final_alpha * keep)[:, :, None]
    out = out * (1 - w) + tinted(fk, TINT[key], tint) * w
    cover = np.maximum(cover, mk)

    # 4. outlines last, so both final poses keep their shape where they lie on each other
    eg = edge(gt_m, outline)[:, :, None]
    out = out * (1 - eg) + np.asarray(GT_EDGE)[None, None] * eg
    ek = edge(mk, outline)[:, :, None]
    out = out * (1 - ek) + np.asarray(TINT[key])[None, None] * ek
    return np.clip(out, 0, 1), cover
