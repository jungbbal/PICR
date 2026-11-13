"""Kinematic replay of the latent figure's example window (Isaac Lab): one env per rollout (recorded / adapted z /
stale z), the same camera in each, so the frames are pixel-aligned and can be stacked into one composite on the host.

Each --traj npz (q [T,9], obj_pos [T,3] robot-root frame, obj_quat [T,4] wxyz) gets its own env.
Nothing is simulated: every frame writes the joint state and the cube pose (kinematic, collision
off, gravity off). Before the sequence, the robot and the cube are dropped 5 m out of sight and one
frame per env is saved as `bg.png` (+ bg_depth.npy): the host uses it to cut the robot out of each frame.

Run with Isaac Lab's python (see README.md for the exact camera used by the figure):
  python.sh replay_arm.py --headless --enable_cameras --traj gt.npz ours.npz stale.npz --out frames
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--traj", nargs="+", required=True)
p.add_argument("--out", required=True)
p.add_argument("--stride", type=int, default=1)
p.add_argument("--width", type=int, default=1400)
p.add_argument("--height", type=int, default=1200)
p.add_argument("--focal", type=float, default=30.0)
p.add_argument("--env_spacing", type=float, default=8.0)
p.add_argument("--drop_ground", action="store_true", help="put the ground plane 50 m down: no horizon")
p.add_argument("--drop_table", action="store_true", help="put the table 50 m down as well")
p.add_argument("--eye", type=float, nargs=3, default=[1.18, -1.52, 0.74])
p.add_argument("--tgt", type=float, nargs=3, default=[0.42, -0.12, 0.38])
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import FRANKA_PANDA_CFG

DT, DEV = 0.02, "cuda:0"
names = [os.path.splitext(os.path.basename(t))[0] for t in a.traj]
trajs = [np.load(t) for t in a.traj]
N = len(trajs)
T = min(len(t["q"]) for t in trajs)


@configclass
class Scene(InteractiveSceneCfg):
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, -50.0 * a.drop_table], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    plane = AssetBaseCfg(prim_path="/World/GroundPlane",
                         init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05 - 50.0 * a.drop_ground]),
                         spawn=GroundPlaneCfg())
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    dome = AssetBaseCfg(prim_path="/World/Light",
                        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.92, 0.95, 1.0)))
    key = AssetBaseCfg(prim_path="/World/Key",
                       spawn=sim_utils.DistantLightCfg(intensity=2500.0, angle=2.0, color=(1.0, 0.97, 0.92)),
                       init_state=AssetBaseCfg.InitialStateCfg(rot=(0.85, 0.30, 0.38, 0.20)))


sim = SimulationContext(SimulationCfg(dt=DT, device=DEV, gravity=(0.0, 0.0, 0.0)))
scene = InteractiveScene(Scene(num_envs=N, env_spacing=a.env_spacing))

cams = {i: Camera(CameraCfg(
    prim_path=f"/World/Cam_{i}", update_period=0, height=a.height, width=a.width, data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=a.focal, focus_distance=1.0,
                                     horizontal_aperture=20.955, clipping_range=(0.01, 60.0)),
)) for i in range(N)}
sim.reset()

robot, cube = scene["robot"], scene["cube"]
origins = scene.env_origins
Q = torch.tensor(np.stack([t["q"][:T] for t in trajs]), device=DEV, dtype=torch.float32)
OP = torch.tensor(np.stack([t["obj_pos"][:T] for t in trajs]), device=DEV, dtype=torch.float32)
OQ = torch.tensor(np.stack([t["obj_quat"][:T] for t in trajs]), device=DEV, dtype=torch.float32)
zeros = torch.zeros(N, Q.shape[2], device=DEV)
for n in names:
    os.makedirs(os.path.join(a.out, n), exist_ok=True)
# the camera, so the host can project a point of the scene (a cube centre) onto these pixels
import json
with open(os.path.join(a.out, "cam.json"), "w") as fh:
    json.dump({"eye": list(map(float, a.eye)), "tgt": list(map(float, a.tgt)), "focal": float(a.focal),
               "aperture": 20.955, "width": int(a.width), "height": int(a.height),
               "names": names, "stride": int(a.stride)}, fh, indent=2)

EYE = torch.tensor(a.eye, device=DEV)
TGT = torch.tensor(a.tgt, device=DEV)


def aim():
    for i in range(N):
        o = origins[i]
        cams[i].set_world_poses_from_view((o + EYE)[None], (o + TGT)[None])


def set_frame(k):
    robot.write_joint_state_to_sim(Q[:, k], zeros)
    robot.set_joint_position_target(Q[:, k])
    pose = torch.cat([OP[:, k] + origins, OQ[:, k]], dim=1)
    cube.write_root_pose_to_sim(pose)
    cube.write_root_velocity_to_sim(torch.zeros(N, 6, device=DEV))
    scene.write_data_to_sim()


def shoot():
    sim.step()
    for c in cams.values():
        c.update(dt=DT)


def grab(i):
    rgb = cams[i].data.output["rgb"][0].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    return rgb[:, :, :3]


def depth(i):
    """Distance to the image plane: the mask the host cuts with, free of the shadows an rgb
    difference also picks up."""
    d = cams[i].data.output["distance_to_image_plane"][0].detach().cpu().numpy()
    return np.nan_to_num(np.squeeze(d), nan=1e3, posinf=1e3).astype(np.float16)


set_frame(0)
aim()
for _ in range(60):  # RTX warm-up
    sim.step()
    scene.update(DT)

# the empty scene: the table and the light without the robot, so the host can cut the robot out
root = robot.data.root_state_w[:, :7].clone()
away = root.clone()
away[:, 2] -= 5.0
robot.write_root_pose_to_sim(away)
cube.write_root_pose_to_sim(torch.cat([OP[:, 0] + origins - torch.tensor([0.0, 0.0, 5.0], device=DEV),
                                       OQ[:, 0]], dim=1))
scene.write_data_to_sim()
sim.forward()
scene.update(DT)
for _ in range(4):
    shoot()
for i, n in enumerate(names):
    Image.fromarray(grab(i)).save(os.path.join(a.out, n, "bg.png"))
    np.save(os.path.join(a.out, n, "bg_depth.npy"), depth(i))
robot.write_root_pose_to_sim(root)
scene.write_data_to_sim()
print("[replay_arm] background written", flush=True)

m = 0
for k in range(0, T, a.stride):
    set_frame(k)
    sim.forward()
    scene.update(DT)
    aim()
    shoot()
    for i, n in enumerate(names):
        Image.fromarray(grab(i)).save(os.path.join(a.out, n, f"f{m:03d}.png"))
        np.save(os.path.join(a.out, n, f"d{m:03d}.npy"), depth(i))
    m += 1
print(f"[replay_arm] wrote {m} frames x {N} envs ({', '.join(names)}) to {a.out}", flush=True)
import threading
threading.Timer(30.0, lambda: os._exit(0)).start()   # app.close() hangs after the frames are written
app.close()
os._exit(0)
