"""A MuJoCo tabletop scene with a Franka Panda arm and a scripted pick-and-place policy.

The robot model comes from MuJoCo Menagerie (Apache-2.0), fetched by setup_assets().
Everything here is simulator-side: physics, rendering, depth export. Nothing calls Reactor.
"""
from __future__ import annotations

import os  # noqa: F401
import subprocess
from pathlib import Path

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
PANDA_DIR = ASSETS / "mujoco_menagerie" / "franka_emika_panda"

WIDTH, HEIGHT = 832, 480        # Cosmos3 Nano Sim Transfer 16:9 canvas
FPS = 30                        # documented simulation cadence
NEAR, FAR = 0.8, 2.2            # fixed depth bounds for the whole episode (meters)

SCENE = """
<mujoco model="cosmos-transfer tabletop">
  <include file="panda.xml"/>
  <visual>
    <global offwidth="{w}" offheight="{h}"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
  </visual>
  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.55 0.55 0.55 1"/>
    <body name="table" pos="0.55 0 0.2">
      <geom type="box" size="0.35 0.45 0.2" rgba="0.75 0.72 0.68 1"/>
    </body>
    <body name="cube" pos="0.55 -0.15 0.425">
      <freejoint/>
      <geom type="box" size="0.025 0.025 0.025" rgba="0.8 0.2 0.2 1" mass="0.05" friction="1.5 0.05 0.01"/>
    </body>
    <body name="bowl" pos="0.5 0.2 0.405">
      <geom type="cylinder" size="0.07 0.005" rgba="0.2 0.4 0.8 1"/>
    </body>
    <camera name="front" pos="1.6 0.0 1.2" xyaxes="0 1 0 -0.57 0 0.82" fovy="45"/>
  </worldbody>
</mujoco>
"""


def setup_assets() -> Path:
    """Fetch only the Franka folders of MuJoCo Menagerie (sparse git clone)."""
    if (PANDA_DIR / "panda.xml").exists() and (ASSETS / "mujoco_menagerie" / "franka_fr3" / "fr3.xml").exists():
        return PANDA_DIR
    repo = ASSETS / "mujoco_menagerie"
    ASSETS.mkdir(exist_ok=True)
    if (repo / ".git").exists():
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "franka_emika_panda", "franka_fr3"], check=True)
        return PANDA_DIR
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
                    "https://github.com/google-deepmind/mujoco_menagerie", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "franka_emika_panda", "franka_fr3"], check=True)
    return PANDA_DIR


class TabletopSim:
    def __init__(self):
        panda = setup_assets()
        scene = panda / "cosmos_transfer_scene.xml"
        scene.write_text(SCENE.format(w=WIDTH, h=HEIGHT))
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.d = mujoco.MjData(self.m)
        key = self.m.keyframe("home").id
        mujoco.mj_resetDataKeyframe(self.m, self.d, key)
        # the cube's free joint is not in the keyframe; restore its start pose
        self.cube_body = self.m.body("cube").id
        mujoco.mj_forward(self.m, self.d)
        self.hand = self.m.body("hand").id
        self.rgb = mujoco.Renderer(self.m, HEIGHT, WIDTH)
        self.depth = mujoco.Renderer(self.m, HEIGHT, WIDTH)
        self.depth.enable_depth_rendering()
        self.cam = self.m.camera("front").id
        self.substeps = int(round(1.0 / FPS / self.m.opt.timestep))
        self.arm_qpos_adr = [self.m.jnt_qposadr[self.m.joint(f"joint{i}").id] for i in range(1, 8)]
        self.arm_dof_adr = [self.m.jnt_dofadr[self.m.joint(f"joint{i}").id] for i in range(1, 8)]

    # ---- state -------------------------------------------------------------------------
    def hand_pos(self) -> np.ndarray:
        # point between the fingertips, ~0.1 m below the hand frame
        R = self.d.xmat[self.hand].reshape(3, 3)
        return self.d.xpos[self.hand] + R @ np.array([0, 0, 0.103])

    def cube_pos(self) -> np.ndarray:
        return self.d.xpos[self.cube_body].copy()

    # ---- control -----------------------------------------------------------------------
    def ik_targets(self, target: np.ndarray, iters: int = 30) -> np.ndarray:
        """Damped least-squares IK on a copy of the state; returns 7 joint targets."""
        d = mujoco.MjData(self.m)
        d.qpos[:] = self.d.qpos
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        down = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)  # gripper pointing down
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            R = d.xmat[self.hand].reshape(3, 3)
            p = d.xpos[self.hand] + R @ np.array([0, 0, 0.103])
            err_p = target - p
            # orientation error: keep the gripper's z-axis pointing down
            err_r = 0.5 * (np.cross(R[:, 2], down[:, 2]))
            mujoco.mj_jac(self.m, d, jacp, jacr, p, self.hand)
            J = np.vstack([jacp[:, self.arm_dof_adr], jacr[:, self.arm_dof_adr]])
            e = np.concatenate([err_p, err_r])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            for k, adr in enumerate(self.arm_qpos_adr):
                d.qpos[adr] += dq[k]
        return np.array([d.qpos[a] for a in self.arm_qpos_adr])

    def set_action(self, joints: np.ndarray, grip_open: bool):
        self.d.ctrl[:7] = joints
        self.d.ctrl[7] = 255 if grip_open else 0

    def step_frame(self):
        mujoco.mj_step(self.m, self.d, nstep=self.substeps)

    # ---- rendering ---------------------------------------------------------------------
    def render_rgb(self) -> np.ndarray:
        self.rgb.update_scene(self.d, camera=self.cam)
        return self.rgb.render()

    def render_depth_png_array(self) -> np.ndarray:
        """Metric depth -> 8-bit RGB visualization (near = white, far = black), fixed bounds."""
        self.depth.update_scene(self.d, camera=self.cam)
        z = self.depth.render().astype(np.float64)  # meters
        near, far = getattr(self, 'near', NEAR), getattr(self, 'far', FAR)
        z[~np.isfinite(z) | (z <= 0)] = far
        v = (far - np.clip(z, near, far)) / (far - near)
        g = np.round(v * 255).astype(np.uint8)
        return np.repeat(g[:, :, None], 3, axis=2)

    def close(self):
        self.rgb.close(); self.depth.close()


class PickAndPlacePolicy:
    """A scripted policy: reach above the cube, descend, grasp, lift, carry to the bowl, release.

    It reads simulator state, not pixels. A learned policy would read the Cosmos RGB frame
    passed in as `observation`, which is why the loop hands it over every step.
    """

    def __init__(self, sim: TabletopSim, total_frames: int):
        cube = sim.cube_pos()
        bowl = np.array([0.5, 0.2, 0.41])
        start = sim.hand_pos()
        hover, lift = 0.15, 0.18
        # (waypoint, gripper open?, share of the episode to reach it)
        self.plan = [
            (cube + [0, 0, hover], True, 0.22),
            (cube + [0, 0, 0.0], True, 0.14),
            (cube + [0, 0, 0.0], False, 0.08),   # close gripper
            (cube + [0, 0, lift], False, 0.14),
            (bowl + [0, 0, lift], False, 0.18),
            (bowl + [0, 0, 0.06], False, 0.10),
            (bowl + [0, 0, 0.06], True, 0.06),   # release
            (bowl + [0, 0, lift], True, 0.08),
        ]
        self.sim, self.start, self.total = sim, start, total_frames
        bounds, acc = [], 0.0
        for _, _, share in self.plan:
            acc += share
            bounds.append(acc)
        self.bounds = [b / acc for b in bounds]

    def act(self, frame_index: int, observation=None):
        t = frame_index / max(1, self.total - 1)
        prev = self.start
        lo = 0.0
        for (wp, grip_open, _), hi in zip(self.plan, self.bounds):
            if t <= hi:
                a = 0.0 if hi == lo else (t - lo) / (hi - lo)
                a = 0.5 - 0.5 * np.cos(np.pi * min(1.0, a))   # ease in/out
                target = prev + a * (wp - prev)
                return self.sim.ik_targets(target), grip_open
            prev, lo = wp, hi
        return self.sim.ik_targets(self.plan[-1][0]), True
