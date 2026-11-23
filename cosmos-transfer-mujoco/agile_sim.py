"""Agile Robots scene: a Franka Research 3 (FR3) with a Franka Hand in a small assembly cell.

The FR3 arm and the Franka Hand come from MuJoCo Menagerie (Apache-2.0); the hand is attached
to the FR3 flange with MjSpec. The task: pick two machined bushings from a parts tray and seat
them in the two sockets of a fixture plate.
"""
from __future__ import annotations

import mujoco
import numpy as np

from sim import ASSETS, HEIGHT, WIDTH, TabletopSim, setup_assets

FR3_DIR = ASSETS / "mujoco_menagerie" / "franka_fr3"
HAND_XML = ASSETS / "mujoco_menagerie" / "franka_emika_panda" / "hand.xml"

BENCH_TOP = 0.40
PARTS = [(0.47, -0.22), (0.53, -0.22)]       # bushings in the tray
SOCKETS = [(0.51, 0.17), (0.59, 0.17)]       # sockets in the fixture
PART_R, PART_H = 0.018, 0.022                # bushing radius, half-height
FIX_TOP = BENCH_TOP + 0.03                   # top of the fixture base plate


def _socket_walls(cx, cy, inner=0.024, t=0.006, h=0.014):
    z = FIX_TOP + h
    walls = []
    for dx, dy, sx, sy in [(inner + t, 0, t, inner + 2 * t), (-(inner + t), 0, t, inner + 2 * t),
                           (0, inner + t, inner, t), (0, -(inner + t), inner, t)]:
        walls.append(f'<geom type="box" pos="{cx + dx:.3f} {cy + dy:.3f} {z:.3f}" size="{sx:.3f} {sy:.3f} {h:.3f}" rgba="0.45 0.47 0.5 1"/>')
    return "\n      ".join(walls)


SCENE = f"""
<mujoco model="agile assembly cell">
  <include file="fr3.xml"/>
  <visual>
    <global offwidth="{WIDTH}" offheight="{HEIGHT}"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
  </visual>
  <worldbody>
    <light pos="0.4 0 2.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.5 0.5 0.52 1"/>
    <!-- robot pedestal -->
    <geom type="cylinder" pos="0 0 -0.001" size="0.11 0.001" rgba="0.3 0.3 0.3 1"/>
    <!-- workbench with an aluminium-profile frame behind it -->
    <geom type="box" pos="0.55 0 {BENCH_TOP / 2:.3f}" size="0.30 0.42 {BENCH_TOP / 2:.3f}" rgba="0.35 0.36 0.38 1"/>
    <geom type="box" pos="-0.30 -0.45 0.80" size="0.02 0.02 0.40" rgba="0.7 0.7 0.72 1"/>
    <geom type="box" pos="-0.30 0.45 0.80" size="0.02 0.02 0.40" rgba="0.7 0.7 0.72 1"/>
    <geom type="box" pos="-0.30 0 1.18" size="0.02 0.47 0.02" rgba="0.7 0.7 0.72 1"/>
    <geom type="box" pos="-0.30 0 0.62" size="0.02 0.47 0.015" rgba="0.7 0.7 0.72 1"/>
    <!-- parts tray -->
    <geom type="box" pos="0.50 -0.22 {BENCH_TOP + 0.004:.3f}" size="0.07 0.06 0.004" rgba="0.1 0.1 0.12 1"/>
    <geom type="box" pos="0.50 -0.28 {BENCH_TOP + 0.015:.3f}" size="0.07 0.004 0.011" rgba="0.1 0.1 0.12 1"/>
    <geom type="box" pos="0.50 -0.16 {BENCH_TOP + 0.015:.3f}" size="0.07 0.004 0.011" rgba="0.1 0.1 0.12 1"/>
    <geom type="box" pos="0.43 -0.22 {BENCH_TOP + 0.015:.3f}" size="0.004 0.06 0.011" rgba="0.1 0.1 0.12 1"/>
    <geom type="box" pos="0.57 -0.22 {BENCH_TOP + 0.015:.3f}" size="0.004 0.06 0.011" rgba="0.1 0.1 0.12 1"/>
    <!-- fixture plate with two sockets -->
    <geom type="box" pos="0.55 0.17 {BENCH_TOP + 0.015:.3f}" size="0.10 0.06 0.015" rgba="0.45 0.47 0.5 1"/>
      {_socket_walls(*SOCKETS[0])}
      {_socket_walls(*SOCKETS[1])}
    <!-- two machined bushings -->
    <body name="part1" pos="{PARTS[0][0]} {PARTS[0][1]} {BENCH_TOP + 0.008 + PART_H:.3f}">
      <freejoint/>
      <geom type="cylinder" size="{PART_R} {PART_H}" rgba="0.75 0.76 0.78 1" mass="0.04" friction="1.5 0.05 0.01"/>
    </body>
    <body name="part2" pos="{PARTS[1][0]} {PARTS[1][1]} {BENCH_TOP + 0.008 + PART_H:.3f}">
      <freejoint/>
      <geom type="cylinder" size="{PART_R} {PART_H}" rgba="0.75 0.76 0.78 1" mass="0.04" friction="1.5 0.05 0.01"/>
    </body>
    <camera name="front" pos="1.15 -0.45 1.05" xyaxes="0.569 0.822 -0.000 -0.458 0.317 0.831" fovy="52"/>
  </worldbody>
</mujoco>
"""


class AgileCellSim(TabletopSim):
    """Same rendering, depth export and IK as TabletopSim, with the FR3 + Franka Hand cell."""

    def __init__(self):
        setup_assets()
        scene = FR3_DIR / "cosmos_transfer_agile_cell.xml"
        scene.write_text(SCENE)
        spec = mujoco.MjSpec.from_file(str(scene))
        hand = mujoco.MjSpec.from_file(str(HAND_XML))
        spec.site("attachment_site").attach_body(hand.body("hand"), "hand_", "")
        for k in list(spec.keys):          # the FR3 keyframe predates the hand and parts
            spec.delete(k)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        home = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
        self.arm_qpos_adr = [self.m.jnt_qposadr[self.m.joint(f"fr3_joint{i}").id] for i in range(1, 8)]
        self.arm_dof_adr = [self.m.jnt_dofadr[self.m.joint(f"fr3_joint{i}").id] for i in range(1, 8)]
        for adr, q in zip(self.arm_qpos_adr, home):
            self.d.qpos[adr] = q
        self.d.ctrl[:7] = home
        self.d.ctrl[7] = 255
        mujoco.mj_forward(self.m, self.d)
        self.hand = self.m.body("hand_hand").id
        self.cube_body = self.m.body("part1").id
        self.parts = [self.m.body("part1").id, self.m.body("part2").id]
        self.rgb = mujoco.Renderer(self.m, HEIGHT, WIDTH)
        self.depth = mujoco.Renderer(self.m, HEIGHT, WIDTH)
        self.depth.enable_depth_rendering()
        self.cam = self.m.camera("front").id
        self.near, self.far = 0.45, 1.6   # fixed for the whole episode
        from sim import FPS
        self.substeps = int(round(1.0 / FPS / self.m.opt.timestep))
        # settle the parts in the tray before the episode starts
        for _ in range(200):
            mujoco.mj_step(self.m, self.d)

    def part_pos(self, i: int) -> np.ndarray:
        return self.d.xpos[self.parts[i]].copy()


class AssemblyPolicy:
    """Scripted policy: for each bushing, reach, grasp, lift, carry, insert, release, retract.

    Like PickAndPlacePolicy it reads simulator state; a learned policy would read the Cosmos
    frame handed to it as `observation`.
    """

    def __init__(self, sim: AgileCellSim, total_frames: int):
        start = sim.hand_pos()
        hover = 0.12
        plan = []
        for i, (sx, sy) in enumerate(SOCKETS):
            p = sim.part_pos(i)
            grip = p + [0, 0, 0.004]
            seat = np.array([sx, sy, FIX_TOP + PART_H + 0.006])
            plan += [
                (grip + [0, 0, hover], True, 0.10),
                (grip, True, 0.07),
                (grip, False, 0.04),                    # close the hand
                (grip + [0, 0, hover], False, 0.07),
                (seat + [0, 0, hover], False, 0.09),
                (seat, False, 0.07),                    # insert
                (seat, True, 0.03),                     # release
                (seat + [0, 0, hover], True, 0.05),
            ]
        self.plan, self.sim, self.start, self.total = plan, sim, start, total_frames
        acc, bounds = 0.0, []
        for _, _, share in plan:
            acc += share
            bounds.append(acc)
        self.bounds = [b / acc for b in bounds]

    def act(self, frame_index: int, observation=None):
        t = frame_index / max(1, self.total - 1)
        prev, lo = self.start, 0.0
        for (wp, grip_open, _), hi in zip(self.plan, self.bounds):
            if t <= hi:
                a = 0.0 if hi == lo else (t - lo) / (hi - lo)
                a = 0.5 - 0.5 * np.cos(np.pi * min(1.0, a))
                return self.sim.ik_targets(prev + a * (wp - prev)), grip_open
            prev, lo = wp, hi
        return self.sim.ik_targets(self.plan[-1][0]), True
