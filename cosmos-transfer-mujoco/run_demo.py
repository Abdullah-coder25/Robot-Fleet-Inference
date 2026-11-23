"""Closed-loop MuJoCo -> NVIDIA Cosmos3 Nano Sim Transfer on Reactor.

Every step: the policy chooses actions, MuJoCo advances four frames and records depth,
Reactor's Cosmos Transfer turns that depth into photoreal RGB, and the newest RGB frame
becomes the policy's next observation. This follows Reactor's documented
initialize / advance loop for connecting a simulator.

  python run_demo.py                 # live run (needs REACTOR_API_KEY)
  python run_demo.py --dry-run       # simulator + depth only, no API calls, no cost
  python run_demo.py --variants      # replay the recorded depth with other appearance prompts
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from sim import FPS, HEIGHT, WIDTH, PickAndPlacePolicy, TabletopSim

PRICE_PER_MIN = 0.102  # published Cosmos3 Nano Sim Transfer price, per session-minute
MODEL = "reactor/cosmos3-nano-sim-transfer"

MAIN_PROMPT = (
    "A real Franka Panda robot arm in a bright robotics lab picks up a small red cube from a "
    "light wooden table and places it on a blue plate. Realistic lighting, soft shadows, "
    "photorealistic camera footage."
)
# Agile Robots scene (FR3 + Franka Hand, assembly cell). Objects and colors are named in every
# prompt so the model keeps them; only the conditions change between variants.
AGILE_BASE = (
    "a white Franka Research 3 robot arm with a white two-finger Franka Hand in a factory assembly "
    "cell, picking small machined aluminum bushings from a black ESD parts tray and pressing them "
    "into the sockets of a gray anodized steel fixture on a dark gray steel workbench, with an "
    "aluminum profile frame behind the robot"
)
AGILE_PROMPT = f"Photorealistic camera footage of {AGILE_BASE}. Bright, even industrial LED lighting."
AGILE_VARIANTS = {
    "night_shift_low_light": f"Photorealistic camera footage of {AGILE_BASE}. Night shift: the cell is dim, "
                             "lit by a single overhead lamp, with deep shadows across the workbench.",
    "skylight_glare": f"Photorealistic camera footage of {AGILE_BASE}. Strong sunlight from a skylight causes "
                      "bright glare and hard reflections on the metal parts and fixture.",
    "cluttered_bench": f"Photorealistic camera footage of {AGILE_BASE}. A busy cell: cables, hand tools, "
                       "cardboard packaging and spare parts are scattered around the edges of the workbench.",
    "brass_parts_blue_fixture": f"Photorealistic camera footage of {AGILE_BASE.replace('aluminum bushings', 'brass bushings').replace('gray anodized steel fixture', 'blue anodized aluminum fixture')}. "
                                "Bright industrial lighting.",
}

VARIANT_PROMPTS = {
    "warehouse": "An industrial robot arm on a steel workbench in a busy warehouse with shelving "
                 "and cardboard boxes in the background, overhead fluorescent lighting, realistic footage.",
    "kitchen": "A robot arm on a marble kitchen counter next to a window with morning sunlight, "
               "a red cube and a blue plate on the counter, realistic home footage.",
    "lab_night": "A robot arm in a dim research lab at night, lit by cool blue monitor light, "
                 "a red cube and a blue plate on a dark table, realistic footage.",
}


def png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class CosmosTransfer:
    """Thin wrapper over the stock Reactor Python SDK session commands."""

    def __init__(self, timeout: float = 180):
        from reactor_sdk import Reactor  # imported here so --dry-run needs no SDK
        key = os.environ.get("REACTOR_API_KEY")
        if not key:
            raise SystemExit("Set REACTOR_API_KEY (or use --dry-run).")
        self.client = Reactor(model_name=os.environ.get("REACTOR_MODEL", MODEL), api_url="https://api.reactor.inc",
                              api_key=key, max_session_duration_seconds=600)
        self.timeout = timeout
        self.results: asyncio.Queue = asyncio.Queue()
        self.episode_id = None

        @self.client.on_message
        def _on_message(message):
            if message.get("type") in ("step_result", "step_failed"):
                self.results.put_nowait(message)

    async def __aenter__(self):
        await self.client.__aenter__()
        await self.client.connect()
        self.opened = time.monotonic()
        return self

    async def __aexit__(self, *exc):
        try:
            await self.client.disconnect()
        finally:
            await self.client.__aexit__(*exc)
        self.session_seconds = time.monotonic() - self.opened

    async def _receive(self, step_index: int):
        async with asyncio.timeout(self.timeout):
            while True:
                msg = await self.results.get()
                data = msg["data"]
                if data["episode_id"] != self.episode_id or data["step_index"] != step_index:
                    continue  # stale or duplicate reply
                if msg["type"] == "step_failed":
                    raise RuntimeError(f"step_failed: {data}")
                return data

    @staticmethod
    def _decode(data, expected_count: int):
        frames = []
        if len(data["frames_png_b64"]) != expected_count:
            raise ValueError("Unexpected number of frames returned")
        for b64, sha in zip(data["frames_png_b64"], data["sha256"], strict=True):
            raw = base64.b64decode(b64, validate=True)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            if hashlib.sha256(img.tobytes()).hexdigest() != sha:
                raise ValueError("Returned RGB frame hash mismatch")
            frames.append(np.asarray(img))
        return frames, data.get("complete", False)

    async def initialize(self, depth0: np.ndarray, prompt: str, total_frames: int, seed: int):
        accepted = await self.client.send_command("initialize", {
            "depth_png_b64": png_b64(depth0), "prompt": prompt, "total_frames": total_frames,
            "aspect_ratio": "16,9", "seed": seed, "preview": False})
        self.episode_id = accepted["data"]["episode_id"]
        return self._decode(await self._receive(0), 1)

    async def advance(self, step_index: int, depths: list[np.ndarray]):
        await self.client.send_command("advance", {
            "episode_id": self.episode_id, "step_index": step_index,
            "depth_png_b64": [png_b64(d) for d in depths]})
        return self._decode(await self._receive(step_index), 4)


def save(frames, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(folder / f"{i:06d}.png")


async def closed_loop(args, out: Path):
    if args.scene == "agile":
        from agile_sim import AgileCellSim, AssemblyPolicy
        sim = AgileCellSim()
        policy = AssemblyPolicy(sim, args.frames)
    else:
        sim = TabletopSim()
        policy = PickAndPlacePolicy(sim, args.frames)
    sim_rgb, depth, cosmos, timings = [sim.render_rgb()], [sim.render_depth_png_array()], [], []

    async def run(cosmos_client):
        obs = None
        if cosmos_client:
            t = time.monotonic()
            first, _ = await cosmos_client.initialize(depth[0], args.prompt, args.frames, args.seed)
            timings.append({"step": 0, "frames": 1, "seconds": round(time.monotonic() - t, 3)})
            cosmos.extend(first); obs = first[-1]
        for step in range(1, 1 + (args.frames - 1) // 4):
            new_depth = []
            for _ in range(4):  # simulation time pauses while Cosmos renders
                joints, grip = policy.act(len(sim_rgb), observation=obs)
                sim.set_action(joints, grip)
                sim.step_frame()
                sim_rgb.append(sim.render_rgb())
                d = sim.render_depth_png_array()
                depth.append(d); new_depth.append(d)
            if cosmos_client:
                t = time.monotonic()
                frames, complete = await cosmos_client.advance(step, new_depth)
                timings.append({"step": step, "frames": 4, "seconds": round(time.monotonic() - t, 3)})
                cosmos.extend(frames); obs = frames[-1]
                print(json.dumps(timings[-1]), flush=True)
        if cosmos_client and not complete:
            raise RuntimeError("Episode did not report complete")

    if args.dry_run:
        await run(None)
        session = 0.0
    else:
        async with CosmosTransfer() as c:
            await run(c)
        session = c.session_seconds
    sim.close()
    save(sim_rgb, out / "sim_rgb"); save(depth, out / "depth")
    if cosmos:
        save(cosmos, out / "cosmos_rgb")
    report = {
        "frames": args.frames, "fps": FPS, "canvas": f"{WIDTH}x{HEIGHT}", "seed": args.seed,
        "scene": args.scene, "prompt": args.prompt, "dry_run": args.dry_run,
        "session_seconds": round(session, 1),
        "estimated_cost_usd": round(session / 60 * PRICE_PER_MIN, 3),
        "step_seconds_median": (sorted(t["seconds"] for t in timings[1:])[len(timings[1:]) // 2]
                                if len(timings) > 1 else None),
        "timings": timings,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "timings"}, indent=2))
    return sim_rgb, depth, cosmos


async def variants(args, out: Path):
    """Replay the first N recorded depth frames under different appearance prompts."""
    depth_dir = out / "depth"
    paths = sorted(depth_dir.glob("*.png"))[: args.variant_frames]
    if len(paths) < args.variant_frames:
        raise SystemExit("Run the main demo first (it records the depth frames).")
    depths = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    summary = {}
    prompts = AGILE_VARIANTS if args.scene == "agile" else VARIANT_PROMPTS
    for name, prompt in prompts.items():
        frames = []
        async with CosmosTransfer() as c:
            first, _ = await c.initialize(depths[0], prompt, args.variant_frames, args.seed)
            frames += first
            for step in range(1, 1 + (args.variant_frames - 1) // 4):
                got, _ = await c.advance(step, depths[1 + 4 * (step - 1): 1 + 4 * step])
                frames += got
        save(frames, out / "variants" / name)
        summary[name] = {"prompt": prompt, "session_seconds": round(c.session_seconds, 1),
                         "estimated_cost_usd": round(c.session_seconds / 60 * PRICE_PER_MIN, 3)}
        print(name, summary[name], flush=True)
    (out / "variants" / "report.json").write_text(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", choices=["tabletop", "agile"], default="tabletop",
                   help="tabletop: Franka Panda pick-and-place; agile: FR3 + Franka Hand assembly cell")
    p.add_argument("--frames", type=int, default=None, help="1 + 4N, up to 597 (default 161 tabletop, 281 agile)")
    p.add_argument("--prompt", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, default=None, help="default: output/<scene>")
    p.add_argument("--dry-run", action="store_true", help="no API calls; simulator and depth only")
    p.add_argument("--variants", action="store_true", help="render appearance variants from recorded depth")
    p.add_argument("--variant-frames", type=int, default=None, help="default 61 tabletop, 121 agile")
    args = p.parse_args()
    if args.frames is None:
        args.frames = 281 if args.scene == "agile" else 161
    if args.prompt is None:
        args.prompt = AGILE_PROMPT if args.scene == "agile" else MAIN_PROMPT
    if args.output is None:
        args.output = Path("output") / args.scene
    if args.variant_frames is None:
        args.variant_frames = 121 if args.scene == "agile" else 61
    if not (1 <= args.frames <= 597 and (args.frames - 1) % 4 == 0):
        raise SystemExit("--frames must be 1 + 4N and at most 597")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.variants:
        asyncio.run(variants(args, args.output))
    else:
        asyncio.run(closed_loop(args, args.output))


if __name__ == "__main__":
    main()
