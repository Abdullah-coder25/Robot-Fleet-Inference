"""Build the side-by-side video, a GIF and the appearance-variants image from a run's output folder.

  python make_video.py --output output
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 30


def font(size):
    for name in ["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def label(arr: np.ndarray, text: str, size=22) -> np.ndarray:
    img = Image.fromarray(arr).convert("RGB")
    d = ImageDraw.Draw(img)
    f = font(size)
    w = d.textlength(text, font=f)
    d.rectangle([10, 10, 26 + w, 18 + size + 6], fill=(0, 0, 0))
    d.text((18, 13), text, fill=(255, 255, 255), font=f)
    return np.asarray(img)


def load(folder: Path):
    return [np.asarray(Image.open(p).convert("RGB")) for p in sorted(folder.glob("*.png"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("output/tabletop"), help="a run folder, e.g. output/agile")
    out = ap.parse_args().output
    sim, depth, cosmos = load(out / "sim_rgb"), load(out / "depth"), load(out / "cosmos_rgb")
    right, right_name = (cosmos, "Cosmos3 Nano Sim Transfer on Reactor") if cosmos else (depth, "Depth sent to Cosmos (dry run)")
    n = min(len(sim), len(right))
    frames = [np.concatenate([label(sim[i], "MuJoCo simulator"), label(right[i], right_name)], axis=1) for i in range(n)]
    imageio.mimsave(out / "side_by_side.mp4", frames, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    # three-panel version showing what is actually sent
    if cosmos:
        tri = [np.concatenate([label(sim[i], "Simulator"), label(depth[i], "Depth sent"), label(cosmos[i], "Cosmos output")], axis=1)[::2, ::2]
               for i in range(n)]
        imageio.mimsave(out / "pipeline.mp4", tri, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    # small GIF for email (every 2nd frame, half size)
    gif = [Image.fromarray(f[::2, ::2]) for f in frames[::2]]
    gif[0].save(out / "side_by_side.gif", save_all=True, append_images=gif[1:], duration=int(2000 / FPS), loop=0, optimize=True)
    # variants grid: one frame per prompt, plus the simulator frame
    vdir = out / "variants"
    if vdir.exists():
        names = sorted(p.name for p in vdir.iterdir() if p.is_dir())
        if names:
            k = min(len(load(vdir / nm)) for nm in names) * 2 // 3
            tiles = [label(sim[k], "Simulator")] + [label(load(vdir / nm)[k], nm.replace("_", " ").title()) for nm in names]
            while len(tiles) % 2:
                tiles.append(np.zeros_like(tiles[0]))
            rows = [np.concatenate(tiles[i:i + 2], axis=1) for i in range(0, len(tiles), 2)]
            Image.fromarray(np.concatenate(rows, axis=0)).save(out / "variants.png")
            vids = [load(vdir / nm) for nm in names]
            m = min(len(v) for v in vids)
            grid = []
            for i in range(m):
                t = [label(sim[i], "Simulator")] + [label(v[i], nm.replace("_", " ").title()) for v, nm in zip(vids, names)]
                while len(t) % 2:
                    t.append(np.zeros_like(t[0]))
                grid.append(np.concatenate([np.concatenate(t[j:j + 2], axis=1) for j in range(0, len(t), 2)], axis=0)[::2, ::2])
            imageio.mimsave(out / "variants.mp4", grid, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    print("wrote", sorted(p.name for p in out.iterdir() if p.is_file()))


if __name__ == "__main__":
    main()
