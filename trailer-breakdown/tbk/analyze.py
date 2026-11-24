"""Trailer analysis helpers: shot detection, contact sheets, gridded frame grabs.

  python -m tbk.analyze shots  media/trailer.mp4  -o work/analysis
  python -m tbk.analyze sheets media/trailer.mp4  -o work/analysis
  python -m tbk.analyze grab   media/trailer.mp4  83.4 [more times...] -o work/analysis
  python -m tbk.analyze audio  media/trailer.mp4  -o work/analysis   (loudness map)
  python -m tbk.analyze strip  media/trailer.mp4  64 84 0.5 -o work/analysis  (frames every 0.5 s)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from .media import canvas_size, grab_frame, probe, read_audio
from .style import font
from .util import fmt_tc


def detect_cuts(src: str, thresh: float = 0.28) -> list[float]:
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", src, "-an", "-vf",
                        f"select='gt(scene,{thresh})',showinfo", "-f", "null", "-"],
                       capture_output=True, text=True)
    return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", p.stderr)]


def shots(src: str, thresh=0.28, min_len=0.25):
    info = probe(src)
    cuts = [c for c in detect_cuts(src, thresh)]
    bounds = [0.0] + cuts + [info["duration"]]
    out = []
    for a, b in zip(bounds, bounds[1:]):
        if out and b - a < min_len:
            out[-1]["end"] = b
            continue
        out.append({"start": round(a, 3), "end": round(b, 3)})
    for i, s in enumerate(out):
        s["id"] = i + 1
        s["dur"] = round(s["end"] - s["start"], 3)
    return out


def _thumbs(src, fps=6, w=480):
    info = probe(src)
    h = int(round(w * info["h"] / info["w"] / 2) * 2)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-an", "-vf", f"fps={fps},scale={w}:{h}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return frames, fps


def sheets(src, outdir, shot_list, cols=3, per_sheet=18):
    """One row per shot: start / middle / end frames, labelled."""
    frames, fps = _thumbs(src)
    h, w = frames.shape[1:3]
    f_lab = font("mono_bold", 20)
    pad, lab_h = 8, 30
    paths = []
    for si in range(0, len(shot_list), per_sheet):
        chunk = shot_list[si:si + per_sheet]
        rows = len(chunk)
        sheet = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + lab_h + pad) + pad), (16, 16, 18))
        d = ImageDraw.Draw(sheet)
        for r, s in enumerate(chunk):
            ts = [s["start"] + 0.08, (s["start"] + s["end"]) / 2, max(s["start"], s["end"] - 0.1)]
            for c, t in enumerate(ts[:cols]):
                idx = min(len(frames) - 1, int(t * fps))
                x, y = pad + c * (w + pad), pad + r * (h + lab_h + pad)
                sheet.paste(Image.fromarray(frames[idx]), (x, y + lab_h))
                lab = f"#{s['id']:03d} {fmt_tc(t, 1)}" if c else f"#{s['id']:03d} {fmt_tc(t, 1)}  ({s['dur']:.1f}s)"
                d.text((x + 2, y + 4), lab, font=f_lab, fill=(232, 176, 74) if c == 0 else (200, 200, 200))
        p = os.path.join(outdir, f"sheet_{si // per_sheet + 1:02d}.jpg")
        sheet.save(p, quality=88)
        paths.append(p)
    return paths


def grid_grab(src, t, outdir, out_h=1080, step=0.05):
    """Full canvas frame with a normalised coordinate grid, for placing highlights."""
    info = probe(src)
    cw, ch = canvas_size(info["h"], out_h)
    arr = grab_frame(src, t, cw, ch)
    img = Image.fromarray(arr).resize((1920, 1080), Image.LANCZOS)
    d = ImageDraw.Draw(img, "RGBA")
    f = font("mono_bold", 16)
    n = int(round(1 / step))
    for i in range(1, n):
        v = i * step
        major = i % 2 == 0
        col = (255, 255, 0, 150) if major else (255, 255, 0, 60)
        d.line([(v * 1920, 0), (v * 1920, 1080)], fill=col, width=1)
        d.line([(0, v * 1080), (1920, v * 1080)], fill=col, width=1)
        if major:
            d.text((v * 1920 + 3, 3), f"{v:.1f}", font=f, fill=(255, 255, 0, 230))
            d.text((3, v * 1080 + 3), f"{v:.1f}", font=f, fill=(255, 255, 0, 230))
    d.text((1700, 1050), fmt_tc(t), font=f, fill=(255, 255, 255, 230))
    p = os.path.join(outdir, f"grab_{t:07.2f}.jpg")
    img.save(p, quality=90)
    return p


def loudness_map(src, hop=0.25):
    a = read_audio(src)
    m = a.mean(axis=1)
    n = int(48000 * hop)
    k = len(m) // n
    rms = np.sqrt((m[:k * n].reshape(k, n) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    return [(round(i * hop, 2), round(float(v), 1)) for i, v in enumerate(db)]




def strip(src, t0, t1, step, outdir, cols=6, w=320):
    """Grid of frames every `step` seconds between t0 and t1."""
    info = probe(src)
    h = int(round(w * info["h"] / info["w"] / 2) * 2)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t0), "-t", str(t1 - t0), "-i", src, "-an", "-vf",
                          f"fps=1/{step},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    rows = (len(frames) + cols - 1) // cols
    lab = 22
    sheet = Image.new("RGB", (cols * (w + 4) + 4, rows * (h + lab + 4) + 4), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    f = font("mono_bold", 15)
    for i, fr in enumerate(frames):
        x, y = 4 + (i % cols) * (w + 4), 4 + (i // cols) * (h + lab + 4)
        sheet.paste(Image.fromarray(fr), (x, y + lab))
        d.text((x + 2, y + 3), fmt_tc(t0 + i * step, 1), font=f, fill=(232, 176, 74))
    p = os.path.join(outdir, f"strip_{t0:06.1f}_{t1:06.1f}.jpg")
    sheet.save(p, quality=88)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["shots", "sheets", "grab", "audio", "strip"])
    ap.add_argument("src")
    ap.add_argument("times", nargs="*", type=float)
    ap.add_argument("-o", "--out", default="work/analysis")
    ap.add_argument("--thresh", type=float, default=0.28)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    shots_path = os.path.join(a.out, "shots.json")
    if a.cmd == "shots" or (a.cmd == "sheets" and not os.path.exists(shots_path)):
        sl = shots(a.src, a.thresh)
        json.dump(sl, open(shots_path, "w"), indent=1)
        print(f"{len(sl)} shots -> {shots_path}")
    if a.cmd == "sheets":
        sl = json.load(open(shots_path))
        for p in sheets(a.src, a.out, sl):
            print(p)
    elif a.cmd == "grab":
        for t in a.times:
            print(grid_grab(a.src, t, a.out))
    elif a.cmd == "strip":  # times: t0 t1 [step]
        t0, t1 = a.times[0], a.times[1]
        print(strip(a.src, t0, t1, a.times[2] if len(a.times) > 2 else 0.5, a.out))
    elif a.cmd == "audio":
        lm = loudness_map(a.src)
        json.dump(lm, open(os.path.join(a.out, "loudness.json"), "w"))
        for t, v in lm:
            if int(t * 4) % 4 == 0:
                print(f"{fmt_tc(t, 0)} {'#' * max(0, int((v + 60) / 2))} {v}")

if __name__ == "__main__":
    main()
