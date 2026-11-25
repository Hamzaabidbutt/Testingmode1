"""YouTube thumbnail (1280x720) composed from trailer frames.

  python -m tbk.thumbnail media/trailer.mp4 out/thumbnail
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .media import grab_frame, probe
from .style import font

W, H = 1280, 720
AMBER = (255, 196, 64)
WHITE = (250, 247, 240)


def frame(src, t, w):
    info = probe(src)
    h = int(round(w * info["h"] / info["w"]))
    arr = grab_frame(src, t, info["w"], info["h"])  # native size, no padding
    img = Image.fromarray(arr).resize((w, h), Image.LANCZOS)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=70, threshold=2))
    return img


def cover(img, cx, cy, w, h):
    """Crop img to w x h centred on (cx, cy) fractions, scaling up if needed."""
    s = max(w / img.width, h / img.height)
    if s > 1:
        img = img.resize((int(img.width * s + 1), int(img.height * s + 1)), Image.LANCZOS)
    x0 = int(min(max(0, cx * img.width - w / 2), img.width - w))
    y0 = int(min(max(0, cy * img.height - h / 2), img.height - h))
    return img.crop((x0, y0, x0 + w, y0 + h))


def punch(img, contrast=1.18, color=1.25, bright=1.05):
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(color)
    return ImageEnhance.Brightness(img).enhance(bright)


def stroked(d, xy, text, fnt, fill, stroke=8, anchor="ls"):
    d.text(xy, text, font=fnt, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=(8, 8, 10))


def make(src, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Doom in the green cathedral (right) and a shocked Steve (left)
    doom = punch(cover(frame(src, 85.2, 1900), 0.50, 0.42, 900, H), 1.25, 1.35, 1.1)
    steve = punch(cover(frame(src, 45.8, 1900), 0.73, 0.40, 700, H), 1.15, 1.15, 1.12)
    base = Image.new("RGB", (W, H))
    base.paste(doom, (W - 900, 0))
    # diagonal feathered blend from Steve into Doom
    mask = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask)
    md.polygon([(0, 0), (640, 0), (500, H), (0, H)], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(28))
    left = Image.new("RGB", (W, H))
    left.paste(steve, (0, 0))
    base = Image.composite(left, base, mask)
    # darken bottom-left for type, vignette
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    grad = np.clip((ys - H * 0.42) / (H * 0.58), 0, 1) ** 1.3 * np.clip(1.25 - xs / W, 0, 1) * 0.85
    vig = 1 - 0.35 * np.clip(np.sqrt(((xs - W / 2) / (W / 2)) ** 2 + ((ys - H / 2) / (H / 2)) ** 2) - 0.6, 0, 1)
    arr = np.asarray(base, np.float32) * (1 - grad[..., None]) * vig[..., None]
    base = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    # green rim glow seam
    d = ImageDraw.Draw(base)
    # inset: the brightened skyline with the two planets
    # brightened skyline crop that contains both planets
    bx0, by0, bx1, by1 = 0.10, 0.12, 0.80, 0.72          # box in source-frame fractions
    sky = frame(src, 93.4, 1280)
    lut = [int(255 * (v / 255) ** (1 / 2.1)) for v in range(256)] * 3
    sky = ImageEnhance.Contrast(sky.point(lut)).enhance(1.35)
    sky = sky.crop((int(bx0 * sky.width), int(by0 * sky.height), int(bx1 * sky.width), int(by1 * sky.height)))
    iw = 420
    ih = int(iw * sky.height / sky.width)
    sky = sky.resize((iw, ih), Image.LANCZOS)
    ins = Image.new("RGB", (iw + 8, ih + 8), AMBER)
    ins.paste(sky, (4, 4))
    di = ImageDraw.Draw(ins)
    for (x, y, r) in ((0.181, 0.388, 0.05), (0.707, 0.285, 0.06)):
        px = (x - bx0) / (bx1 - bx0) * iw + 4
        py = (y - by0) / (by1 - by0) * ih + 4
        pr = r / (bx1 - bx0) * iw * 1.15
        di.ellipse([px - pr, py - pr, px + pr, py + pr], outline=(255, 60, 60), width=5)
    shadow = Image.new("RGBA", (ins.width + 40, ins.height + 40), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rectangle([20, 20, ins.width + 20, ins.height + 20], fill=(0, 0, 0, 200))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    bx, by = W - ins.width - 36, 34
    base = base.convert("RGBA")
    base.alpha_composite(shadow, (bx - 14, by - 10))
    base = base.convert("RGB")
    base.paste(ins, (bx, by))
    d = ImageDraw.Draw(base)
    stroked(d, (bx + ins.width / 2, by + ins.height + 46), "2 EARTHS?", font("display_black", 44), WHITE, 7, "ms")
    # type
    stroked(d, (52, 470), "STOLEN", font("display_black", 150), WHITE, 10)
    stroked(d, (52, 625), "LIVES", font("display_black", 150), AMBER, 10)
    # pill: 14 DETAILS YOU MISSED
    f = font("display_black", 40)
    txt = "14 DETAILS YOU MISSED"
    tw = d.textlength(txt, font=f)
    px0, py0 = 56, 652
    d.rounded_rectangle([px0, py0, px0 + tw + 44, py0 + 56], radius=12, fill=(220, 30, 40))
    d.text((px0 + 22, py0 + 28), txt, font=f, fill=WHITE, anchor="lm")
    p = os.path.join(out_dir, "thumbnail_stolen_lives.jpg")
    base.save(p, quality=92, optimize=True)
    base.save(os.path.join(out_dir, "thumbnail_stolen_lives.png"))
    return p


if __name__ == "__main__":
    print(make(sys.argv[1], sys.argv[2]))
