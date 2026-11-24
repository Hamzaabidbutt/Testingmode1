"""Visual identity: palette, fonts and typography helpers.

Everything is sized relative to a 1080-line frame and scaled by `Style.k`, so a
4K render keeps identical proportions.
"""
from __future__ import annotations

import functools
import subprocess

from PIL import ImageFont

FONT_QUERIES = {
    "display_black": "Inter Display:style=Black",
    "display_bold": "Inter Display:style=Bold",
    "display_semi": "Inter Display:style=SemiBold",
    "display_light": "Inter Display:style=Light",
    "sans": "Inter:style=Regular",
    "sans_medium": "Inter:style=Medium",
    "sans_semi": "Inter:style=SemiBold",
    "mono": "JetBrains Mono:style=Regular",
    "mono_bold": "JetBrains Mono:style=Bold",
    "serif_italic": "Caladea:style=Italic",
}

DEFAULT_PALETTE = {
    "bg": "#0A0C0F",          # near-black, slightly cool
    "bg2": "#12161B",
    "text": "#EEEAE3",        # warm off-white
    "muted": "#9AA0A8",
    "accent": "#E8B04A",      # amber: the channel colour
    "confirmed": "#4FD1C5",   # teal: on-screen / officially confirmed
    "theory": "#E8B04A",      # amber: evidence-backed theory
    "speculation": "#E5484D", # red: speculation, low confidence
    "hud": "#CFE3F2",
}


@functools.lru_cache(maxsize=None)
def font_file(key: str) -> str:
    q = FONT_QUERIES.get(key, key)
    try:
        return subprocess.run(["fc-match", "-f", "%{file}", q], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # pragma: no cover - fall back to PIL default
        return "DejaVuSans.ttf"


@functools.lru_cache(maxsize=512)
def font(key: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_file(key), max(6, int(size)))


class Style:
    def __init__(self, W: int, H: int, palette: dict | None = None):
        self.W, self.H = W, H
        self.k = H / 1080.0
        self.pal = dict(DEFAULT_PALETTE)
        if palette:
            self.pal.update(palette)

    def px(self, v: float) -> int:
        return int(round(v * self.k))

    def f(self, key: str, size: float):
        return font(key, self.px(size))

    def c(self, name: str):
        from .util import rgb
        return rgb(self.pal.get(name, name))


# ---------------------------------------------------------------- tracked text


def text_width(fnt, text: str, tracking: float = 0.0) -> float:
    """Width of text with extra letter spacing (`tracking` in px)."""
    if not text:
        return 0.0
    return sum(fnt.getlength(ch) for ch in text) + tracking * (len(text) - 1)


def draw_tracked(draw, xy, text, fnt, fill, tracking: float = 0.0, anchor="ls"):
    """Draw text char-by-char with letter spacing. anchor: l/m/r + s(baseline)/m."""
    x, y = xy
    w = text_width(fnt, text, tracking)
    if anchor[0] == "m":
        x -= w / 2
    elif anchor[0] == "r":
        x -= w
    va = anchor[1] if len(anchor) > 1 else "s"
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill, anchor="l" + va)
        x += fnt.getlength(ch) + tracking
    return w


def wrap(fnt, text: str, max_w: float) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if fnt.getlength(t) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
