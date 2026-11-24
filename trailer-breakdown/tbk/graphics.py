"""Virtual camera + motion-graphics overlays.

Coordinates
-----------
* canvas space: (x, y) in 0..1 of the *source frame* (what the trailer shows).
  Highlights given in canvas space stay pinned to the object while the camera
  zooms/pans, because they are mapped through the current View.
* screen space: (x, y) in 0..1 of the output frame (lower thirds, HUD, text).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

from .style import Style, draw_tracked, text_width, wrap
from .util import clamp, ease, env, interp_keys, lerp, norm_keys, parse_time, rgb

# ====================================================================== camera


class View:
    """Maps canvas-normalised coords to output pixels for one frame."""

    def __init__(self, box, cw, ch, rect):
        self.box = box                      # (x0, y0, x1, y1) in canvas px
        self.cw, self.ch = cw, ch
        self.rect = rect                    # (ox, oy, w, h) screen rect in output px
        self.s = rect[2] / (box[2] - box[0])  # output px per canvas px

    @property
    def zoom(self):
        return self.cw / (self.box[2] - self.box[0])

    def pt(self, x, y):
        X = self.rect[0] + (x * self.cw - self.box[0]) * self.s
        Y = self.rect[1] + (y * self.ch - self.box[1]) * self.s
        return X, Y

    def ln(self, r):
        """canvas-width fraction -> output px."""
        return r * self.cw * self.s


class Camera:
    FIELDS = {"x": 0.5, "y": 0.5, "z": 1.0}

    def __init__(self, keys, seg_dur, cw, ch, shake=None):
        self.keys = norm_keys(keys or [{"t": 0}], seg_dur, self.FIELDS)
        self.cw, self.ch = cw, ch
        self.shake = []
        for s in shake or []:
            s = dict(s)
            s["start"] = parse_time(s.get("start", 0), seg_dur)
            s["dur"] = float(s.get("dur", 0.6))
            s["amp"] = float(s.get("amp", 0.012))
            self.shake.append(s)

    def box(self, t):
        k = interp_keys(self.keys, t, ("x", "y", "z"))
        z = max(1.0, k["z"])
        sx = sy = 0.0
        for s in self.shake:
            u = (t - s["start"]) / s["dur"]
            if 0 <= u <= 1:
                a = s["amp"] * (1 - u) ** 2
                sx += a * (math.sin(t * 71.3) * 0.6 + math.sin(t * 37.9 + 1.3) * 0.4)
                sy += a * (math.sin(t * 59.1 + 0.7) * 0.6 + math.sin(t * 43.3 + 2.1) * 0.4)
                z = max(z, 1.0 + 2.2 * s["amp"])
        bw, bh = self.cw / z, self.ch / z
        x0 = clamp(k["x"] * self.cw - bw / 2 + sx * self.cw, 0, self.cw - bw)
        y0 = clamp(k["y"] * self.ch - bh / 2 + sy * self.ch, 0, self.ch - bh)
        return (x0, y0, x0 + bw, y0 + bh)


def render_view(frame: np.ndarray, box, size) -> Image.Image:
    """Sub-pixel accurate crop+resize (no zoompan jitter)."""
    img = Image.fromarray(frame)
    return img.resize(size, Image.BICUBIC, box=box, reducing_gap=None)


# ====================================================================== stamping


def stamp(layer: Image.Image, bbox, fn, ss=3, glow=0.0, glow_alpha=0.55, opacity=1.0):
    """Draw `fn(draw, T, ss)` supersampled into bbox and composite onto layer.

    T(x, y) maps output px -> tile px. Gives anti-aliased strokes cheaply.
    """
    pad = int(math.ceil(glow * 3)) + 2
    if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > 250_000 and ss > 2:
        ss = 2  # big tiles: 2x supersampling is plenty and 2.25x cheaper
    x0 = int(math.floor(bbox[0])) - pad
    y0 = int(math.floor(bbox[1])) - pad
    x1 = int(math.ceil(bbox[2])) + pad
    y1 = int(math.ceil(bbox[3])) + pad
    W, H = layer.size
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if cx1 <= cx0 or cy1 <= cy0 or opacity <= 0.003:
        return
    w, h = x1 - x0, y1 - y0
    tile = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)

    def T(x, y):
        return ((x - x0) * ss, (y - y0) * ss)

    fn(d, T, ss)
    tile = tile.reduce(ss) if ss > 1 else tile  # exact box-filter supersampling, fast
    if opacity < 0.999:
        a = tile.getchannel("A").point(lambda v: int(v * opacity))
        tile.putalpha(a)
    crop = (cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0)
    if glow > 0:
        g = tile.filter(ImageFilter.GaussianBlur(glow))
        g.putalpha(g.getchannel("A").point(lambda v: int(v * glow_alpha)))
        layer.alpha_composite(g.crop(crop), (cx0, cy0))
    layer.alpha_composite(tile.crop(crop), (cx0, cy0))


def text_layer(layer, xy, text, fnt, fill, tracking=0.0, anchor="ls", opacity=1.0, shadow=0.0):
    """Tracked text with optional soft shadow, composited with opacity."""
    if not text or opacity <= 0.003:
        return
    tile, sh, w, asc, desc, pad = _text_tile(text, fnt, tuple(rgb(fill)), round(float(tracking), 2),
                                             round(float(shadow), 2))
    if opacity < 0.999:
        tile = tile.copy()
        tile.putalpha(tile.getchannel("A").point(lambda v: int(v * opacity)))
        if sh is not None:
            sh = sh.copy()
            sh.putalpha(sh.getchannel("A").point(lambda v: int(v * opacity)))
    x, y = xy
    if anchor[0] == "m":
        x -= w / 2
    elif anchor[0] == "r":
        x -= w
    if len(anchor) > 1 and anchor[1] == "m":
        y += (asc - desc) / 2  # y was the visual middle; convert to baseline
    ox, oy = int(round(x - pad)), int(round(y - pad - asc))
    if sh is not None:
        _paste(layer, sh, ox + 1, oy + 2)
    _paste(layer, tile, ox, oy)
    return w


@lru_cache(maxsize=4096)
def _text_tile(text, fnt, fill, tracking, shadow):
    """Rendered text (+ soft shadow) tiles, cached: titles repeat every frame."""
    w = text_width(fnt, text, tracking)
    asc, desc = fnt.getmetrics()
    pad = int(shadow * 3) + 4
    tw, th = int(w) + 2 * pad + 2, asc + desc + 2 * pad
    tile = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    draw_tracked(ImageDraw.Draw(tile), (pad, pad + asc), text, fnt, fill + (255,), tracking, "ls")
    sh = None
    if shadow > 0:
        sh = Image.new("RGBA", tile.size, (0, 0, 0, 0))
        sh.putalpha(tile.getchannel("A").point(lambda v: int(v * 0.75)))
        sh = sh.filter(ImageFilter.GaussianBlur(shadow))
    return tile, sh, w, asc, desc, pad


def _paste(layer, tile, ox, oy):
    W, H = layer.size
    cx0, cy0 = max(0, ox), max(0, oy)
    cx1, cy1 = min(W, ox + tile.width), min(H, oy + tile.height)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    layer.alpha_composite(tile.crop((cx0 - ox, cy0 - oy, cx1 - ox, cy1 - oy)), (cx0, cy0))


def gradient_plate(layer, x0, y0, x1, y1, color=(0, 0, 0), a0=0.6, a1=0.0, horizontal=True):
    w, h = int(x1 - x0), int(y1 - y0)
    if w <= 0 or h <= 0:
        return
    n = w if horizontal else h
    ramp = np.linspace(a0, a1, n, dtype=np.float32)
    ramp = (ramp ** 1.4 if a0 > a1 else ramp) * 255
    alpha = np.tile(ramp[None, :], (h, 1)) if horizontal else np.tile(ramp[:, None], (1, w))
    tile = Image.new("RGBA", (w, h), tuple(color) + (0,))
    tile.putalpha(Image.fromarray(alpha.clip(0, 255).astype(np.uint8)))
    _paste(layer, tile, int(x0), int(y0))


# ====================================================================== pixel effects
# All full-frame work goes through PIL's C routines (no float numpy on 1080p frames).


@lru_cache(maxsize=8)
def _grid(w, h):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return xs, ys


@lru_cache(maxsize=8)
def _vignette_img(W, H, strength=0.45):
    w, h = W // 4, H // 4
    xs, ys = _grid(w, h)
    d = np.sqrt(((xs - w / 2) / (w / 2)) ** 2 + ((ys - h / 2) / (h / 2)) ** 2) / math.sqrt(2)
    m = 1 - strength * np.clip((d - 0.35) / 0.65, 0, 1) ** 1.6
    L = Image.fromarray((m * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    return Image.merge("RGB", (L, L, L))


def apply_vignette(img: Image.Image, strength=0.45) -> Image.Image:
    if strength <= 0:
        return img
    return ImageChops.multiply(img, _vignette_img(img.width, img.height, round(strength, 2)))


def apply_spotlight(img: Image.Image, view: View, o: dict, t: float, seg_dur: float) -> Image.Image:
    """Keep an elliptical region crisp; blur, desaturate and dim everything else."""
    start, end = parse_time(o.get("start", 0), seg_dur), parse_time(o.get("end", "end"), seg_dur)
    a = env(t, start, end, o.get("fade_in", 0.5), o.get("fade_out", 0.4))
    if a <= 0:
        return img
    a = ease("inoutsine", a)
    W, H = img.size
    q = 4
    w, h = W // q, H // q
    cx, cy = view.pt(o["x"], o["y"])
    rx = view.ln(o.get("rx", o.get("r", 0.12)))
    ry = view.ln(o.get("ry", o.get("rx", o.get("r", 0.12))))
    feather = float(o.get("feather", 0.45))
    xs, ys = _grid(w, h)
    d = np.sqrt(((xs - cx / q) / max(1, rx / q)) ** 2 + ((ys - cy / q) / max(1, ry / q)) ** 2)
    m = np.clip((1 + feather - d) / feather, 0, 1)
    m = m * m * (3 - 2 * m)
    mask = Image.fromarray((m * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    dim = float(o.get("dim", 0.62))
    if o.get("blur", True):
        out = img.resize((W // 10, H // 10), Image.BILINEAR).filter(ImageFilter.GaussianBlur(1.2))
    else:
        out = img
    if o.get("desaturate", True):
        out = ImageEnhance.Color(out).enhance(0.35)
    out = ImageEnhance.Brightness(out).enhance(1 - dim)
    if out.size != img.size:
        out = out.resize((W, H), Image.BILINEAR)
    if a < 0.999:
        out = Image.blend(img, out, a)
    return Image.composite(img, out, mask)


class Grain:
    """Monochrome film grain, applied with saturating add/subtract."""

    def __init__(self, W, H, n=6, seed=7):
        self.W, self.H, self.n, self.seed = W, H, n, seed
        self.cache = {}

    def _tiles(self, strength):
        key = round(strength, 3)
        if key not in self.cache:
            rng = np.random.default_rng(self.seed)
            amp = strength * 128
            tiles = []
            for _ in range(self.n):
                g = rng.normal(0, 1, (self.H // 2, self.W // 2)).astype(np.float32) * amp
                pos = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8)).resize((self.W, self.H), Image.BILINEAR)
                neg = Image.fromarray(np.clip(-g, 0, 255).astype(np.uint8)).resize((self.W, self.H), Image.BILINEAR)
                tiles.append((Image.merge("RGB", (pos,) * 3), Image.merge("RGB", (neg,) * 3)))
            self.cache[key] = tiles
        return self.cache[key]

    def apply(self, img: Image.Image, i, strength=0.05) -> Image.Image:
        if strength <= 0:
            return img
        pos, neg = self._tiles(strength)[i % self.n]
        return ImageChops.subtract(ImageChops.add(img, pos), neg)


# ====================================================================== overlays


def _times(o, seg_dur):
    return parse_time(o.get("start", 0), seg_dur), parse_time(o.get("end", "end"), seg_dur)


def ov_circle(layer, view, o, t, seg_dur, st: Style, ctx):
    start, end = _times(o, seg_dur)
    a = env(t, start, end, 0.12, o.get("fade_out", 0.25))
    if a <= 0:
        return
    p = ease("out", (t - start) / float(o.get("draw", 0.55)))
    cx, cy = view.pt(o["x"], o["y"])
    R = view.ln(o.get("r", 0.06)) * (1.18 - 0.18 * p)
    if o.get("pulse"):
        R *= 1 + 0.025 * math.sin((t - start) * 5)
    col = rgb(st.c(o.get("color", "accent")), 1.0)
    wpx = st.k * o.get("width", 5)

    def fn(d, T, ss):
        x0, y0 = T(cx - R, cy - R)
        x1, y1 = T(cx + R, cy + R)
        d.arc([x0, y0, x1, y1], -90, -90 + 360 * p, fill=col, width=max(1, int(wpx * ss)))
    stamp(layer, (cx - R - wpx, cy - R - wpx, cx + R + wpx, cy + R + wpx), fn, glow=4 * st.k, opacity=a)
    if o.get("label"):
        _side_label(layer, o["label"], cx + R * 0.75, cy - R * 0.85, st, a, p, o.get("label_side", "right"))


def _side_label(layer, text, x, y, st, a, p, side="right"):
    f = st.f("mono_bold", 20)
    dx = st.px(26) * (1 if side == "right" else -1)
    col = st.c("accent")

    def fn(d, T, ss):
        d.line([T(x, y), T(x + dx * p, y - abs(dx) * 0.6 * p)], fill=rgb(col, 1), width=int(2 * st.k * ss))
    stamp(layer, (min(x, x + dx) - 4, y - abs(dx) - 4, max(x, x + dx) + 4, y + 4), fn, opacity=a)
    if p > 0.6:
        anchor = "ls" if side == "right" else "rs"
        text_layer(layer, (x + dx + st.px(8) * (1 if side == "right" else -1), y - abs(dx) * 0.6 + st.px(6)),
                   text.upper(), f, st.c("text"), tracking=2 * st.k, anchor=anchor,
                   opacity=a * clamp((p - 0.6) / 0.4), shadow=3 * st.k)


def ov_bracket(layer, view, o, t, seg_dur, st: Style, ctx):
    start, end = _times(o, seg_dur)
    a = env(t, start, end, 0.2, o.get("fade_out", 0.25))
    if a <= 0:
        return
    p = ease("outexpo", (t - start) / 0.5)
    x0, y0 = view.pt(o["x"] - o["w"] / 2, o["y"] - o["h"] / 2)
    x1, y1 = view.pt(o["x"] + o["w"] / 2, o["y"] + o["h"] / 2)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = 1.25 - 0.25 * p
    x0, x1 = cx + (x0 - cx) * s, cx + (x1 - cx) * s
    y0, y1 = cy + (y0 - cy) * s, cy + (y1 - cy) * s
    L = 0.22 * min(x1 - x0, y1 - y0)
    col = rgb(st.c(o.get("color", "accent")), 1.0)
    wpx = st.k * o.get("width", 4)

    def fn(d, T, ss):
        lw = max(1, int(wpx * ss))
        for (px, py, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            d.line([T(px + sx * L, py), T(px, py), T(px, py + sy * L)], fill=col, width=lw, joint="curve")
    stamp(layer, (x0 - wpx, y0 - wpx, x1 + wpx, y1 + wpx), fn, glow=3 * st.k, opacity=a)
    if o.get("label"):
        text_layer(layer, (x0, y0 - st.px(12)), o["label"].upper(), st.f("mono_bold", 19),
                   st.c(o.get("color", "accent")), tracking=2.5 * st.k, opacity=a * p, shadow=3 * st.k)


def _pt(view, st, p, space):
    if space == "screen":
        return p[0] * st.W, p[1] * st.H
    return view.pt(*p)


def ov_arrow(layer, view, o, t, seg_dur, st: Style, ctx):
    start, end = _times(o, seg_dur)
    a = env(t, start, end, 0.08, o.get("fade_out", 0.25))
    if a <= 0:
        return
    p = ease("out", (t - start) / float(o.get("draw", 0.45)))
    space = o.get("space", "canvas")
    tx, ty = _pt(view, st, o["from"], space)
    hx, hy = _pt(view, st, o["to"], space)
    ex, ey = lerp(tx, hx, p), lerp(ty, hy, p)
    ang = math.atan2(ey - ty, ex - tx)
    hs = st.px(o.get("head", 22))
    col = rgb(st.c(o.get("color", "accent")), 1.0)
    wpx = st.k * o.get("width", 5)

    def fn(d, T, ss):
        bx, by = ex - math.cos(ang) * hs * 0.8, ey - math.sin(ang) * hs * 0.8
        d.line([T(tx, ty), T(bx, by)], fill=col, width=max(1, int(wpx * ss)))
        pts = [(ex, ey),
               (ex - hs * math.cos(ang - 0.45), ey - hs * math.sin(ang - 0.45)),
               (ex - hs * math.cos(ang + 0.45), ey - hs * math.sin(ang + 0.45))]
        d.polygon([T(*q) for q in pts], fill=col)
    stamp(layer, (min(tx, ex) - hs, min(ty, ey) - hs, max(tx, ex) + hs, max(ty, ey) + hs), fn,
          glow=4 * st.k, opacity=a)


def ov_line(layer, view, o, t, seg_dur, st: Style, ctx):
    start, end = _times(o, seg_dur)
    a = env(t, start, end, 0.1, 0.25)
    if a <= 0:
        return
    p = ease("inout", (t - start) / float(o.get("draw", 0.6)))
    space = o.get("space", "canvas")
    x0, y0 = _pt(view, st, o["from"], space)
    x1, y1 = _pt(view, st, o["to"], space)
    ex, ey = lerp(x0, x1, p), lerp(y0, y1, p)
    col = rgb(st.c(o.get("color", "accent")), 1.0)
    wpx = st.k * o.get("width", 3)
    dashed = o.get("dashed", True)

    def fn(d, T, ss):
        if not dashed:
            d.line([T(x0, y0), T(ex, ey)], fill=col, width=max(1, int(wpx * ss)))
            return
        L = math.hypot(ex - x0, ey - y0)
        seg, gap = st.px(16), st.px(10)
        s = 0.0
        while s < L:
            e = min(L, s + seg)
            ax, ay = x0 + (ex - x0) * s / L, y0 + (ey - y0) * s / L
            bx, by = x0 + (ex - x0) * e / L, y0 + (ey - y0) * e / L
            d.line([T(ax, ay), T(bx, by)], fill=col, width=max(1, int(wpx * ss)))
            s = e + gap
        r = wpx * 2.2
        for (qx, qy) in ((x0, y0),) + (((x1, y1),) if p >= 1 else ()):
            d.ellipse([*T(qx - r, qy - r), *T(qx + r, qy + r)], fill=col)
    stamp(layer, (min(x0, x1) - 10, min(y0, y1) - 10, max(x0, x1) + 10, max(y0, y1) + 10), fn,
          glow=3 * st.k, opacity=a)


def ov_callout(layer, view, o, t, seg_dur, st: Style, ctx):
    start, end = _times(o, seg_dur)
    a = env(t, start, end, 0.15, 0.3)
    if a <= 0:
        return
    u = t - start
    px_, py_ = view.pt(o["x"], o["y"])
    dx, dy = o.get("dx", 0.1) * st.W, o.get("dy", -0.1) * st.H
    ex, ey = px_ + dx, py_ + dy
    side = 1 if dx >= 0 else -1
    run_len = st.px(o.get("run", 60))
    p1 = ease("out", u / 0.35)
    p2 = ease("out", (u - 0.3) / 0.3)
    col = rgb(st.c(o.get("color", "accent")), 1.0)

    def fn(d, T, ss):
        r = st.k * 7
        ring = r * (1 + 1.6 * ((u * 1.2) % 1.0))
        ra = int(255 * (1 - (u * 1.2) % 1.0) * 0.8)
        d.ellipse([*T(px_ - ring, py_ - ring), *T(px_ + ring, py_ + ring)], outline=col[:3] + (ra,),
                  width=max(1, int(2 * st.k * ss)))
        d.ellipse([*T(px_ - r, py_ - r), *T(px_ + r, py_ + r)], fill=col)
        mx, my = lerp(px_, ex, p1), lerp(py_, ey, p1)
        lw = max(1, int(2.5 * st.k * ss))
        d.line([T(px_, py_), T(mx, my)], fill=col, width=lw)
        if p2 > 0:
            d.line([T(ex, ey), T(ex + side * run_len * p2, ey)], fill=col, width=lw)
    bb = (min(px_, ex, ex + side * run_len) - 30, min(py_, ey) - 30, max(px_, ex, ex + side * run_len) + 30,
          max(py_, ey) + 30)
    stamp(layer, bb, fn, glow=3 * st.k, opacity=a)
    if p2 > 0.2:
        ta = a * clamp((p2 - 0.2) / 0.8)
        tx = ex + side * (run_len + st.px(14))
        anchor = "l" if side > 0 else "r"
        text_layer(layer, (tx, ey + st.px(10)), o["text"], st.f("sans_semi", o.get("size", 34)),
                   st.c("text"), anchor=anchor + "s", opacity=ta, shadow=4 * st.k)
        if o.get("sub"):
            text_layer(layer, (tx, ey + st.px(42)), o["sub"].upper(), st.f("mono", 19), st.c("muted"),
                       tracking=2 * st.k, anchor=anchor + "s", opacity=ta, shadow=3 * st.k)


def ov_lower_third(layer, view, o, t, seg_dur, st: Style, ctx):
    start, end = _times(o, seg_dur)
    if not (start <= t <= end):
        return
    u, rem = t - start, end - t
    pin = ease("outexpo", u / 0.6)
    pout = ease("in", clamp(1 - rem / 0.35))
    a = 1 - pout
    x = st.px(o.get("x", 120))
    yb = st.H - st.px(o.get("y_from_bottom", 150))
    bar_h = st.px(78)
    gradient_plate(layer, 0, yb - bar_h - st.px(30), st.px(980), yb + st.px(50), a0=0.55 * a * pin, a1=0)
    col = rgb(st.c("accent"), 1)
    h = bar_h * pin

    def fn(d, T, ss):
        d.rectangle([*T(x, yb - bar_h / 2 - h / 2 + st.px(10)), *T(x + st.px(6), yb - bar_h / 2 + h / 2 + st.px(10))],
                    fill=col)
    stamp(layer, (x - 2, yb - bar_h - 10, x + st.px(8), yb + st.px(40)), fn, opacity=a)
    slide = (1 - pin) * st.px(-40)
    name = o.get("name", "").upper()
    text_layer(layer, (x + st.px(28) + slide, yb - st.px(18)), name, st.f("display_bold", o.get("size", 54)),
               st.c("text"), tracking=2.5 * st.k, opacity=a * pin, shadow=5 * st.k)
    ra = a * ease("out", (u - 0.25) / 0.4)
    if o.get("role"):
        text_layer(layer, (x + st.px(30), yb + st.px(22)), o["role"].upper(), st.f("mono_bold", 20),
                   st.c(o.get("role_color", "accent")), tracking=4 * st.k, opacity=ra, shadow=3 * st.k)


def ov_location(layer, view, o, t, seg_dur, st: Style, ctx):
    start, end = _times(o, seg_dur)
    a = env(t, start, end, 0.1, 0.35)
    if a <= 0:
        return
    u = t - start
    x, y = st.px(110), st.px(118)
    text = o["text"].upper()
    n = int(clamp(u * float(o.get("cps", 26)) / max(1, len(text))) * len(text))
    col = rgb(st.c("accent"), 1)

    def fn(d, T, ss):
        r = st.k * 7
        d.ellipse([*T(x - r, y - st.px(12) - r), *T(x + r, y - st.px(12) + r)], outline=col, width=int(2.5 * st.k * ss))
        rr = r * 0.4
        d.ellipse([*T(x - rr, y - st.px(12) - rr), *T(x + rr, y - st.px(12) + rr)], fill=col)
        L = st.px(420) * ease("outexpo", u / 0.8)
        d.line([T(x - r, y + st.px(18)), T(x - r + L, y + st.px(18))], fill=col[:3] + (150,), width=max(1, int(1.5 * st.k * ss)))
    stamp(layer, (x - 20, y - 40, x + st.px(440), y + 30), fn, opacity=a)
    shown = text[:n] + ("▌" if n < len(text) and int(u * 8) % 2 == 0 else "")
    text_layer(layer, (x + st.px(22), y - st.px(3)), shown, st.f("mono_bold", 26), st.c("text"),
               tracking=3 * st.k, opacity=a, shadow=4 * st.k)
    if o.get("sub") and n >= len(text):
        text_layer(layer, (x + st.px(22), y + st.px(46)), o["sub"].upper(), st.f("mono", 18), st.c("muted"),
                   tracking=3 * st.k, opacity=a * ease("out", (u - len(text) / 26) / 0.4), shadow=3 * st.k)


def _rich_words(text):
    """'*several words*' marks emphasis (accent colour); spans may cover many words."""
    out, on = [], False
    for w in text.split():
        starts = w.startswith("*")
        if starts:
            on = True
        emph = on
        if w.rstrip(".,!?;:'\"").endswith("*") and (len(w.strip("*.,!?;:'\"")) > 0):
            on = False
        out.append((w.replace("*", ""), emph))
    return out


def draw_rich_block(layer, lines_words, fnt, cx, y_top, line_h, st, t_reveal, total_reveal, a, color="text",
                    align="center", x_left=None):
    """Word-by-word reveal of pre-wrapped lines of (word, emph)."""
    flat = [w for ln in lines_words for w in ln]
    n = max(1, len(flat))
    i = 0
    space = fnt.getlength(" ")
    for li, ln in enumerate(lines_words):
        widths = [fnt.getlength(w) for w, _ in ln]
        lw = sum(widths) + space * (len(ln) - 1)
        x = cx - lw / 2 if align == "center" else x_left
        y = y_top + li * line_h
        for (w, emph), ww in zip(ln, widths):
            wa = a * ease("out", (t_reveal - total_reveal * i / n) / 0.35) if total_reveal > 0 else a
            lift = (1 - ease("out", (t_reveal - total_reveal * i / n) / 0.45)) * st.px(14) if total_reveal > 0 else 0
            text_layer(layer, (x, y + lift), w, fnt, st.c("accent" if emph else color), opacity=wa,
                       shadow=4 * st.k)
            x += ww + space
            i += 1


def wrap_rich(fnt, text, max_w):
    words = _rich_words(text)
    lines, cur, cur_w = [], [], 0.0
    sp = fnt.getlength(" ")
    for w, e in words:
        ww = fnt.getlength(w)
        if cur and cur_w + sp + ww > max_w:
            lines.append(cur)
            cur, cur_w = [], 0.0
        cur.append((w, e))
        cur_w += (sp if len(cur) > 1 else 0) + ww
    if cur:
        lines.append(cur)
    return lines


def ov_quote(layer, view, o, t, seg_dur, st: Style, ctx):
    start, end = _times(o, seg_dur)
    a = env(t, start, end, 0.2, 0.35)
    if a <= 0:
        return
    u = t - start
    fnt = st.f("serif_italic", o.get("size", 52))
    lines = wrap_rich(fnt, o["text"], st.px(1350))
    lh = st.px(o.get("size", 52) * 1.22)
    block_h = lh * len(lines)
    pos = o.get("pos", "bottom")
    if pos == "center":
        y_top = st.H / 2 - block_h / 2 + st.px(20)
    else:
        y_top = st.H - st.px(140) - block_h
        gradient_plate(layer, 0, y_top - st.px(160), st.W, st.H, a0=0.0, a1=0.72 * a, horizontal=False)
    if pos == "center":
        text_layer(layer, (st.W / 2 - st.px(700), y_top + st.px(30)), "“", st.f("serif_italic", 150),
                   st.c("accent"), opacity=a * 0.9)
    reveal = float(o.get("reveal", min(1.4, (end - start) * 0.4)))
    draw_rich_block(layer, lines, fnt, st.W / 2, y_top + lh * 0.8, lh, st, u, reveal, a)
    if o.get("speaker"):
        text_layer(layer, (st.W / 2, y_top + block_h + st.px(42)), "—  " + o["speaker"].upper(),
                   st.f("mono_bold", 20), st.c("accent"), tracking=4 * st.k, anchor="ms",
                   opacity=a * ease("out", (u - reveal * 0.7) / 0.4), shadow=3 * st.k)


TAGS = {
    "confirmed": ("confirmed", "●", "CONFIRMED"),
    "theory": ("theory", "◆", "THEORY"),
    "speculation": ("speculation", "▲", "SPECULATION"),
    "clue": ("accent", "●", "CLUE"),
    "easter_egg": ("accent", "●", "EASTER EGG"),
}


def ov_tag(layer, view, o, t, seg_dur, st: Style, ctx):
    start, end = _times(o, seg_dur)
    a = env(t, start, end, 0.25, 0.3)
    if a <= 0:
        return
    key = o.get("kind", "theory").lower()
    colname, glyph, label = TAGS.get(key, ("accent", "◆", key.upper()))
    label = o.get("label", label).upper()
    col = st.c(colname)
    f = st.f("mono_bold", 20)
    lw = text_width(f, label, 3.5 * st.k)
    pad_x, h = st.px(18), st.px(44)
    gw = st.px(22)
    bw = lw + pad_x * 2 + gw
    x1 = st.W * float(o["x"]) if "x" in o else st.W - st.px(104)
    x0 = x1 - bw
    y0 = st.px(o.get("y", 122)) - (1 - ease("out", (t - start) / 0.4)) * st.px(16)

    def fn(d, T, ss):
        d.rounded_rectangle([*T(x0, y0), *T(x1, y0 + h)], radius=int(h / 2 * ss), fill=(8, 10, 13, 170),
                            outline=rgb(col, 1), width=max(1, int(2 * st.k * ss)))
    stamp(layer, (x0, y0, x1, y0 + h), fn, opacity=a)
    text_layer(layer, (x0 + pad_x, y0 + h / 2 + st.px(7)), glyph, st.f("sans_semi", 18), col, opacity=a)
    text_layer(layer, (x0 + pad_x + gw, y0 + h / 2 + st.px(7)), label, f, col, tracking=3.5 * st.k, opacity=a)
    if o.get("text"):
        text_layer(layer, (x1, y0 + h + st.px(34)), o["text"], st.f("sans_medium", 22), st.c("muted"),
                   anchor="rs", opacity=a * ease("out", (t - start - 0.2) / 0.4), shadow=3 * st.k)


def ov_hud(layer, view, o, t, seg_dur, st: Style, ctx):
    start, end = _times(o, seg_dur)
    a = env(t, start, end, 0.25, 0.3) * float(o.get("opacity", 0.8))
    if a <= 0:
        return
    m = st.px(48)
    L = st.px(46)
    p = ease("outexpo", (t - start) / 0.6)
    inset = m + (1 - p) * st.px(30)
    col = rgb(st.c("hud"), 1)
    W, H = st.W, st.H

    def fn_corner(x, y, sx, sy):
        def fn(d, T, ss):
            lw = max(1, int(2 * st.k * ss))
            d.line([T(x + sx * L, y), T(x, y), T(x, y + sy * L)], fill=col, width=lw)
        return fn
    for (x, y, sx, sy) in ((inset, inset, 1, 1), (W - inset, inset, -1, 1), (inset, H - inset, 1, -1),
                           (W - inset, H - inset, -1, -1)):
        stamp(layer, (x - L - 4, y - L - 4, x + L + 4, y + L + 4), fn_corner(x, y, sx, sy), opacity=a)
    f = st.f("mono", 17)
    blink = 1.0 if int((t - start) * 2) % 2 == 0 else 0.35
    text_layer(layer, (m + st.px(22), m + st.px(36)), "●", f, st.c("speculation"), opacity=a * blink)
    text_layer(layer, (m + st.px(44), m + st.px(36)), o.get("label", "FRAME ANALYSIS").upper(), f, st.c("hud"),
               tracking=3 * st.k, opacity=a, shadow=3 * st.k)
    if ctx.get("src_t") is not None:
        from .util import fmt_tc
        text_layer(layer, (W - m - st.px(22), m + st.px(36)), "SRC " + fmt_tc(ctx["src_t"]), f, st.c("hud"),
                   tracking=2 * st.k, anchor="rs", opacity=a, shadow=3 * st.k)
    text_layer(layer, (W - m - st.px(22), H - m - st.px(22)), f"ZOOM {view.zoom:0.1f}×", f, st.c("hud"),
               tracking=2 * st.k, anchor="rs", opacity=a, shadow=3 * st.k)
    if o.get("crosshair"):
        cx, cy = view.pt(*o["crosshair"])
        r = st.px(18)

        def fnx(d, T, ss):
            lw = max(1, int(1.5 * st.k * ss))
            for (ax, ay, bx, by) in ((cx - 2 * r, cy, cx - r * 0.5, cy), (cx + r * 0.5, cy, cx + 2 * r, cy),
                                     (cx, cy - 2 * r, cx, cy - r * 0.5), (cx, cy + r * 0.5, cx, cy + 2 * r)):
                d.line([T(ax, ay), T(bx, by)], fill=col, width=lw)
        stamp(layer, (cx - 2 * r - 4, cy - 2 * r - 4, cx + 2 * r + 4, cy + 2 * r + 4), fnx, opacity=a)


def ov_counter(layer, view, o, t, seg_dur, st: Style, ctx):
    start, end = _times(o, seg_dur)
    a = env(t, start, end, 0.25, 0.3)
    if a <= 0:
        return
    label = o.get("label", "DETAIL").upper()
    n = str(o.get("n", 1)).zfill(2)
    tot = o.get("total")
    s = f"{label}  {n}" + (f" / {str(tot).zfill(2)}" if tot else "")
    y = st.px(o.get("y", 96)) - (1 - ease("out", (t - start) / 0.4)) * st.px(14)
    f = st.f("mono_bold", 20)
    w = text_width(f, s, 5 * st.k)
    px_, h = st.px(18), st.px(38)

    def fn(d, T, ss):
        d.rounded_rectangle([*T(st.W / 2 - w / 2 - px_, y - st.px(26)), *T(st.W / 2 + w / 2 + px_, y - st.px(26) + h)],
                            radius=int(h / 2 * ss), fill=(8, 10, 13, 165))
    stamp(layer, (st.W / 2 - w / 2 - px_, y - st.px(26), st.W / 2 + w / 2 + px_, y - st.px(26) + h), fn, opacity=a)
    text_layer(layer, (st.W / 2, y), s, f, st.c("accent"), tracking=5 * st.k, anchor="ms", opacity=a)


def ov_text(layer, view, o, t, seg_dur, st: Style, ctx):
    start, end = _times(o, seg_dur)
    a = env(t, start, end, o.get("fade_in", 0.3), o.get("fade_out", 0.3))
    if a <= 0:
        return
    txt = o["text"].upper() if o.get("upper") else o["text"]
    fnt = st.f(o.get("font", "display_bold"), o.get("size", 60))
    lift = (1 - ease("out", (t - start) / 0.5)) * st.px(18)
    x, y = o.get("x", 0.5) * st.W, o.get("y", 0.5) * st.H + lift
    text_layer(layer, (x, y), txt, fnt, st.c(o.get("color", "text")), tracking=o.get("tracking", 0) * st.k,
               anchor=o.get("anchor", "ms"), opacity=a, shadow=o.get("shadow", 5) * st.k)


def ov_caption(layer, view, o, t, seg_dur, st: Style, ctx):
    """Burned-in narration caption chunk (auto-generated)."""
    start, end = o["start"], o["end"]
    a = env(t, start, end, 0.08, 0.08)
    if a <= 0:
        return
    fnt = st.f("sans_semi", 38)
    lines = wrap(fnt, o["text"], st.px(1300))
    lh = st.px(50)
    y = st.H - st.px(70) - lh * (len(lines) - 1)
    for i, ln in enumerate(lines):
        text_layer(layer, (st.W / 2, y + i * lh), ln, fnt, st.c("text"), anchor="ms", opacity=a, shadow=6 * st.k)


OVERLAYS = {
    "circle": ov_circle, "bracket": ov_bracket, "box": ov_bracket, "arrow": ov_arrow, "line": ov_line,
    "callout": ov_callout, "lower_third": ov_lower_third, "location": ov_location, "quote": ov_quote,
    "tag": ov_tag, "hud": ov_hud, "counter": ov_counter, "text": ov_text, "caption": ov_caption,
}
PIXEL_EFFECTS = {"spotlight"}


def apply_overlays(img: Image.Image, view: View, overlays, t, seg_dur, st: Style, ctx) -> Image.Image:
    """Pixel effects first (spotlight), then vector layer."""
    if not overlays:
        return img
    for o in overlays:
        if o["type"] in PIXEL_EFFECTS:
            img = apply_spotlight(img, view, o, t, seg_dur)
    vec = [o for o in overlays if o["type"] not in PIXEL_EFFECTS]
    if not vec:
        return img
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    for o in vec:
        fn = OVERLAYS.get(o["type"])
        if fn is None:
            raise ValueError(f"unknown overlay type {o['type']!r}")
        fn(layer, view, o, t, seg_dur, st, ctx)
    base = img.convert("RGBA")
    base.alpha_composite(layer)
    return base.convert("RGB")
