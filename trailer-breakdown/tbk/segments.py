"""Segment types. Each yields exactly `n` RGB frames (H, W, 3) uint8."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from . import graphics as G
from .media import FrameReader, grab_frame
from .style import Style, text_width
from .util import clamp, ease, env, interp_keys, lerp, norm_keys, parse_time, rgb


class Ctx:
    def __init__(self, src, info, W, H, fps, style: Style, cw, ch):
        self.src, self.info = src, info
        self.W, self.H, self.fps = W, H, fps
        self.st = style
        self.cw, self.ch = cw, ch
        self._grain = None
        # zoom at which a letterboxed (e.g. 2.39:1) picture fills a 16:9 frame
        src_ar = info["w"] / info["h"]
        content_h = min(1.0, (cw / src_ar) / ch)
        self.fill_zoom = 1.0 / content_h

    @property
    def grain(self):
        if self._grain is None:
            self._grain = G.Grain(self.W, self.H)
        return self._grain


# ---------------------------------------------------------------- natural durations


def speed_at(spec, t):
    sp = spec.get("speed", 1.0)
    if isinstance(sp, (int, float)):
        return float(sp)
    keys = [dict(k, t=parse_time(k.get("t", 0))) for k in sp]
    keys.sort(key=lambda k: k["t"])
    return max(0.02, interp_keys(keys, t, ("s",), "inoutsine")["s"])


def clip_time_map(spec, fps: float, n_frames: int | None = None):
    """Source time for each output frame; holds at `out` once reached."""
    t_in, t_out = parse_time(spec["in"]), parse_time(spec["out"])
    src, times, i = t_in, [], 0
    limit = n_frames if n_frames is not None else 100000
    while i < limit:
        if n_frames is None and src >= t_out - 1e-6:
            break
        times.append(min(src, t_out - 0.5 / fps))
        src += speed_at(spec, i / fps) / fps
        i += 1
    return np.array(times)


def natural_duration(spec, fps: float) -> float:
    t = spec["type"]
    if t == "clip":
        return len(clip_time_map(spec, fps)) / fps
    return float(spec.get("dur", {"title": 3.0, "card": 3.5, "black": 0.6, "freeze": 4.0, "split": 5.0,
                                  "timeline": 6.0}.get(t, 3.0)))


# ---------------------------------------------------------------- base


class Segment:
    footage = False

    def __init__(self, spec, ctx: Ctx, dur: float, n: int):
        self.spec, self.ctx, self.dur, self.n = spec, ctx, dur, n
        self.st = ctx.st
        self.W, self.H = ctx.W, ctx.H
        self.overlays = list(spec.get("overlays", []))
        self.grain_amt = float(spec.get("grain", 0.0 if self.footage else 0.05))
        self.vig = float(spec.get("vignette", 0.0 if self.footage else 0.4))
        self.screen_view = G.View((0, 0, self.W, self.H), self.W, self.H, (0, 0, self.W, self.H))

    def frames(self):
        for i in range(self.n):
            t = i / float(self.ctx.fps)
            yield self.frame(i, t)

    def finish(self, img: Image.Image, i, t, view=None, ctx_extra=None) -> np.ndarray:
        img = G.apply_overlays(img, view or self.screen_view, self.overlays, t, self.dur, self.st,
                               ctx_extra or {})
        if self.vig > 0:
            img = G.apply_vignette(img, self.vig)
        if self.grain_amt > 0:
            img = self.ctx.grain.apply(img, i, self.grain_amt)
        return np.asarray(img)

    def close(self):
        pass


# ---------------------------------------------------------------- footage


class Clip(Segment):
    footage = True

    def __init__(self, spec, ctx, dur, n):
        super().__init__(spec, ctx, dur, n)
        self.times = clip_time_map(spec, float(ctx.fps), n)
        self.cam = G.Camera(spec.get("camera"), dur, ctx.cw, ctx.ch, spec.get("shake"))
        self.reader = FrameReader(ctx.src, float(self.times[0]), ctx.cw, ctx.ch, ctx.info["fps"])
        sp = spec.get("speed", 1.0)
        self.blend = not isinstance(sp, (int, float)) or float(sp) < 0.95

    def frame(self, i, t):
        src_t = float(self.times[i])
        f = self.reader.get_blend(src_t) if self.blend else self.reader.get(src_t)
        view = G.View(self.cam.box(t), self.ctx.cw, self.ctx.ch, (0, 0, self.W, self.H))
        img = G.render_view(f, view.box, (self.W, self.H))
        return self.finish(img, i, t, view, {"src_t": src_t})

    def close(self):
        self.reader.close()


class Freeze(Segment):
    footage = True

    def __init__(self, spec, ctx, dur, n):
        super().__init__(spec, ctx, dur, n)
        self.at = parse_time(spec["at"])
        self.img = grab_frame(ctx.src, self.at, ctx.cw, ctx.ch)
        cam = spec.get("camera")
        if not cam:  # slow, purposeful push-in so a held frame never feels dead
            cam = [{"t": 0, "z": 1.0}, {"t": "end", "z": 1.06, "ease": "linear"}]
        self.cam = G.Camera(cam, dur, ctx.cw, ctx.ch, spec.get("shake"))
        self.flash = spec.get("flash", True)
        if "vignette" not in spec:
            self.vig = 0.3
        self.enh = None
        e = spec.get("enhance")
        if e:  # reveal hidden detail: brighten/contrast the frame, optionally animated
            im = Image.fromarray(self.img)
            if e.get("gamma"):
                g = float(e["gamma"])
                lut = [int(round(255 * (v / 255) ** (1 / g))) for v in range(256)] * 3
                im = im.point(lut)
            if e.get("brightness"):
                im = ImageEnhance.Brightness(im).enhance(float(e["brightness"]))
            if e.get("contrast"):
                im = ImageEnhance.Contrast(im).enhance(float(e["contrast"]))
            if e.get("saturation"):
                im = ImageEnhance.Color(im).enhance(float(e["saturation"]))
            self.enh = (Image.fromarray(self.img), im, parse_time(e.get("start", 0), dur), float(e.get("ramp", 0)))

    def source_at(self, t):
        if not self.enh:
            return self.img
        orig, enh, t0, ramp = self.enh
        p = 1.0 if ramp <= 0 and t >= t0 else ease("inoutsine", (t - t0) / ramp) if ramp > 0 else 0.0
        if p <= 0:
            return self.img
        if p >= 1:
            return np.asarray(enh)
        return np.asarray(Image.blend(orig, enh, p))

    def frame(self, i, t):
        view = G.View(self.cam.box(t), self.ctx.cw, self.ctx.ch, (0, 0, self.W, self.H))
        img = G.render_view(self.source_at(t), view.box, (self.W, self.H))
        arr = self.finish(img, i, t, view, {"src_t": self.at})
        if self.flash and t < 0.2:
            k = 0.32 * (1 - t / 0.2) ** 2
            arr = (arr.astype(np.float32) * (1 - k) + 255 * k).astype(np.uint8)
        return arr


# ---------------------------------------------------------------- graphics backgrounds


def make_bg(ctx: Ctx, at=None, darken=0.32, blur=16) -> Image.Image:
    W, H, st = ctx.W, ctx.H, ctx.st
    if at is not None:
        f = grab_frame(ctx.src, parse_time(at), ctx.cw, ctx.ch)
        img = Image.fromarray(f).resize((W // 2, H // 2), Image.BILINEAR)
        img = img.filter(ImageFilter.GaussianBlur(blur * st.k / 2)).resize((W, H), Image.BICUBIC)
        arr = np.asarray(img, np.float32)
        g = arr.mean(axis=2, keepdims=True)
        arr = lerp(arr, g, 0.35) * darken
        tint = np.array(rgb(st.pal["bg"]), np.float32)
        arr = arr * 0.85 + tint * 0.15
        return Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    ys, xs = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32)
    d = np.sqrt(((xs - W / 8) / (W / 8)) ** 2 + ((ys - H / 8) / (H / 8)) ** 2)
    c0, c1 = np.array(rgb(st.pal["bg2"]), np.float32), np.array(rgb(st.pal["bg"]), np.float32)
    m = np.clip(d / 1.2, 0, 1)[..., None]
    arr = c0 * (1 - m) + c1 * m
    return Image.fromarray(arr.astype(np.uint8)).resize((W, H), Image.BICUBIC)


class Graphic(Segment):
    """Base for full-screen motion-graphics segments with optional trailer backdrop."""

    def __init__(self, spec, ctx, dur, n):
        super().__init__(spec, ctx, dur, n)
        self.bg = make_bg(ctx, spec.get("bg_at"))
        self.push = float(spec.get("bg_push", 0.05))

    def bg_frame(self, t):
        z = 1 + self.push * (t / max(self.dur, 1e-3))
        bw, bh = self.W / z, self.H / z
        box = ((self.W - bw) / 2, (self.H - bh) / 2, (self.W + bw) / 2, (self.H + bh) / 2)
        return self.bg.resize((self.W, self.H), Image.BILINEAR, box=box)

    def frame(self, i, t):
        img = self.bg_frame(t)
        layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        self.draw(layer, t)
        base = img.convert("RGBA")
        base.alpha_composite(layer)
        return self.finish(base.convert("RGB"), i, t)


class Title(Graphic):
    """Chapter card: number, kicker, title, subtitle."""

    def draw(self, layer, t):
        s, st, W, H = self.spec, self.st, self.W, self.H
        out = ease("in", clamp(1 - (self.dur - t) / 0.4))
        A = 1 - out

        def item(delay):
            p = ease("outexpo", (t - delay) / 0.9)
            return A * clamp((t - delay) / 0.35), (1 - p) * st.px(28)

        cy = H * 0.52
        if s.get("number") is not None:
            a, dy = item(0.0)
            G.text_layer(layer, (W / 2, cy - st.px(150) + dy), str(s["number"]).zfill(2),
                         st.f("display_light", 120), st.c("accent"), tracking=6 * st.k, anchor="ms",
                         opacity=a * 0.9)
        if s.get("kicker"):
            a, dy = item(0.12)
            f = st.f("mono_bold", 22)
            kw = text_width(f, s["kicker"].upper(), 8 * st.k)
            G.text_layer(layer, (W / 2, cy - st.px(70) + dy), s["kicker"].upper(), f, st.c("accent"),
                         tracking=8 * st.k, anchor="ms", opacity=a)
            L = st.px(90) * ease("outexpo", (t - 0.3) / 0.8)
            col = rgb(st.c("accent"), 1)
            yl = cy - st.px(78) + dy

            def fn(d, T, ss):
                lw = max(1, int(2 * st.k * ss))
                d.line([T(W / 2 - kw / 2 - st.px(24), yl), T(W / 2 - kw / 2 - st.px(24) - L, yl)], fill=col, width=lw)
                d.line([T(W / 2 + kw / 2 + st.px(24), yl), T(W / 2 + kw / 2 + st.px(24) + L, yl)], fill=col, width=lw)
            G.stamp(layer, (W / 2 - kw / 2 - st.px(130), yl - 6, W / 2 + kw / 2 + st.px(130), yl + 6), fn, opacity=a)
        if s.get("title"):
            a, dy = item(0.22)
            tr = (6 + 3 * t / self.dur) * st.k
            size = s.get("size", 104)
            f = st.f("display_bold", size)
            title = s["title"].upper()
            while text_width(f, title, tr) > W * 0.86 and size > 40:
                size -= 4
                f = st.f("display_bold", size)
            G.text_layer(layer, (W / 2, cy + st.px(40) + dy), title, f, st.c("text"), tracking=tr, anchor="ms",
                         opacity=a, shadow=8 * st.k)
        if s.get("sub"):
            a, dy = item(0.45)
            G.text_layer(layer, (W / 2, cy + st.px(110) + dy), s["sub"], st.f("sans", 32), st.c("muted"),
                         anchor="ms", opacity=a, shadow=4 * st.k)


class Card(Graphic):
    """Statement / question card. '*word*' = accent emphasis."""

    def draw(self, layer, t):
        s, st, W, H = self.spec, self.st, self.W, self.H
        A = 1 - ease("in", clamp(1 - (self.dur - t) / 0.4))
        size = s.get("size", 74)
        f = st.f(s.get("font", "display_bold"), size)
        lines = G.wrap_rich(f, s["text"], W * 0.78)
        lh = st.px(size * 1.18)
        top = H / 2 - lh * len(lines) / 2 + lh * 0.75
        if s.get("kicker"):
            G.text_layer(layer, (W / 2, top - lh - st.px(10)), s["kicker"].upper(), st.f("mono_bold", 22),
                         st.c("accent"), tracking=8 * st.k, anchor="ms", opacity=A * clamp(t / 0.4))
        reveal = float(s.get("reveal", min(2.0, self.dur * 0.45)))
        G.draw_rich_block(layer, lines, f, W / 2, top, lh, st, t, reveal, A)
        if s.get("sub"):
            G.text_layer(layer, (W / 2, top + lh * len(lines) + st.px(30)), s["sub"], st.f("sans", 32),
                         st.c("muted"), anchor="ms", opacity=A * ease("out", (t - reveal) / 0.5))


class Black(Graphic):
    def __init__(self, spec, ctx, dur, n):
        Segment.__init__(self, spec, ctx, dur, n)
        self.bg = Image.new("RGB", (self.W, self.H), (0, 0, 0))
        self.push = 0
        self.grain_amt = float(spec.get("grain", 0))
        self.vig = 0

    def draw(self, layer, t):
        if self.spec.get("text"):
            a = env(t, 0, self.dur, 0.4, 0.4)
            G.text_layer(layer, (self.W / 2, self.H / 2), self.spec["text"], self.st.f("mono_bold", 24),
                         self.st.c("muted"), tracking=6 * self.st.k, anchor="mm", opacity=a)


class Split(Graphic):
    """Side-by-side comparison of two moments."""

    def __init__(self, spec, ctx, dur, n):
        spec = dict(spec)
        spec.setdefault("bg_at", None)
        super().__init__(spec, ctx, dur, n)
        self.panels = []
        pw = int(self.W * 0.44)
        ph = int(pw * 9 / 16)
        gap = int(self.W * 0.04)
        x0 = (self.W - 2 * pw - gap) // 2
        y0 = int(self.H * 0.47 - ph / 2)
        for j, side in enumerate(("left", "right")):
            p = spec[side]
            img = grab_frame(ctx.src, parse_time(p["at"]), ctx.cw, ctx.ch)
            fz = ctx.fill_zoom
            cam = p.get("camera") or [{"t": 0, "z": fz}, {"t": "end", "z": fz * 1.05, "ease": "linear"}]
            if p.get("camera"):  # panel zoom is relative to the filled picture
                cam = [dict(k, z=k.get("z", 1.0) * fz) for k in cam]
            self.panels.append({
                "spec": p, "img": img, "cam": G.Camera(cam, dur, ctx.cw, ctx.ch),
                "rect": (x0 + j * (pw + gap), y0, pw, ph), "delay": 0.12 * j,
                "overlays": p.get("overlays", []),
            })

    def frame(self, i, t):
        st, W, H = self.st, self.W, self.H
        img = self.bg_frame(t)
        A = 1 - ease("in", clamp(1 - (self.dur - t) / 0.35))
        for j, P in enumerate(self.panels):
            ox, oy, pw, ph = P["rect"]
            p = ease("outexpo", (t - P["delay"]) / 0.7)
            slide = (1 - p) * st.px(60) * (-1 if j == 0 else 1)
            rect = (ox + slide, oy, pw, ph)
            box = P["cam"].box(t)
            view = G.View(box, self.ctx.cw, self.ctx.ch, rect)
            panel = G.render_view(P["img"], box, (pw, ph))
            if P["overlays"]:
                local = G.View(box, self.ctx.cw, self.ctx.ch, (0, 0, pw, ph))
                pst = Style(pw, ph, st.pal)
                pst.k = st.k
                panel = G.apply_overlays(panel, local, P["overlays"], t, self.dur, pst, {})
            a = A * clamp((t - P["delay"]) / 0.3)
            if a < 1:
                bgc = img.crop((int(rect[0]), oy, int(rect[0]) + pw, oy + ph))
                panel = Image.blend(bgc, panel, max(0.0, a))
            img.paste(panel, (int(rect[0]), oy))
            P["view"] = view
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        for j, P in enumerate(self.panels):
            ox, oy, pw, ph = P["rect"]
            a = A * clamp((t - P["delay"] - 0.2) / 0.4)
            col = rgb(st.c("muted"), 1)

            def fn(d, T, ss, ox=ox, oy=oy, pw=pw, ph=ph):
                d.rectangle([*T(ox, oy), *T(ox + pw, oy + ph)], outline=col, width=max(1, int(1.5 * st.k * ss)))
            G.stamp(layer, (ox - 3, oy - 3, ox + pw + 3, oy + ph + 3), fn, opacity=a * 0.6)
            sp = P["spec"]
            if sp.get("label"):
                G.text_layer(layer, (ox, oy + ph + st.px(44)), sp["label"].upper(), st.f("mono_bold", 20),
                             st.c("accent"), tracking=4 * st.k, opacity=a)
            if sp.get("caption"):
                G.text_layer(layer, (ox, oy + ph + st.px(84)), sp["caption"], st.f("sans_medium", 28),
                             st.c("text"), opacity=a)
        s = self.spec
        if s.get("title"):
            G.text_layer(layer, (W / 2, st.px(120)), s["title"].upper(), st.f("display_bold", 44), st.c("text"),
                         tracking=5 * st.k, anchor="ms", opacity=A * clamp(t / 0.4), shadow=5 * st.k)
        if s.get("match"):
            a = A * ease("out", (t - 0.6) / 0.4)
            cx, cy = W / 2, self.panels[0]["rect"][1] + self.panels[0]["rect"][3] / 2
            r = st.px(30)
            col = rgb(st.c("accent"), 1)

            def fnb(d, T, ss):
                d.ellipse([*T(cx - r, cy - r), *T(cx + r, cy + r)], fill=(10, 12, 15, 235), outline=col,
                          width=max(1, int(2 * st.k * ss)))
            G.stamp(layer, (cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4), fnb, opacity=a)
            G.text_layer(layer, (cx, cy), s["match"], st.f("display_semi", 26), st.c("accent"), anchor="mm",
                         opacity=a)
        base = img.convert("RGBA")
        base.alpha_composite(layer)
        return self.finish(base.convert("RGB"), i, t)


class Timeline(Graphic):
    """Chronology graphic, optionally with trailer thumbnails per event."""

    def __init__(self, spec, ctx, dur, n):
        super().__init__(spec, ctx, dur, n)
        self.thumbs = []
        tw = int(self.W * 0.15)
        th = int(tw * 9 / 16)
        for ev in spec["events"]:
            if ev.get("at") is not None:
                f = grab_frame(ctx.src, parse_time(ev["at"]), ctx.cw, ctx.ch)
                self.thumbs.append(Image.fromarray(f).resize((tw, th), Image.LANCZOS))
            else:
                self.thumbs.append(None)

    def draw(self, layer, t):
        s, st, W, H = self.spec, self.st, self.W, self.H
        A = 1 - ease("in", clamp(1 - (self.dur - t) / 0.4))
        if s.get("title"):
            G.text_layer(layer, (W / 2, st.px(160)), s["title"].upper(), st.f("display_bold", 52), st.c("text"),
                         tracking=6 * st.k, anchor="ms", opacity=A * clamp(t / 0.4), shadow=5 * st.k)
        if s.get("kicker"):
            G.text_layer(layer, (W / 2, st.px(92)), s["kicker"].upper(), st.f("mono_bold", 20), st.c("accent"),
                         tracking=8 * st.k, anchor="ms", opacity=A * clamp(t / 0.4))
        evs = s["events"]
        n = len(evs)
        x0, x1 = W * 0.09, W * 0.91
        y = H * 0.64
        pl = ease("inout", (t - 0.2) / 0.9)
        col = rgb(st.c("accent"), 1)
        mut = rgb(st.c("muted"), 1)

        def fnl(d, T, ss):
            d.line([T(x0, y), T(lerp(x0, x1, pl), y)], fill=mut[:3] + (180,), width=max(1, int(2 * st.k * ss)))
        G.stamp(layer, (x0 - 4, y - 6, x1 + 4, y + 6), fnl, opacity=A)
        stagger = min(0.6, (self.dur * 0.55) / max(1, n))
        hi = s.get("highlight")
        for j, ev in enumerate(evs):
            x = lerp(x0, x1, (j + 0.5) / n)
            d0 = 0.5 + j * stagger
            a = A * clamp((t - d0) / 0.35)
            if a <= 0:
                continue
            p = ease("outback", (t - d0) / 0.45)
            ecol = rgb(st.c(ev.get("tag", "accent") if ev.get("tag") in ("confirmed", "theory", "speculation")
                            else "accent"), 1)
            r = st.px(10) * p

            def fnn(d, T, ss, x=x, r=r, ecol=ecol, j=j):
                d.ellipse([*T(x - r, y - r), *T(x + r, y + r)], fill=ecol)
                if hi == j:
                    rr = r * (1.8 + 0.4 * math.sin(t * 5))
                    d.ellipse([*T(x - rr, y - rr), *T(x + rr, y + rr)], outline=ecol, width=max(1, int(2 * st.k * ss)))
            G.stamp(layer, (x - st.px(30), y - st.px(30), x + st.px(30), y + st.px(30)), fnn, opacity=a)
            lift = (1 - ease("out", (t - d0) / 0.5)) * st.px(16)
            th = self.thumbs[j]
            label_y = y - st.px(40)
            if th is not None:
                tx, ty = int(x - th.width / 2), int(y - st.px(70) - th.height + lift)
                tile = th.convert("RGBA")
                tile.putalpha(int(255 * a))
                G._paste(layer, tile, tx, ty)

                def fnt(d, T, ss, tx=tx, ty=ty, th=th, ecol=ecol):
                    d.rectangle([*T(tx, ty), *T(tx + th.width, ty + th.height)], outline=ecol,
                                width=max(1, int(1.5 * st.k * ss)))
                    d.line([T(x, ty + th.height), T(x, y - st.px(14))], fill=ecol[:3] + (160,),
                           width=max(1, int(1.5 * st.k * ss)))
                G.stamp(layer, (tx - 3, ty - 3, tx + th.width + 3, y), fnt, opacity=a)
                label_y = ty - st.px(18)
            G.text_layer(layer, (x, label_y + lift), ev["label"], st.f("display_semi", 30), st.c("text"),
                         anchor="ms", opacity=a, shadow=4 * st.k)
            if ev.get("sub"):
                G.text_layer(layer, (x, y + st.px(52) + lift), ev["sub"].upper(), st.f("mono", 17), st.c("muted"),
                             tracking=2.5 * st.k, anchor="ms", opacity=a)


TYPES = {"clip": Clip, "freeze": Freeze, "title": Title, "card": Card, "black": Black, "split": Split,
         "timeline": Timeline}


def build(spec, ctx, dur, n) -> Segment:
    cls = TYPES.get(spec["type"])
    if cls is None:
        raise ValueError(f"unknown segment type {spec['type']!r}")
    return cls(spec, ctx, dur, n)
