"""Small shared helpers: time parsing, easing, keyframe interpolation, colours."""
from __future__ import annotations

import math
import re
from fractions import Fraction

# ---------------------------------------------------------------- time


def parse_time(v, seg_dur: float | None = None) -> float:
    """Accept 12.5, "12.5", "1:23.4", "01:02:03.5", "end", "end-0.5", "50%"."""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    if s.startswith("end"):
        if seg_dur is None:
            raise ValueError(f"'{v}' needs a segment duration")
        rest = s[3:].strip()
        return seg_dur + (float(rest) if rest else 0.0)
    if s.endswith("%"):
        if seg_dur is None:
            raise ValueError(f"'{v}' needs a segment duration")
        return seg_dur * float(s[:-1]) / 100.0
    parts = s.split(":")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total


def fmt_tc(seconds: float, frac: int = 2) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(seconds, 60)
    if frac:
        return f"{int(m):02d}:{s:0{3 + frac}.{frac}f}"
    return f"{int(m):02d}:{int(s):02d}"


def fmt_chapter(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_srt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_fps(v) -> Fraction:
    if isinstance(v, Fraction):
        return v
    if isinstance(v, (int, float)):
        return Fraction(v).limit_denominator(1001)
    s = str(v)
    if "/" in s:
        a, b = s.split("/")
        return Fraction(int(a), int(b))
    return Fraction(float(s)).limit_denominator(1001)


# ---------------------------------------------------------------- easing


def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def lerp(a, b, t):
    return a + (b - a) * t


EASES = {
    "linear": lambda t: t,
    "in": lambda t: t * t * t,
    "out": lambda t: 1 - (1 - t) ** 3,
    "inout": lambda t: 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2,
    "outexpo": lambda t: 1.0 if t >= 1 else 1 - 2 ** (-10 * t),
    "inexpo": lambda t: 0.0 if t <= 0 else 2 ** (10 * t - 10),
    "inoutsine": lambda t: -(math.cos(math.pi * t) - 1) / 2,
    "outback": lambda t: 1 + 2.70158 * (t - 1) ** 3 + 1.70158 * (t - 1) ** 2,
}


def ease(name: str, t: float) -> float:
    return EASES.get((name or "inout").replace("_", "").replace("-", "").lower(), EASES["inout"])(clamp(t))


def env(t: float, start: float, end: float, fin: float = 0.3, fout: float = 0.3) -> float:
    """Opacity envelope for an element visible on [start, end] with fades."""
    if t < start or t > end:
        return 0.0
    a = 1.0
    if fin > 0:
        a = min(a, (t - start) / fin)
    if fout > 0:
        a = min(a, (end - t) / fout)
    return clamp(a)


# ---------------------------------------------------------------- keyframes


def interp_keys(keys: list[dict], t: float, fields: tuple[str, ...], default_ease="inout") -> dict:
    """keys: [{t:sec, field:val, ease:name}, ...] sorted by t (already parsed)."""
    if not keys:
        return {}
    if t <= keys[0]["t"]:
        return {f: keys[0][f] for f in fields}
    if t >= keys[-1]["t"]:
        return {f: keys[-1][f] for f in fields}
    for a, b in zip(keys, keys[1:]):
        if a["t"] <= t <= b["t"]:
            span = b["t"] - a["t"]
            p = 0.0 if span <= 0 else (t - a["t"]) / span
            p = ease(b.get("ease", default_ease), p)
            return {f: lerp(a[f], b[f], p) for f in fields}
    return {f: keys[-1][f] for f in fields}


def norm_keys(keys, seg_dur: float, fields: dict) -> list[dict]:
    """Fill defaults, parse times, carry forward missing fields, sort."""
    if isinstance(keys, dict):
        keys = [keys]
    out = []
    prev = dict(fields)
    for k in keys or []:
        k = dict(k)
        k["t"] = parse_time(k.get("t", 0), seg_dur)
        for f in fields:
            if f not in k:
                k[f] = prev[f]
        prev = {f: k[f] for f in fields}
        out.append(k)
    if not out:
        out = [dict(fields, t=0.0)]
    out.sort(key=lambda k: k["t"])
    return out


# ---------------------------------------------------------------- colour


def rgb(c, alpha: float | None = None):
    if isinstance(c, (tuple, list)):
        c = tuple(int(x) for x in c)
    else:
        s = str(c).lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        c = tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    if alpha is None:
        return c[:3]
    return c[:3] + (int(round(255 * clamp(alpha))),)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "x"
