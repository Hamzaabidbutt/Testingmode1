"""Frame-level transitions. Each takes tail frame A, head frame B, progress p in (0,1)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .util import ease


def _f(a):
    return a.astype(np.float32)


def _u8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def hblur(arr, k):
    """Directional (horizontal) motion blur via squash/stretch resampling."""
    if k < 1.5:
        return arr
    H, W = arr.shape[:2]
    img = Image.fromarray(arr)
    return np.asarray(img.resize((max(8, int(W / k)), H), Image.BILINEAR).resize((W, H), Image.BILINEAR))


def radial_zoom(arr, z, blur):
    H, W = arr.shape[:2]
    img = Image.fromarray(arr)
    if z != 1:
        bw, bh = W / z, H / z
        img = img.resize((W, H), Image.BICUBIC, box=((W - bw) / 2, (H - bh) / 2, (W + bw) / 2, (H + bh) / 2))
    if blur > 1.5:
        img = img.resize((max(8, int(W / blur)), max(8, int(H / blur))), Image.BILINEAR).resize((W, H), Image.BILINEAR)
    return np.asarray(img)


def fade(A, B, p, **kw):
    p = ease("inoutsine", p)
    return _u8(_f(A) * (1 - p) + _f(B) * p)


def dip(A, B, p, color=0, **kw):
    if p < 0.5:
        k = ease("inoutsine", p * 2)
        return _u8(_f(A) * (1 - k) + color * k)
    k = ease("inoutsine", (p - 0.5) * 2)
    return _u8(color * (1 - k) + _f(B) * k)


def flash(A, B, p, **kw):
    w = np.exp(-((p - 0.45) / 0.18) ** 2)  # white peak just before the cut
    base = _f(A) if p < 0.45 else _f(B)
    return _u8(base * (1 - w) + 255 * w)


def whip(A, B, p, direction=1, **kw):
    """Whip pan: A shoots off, B arrives, with motion blur matched to speed."""
    H, W = A.shape[:2]
    e = ease("inout", p)
    speed = 6 * p * (1 - p)  # derivative-shaped blur envelope
    k = 1 + 70 * speed
    off = int(round(e * W)) * direction
    out = np.empty_like(A)
    if direction > 0:
        out[:, :W - off] = A[:, off:] if off < W else out[:, :0]
        out[:, W - off:] = B[:, :off]
    else:
        off = -off
        out[:, off:] = A[:, :W - off]
        out[:, :off] = B[:, W - off:]
    return hblur(out, k)


def glitch(A, B, p, seed=0, **kw):
    """Digital glitch: RGB split + slice displacement around a hard switch."""
    H, W = A.shape[:2]
    rng = np.random.default_rng(seed + int(p * 1000))
    base = (A if p < 0.5 else B).copy()
    amt = np.sin(np.pi * p)
    # slice displacement
    n = int(4 + 10 * amt)
    for _ in range(n):
        y0 = rng.integers(0, H - 8)
        h = int(rng.integers(4, max(6, int(H * 0.08))))
        dx = int(rng.normal(0, W * 0.05 * amt))
        src = B if rng.random() < p else A
        base[y0:y0 + h] = np.roll(src[y0:y0 + h], dx, axis=1)
    # chromatic split
    s = int(W * 0.012 * amt) + 1
    out = base.copy()
    out[..., 0] = np.roll(base[..., 0], s, axis=1)
    out[..., 2] = np.roll(base[..., 2], -s, axis=1)
    # scanline flicker
    if amt > 0.3:
        out[::3] = (out[::3].astype(np.float32) * (1 - 0.25 * amt)).astype(np.uint8)
    return out


def zoom(A, B, p, **kw):
    """Push-in: A rushes forward with zoom blur, B settles from slight overscale."""
    if p < 0.5:
        q = ease("in", p * 2)
        X = radial_zoom(A, 1 + 0.35 * q, 1 + 14 * q)
        Y = None
    else:
        q = ease("out", (p - 0.5) * 2)
        X = radial_zoom(B, 1.25 - 0.25 * q, 1 + 14 * (1 - q))
        Y = None
    mix = np.clip((p - 0.4) / 0.2, 0, 1)
    if 0 < mix < 1:
        Y = radial_zoom(B, 1.25, 15) if p < 0.5 else radial_zoom(A, 1.35, 15)
        return _u8(_f(X) * (1 - mix) + _f(Y) * mix) if p < 0.5 else _u8(_f(Y) * (1 - mix) + _f(X) * mix)
    return X


TRANSITIONS = {"fade": fade, "dip": dip, "flash": flash, "whip": whip, "whip_left": whip, "glitch": glitch,
               "zoom": zoom}


def apply(kind, A, B, p, seed=0):
    if kind == "whip_left":
        return whip(A, B, p, direction=-1)
    fn = TRANSITIONS.get(kind)
    if fn is None:
        raise ValueError(f"unknown transition {kind!r}")
    return fn(A, B, p, seed=seed)
