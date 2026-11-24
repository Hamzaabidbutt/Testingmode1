"""Sound design + mix: synthesised SFX, an original tension score, ducking, loudness.

Everything here is generated procedurally, so the edit ships with no third-party
music or sound-library licensing issues. Swap in a licensed track with
`project.music = "path/to/track.wav"` if you prefer.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
from scipy import signal

from .media import SR, read_audio, write_wav


def _sos(kind, f, order=4):
    return signal.butter(order, f, btype=kind, fs=SR, output="sos")


def _rng(seed):
    return np.random.default_rng(seed)


def _st(x, width=0.0, seed=0):
    """mono -> stereo with optional decorrelated width."""
    if width <= 0:
        return np.stack([x, x], 1)
    d = int(SR * 0.011)
    r = np.concatenate([np.zeros(d), x[:-d]]) if len(x) > d else x
    return np.stack([x * (1 - width * 0.3) + r * width * 0.3, r * (1 - width * 0.3) + x * width * 0.3], 1)


def _ir(length, seed=1, damp=6000):
    rng = _rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    ir = rng.normal(0, 1, (n, 2)) * np.exp(-t * 6.9 / length)[:, None]
    ir = signal.sosfilt(_sos("lowpass", damp, 2), ir, axis=0)
    return ir / np.sqrt((ir ** 2).sum(axis=0))


def reverb(x, length=2.4, wet=0.3, seed=1, damp=6000):
    if x.ndim == 1:
        x = _st(x)
    w = signal.oaconvolve(x, _ir(length, seed, damp), axes=0)[: len(x)]
    return x * (1 - wet) + w * wet


def _norm(x, peak=0.9):
    m = np.abs(x).max()
    return x * (peak / m) if m > 0 else x


def _sweep_noise(n, f0, f1, seed, bands=10, q=1.6):
    """Band-pass noise whose centre glides f0 -> f1 (crossfaded band bank)."""
    rng = _rng(seed)
    noise = rng.normal(0, 1, n)
    freqs = np.geomspace(min(f0, f1), max(f0, f1), bands)
    layers = []
    for fc in freqs:
        lo, hi = fc / (1 + 1 / q), min(fc * (1 + 1 / q), SR / 2 - 100)
        layers.append(signal.sosfilt(signal.butter(2, [lo, hi], btype="band", fs=SR, output="sos"), noise))
    pos = np.linspace(0, 1, n)
    pos = pos if f1 >= f0 else 1 - pos
    idx = pos * (bands - 1)
    out = np.zeros(n)
    for i, L in enumerate(layers):
        w = np.clip(1 - np.abs(idx - i), 0, 1)
        out += L * w
    return out


# ====================================================================== SFX


def sfx_whoosh(d=0.6, seed=1, up=True):
    n = int(d * SR)
    t = np.linspace(0, 1, n)
    x = _sweep_noise(n, 250, 3500, seed) if up else _sweep_noise(n, 3500, 250, seed)
    env = np.sin(np.pi * t ** 0.8) ** 2
    x = x * env
    return _norm(reverb(_st(x, 0.8, seed), 0.8, 0.2, seed), 0.7)


def sfx_impact(d=2.5, seed=2):
    n = int(d * SR)
    t = np.arange(n) / SR
    f = 40 + 70 * np.exp(-t / 0.05)
    ph = 2 * np.pi * np.cumsum(f) / SR
    boom = np.sin(ph) * np.exp(-t / 0.7)
    boom = np.tanh(boom * 2.2) * 0.8
    rng = _rng(seed)
    crack = rng.normal(0, 1, n) * np.exp(-t / 0.03)
    crack = signal.sosfilt(_sos("lowpass", 2500, 2), crack)
    x = boom + crack * 0.6
    return _norm(reverb(_st(x, 0.5, seed), 2.8, 0.28, seed, 3500), 0.95)


def sfx_hit(d=1.6, seed=3):
    """Shorter, tighter bass hit for scene changes."""
    n = int(d * SR)
    t = np.arange(n) / SR
    f = 48 + 40 * np.exp(-t / 0.03)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.35)
    x = np.tanh(x * 1.8)
    return _norm(reverb(_st(x), 1.2, 0.15, seed, 3000), 0.85)


def sfx_riser(d=2.5, seed=4):
    n = int(d * SR)
    t = np.linspace(0, 1, n)
    noise = _sweep_noise(n, 300, 6000, seed, 12)
    f = 140 * (6 ** t)
    trem = 1 + 0.35 * np.sin(2 * np.pi * np.cumsum(4 + 22 * t ** 2) / SR)
    tone = (np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.5 * np.sin(2 * np.pi * np.cumsum(f * 1.5) / SR)) * trem
    env = t ** 2.6
    x = (noise * 0.9 + tone * 0.25) * env
    x[-int(0.02 * SR):] *= np.linspace(1, 0, int(0.02 * SR))
    return _norm(reverb(_st(x, 0.9, seed), 1.5, 0.25, seed), 0.6)


def sfx_swell(d=1.6, seed=5):
    """Reverse-cymbal style suck-in, peaks at the end."""
    n = int(d * SR)
    t = np.linspace(0, 1, n)
    rng = _rng(seed)
    x = signal.sosfilt(_sos("highpass", 1500, 2), rng.normal(0, 1, n)) * np.exp(5 * (t - 1))
    x = _st(x, 0.8, seed)
    return _norm(reverb(x[::-1], 1.8, 0.4, seed)[::-1], 0.55)


def sfx_tick(seed=6):
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * 2200 * t) * np.exp(-t / 0.008) + np.sin(2 * np.pi * 3300 * t) * np.exp(-t / 0.004) * 0.5
    return _norm(reverb(_st(x), 0.4, 0.15, seed), 0.35)


def sfx_glitch(d=0.4, seed=7):
    rng = _rng(seed)
    n = int(d * SR)
    x = np.zeros(n)
    pos = 0
    while pos < n:
        L = int(rng.uniform(0.01, 0.05) * SR)
        kind = rng.integers(0, 3)
        tt = np.arange(min(L, n - pos)) / SR
        if kind == 0:
            seg = np.sign(np.sin(2 * np.pi * rng.uniform(200, 1800) * tt))
        elif kind == 1:
            seg = rng.normal(0, 1, len(tt))
        else:
            seg = np.zeros(len(tt))
        x[pos:pos + len(tt)] = seg * rng.uniform(0.3, 1.0)
        pos += len(tt)
    x = np.round(x * 6) / 6  # bitcrush
    return _norm(_st(signal.sosfilt(_sos("highpass", 150, 2), x), 0.6, seed), 0.4)


def sfx_shutter(seed=8):
    rng = _rng(seed)
    n = int(0.25 * SR)
    x = np.zeros(n)
    for off, a in ((0, 1.0), (0.045, 0.7)):
        i = int(off * SR)
        L = int(0.012 * SR)
        x[i:i + L] += rng.normal(0, 1, L) * np.exp(-np.arange(L) / (0.002 * SR)) * a
    x = signal.sosfilt(_sos("highpass", 1800, 2), x)
    return _norm(reverb(_st(x), 0.5, 0.2, seed), 0.45)


def sfx_sub_drop(d=1.8, seed=9):
    n = int(d * SR)
    t = np.arange(n) / SR
    f = 30 + 50 * np.exp(-t / 0.5)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.9) * np.minimum(1, t / 0.01)
    return _norm(_st(x), 0.8)


SFX = {
    "whoosh": lambda s: sfx_whoosh(0.55, s), "whoosh_long": lambda s: sfx_whoosh(1.0, s),
    "impact": sfx_impact, "hit": sfx_hit, "riser": sfx_riser, "riser_short": lambda s: sfx_riser(1.3, s),
    "swell": sfx_swell, "tick": lambda s: sfx_tick(s), "glitch": sfx_glitch, "shutter": lambda s: sfx_shutter(s),
    "sub_drop": sfx_sub_drop,
}
SFX_DEFAULT_GAIN = {"whoosh": -14, "whoosh_long": -13, "impact": -7, "hit": -10, "riser": -13, "riser_short": -15,
                    "swell": -14, "tick": -22, "glitch": -16, "shutter": -15, "sub_drop": -10}


def get_sfx(name, cache_dir, seed=0):
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, f"sfx_{name}_{seed}.npy")
    if os.path.exists(p):
        return np.load(p)
    x = SFX[name](seed + 11).astype(np.float32)
    np.save(p, x)
    return x


# ====================================================================== score

NOTE = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7,
        "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


def hz(name):
    n, o = name[:-1], int(name[-1])
    return 440.0 * 2 ** ((NOTE[n] + 12 * (o + 1) - 69) / 12)


# D minor -> Bb -> G minor -> A (sus to major): dark, unresolved, cinematic
CHORDS = [["D2", "A2", "D3", "F3", "A3"], ["Bb1", "F2", "D3", "F3", "Bb3"], ["G1", "D2", "G2", "Bb2", "D3"],
          ["A1", "E2", "A2", "D3", "E3"], ["D2", "A2", "F3", "A3", "D4"], ["Bb1", "F2", "Bb2", "D3", "F3"],
          ["C2", "G2", "C3", "E3", "G3"], ["A1", "E2", "A2", "C#3", "E3"]]

_TABLE = None


def _saw_table(n=4096, harm=48):
    global _TABLE
    if _TABLE is None:
        ph = np.arange(n) / n
        _TABLE = sum(np.sin(2 * np.pi * k * ph) / k for k in range(1, harm + 1)) * (2 / np.pi)
    return _TABLE


def _osc(freq, n, seed, table=None):
    table = _saw_table() if table is None else table
    ph0 = _rng(seed).random()
    ph = (ph0 + freq * np.arange(n) / SR) % 1.0
    return np.interp(ph * len(table), np.arange(len(table)), table, period=len(table))


def score(duration: float, intensity: np.ndarray, seed=21, chord_len=8.0) -> np.ndarray:
    """Original ambient tension score. `intensity` sampled at 10 Hz, 0..3."""
    n = int(duration * SR)
    t_ctrl = np.arange(len(intensity)) / 10.0
    I = np.interp(np.arange(n) / SR, t_ctrl, intensity).astype(np.float32)
    pad = np.zeros(n, np.float32)
    xf = int(1.5 * SR)
    L = int(chord_len * SR)
    for ci, start in enumerate(range(0, n, L)):
        chord = CHORDS[ci % len(CHORDS)]
        seg_n = min(n - start, L + xf)
        block = np.zeros(seg_n, np.float32)
        for vi, nm in enumerate(chord):
            f = hz(nm)
            for dt in (-0.07, 0.0, 0.07):  # detune in semitones (~7 cents)
                block += _osc(f * 2 ** (dt / 12), seg_n, seed + ci * 31 + vi * 7) / (len(chord) * 3)
        env = np.ones(seg_n, np.float32)
        a = min(xf, seg_n)
        env[:a] = np.linspace(0, 1, a) if start > 0 else np.linspace(0.2, 1, a)
        if seg_n > L:
            env[L:] = np.linspace(1, 0, seg_n - L)
        pad[start:start + seg_n] += block * env
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n) / SR / 11.0)
    pad_dark = signal.sosfilt(_sos("lowpass", 520, 2), pad)
    pad_bright = signal.sosfilt(_sos("lowpass", 1700, 2), pad)
    bright = np.clip((I - 1) / 2, 0, 1) * (0.6 + 0.4 * lfo)
    pad = pad_dark * (1 - bright) + pad_bright * bright
    # sub drone on the root
    tt = np.arange(n) / SR
    sub = np.sin(2 * np.pi * hz("D1") * tt) * 0.35
    # heartbeat pulse (intensity >= 1.5)
    bpm = 66
    beat = np.zeros(n, np.float32)
    kick_n = int(0.35 * SR)
    kt = np.arange(kick_n) / SR
    kick = np.sin(2 * np.pi * np.cumsum(45 + 60 * np.exp(-kt / 0.03)) / SR) * np.exp(-kt / 0.12)
    period = 60 / bpm
    for k, bt in enumerate(np.arange(0, duration, period)):
        for off, amp in ((0.0, 1.0), (0.22, 0.6)):
            i = int((bt + off) * SR)
            if i + kick_n < n:
                beat[i:i + kick_n] += kick * amp
    beat *= np.clip(I - 1.2, 0, 1)
    # staccato low-string ostinato (intensity >= 2.3)
    ost = np.zeros(n, np.float32)
    eighth = 60 / 132 / 2
    note_n = int(0.16 * SR)
    nt = np.arange(note_n) / SR
    env_s = np.exp(-nt / 0.06) * np.minimum(1, nt / 0.004)
    pattern = ["D2", "D2", "F2", "D2", "A2", "D2", "F2", "E2"]
    tone_cache = {}
    for k, bt in enumerate(np.arange(0, duration, eighth)):
        i = int(bt * SR)
        if i + note_n >= n or I[i] < 2.2:
            continue
        nm = pattern[k % len(pattern)]
        if nm not in tone_cache:
            tone_cache[nm] = _osc(hz(nm), note_n, 3) * env_s
        ost[i:i + note_n] += tone_cache[nm] * min(1.0, (I[i] - 2.2) / 0.6)
    ost = signal.sosfilt(_sos("lowpass", 1400, 2), ost)
    # airy texture
    air = _rng(seed).normal(0, 1, n).astype(np.float32)
    air = signal.sosfilt(signal.butter(2, [2500, 6000], btype="band", fs=SR, output="sos"), air) * 0.05
    air *= (0.5 + 0.5 * np.sin(2 * np.pi * tt / 7.3)) * np.clip(I, 0, 1)
    level = np.clip(I, 0, 1) ** 1.2
    mono_pad = (pad * 0.9 + sub * 0.5) * level
    mix = _st(mono_pad, 0.9, seed) + _st(air, 0.9, seed + 1) + _st(beat * 0.9 + ost * 0.45, 0.2, seed + 2)
    mix = reverb(mix, 3.2, 0.32, seed, 4500)
    return (_norm(mix, 0.5)).astype(np.float32)


# ====================================================================== mixing


def db(x):
    return 10 ** (x / 20)


def smooth_env(mask: np.ndarray, attack=0.06, release=0.4, rate=100):
    """One-pole attack/release smoothing of a 0..1 control signal sampled at `rate` Hz."""
    out = np.zeros_like(mask, dtype=np.float32)
    a_att = np.exp(-1 / (attack * rate))
    a_rel = np.exp(-1 / (release * rate))
    v = 0.0
    for i, m in enumerate(mask):
        a = a_att if m > v else a_rel
        v = a * v + (1 - a) * m
        out[i] = v
    return out


class Mixer:
    def __init__(self, duration: float):
        self.n = int(np.ceil(duration * SR)) + SR
        self.bus = {k: np.zeros((self.n, 2), np.float32) for k in ("vo", "src", "music", "sfx")}

    def place(self, bus, x, t, gain_db=0.0, fade_in=0.0, fade_out=0.0):
        if x is None or len(x) == 0:
            return
        x = np.asarray(x, np.float32)
        if x.ndim == 1:
            x = np.stack([x, x], 1)
        x = x * db(gain_db)
        fi, fo = min(int(fade_in * SR), len(x)), min(int(fade_out * SR), len(x))
        if fi > 0:
            x[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)[:, None]
        if fo > 0:
            x[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)[:, None]
        i = int(round(t * SR))
        if i < 0:
            x = x[-i:]
            i = 0
        j = min(self.n, i + len(x))
        if j > i:
            self.bus[bus][i:j] += x[: j - i]

    def render(self, music_level_db: np.ndarray, src_level_db: np.ndarray, rate=100, duck_music=-9.0,
               duck_src=-13.0):
        """music/src level curves at `rate` Hz (dB). VO ducks music and trailer audio."""
        hop = SR // rate
        k = self.n // hop
        vo = self.bus["vo"][: k * hop].reshape(k, hop, 2)
        vo_rms = np.sqrt((vo ** 2).mean(axis=(1, 2)) + 1e-12)
        active = (vo_rms > db(-50)).astype(np.float32)
        # dilate so ducking anticipates speech slightly
        active = np.convolve(active, np.ones(15), "same") > 0
        duck = smooth_env(active.astype(np.float32), 0.08, 0.5, rate)

        def curve(levels_db, duck_db):
            lv = levels_db[:k] if len(levels_db) >= k else np.pad(levels_db, (0, k - len(levels_db)), mode="edge")
            g = db(lv) * db(duck_db * duck)
            return np.repeat(g, hop)[:, None]

        out = np.zeros((self.n, 2), np.float32)
        m = k * hop
        out[:m] += self.bus["vo"][:m]
        out[:m] += self.bus["src"][:m] * curve(src_level_db, duck_src)
        out[:m] += self.bus["music"][:m] * curve(music_level_db, duck_music)
        out[:m] += self.bus["sfx"][:m]
        # gentle bus glue + safety
        out = np.tanh(out * 1.1) / 1.1
        return out


def loudnorm(in_wav, out_wav, target=-14.0, tp=-1.5, lra=11):
    p = subprocess.run(["ffmpeg", "-v", "info", "-y", "-i", in_wav, "-af",
                        f"loudnorm=I={target}:TP={tp}:LRA={lra}:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    txt = p.stderr[p.stderr.rfind("{"): p.stderr.rfind("}") + 1]
    m = json.loads(txt)
    af = (f"loudnorm=I={target}:TP={tp}:LRA={lra}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:"
          f"linear=true:print_format=summary")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", in_wav, "-af", af, "-ar", str(SR), "-c:a", "pcm_s24le",
                    out_wav], check=True)
    return m
