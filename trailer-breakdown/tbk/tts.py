"""Narration voice: offline TTS (RHVoice / Pico / espeak-ng) or drop-in recordings.

Recorded or third-party VO always wins: put `vo/<segment-id>.wav` (any format ffmpeg
reads: .wav/.mp3/.m4a) next to the plan and it replaces the synthetic line.

Script markup understood in `vo` text:
  "..."        -> 450 ms pause         "[pause]" / "[pause 800]" -> explicit pause
  "*word*"     -> emphasis marker for captions only (stripped for speech)
"""
from __future__ import annotations

import hashlib
import html
import os
import re
import shutil
import subprocess

import numpy as np
from scipy import signal

from .media import SR, read_audio, write_wav

PAUSE_RE = re.compile(r"\[pause(?:\s+(\d+))?\]|\.\.\.|…")

# words the synthesiser tends to mangle -> phonetic respellings
RESPELL = {
    "MCU": "M C U", "Marvel's": "Marvels", "Wakanda": "Wah-kahn-dah", "Wakandans": "Wah-kahn-dans",
    "Talokan": "Tah-low-kahn", "Talokanil": "Tah-low-kahn-eel", "Latverian": "Lat-veery-an",
    "Latveria": "Lat-veery-ah", "Mjolnir": "Myol-neer", "Shang-Chi": "Shang Chee", "Shuri": "Shoo-ree",
    "X-Men": "X Men", "Endgame": "End game", "Doom's": "Dooms", "Russo": "Roosso", "Namor": "Nay-more",
    "Pascal": "Pass-kal", "VisionQuest": "Vision Quest", "Thunderbolts*": "Thunderbolts",
    "Sentinels": "Sentinels", "Gambit": "Gam-bit", "Cyclops": "Sigh-clops", "Clea": "Clay-ah",
    "Romijn": "Ro-main", "Steinfeld": "Stine-feld", "Tatum": "Tay-tum", "Hailee": "Hay-lee",
    "Maximoff": "Max-im-off", "Marsden": "Mars-den", "Atwell": "At-well",
}


def clean_for_speech(text: str) -> str:
    t = text.replace("*", "")
    for k, v in RESPELL.items():
        t = re.sub(rf"(?<![\w-]){re.escape(k)}(?![\w-])", v, t)
    t = t.replace("—", ", ").replace(" - ", ", ")
    return t


def to_ssml(text: str) -> str:
    out, pos = [], 0
    for m in PAUSE_RE.finditer(text):
        out.append(html.escape(text[pos:m.start()]))
        ms = int(m.group(1)) if m.group(1) else (450 if m.group(0) in ("...", "…") else 600)
        out.append(f'<break time="{ms}ms"/>')
        pos = m.end()
    out.append(html.escape(text[pos:]))
    return "<speak>" + "".join(out) + "</speak>"


def caption_text(text: str) -> str:
    t = PAUSE_RE.sub(lambda m: "..." if m.group(0) in ("...", "…") else " ", text)
    return re.sub(r"\s+", " ", t.replace("*", "")).strip()


def _synth_rhvoice(text, path, voice="alan", rate=90, pitch=96):
    ssml = to_ssml(clean_for_speech(text))
    subprocess.run(["RHVoice-test", "-s", "-p", voice, "-r", str(rate), "-t", str(pitch), "-o", path],
                   input=ssml.encode(), check=True, capture_output=True)


def _synth_pico(text, path, **kw):
    t = PAUSE_RE.sub(", ", clean_for_speech(text))
    subprocess.run(["pico2wave", "-l", "en-US", "-w", path, t], check=True, capture_output=True)


def _synth_espeak(text, path, voice="en-us+m3", rate=150, **kw):
    t = PAUSE_RE.sub(", ", clean_for_speech(text))
    subprocess.run(["espeak-ng", "-v", voice, "-s", str(rate), "-w", path, t], check=True, capture_output=True)


ENGINES = {"rhvoice": _synth_rhvoice, "pico": _synth_pico, "espeak": _synth_espeak}


# ---------------------------------------------------------------- voice processing


def _sos(kind, f, order=2):
    return signal.butter(order, f, btype=kind, fs=SR, output="sos")


def peaking(f0, gain_db, q=0.9):
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / SR
    al = np.sin(w) / (2 * q)
    b = [1 + al * A, -2 * np.cos(w), 1 - al * A]
    a = [1 + al / A, -2 * np.cos(w), 1 - al / A]
    return np.array(b) / a[0], np.array(a) / a[0]


def compress(x, thresh_db=-24, ratio=3.0, attack=0.005, release=0.12):
    env = np.abs(x)
    # one-pole envelope follower (vectorised via lfilter on rectified signal)
    a_rel = np.exp(-1 / (release * SR))
    env = signal.lfilter([1 - a_rel], [1, -a_rel], env)
    env_db = 20 * np.log10(env + 1e-9)
    over = np.maximum(0, env_db - thresh_db)
    gain_db = -over * (1 - 1 / ratio)
    return x * 10 ** (gain_db / 20)


def room_ir(length=0.35, seed=3):
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    ir = rng.normal(0, 1, (n, 2)) * np.exp(-t / (length / 5))[:, None]
    ir = signal.sosfilt(_sos("lowpass", 5000), ir, axis=0)
    return ir / np.abs(ir).sum(axis=0) * 6


def process_voice(mono: np.ndarray) -> np.ndarray:
    """Broadcast-style chain: HPF, warmth, presence, compression, light room."""
    x = signal.sosfilt(_sos("highpass", 75), mono)
    b, a = peaking(180, 2.5, 0.8)
    x = signal.lfilter(b, a, x)
    b, a = peaking(3200, 2.0, 1.0)
    x = signal.lfilter(b, a, x)
    b, a = peaking(7000, -2.5, 1.2)  # tame synthetic fizz
    x = signal.lfilter(b, a, x)
    x = compress(x)
    rms = np.sqrt(np.mean(x[np.abs(x) > 1e-4] ** 2) + 1e-12)
    x = x * (10 ** (-19 / 20) / rms)
    st = np.stack([x, x], 1)
    wet = signal.oaconvolve(st, room_ir(), axes=0)[: len(x)]
    out = st * 0.92 + wet * 0.10
    return np.clip(out, -0.98, 0.98).astype(np.float32)


def load_any(path: str) -> np.ndarray:
    a = read_audio(path)
    return a.mean(axis=1)


def synth_line(seg_id: str, text: str, cache_dir: str, vo_dir: str | None, engine="rhvoice", **opts):
    """Returns (stereo float32 at SR, source_label)."""
    if vo_dir:
        for ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg"):
            p = os.path.join(vo_dir, seg_id + ext)
            if os.path.exists(p):
                mono = load_any(p)
                return process_voice(mono) if opts.get("process_recorded", True) else np.stack([mono, mono], 1), p
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha1(f"{engine}|{sorted(opts.items())}|{text}".encode()).hexdigest()[:16]
    out = os.path.join(cache_dir, f"{seg_id}_{key}.wav")
    if not os.path.exists(out):
        raw = out + ".raw.wav"
        fn = ENGINES.get(engine)
        if fn is None or not shutil.which({"rhvoice": "RHVoice-test", "pico": "pico2wave",
                                           "espeak": "espeak-ng"}[engine]):
            raise RuntimeError(f"TTS engine {engine!r} not available")
        fn(text, raw, **{k: v for k, v in opts.items() if k in ("voice", "rate", "pitch")})
        mono = load_any(raw)
        # trim leading/trailing silence
        idx = np.where(np.abs(mono) > 0.01)[0]
        if len(idx):
            mono = mono[max(0, idx[0] - int(0.02 * SR)): idx[-1] + int(0.08 * SR)]
        write_wav(out, process_voice(mono))
        os.remove(raw)
    return read_audio(out), out
