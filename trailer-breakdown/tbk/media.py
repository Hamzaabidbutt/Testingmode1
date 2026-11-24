"""ffmpeg plumbing: probing, streaming frame reads, audio reads, encoding pipes."""
from __future__ import annotations

import json
import subprocess
from fractions import Fraction

import numpy as np

SR = 48000


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def probe(path: str) -> dict:
    out = run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
              capture_output=True, text=True).stdout
    info = json.loads(out)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        raise RuntimeError(f"{path}: no video stream")
    fr = v.get("avg_frame_rate") or v.get("r_frame_rate") or "24/1"
    if fr in ("0/0", ""):
        fr = v.get("r_frame_rate", "24/1")
    num, den = (int(x) for x in fr.split("/"))
    sar = v.get("sample_aspect_ratio", "1:1")
    w, h = int(v["width"]), int(v["height"])
    if sar not in ("0:1", "1:1", "N/A"):
        sn, sd = (int(x) for x in sar.split(":"))
        w = int(round(w * sn / sd))
    return {
        "w": w, "h": h, "fps": Fraction(num, den),
        "duration": float(info["format"].get("duration", v.get("duration", 0)) or 0),
        "has_audio": a is not None,
    }


def canvas_size(src_h: int, out_h: int) -> tuple[int, int]:
    """16:9 working canvas. Keep source detail for zooms (capped at 2160 lines)."""
    h = max(out_h, min(src_h, 2160))
    h -= h % 2
    w = int(round(h * 16 / 9))
    w -= w % 2
    return w, h


def _scale_pad(cw, ch):
    return (f"scale={cw}:{ch}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={cw}:{ch}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format=rgb24")


class FrameReader:
    """Sequential reader. `get(t)` must be called with non-decreasing source times."""

    def __init__(self, path: str, start: float, cw: int, ch: int, src_fps: Fraction):
        self.cw, self.ch = cw, ch
        self.start = max(0.0, start)
        self.dt = 1.0 / float(src_fps)
        self.fsize = cw * ch * 3
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-ss", f"{self.start:.4f}", "-i", path, "-an", "-sn",
             "-vf", _scale_pad(cw, ch), "-f", "rawvideo", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=self.fsize * 2)
        self.idx = -1
        self.cur = None
        self.prev = None
        self.eof = False

    def _read(self):
        buf = self.proc.stdout.read(self.fsize)
        if len(buf) < self.fsize:
            self.eof = True
            return False
        self.prev = self.cur
        self.cur = np.frombuffer(buf, np.uint8).reshape(self.ch, self.cw, 3)
        self.idx += 1
        return True

    def get(self, t: float) -> np.ndarray:
        """Frame showing at source time t (holds the last frame past EOF)."""
        target = int(np.floor((t - self.start) / self.dt + 1e-6))
        if self.cur is None:
            if not self._read():
                raise RuntimeError(f"could not decode frame at {t:.3f}s")
        while self.idx < target and not self.eof:
            if not self._read():
                break
        return self.cur

    def get_blend(self, t: float) -> np.ndarray:
        """Like get(), but cross-blends neighbouring frames for smooth slow motion."""
        f = max(0.0, (t - self.start) / self.dt)
        i0 = int(np.floor(f + 1e-6))
        frac = f - i0
        if self.cur is None and not self._read():
            raise RuntimeError(f"could not decode frame at {t:.3f}s")
        while self.idx < i0 + 1 and not self.eof:
            if not self._read():
                break
        if self.idx == i0 + 1 and self.prev is not None:
            if frac < 0.03:
                return self.prev
            if frac > 0.97:
                return self.cur
            from PIL import Image
            return np.asarray(Image.blend(Image.fromarray(self.prev), Image.fromarray(self.cur), frac))
        return self.cur

    def close(self):
        try:
            self.proc.stdout.close()
            self.proc.kill()
            self.proc.wait()
        except Exception:
            pass


def grab_frame(path: str, t: float, cw: int, ch: int) -> np.ndarray:
    out = run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.4f}", "-i", path, "-frames:v", "1",
               "-an", "-vf", _scale_pad(cw, ch), "-f", "rawvideo", "-"], capture_output=True).stdout
    if len(out) < cw * ch * 3:
        raise RuntimeError(f"no frame at {t:.3f}s in {path}")
    return np.frombuffer(out[:cw * ch * 3], np.uint8).reshape(ch, cw, 3)


def read_audio(path: str, start: float = 0.0, dur: float | None = None) -> np.ndarray:
    """float32 stereo array (n, 2) at SR."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.4f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.4f}"]
    cmd += ["-i", path, "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    try:
        raw = run(cmd, capture_output=True).stdout
    except subprocess.CalledProcessError:
        return np.zeros((0, 2), np.float32)
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()


def write_wav(path: str, data: np.ndarray, sr: int = SR):
    data = np.asarray(data, np.float32)
    if data.ndim == 1:
        data = np.stack([data, data], 1)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac",
                          str(data.shape[1]), "-i", "-", "-c:a", "pcm_f32le", path],
                         stdin=subprocess.PIPE)
    p.stdin.write(data.tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError(f"failed writing {path}")


class VideoWriter:
    """Raw RGB frames in, encoded file out."""

    def __init__(self, path: str, w: int, h: int, fps: Fraction, *, crf=12, preset="veryfast",
                 audio: str | None = None, final=False, codec="libx264"):
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", f"{fps.numerator}/{fps.denominator}", "-i", "-"]
        if audio:
            cmd += ["-i", audio]
        cmd += ["-c:v", codec, "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
        if final:
            cmd += ["-profile:v", "high", "-tune", "film", "-g", str(int(round(float(fps) * 2))),
                    "-bf", "2", "-colorspace", "bt709", "-color_primaries", "bt709",
                    "-color_trc", "bt709", "-movflags", "+faststart"]
        if audio:
            cmd += ["-c:a", "aac", "-b:a", "320k", "-ar", str(SR), "-shortest"]
        cmd += [path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=w * h * 3 * 2)
        self.n = 0

    def write(self, frame):
        if not isinstance(frame, (bytes, bytearray)):
            frame = np.ascontiguousarray(frame, np.uint8).tobytes()
        self.proc.stdin.write(frame)
        self.n += 1

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait():
            raise RuntimeError("ffmpeg encoder failed")


class SegReader:
    """Reads an intermediate segment back as RGB frames."""

    def __init__(self, path: str, w: int, h: int):
        self.w, self.h = w, h
        self.fsize = w * h * 3
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
                                      "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE,
                                     bufsize=self.fsize * 2)
        self.last = None

    def next(self) -> np.ndarray:
        buf = self.proc.stdout.read(self.fsize)
        if len(buf) < self.fsize:
            if self.last is None:
                raise RuntimeError("empty segment")
            return self.last  # pad by holding last frame
        self.last = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
        return self.last

    def close(self):
        try:
            self.proc.stdout.close()
            self.proc.kill()
            self.proc.wait()
        except Exception:
            pass
