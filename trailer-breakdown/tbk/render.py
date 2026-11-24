"""Render a breakdown plan (JSON) into a finished video.

  python -m tbk.render plans/doomsday.json                # full render
  python -m tbk.render plans/doomsday.json --preview      # 960x540, fast encode
  python -m tbk.render plans/doomsday.json --only 3-9     # render a segment range (with audio)
  python -m tbk.render plans/doomsday.json --stills hook1@1.2,f3@2.0   # PNG frames for review
  python -m tbk.render plans/doomsday.json --timing       # print resolved timeline only
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

from . import audio as A
from . import segments as S
from . import transitions as TR
from .media import SR, SegReader, VideoWriter, canvas_size, probe, read_audio, write_wav
from .style import Style
from .tts import caption_text, synth_line
from .util import fmt_chapter, fmt_srt, fmt_tc, parse_fps, parse_time

MUSIC_LEVELS = {"off": 0.0, "low": 1.0, "mid": 1.7, "high": 2.4, "max": 3.0}
TRANS_SFX = {"whip": ("whoosh", -0.5), "whip_left": ("whoosh", -0.5), "glitch": ("glitch", -0.5),
             "zoom": ("whoosh_long", -0.6), "flash": ("hit", 0.0)}


def code_hash():
    h = hashlib.sha1()
    for p in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))):
        h.update(open(p, "rb").read())
    return h.hexdigest()[:10]


class Project:
    def __init__(self, plan_path, preview=False):
        self.plan_path = os.path.abspath(plan_path)
        self.root = os.path.dirname(os.path.dirname(self.plan_path))  # project dir (plans/..)
        self.plan = json.load(open(plan_path))
        P = self.plan["project"]
        self.P = P
        self.src = self.path(P["source"])
        self.info = probe(self.src)
        W, H = P.get("resolution", [1920, 1080])
        if preview:
            W, H = W // 2, H // 2
        self.W, self.H = W, H
        self.preview = preview
        fps = P.get("fps", "source")
        self.fps = self.info["fps"] if fps == "source" else parse_fps(fps)
        self.ffps = float(self.fps)
        self.cw, self.ch = canvas_size(self.info["h"], H)
        self.palette = self.plan.get("style", {}).get("palette")
        self.work = self.path(P.get("work_dir", "work"))
        os.makedirs(self.work, exist_ok=True)
        self.voice = P.get("voice", {"engine": "rhvoice", "voice": "alan", "rate": 90, "pitch": 96})
        self.segs = [dict(s) for s in self.plan["segments"]]
        for i, s in enumerate(self.segs):
            s.setdefault("id", f"s{i:03d}")
        self.code = code_hash()

    def path(self, p):
        return p if os.path.isabs(p) else os.path.join(self.root, p)

    # ------------------------------------------------------------------ timing
    def resolve(self):
        fps = self.ffps
        vo_dir = self.path(self.P.get("vo_dir", "vo"))
        cache = os.path.join(self.work, "tts")
        for s in self.segs:
            s["_vo"] = None
            if s.get("vo"):
                opts = {k: v for k, v in self.voice.items() if k != "engine"}
                a, label = synth_line(s["id"], s["vo"], cache, vo_dir, self.voice.get("engine", "rhvoice"), **opts)
                s["_vo"], s["_vo_src"] = a, label
                s["_vo_len"] = len(a) / SR
        # transitions (into each segment)
        for i, s in enumerate(self.segs):
            tr = s.get("transition", {"type": "cut"})
            if isinstance(tr, str):
                tr = {"type": tr}
            tr.setdefault("dur", {"cut": 0, "fade": 0.6, "dip": 0.8, "flash": 0.3, "whip": 0.36, "whip_left": 0.36,
                                  "glitch": 0.3, "zoom": 0.5}.get(tr["type"], 0.4))
            if i == 0 or tr["type"] == "cut":
                tr = {"type": "cut", "dur": 0}
            s["_tr"] = tr
        # durations
        for i, s in enumerate(self.segs):
            nat = S.natural_duration(s, fps)
            dur = float(s["dur"]) if "dur" in s and s["type"] != "clip" else nat
            tr_in = s["_tr"]["dur"]
            s["_vo_off"] = float(s.get("vo_offset", max(0.25, tr_in * 0.7)))
            if s["_vo"] is not None:
                nxt = self.segs[i + 1]["_tr"]["dur"] if i + 1 < len(self.segs) else 0
                need = s["_vo_off"] + s["_vo_len"] + float(s.get("vo_tail", 0.45)) + nxt * 0.5
                sp = s.get("speed", 1.0)
                if (need > dur + 1e-3 and s["type"] == "clip" and s.get("fit", "speed") == "speed"
                        and isinstance(sp, (int, float))):
                    factor = dur / need  # slow the shot down so it covers the narration
                    if factor >= float(s.get("min_speed", 0.45)):
                        s["speed"] = round(float(sp) * factor * 0.995, 4)
                        s["_fitted"] = s["speed"]
                        dur = S.natural_duration(s, fps)
                if need > dur + 1e-3:
                    if s["type"] == "clip" and not s.get("allow_hold", True):
                        raise ValueError(f"{s['id']}: narration ({need:.2f}s) longer than clip ({dur:.2f}s)")
                    s["_held"] = need - dur if s["type"] == "clip" else 0
                    dur = need
            if "min_dur" in s:
                dur = max(dur, float(s["min_dur"]))
            s["_dur"] = dur
            s["_n"] = max(2, int(round(dur * fps)))
        # overlays timed in trailer time (start_src/end_src) follow the clip's final speed
        for s in self.segs:
            if s["type"] != "clip":
                continue
            t_in, sp = parse_time(s["in"]), s.get("speed", 1.0)
            if not isinstance(sp, (int, float)):
                continue
            for o in s.get("overlays", []):
                for k_src, k in (("start_src", "start"), ("end_src", "end")):
                    if k_src in o:
                        o[k] = round(max(0.0, (parse_time(o[k_src]) - t_in) / sp), 3)
        start = 0
        for i, s in enumerate(self.segs):
            m = int(round(s["_tr"]["dur"] * fps))
            if i > 0:
                m = min(m, self.segs[i - 1]["_n"] // 2, s["_n"] // 2)
                start = start + self.segs[i - 1]["_n"] - m
            s["_m"] = m
            s["_start_f"] = start
            s["_t"] = start / fps
            s["_dur"] = s["_n"] / fps
        self.total_f = self.segs[-1]["_start_f"] + self.segs[-1]["_n"]
        self.total = self.total_f / fps
        return self

    def timing_report(self):
        rows = []
        for i, s in enumerate(self.segs):
            held = f" HOLD+{s['_held']:.1f}s" if s.get("_held") else ""
            held += f" speed {s['_fitted']:.2f}" if s.get("_fitted") else ""
            vo = f" vo {s['_vo_len']:.1f}s" if s["_vo"] is not None else ""
            rows.append(f"{i:3d} {fmt_tc(s['_t'])} {s['_dur']:5.2f}s {s['type']:8s} {s['id']:18s} "
                        f"[{s['_tr']['type']}]{vo}{held}")
        rows.append(f"TOTAL {fmt_tc(self.total)} ({self.total_f} frames @ {self.ffps:.3f})")
        return "\n".join(rows)

    # ------------------------------------------------------------------ video
    def seg_key(self, s):
        spec = {k: v for k, v in s.items() if not k.startswith("_")}
        blob = json.dumps([spec, s["_n"], self.W, self.H, str(self.fps), self.palette, self.code,
                           os.path.getmtime(self.src)], sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:14]

    def seg_path(self, i, s):
        d = os.path.join(self.work, "segs_preview" if self.preview else "segs")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{i:03d}_{s['id']}_{self.seg_key(s)}.mkv")

    def job(self, i, s):
        spec = {k: v for k, v in s.items() if not k.startswith("_")}
        if self.P.get("burn_captions") and s["_vo"] is not None:
            spec.setdefault("overlays", [])
            spec["overlays"] = spec["overlays"] + self.caption_overlays(s)
        return {"spec": spec, "n": s["_n"], "dur": s["_dur"], "out": self.seg_path(i, s), "src": self.src,
                "info": self.info, "W": self.W, "H": self.H, "fps": self.fps, "cw": self.cw, "ch": self.ch,
                "palette": self.palette, "preview": self.preview}

    def render_segments(self, idxs, workers=4):
        jobs = [self.job(i, self.segs[i]) for i in idxs]
        todo = [j for j in jobs if not os.path.exists(j["out"])]
        print(f"segments: {len(jobs)} total, {len(jobs) - len(todo)} cached, {len(todo)} to render", flush=True)
        if todo:
            todo.sort(key=lambda j: -j["n"])  # longest first for better packing
            t0 = time.time()
            with Pool(workers) as pool:
                for k, msg in enumerate(pool.imap_unordered(render_job, todo)):
                    print(f"  [{k + 1}/{len(todo)}] {msg}  ({time.time() - t0:.0f}s)", flush=True)
        return [j["out"] for j in jobs]

    def assemble(self, idxs, paths, out_path, audio_path):
        segs = [self.segs[i] for i in idxs]
        crf = 23 if self.preview else int(self.P.get("crf", 17))
        preset = "veryfast" if self.preview else self.P.get("preset", "medium")
        wr = VideoWriter(out_path, self.W, self.H, self.fps, crf=crf, preset=preset, audio=audio_path, final=True)
        tail = []
        for k, (s, p) in enumerate(zip(segs, paths)):
            rd = SegReader(p, self.W, self.H)
            m = s["_m"] if k > 0 else 0
            m_next = segs[k + 1]["_m"] if k + 1 < len(segs) else 0
            n = s["_n"]
            for f in range(n):
                fr = rd.next()
                if f < m and tail:
                    A_ = tail[f] if f < len(tail) else tail[-1]
                    fr = TR.apply(s["_tr"]["type"], A_, fr, (f + 0.5) / m, seed=k * 1000 + f)
                    wr.write(fr)
                elif f >= n - m_next:
                    if f == n - m_next:
                        tail = []
                    tail.append(fr.copy())
                else:
                    wr.write(fr)
            rd.close()
        wr.close()

    # ------------------------------------------------------------------ audio
    def caption_chunks(self, s):
        text = caption_text(s["vo"])
        words = text.split()
        chunks, cur = [], []
        for w in words:
            cur.append(w)
            if len(" ".join(cur)) > 38 or w.endswith((".", "?", "!")) and len(cur) > 3:
                chunks.append(" ".join(cur))
                cur = []
        if cur:
            chunks.append(" ".join(cur))
        total_chars = sum(len(c) for c in chunks) or 1
        t = s["_vo_off"]
        out = []
        for c in chunks:
            d = s["_vo_len"] * len(c) / total_chars
            out.append((t, t + d, c))
            t += d
        return out

    def caption_overlays(self, s):
        return [{"type": "caption", "start": a, "end": b, "text": c} for a, b, c in self.caption_chunks(s)]

    def mix(self, idxs, out_wav):
        segs = [self.segs[i] for i in idxs]
        t_base = segs[0]["_t"]
        last = segs[-1]
        dur = last["_t"] + last["_dur"] - t_base
        mx = A.Mixer(dur)
        sfx_cache = os.path.join(self.work, "sfx")
        rate = 100
        k = int(np.ceil(dur * rate)) + rate
        intensity = np.zeros(k, np.float32)
        music_db = np.zeros(k, np.float32)
        src_db = np.zeros(k, np.float32)
        src_cache = {}

        def src_audio():
            if "a" not in src_cache:
                src_cache["a"] = read_audio(self.src)
            return src_cache["a"]

        def sfx(name, t, gain=None, seed=0):
            x = A.get_sfx(name, sfx_cache, seed)
            mx.place("sfx", x, t, A.SFX_DEFAULT_GAIN.get(name, -14) + (gain or 0))

        for k_i, s in enumerate(segs):
            t0 = s["_t"] - t_base
            d = s["_dur"]
            m_in = s["_m"] / self.ffps if k_i > 0 else 0
            m_out = segs[k_i + 1]["_m"] / self.ffps if k_i + 1 < len(segs) else 0.05
            a, b = int(t0 * rate), int((t0 + d) * rate)
            # --- music intensity
            has_src = (s["type"] == "clip" and s.get("audio", True) is not False)
            featured = has_src and s["_vo"] is None
            lvl = s.get("music", "off" if featured else "low")
            intensity[a:b] = MUSIC_LEVELS.get(lvl, lvl) if isinstance(lvl, str) else float(lvl)
            music_db[a:b] = float(s.get("music_db", -12.0 if featured else 0.0))
            src_db[a:b] = float(s.get("audio_db", 0.0))
            # --- voice
            if s["_vo"] is not None:
                mx.place("vo", s["_vo"], t0 + s["_vo_off"], float(s.get("vo_gain_db", 0)))
            # --- trailer audio for clips (follows speed ramps; fades over held frames)
            if has_src:
                times = S.clip_time_map(s, self.ffps, s["_n"])
                full = src_audio()
                ts = np.arange(int(d * SR)) / SR
                src_t = np.interp(ts, np.arange(len(times)) / self.ffps, times)
                idx = src_t * SR
                i0 = np.clip(idx.astype(np.int64), 0, len(full) - 2)
                fr = (idx - i0)[:, None]
                seg = full[i0] * (1 - fr) + full[i0 + 1] * fr
                natural = (len(times) - (s.get("_held") or 0) * self.ffps) / self.ffps
                if s.get("_held"):
                    h = int(natural * SR)
                    fade = np.ones(len(seg), np.float32)
                    fl = int(0.25 * SR)
                    fade[h:h + fl] = np.linspace(1, 0, len(fade[h:h + fl]))
                    fade[h + fl:] = 0
                    seg = seg * fade[:, None]
                mx.place("src", seg, t0, float(s.get("audio_gain_db", 0.0)),
                         fade_in=max(0.02, m_in), fade_out=max(0.04, m_out))
            # --- original audio under a freeze/graphic (e.g. let a line of dialogue play)
            sa = s.get("src_audio")
            if sa:
                f0 = parse_time(sa["from"])
                f1 = parse_time(sa["to"]) if sa.get("to") is not None else f0 + d
                full = src_audio()
                seg = full[int(f0 * SR): int(f1 * SR)]
                mx.place("src", seg, t0 + parse_time(sa.get("at", 0), d), float(sa.get("gain_db", 0.0)),
                         fade_in=float(sa.get("fade_in", 0.05)), fade_out=float(sa.get("fade_out", 0.2)))
            # --- automatic sound design
            tr = s["_tr"]["type"]
            if tr in TRANS_SFX and k_i > 0:
                nm, off = TRANS_SFX[tr]
                sfx(nm, t0 + off * m_in if nm != "hit" else t0 + m_in * 0.45, seed=k_i % 3)
            if s["type"] == "freeze" and s.get("flash", True):
                sfx("shutter", t0 + m_in * 0.5)
            if s["type"] == "title":
                sfx("impact", t0 + m_in * 0.4)
                sfx("riser_short", t0 + m_in * 0.4 - 1.3, -2)
            if s.get("reveal"):
                sfx("riser", t0 - 2.5 + m_in * 0.4)
                sfx("impact", t0 + m_in * 0.4)
            for o in s.get("overlays", []):
                if o["type"] in ("circle", "bracket", "box") and not o.get("silent"):
                    sfx("tick", t0 + parse_time(o.get("start", 0), d))
                if o["type"] in ("lower_third", "tag", "quote", "callout") and not o.get("silent"):
                    sfx("whoosh", t0 + parse_time(o.get("start", 0), d) - 0.15, -8, seed=2)
            for e in s.get("sfx", []):
                sfx(e["name"], t0 + parse_time(e.get("at", 0), d), e.get("gain_db"), e.get("seed", 0))
        # music: user track or generated score
        music = self.P.get("music", "score")
        if music == "score":
            sm = np.convolve(intensity, np.ones(60) / 60, "same")  # 0.6 s ramps
            bed = A.score(dur + 1, sm[::10] if len(sm) else sm)
        elif music and music != "none":
            bed = read_audio(self.path(music))
            reps = int(np.ceil((dur + 1) * SR / max(1, len(bed))))
            bed = np.tile(bed, (reps, 1))
        else:
            bed = None
        if bed is not None:
            mx.place("music", bed[: mx.n], 0, float(self.P.get("music_gain_db", -4)))
        sm_db = np.convolve(music_db, np.ones(40) / 40, "same")
        sm_src = np.convolve(src_db, np.ones(10) / 10, "same")
        out = mx.render(sm_db, sm_src, rate, float(self.P.get("duck_music_db", -9)),
                        float(self.P.get("duck_src_db", -16)))
        out = out[: int(dur * SR)]
        raw = out_wav + ".raw.wav"
        write_wav(raw, out)
        m = A.loudnorm(raw, out_wav, float(self.P.get("lufs", -14)))
        os.remove(raw)
        print(f"audio: input {m['input_i']} LUFS -> {self.P.get('lufs', -14)} LUFS", flush=True)

    # ------------------------------------------------------------------ exports
    def exports(self, base):
        srt, n = [], 1
        for s in self.segs:
            if s["_vo"] is None:
                continue
            for a, b, c in self.caption_chunks(s):
                srt.append(f"{n}\n{fmt_srt(s['_t'] + a)} --> {fmt_srt(s['_t'] + b)}\n{c}\n")
                n += 1
        open(base + ".srt", "w").write("\n".join(srt))
        chapters = []
        for s in self.segs:
            if s.get("chapter") or s["type"] == "title":
                name = s.get("chapter") or (s.get("title") or "").title()
                chapters.append((s["_t"], name))
        if not chapters or chapters[0][0] > 0.5:
            chapters.insert(0, (0.0, self.P.get("intro_chapter", "Intro")))
        else:
            chapters[0] = (0.0, chapters[0][1])
        open(base + ".chapters.txt", "w").write("\n".join(f"{fmt_chapter(t)} {n}" for t, n in chapters) + "\n")
        lines = [f"# {self.P.get('title', 'Breakdown')} - narration script\n",
                 f"Runtime {fmt_tc(self.total, 0)}. Timecodes are positions in the finished video.\n"]
        for s in self.segs:
            if s["_vo"] is None:
                continue
            lines.append(f"**[{fmt_tc(s['_t'], 1)}] `{s['id']}`** ({s['type']})  \n{caption_text(s['vo'])}\n")
        open(base + ".script.md", "w").write("\n".join(lines))


# ---------------------------------------------------------------------- worker


def make_ctx(j):
    st = Style(j["W"], j["H"], j["palette"])
    return S.Ctx(j["src"], j["info"], j["W"], j["H"], j["fps"], st, j["cw"], j["ch"])


def render_job(j):
    t0 = time.time()
    ctx = make_ctx(j)
    seg = S.build(j["spec"], ctx, j["dur"], j["n"])
    tmp = j["out"] + ".tmp.mkv"
    wr = VideoWriter(tmp, j["W"], j["H"], j["fps"], crf=14 if not j["preview"] else 20, preset="veryfast")
    try:
        for fr in seg.frames():
            wr.write(fr)
    finally:
        seg.close()
        wr.close()
    os.replace(tmp, j["out"])
    return f"{os.path.basename(j['out'])} {j['n']} frames in {time.time() - t0:.1f}s"


def stills(proj: Project, specs: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    by_id = {s["id"]: (i, s) for i, s in enumerate(proj.segs)}
    for item in specs.split(","):
        sid, t = item.split("@")
        i, s = by_id[sid]
        j = proj.job(i, s)
        ctx = make_ctx(j)
        seg = S.build(j["spec"], ctx, j["dur"], j["n"])
        target = min(s["_n"] - 1, int(parse_time(t, s["_dur"]) * proj.ffps))
        if isinstance(seg, S.Clip):  # decode up to the target without rendering
            for f in range(target):
                seg.reader.get(float(seg.times[f]))
        fr = seg.frame(target, target / proj.ffps)
        seg.close()
        p = os.path.join(outdir, f"{sid}@{t}.png")
        Image.fromarray(fr).save(p)
        print(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--only", help="segment index range a-b (inclusive)")
    ap.add_argument("--stills", help="id@time,id@time")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    proj = Project(a.plan, preview=a.preview).resolve()
    print(proj.timing_report(), flush=True)
    if a.timing:
        return
    if a.stills:
        stills(proj, a.stills, os.path.join(proj.work, "stills"))
        return
    idxs = list(range(len(proj.segs)))
    if a.only:
        lo, hi = (int(x) for x in a.only.split("-"))
        idxs = idxs[lo:hi + 1]
    out = a.out or proj.path(proj.P.get("output", "out/breakdown.mp4"))
    if a.preview and not a.out:
        out = out.replace(".mp4", "_preview.mp4")
    if a.only and not a.out:
        out = out.replace(".mp4", f"_{a.only}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    paths = proj.render_segments(idxs, a.workers)
    wav = os.path.join(proj.work, "mix_preview.wav" if a.preview else "mix.wav")
    proj.mix(idxs, wav)
    print("assembling...", flush=True)
    proj.assemble(idxs, paths, out, wav)
    if not a.only:
        proj.exports(os.path.splitext(out)[0])
    print(f"done: {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
