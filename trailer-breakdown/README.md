# Trailer Breakdown Studio

A scriptable editing pipeline that turns a movie trailer into a narrated, YouTube-ready
breakdown video. You get freeze-frame analysis with highlights, cinematic camera moves,
motion graphics, an original score, sound design and a loudness-normalised mix.

The first production is **`plans/doomsday_final_trailer.json`**, an 11:47 breakdown of the
*Avengers: Doomsday* final trailer (released 23 Nov 2026).

```
trailer.mp4 ─┐
plan.json ───┼─► tbk.render ─► breakdown.mp4  (1080p/4K, H.264 High, AAC 320k, -14 LUFS)
vo/*.wav ────┘                ─► breakdown.srt          narration captions
  (optional)                  ─► breakdown.chapters.txt  paste into the YouTube description
                              ─► breakdown.script.md     narration script with timecodes
```

## Requirements

- Python 3.10+ with `pip install -r requirements.txt` (numpy, pillow, scipy)
- `ffmpeg` / `ffprobe`
- A narration voice, in order of preference:
  1. **Your own recordings**: drop `vo/<segment-id>.wav` (or .mp3/.m4a) next to `plans/`.
     The edit re-times itself to your takes automatically.
  2. Offline TTS: `apt install rhvoice rhvoice-english` (default voice: `alan`).
     `pico` and `espeak` also work.
- Fonts: Inter / Inter Display, JetBrains Mono, Caladea (resolved with `fc-match`).

## Workflow

```bash
# 1. Analyse the trailer
python -m tbk.analyze sheets media/trailer.mp4 -o work/analysis     # shot list + contact sheets
python -m tbk.analyze grab   media/trailer.mp4 83.4 92.8 -o work/analysis   # gridded frames for x/y
python -m tbk.analyze audio  media/trailer.mp4 -o work/analysis     # loudness map

# 2. Write the plan (see below), then check timing. VO is synthesised and the edit is fitted to it.
python -m tbk.render plans/my_plan.json --timing

# 3. Review stills of any moment (id@seconds or id@percent)
python -m tbk.render plans/my_plan.json --stills p3_sky@85%,p5_twocaps@2.5

# 4. Render a section, a fast preview, or the final video
python -m tbk.render plans/my_plan.json --only 20-30
python -m tbk.render plans/my_plan.json --preview
python -m tbk.render plans/my_plan.json
```

Segments are rendered in parallel and cached by content hash. After editing one segment,
a re-render only rebuilds that segment, then re-mixes and re-encodes.

**Better source = better video.** All coordinates are resolution-independent (0–1 of the
frame). Replace `media/trailer.mp4` with a 1080p or 4K copy of the same trailer and re-run.
Nothing else changes. Set `"resolution": [3840, 2160]` for a 4K master.

## Plan format

```jsonc
{
  "project": {
    "source": "media/trailer.mp4", "output": "out/breakdown.mp4",
    "resolution": [1920, 1080], "fps": "source",
    "voice": {"engine": "rhvoice", "voice": "alan", "rate": 100, "pitch": 96},
    "music": "score",              // generated score, a path to your own track, or "none"
    "lufs": -14, "crf": 17, "burn_captions": false
  },
  "segments": [ ... ]
}
```

### Segment types

| type | what it is | key fields |
|---|---|---|
| `clip` | trailer footage | `in`, `out`, `speed` (number or `[{t, s}]` ramp), `camera`, `shake` |
| `freeze` | held frame for analysis (flash + shutter sfx) | `at`, `camera`, `enhance` (`gamma`/`brightness`/`contrast`, `start`, `ramp`) |
| `title` | chapter card | `number`, `kicker`, `title`, `sub`, `bg_at` |
| `card` | statement / question; `*words*` render in the accent colour | `text`, `kicker`, `sub`, `size` |
| `split` | side-by-side comparison | `left`/`right`: `at`, `label`, `caption`, `camera`, `overlays`; `title`, `match` |
| `timeline` | chronology with trailer thumbnails | `events`: `label`, `sub`, `at`, `tag`; `highlight` |
| `black` | breath / silence | `dur`, `text` |

Common fields: `id`, `vo` (narration), `transition` (`cut`, `fade`, `dip`, `flash`, `whip`,
`whip_left`, `glitch`, `zoom`), `music` (`off`/`low`/`mid`/`high`/`max`), `reveal: true` (riser and
impact), `sfx: [{name, at}]`, `src_audio` (play original audio under a freeze), `chapter`.

Narration markup: `...` is a short pause, `[pause 600]` is an explicit pause in ms, and
`*word*` is emphasis in captions. When narration is longer than a clip, the clip is slowed to fit
with frame blending (`min_speed`, default 0.45). Past that limit it holds the last frame.

### Camera

`"camera": [{"t": 0, "z": 1.0}, {"t": "40%", "x": 0.43, "y": 0.56, "z": 1.7, "ease": "inout"}]`.
`x`/`y` are the point of interest (0–1 of the source frame) and `z` is the zoom. Times are seconds,
`"end"`, `"end-0.5"` or a percentage of the segment, so moves follow the narration.

### Overlays

Canvas-space overlays stay pinned to the object while the camera moves:

| overlay | fields |
|---|---|
| `circle` | `x`, `y`, `r`, `label`, `label_side`, `pulse` (animated draw-on) |
| `bracket` | `x`, `y`, `w`, `h`, `label` (HUD corner brackets) |
| `spotlight` | `x`, `y`, `r` or `rx`/`ry`, `dim`, `blur`, `desaturate` |
| `arrow`, `line` | `from`, `to`, `space` (`canvas`/`screen`), `dashed` |
| `callout` | point `x`, `y` + `text`, `sub`, `dx`, `dy` (leader line to a label) |

Screen-space overlays:

| overlay | fields |
|---|---|
| `lower_third` | `name`, `role` |
| `location` | `text`, `sub` (typewriter HUD label) |
| `quote` | `text`, `speaker`, `pos` (`bottom`/`center`), `reveal` (word-by-word dialogue callout) |
| `tag` | `kind`: `confirmed` / `theory` / `speculation` / `clue`, plus `label`, `text` |
| `hud`, `counter`, `text`, `caption` | analysis frame, "DETAIL 07 / 14", free text, captions |

All overlays take `start`/`end` (segment time). On clips they can use `start_src`/`end_src` in
trailer time instead, which stays correct when the clip is speed-fitted.

## Accuracy rules used in the Doomsday plan

- Dialogue is quoted only where it was both reported by press coverage of the trailer *and*
  located in the audio by forced alignment (`pocketsphinx`). Lines that could not be placed
  appear as attributed quote cards rather than being synced to the wrong shot.
- Every observation is shown on the exact frame it refers to.
- Claims are tagged on screen: **CONFIRMED** (visible on screen or officially stated),
  **THEORY** (evidence-based interpretation), **SPECULATION** (fan theory, unsupported).
