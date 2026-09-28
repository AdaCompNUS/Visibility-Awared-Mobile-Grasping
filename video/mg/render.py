"""Render the video: sections in parallel (frames piped to ffmpeg), then audio + concat.

  pixi run python -m mg.render preview   -> 640x360 quick pass of every section (every 3rd frame)
  pixi run python -m mg.render stills    -> a few PNG stills per section for review
  pixi run python -m mg.render full      -> full 1080p30 render + audio mux -> out/final/*.mp4
  pixi run python -m mg.render section <name> [scale]
"""
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from . import core as C
from . import scenes

OUT = os.path.join(C.ROOT, "out")
SEC_DIR = os.path.join(OUT, "sections")
FINAL_DIR = os.path.join(OUT, "final")
NARR = os.path.join(C.ROOT, "narration")
FPS = C.FPS


def _ffmpeg_writer(path, w, h, fps=FPS, crf=16):
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_section(args):
    name, scale, step, outdir = args
    sec = [s for s in scenes.SECTIONS if s[0] == name][0]
    _, dur, fn, _, _ = sec
    ctx = scenes.Ctx()
    n = int(round(dur * FPS))
    w, h = int(C.W * scale) // 2 * 2, int(C.H * scale) // 2 * 2
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.mp4")
    fps_out = FPS / step
    p = _ffmpeg_writer(path, w, h, fps_out)
    t0 = time.time()
    for i in range(0, n, step):
        img = fn(i / FPS, ctx)
        if scale != 1.0:
            img = img.resize((w, h), Image.BILINEAR)
        p.stdin.write(np.asarray(img.convert("RGB")).tobytes())
    p.stdin.close()
    p.wait()
    dt = time.time() - t0
    return name, n // step, dt, path


def stills(names=None, scale=0.5):
    ctx = scenes.Ctx()
    d = os.path.join(OUT, "stills")
    os.makedirs(d, exist_ok=True)
    for name, dur, fn, _, _ in scenes.SECTIONS:
        if names and name not in names:
            continue
        for k in range(8):
            t = dur * (k + 0.5) / 8
            img = fn(t, ctx)
            img.convert("RGB").resize((int(C.W * scale), int(C.H * scale))).save(
                os.path.join(d, f"{name}_{k}_{t:05.1f}.jpg"), quality=90
            )
        print("stills", name)


def build_audio(total, path):
    """Narration placed per section + procedural music bed ducked under speech."""
    from scipy.io import wavfile

    sr = 48000
    n = int(total * sr) + sr
    mix = np.zeros((n, 2), np.float32)
    speech_mask = np.zeros(n, np.float32)
    t0 = 0.0
    for name, dur, fn, narr, offset in scenes.SECTIONS:
        mp3 = os.path.join(NARR, narr + ".mp3")
        wav = os.path.join(OUT, "tmp_" + narr + ".wav")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                mp3,
                "-ar",
                str(sr),
                "-ac",
                "2",
                "-af",
                "loudnorm=I=-16:TP=-1.5:LRA=9",
                wav,
            ],
            check=True,
        )
        r, a = wavfile.read(wav)
        a = a.astype(np.float32) / (32768.0 if a.dtype == np.int16 else 1.0)
        s0 = int((t0 + offset) * sr)
        e0 = min(n, s0 + len(a))
        mix[s0:e0] += a[: e0 - s0] * 0.95
        speech_mask[s0:e0] = 1.0
        t0 += dur
    music_path = os.path.join(OUT, "music_bed.wav")
    if not os.path.exists(music_path):
        from . import music

        wavfile.write(music_path, sr, music.build(total + 2))
    r, m = wavfile.read(music_path)
    m = m.astype(np.float32)
    if m.dtype != np.float32 or np.abs(m).max() > 1.5:
        m = m / 32768.0
    if len(m) < n:
        m = np.concatenate([m, np.zeros((n - len(m), 2), np.float32)])
    m = m[:n]
    # ducking envelope: -13 dB under speech, smoothed
    from scipy.ndimage import maximum_filter1d, uniform_filter1d

    env = maximum_filter1d(speech_mask, int(0.6 * sr))
    env = uniform_filter1d(env, int(0.5 * sr))
    gain = 10 ** (-12 / 20) * env + (1 - env)
    music_level = 10 ** (-11.5 / 20)
    mix += m * (gain * music_level)[:, None]
    # gentle master fade at the very end
    tail = int(2.5 * sr)
    fade = np.linspace(1, 0, tail)
    end = int(total * sr)
    mix[end - tail : end] *= fade[:, None]
    mix[end:] = 0
    peak = np.abs(mix).max()
    if peak > 0.98:
        mix *= 0.98 / peak
    wavfile.write(path, sr, (mix * 32767).astype(np.int16))
    return path


def assemble(scale=1.0, step=1, workers=None, tag="final", only=None):
    os.makedirs(SEC_DIR, exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)
    secdir = os.path.join(SEC_DIR, tag)
    names = [s[0] for s in scenes.SECTIONS]
    jobs = [(nm, scale, step, secdir) for nm in names if (only is None or nm in only)]
    t0 = time.time()
    with mp.Pool(workers or min(8, mp.cpu_count())) as pool:
        for name, nf, dt, path in pool.imap_unordered(render_section, jobs):
            print(
                f"  {name:12s} {nf:5d} frames in {dt:6.1f}s ({nf / max(dt, 1e-6):.1f} fps)",
                flush=True,
            )
    print("sections done in", round(time.time() - t0), "s")
    total = sum(s[1] for s in scenes.SECTIONS)
    # concat
    lst = os.path.join(secdir, "list.txt")
    with open(lst, "w") as f:
        for nm in names:
            f.write(f"file '{os.path.join(secdir, nm + '.mp4')}'\n")
    silent = os.path.join(secdir, "video_only.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            lst,
            "-c",
            "copy",
            silent,
        ],
        check=True,
    )
    audio = build_audio(total, os.path.join(secdir, "audio.wav"))
    final = os.path.join(FINAL_DIR, f"visibility_aware_mobile_grasping_{tag}.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            silent,
            "-i",
            audio,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            final,
        ],
        check=True,
    )
    print("final:", final, "duration", round(total, 1), "s")
    return final


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "preview"
    if mode == "preview":
        assemble(scale=1 / 3, step=3, tag="preview", workers=4)
    elif mode == "full":
        assemble(scale=1.0, step=1, tag="final", only=(sys.argv[2:] or None))
    elif mode == "stills":
        stills(sys.argv[2:] or None)
    elif mode == "section":
        nm = sys.argv[2]
        sc = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
        print(render_section((nm, sc, 1, os.path.join(SEC_DIR, "single"))))
