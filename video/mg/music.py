"""Procedural ambient music bed (no external samples).

Slow evolving pad chords + soft sub pulse + sparse bell notes, rendered to a
stereo WAV. Designed to sit under narration at low level.
"""
import math
import os
import sys

import numpy as np
from scipy.signal import butter, sosfilt

SR = 48000


def note(
    f0,
    dur,
    sr=SR,
    harmonics=((1, 1.0), (2, 0.35), (3, 0.18), (4, 0.08)),
    detune=0.004,
    att=1.8,
    rel=2.5,
):
    n = int(dur * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    for h, a in harmonics:
        for d in (-detune, 0.0, detune):
            ph = np.random.uniform(0, 2 * math.pi)
            out += a * np.sin(2 * math.pi * f0 * h * (1 + d) * t + ph)
    env = np.minimum(1.0, t / att) * np.minimum(1.0, (dur - t) / rel)
    env = np.clip(env, 0, 1) ** 1.5
    return out * env / len(harmonics)


def lowpass(x, fc, order=2):
    sos = butter(order, fc / (SR / 2), btype="low", output="sos")
    return sosfilt(sos, x)


def highpass(x, fc, order=2):
    sos = butter(order, fc / (SR / 2), btype="high", output="sos")
    return sosfilt(sos, x)


def reverb(x, sr=SR, decay=2.2, taps=24, seed=3):
    rng = np.random.default_rng(seed)
    n = int(decay * sr)
    ir = np.zeros(n)
    for _ in range(taps):
        p = int(rng.uniform(0.01, decay) * sr)
        if p < n:
            ir[p] += rng.uniform(0.2, 1.0) * math.exp(-3.0 * p / n)
    ir[0] = 1.0
    ir = ir / np.sum(np.abs(ir))
    from scipy.signal import fftconvolve

    return fftconvolve(x, ir)[: len(x)]


def build(duration, seed=7, bpm=72):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    n = int(duration * SR)
    L = np.zeros(n)
    R = np.zeros(n)
    # chord progression (A minor world): Am - F - C - G   | Am - F - Dm - E(sus)
    A = 110.0

    def hz(semi):
        return A * 2 ** (semi / 12)

    chords = [
        [0, 3, 7, 12, 19],  # Am (add 5th above)
        [-4, 0, 3, 8, 15],  # F
        [-9, -2, 3, 7, 15],  # C
        [-2, 2, 5, 9, 17],  # G
        [0, 3, 7, 14, 19],  # Am9
        [-4, 0, 3, 8, 17],  # Fmaj7
        [-7, -2, 5, 9, 12],  # Dm
        [-5, -1, 2, 7, 14],  # Esus/E
    ]
    bar = 60.0 / bpm * 4
    cd = bar * 2  # 2 bars per chord
    t = 0.0
    i = 0
    while t < duration:
        ch = chords[i % len(chords)]
        dur = min(cd + 1.5, duration - t + 0.5)
        for k, semi in enumerate(ch):
            f0 = hz(semi) * (0.5 if k == 0 else 1.0)
            sig = note(f0, dur, att=2.2, rel=2.5)
            sig = lowpass(sig, 900 + 300 * math.sin(i))
            s0 = int(t * SR)
            e0 = min(n, s0 + len(sig))
            pan = 0.5 + 0.35 * math.sin(k * 1.7 + i)
            L[s0:e0] += sig[: e0 - s0] * (1 - pan) * 0.22
            R[s0:e0] += sig[: e0 - s0] * pan * 0.22
        t += cd
        i += 1
    # soft sub pulse on beats 1 and 3
    beat = 60.0 / bpm
    tb = 0.0
    while tb < duration:
        f0 = hz(chords[int(tb // cd) % len(chords)][0]) * 0.5
        dur = 0.9
        m = int(dur * SR)
        tt = np.arange(m) / SR
        sig = (
            np.sin(2 * math.pi * f0 * tt)
            * np.exp(-tt * 4.5)
            * np.minimum(1.0, tt / 0.01)
        )
        s0 = int(tb * SR)
        e0 = min(n, s0 + m)
        L[s0:e0] += sig[: e0 - s0] * 0.16
        R[s0:e0] += sig[: e0 - s0] * 0.16
        tb += beat * 2
    # sparse bells (pentatonic over the chord), reverberated
    bells = np.zeros(n)
    tb = 4.0
    while tb < duration - 3:
        ch = chords[int(tb // cd) % len(chords)]
        semi = ch[rng.integers(1, len(ch))] + 24 + rng.choice([0, 0, 12])
        f0 = hz(semi)
        dur = 3.0
        m = int(dur * SR)
        tt = np.arange(m) / SR
        sig = (
            np.sin(2 * math.pi * f0 * tt) + 0.3 * np.sin(2 * math.pi * f0 * 2.01 * tt)
        ) * np.exp(-tt * 2.2)
        s0 = int(tb * SR)
        e0 = min(n, s0 + m)
        bells[s0:e0] += sig[: e0 - s0] * 0.05
        tb += rng.uniform(2.5, 6.0)
    bells = reverb(bells, decay=3.0)
    L += bells * 0.8
    R += np.roll(bells, 400) * 0.8
    # air: filtered noise swell
    noise = rng.normal(0, 1, n)
    noise = lowpass(highpass(noise, 3000), 7000) * 0.006
    swell = 0.5 + 0.5 * np.sin(2 * math.pi * np.arange(n) / SR / 23.0)
    L += noise * swell
    R += np.roll(noise, 900) * swell
    # gentle reverb on the whole bed
    L = reverb(L, decay=1.6, seed=11)
    R = reverb(R, decay=1.6, seed=12)
    # global fade in/out
    tt = np.arange(n) / SR
    env = np.minimum(1.0, tt / 3.0) * np.minimum(1.0, (duration - tt) / 5.0)
    L *= env
    R *= env
    mix = np.stack([L, R], axis=1)
    mix /= max(1e-9, np.abs(mix).max())
    return (mix * 0.85).astype(np.float32)


if __name__ == "__main__":
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 190.0
    out = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "out",
            "music_bed.wav",
        )
    )
    mix = build(duration)
    from scipy.io import wavfile

    os.makedirs(os.path.dirname(out), exist_ok=True)
    wavfile.write(out, SR, mix)
    print("wrote", out, mix.shape, "peak", float(np.abs(mix).max()))
