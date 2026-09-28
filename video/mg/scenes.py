"""All sections of the paper video, drawn frame-by-frame with mg.core primitives.

Each section drawer has the signature draw(t, ctx) -> RGBA PIL image, where t is
the local time in seconds and ctx is a SectionContext (frame sources are lazily
opened and cached per process).
"""
import glob
import json
import math
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import core as C
from . import sources as S

W, H = C.W, C.H
OUT = os.path.join(C.ROOT, "out")
DEMOS = os.path.join(S.DROPBOX, "real robot demo", "all demos")
FINAL_ASSETS = os.path.join(S.DROPBOX, "final assets")
PAPER_FIG = os.path.expanduser("~/paper/Grasp-Anywhere/figures")

CLIP_PERSON = "Video 2026-1-21, 14 17 02.mov"  # sofa: person crosses the route
CLIP_CHAIRS = "Video 2026-1-21, 13 46 34.mov"  # workstation: tight chairs
CLIP_KITCHEN_BLOCK = (
    "Video 2026-1-26, 10 25 31.mov"  # kitchen: person blocks, robot repositions
)
CLIP_KITCHEN2 = "Video 2026-1-26, 10 11 50.mov"
CLIP_SOFA2 = "Video 2026-1-26, 11 15 51.mov"
CLIP_TABLE = "Video 2026-1-26, 09 32 09.mov"
KITCHEN_LIFT_A, KITCHEN_LIFT_B = 108.0, 124.0  # refined after inspecting the clip


# --------------------------------------------------------------------------- helpers
class Ctx:
    def __init__(self):
        self._vid = {}
        self._tracks = {}

    def video(self, name):
        if name not in self._vid:
            p = name if os.path.isabs(name) else os.path.join(DEMOS, name)
            self._vid[name] = S.VideoSource(p)
        return self._vid[name]

    def people(self, name, a, b):
        key = ("people", name, a, b)
        if key not in self._tracks:
            try:
                self._tracks[key] = S.person_track(
                    os.path.join(DEMOS, name), a, b, step=2, cache=True
                )
            except Exception:
                self._tracks[key] = {}
        return self._tracks[key]

    def track(self, name, a, b):
        key = (name, a, b)
        if key not in self._tracks:
            try:
                self._tracks[key] = S.face_track(
                    os.path.join(DEMOS, name), a, b, step=3, cache=True
                )
            except Exception:
                self._tracks[key] = {}
        return self._tracks[key]


def render_frame(path_pattern, idx, nearest=True):
    """Load a Blender output frame; fall back to the nearest existing one."""
    p = path_pattern % idx
    if os.path.exists(p) and os.path.getsize(p) > 2000:
        try:
            return Image.open(p).convert("RGBA")
        except Exception:
            pass
    files = _frame_index(
        os.path.dirname(path_pattern), os.path.basename(path_pattern).split("%")[0]
    )
    if len(files) == 0:
        return C.new_canvas((10, 12, 18))
    j = int(np.argmin(np.abs(files - idx)))
    return Image.open(path_pattern % int(files[j])).convert("RGBA")


@lru_cache(maxsize=64)
def _frame_index(d, prefix="f"):
    fs = glob.glob(os.path.join(d, prefix + "*.png"))
    idx = []
    import re

    for f in fs:
        try:
            m = re.search(r"(\d{4})\.png$", os.path.basename(f))
            if m and os.path.getsize(f) > 2000:
                idx.append(int(m.group(1)))
        except (ValueError, OSError):
            pass
    return np.array(sorted(idx)) if idx else np.array([])


def episode_frame(src_idx):
    return render_frame(os.path.join(OUT, "episode2", "f%04d.png"), int(src_idx))


def headcam_frame(prefix, idx):
    """Head-camera inset frames (rgb_/depthc_/belief_) rendered from the recorded head pose."""
    return render_frame(os.path.join(OUT, "headcam2", prefix + "%04d.png"), int(idx))


@lru_cache(maxsize=1)
def objvis_occlusion_window():
    """(t_start, t_end) in seconds of the self-occluding phase of the objvis studio shot."""
    try:
        d = json.load(open(os.path.join(OUT, "studio_objvis_timing.json")))
        occ = d["occluded"]
        idx = [i for i, v in enumerate(occ) if v]
        if not idx:
            return None
        return (idx[0] / d["fps"], idx[-1] / d["fps"])
    except Exception:
        return None


def studio_frame(shot, idx):
    return render_frame(os.path.join(OUT, f"studio_{shot}", "f%04d.png"), int(idx))


def real_clip(ctx, name, t_src, a=None, b=None, blur=True, size=(W, H)):
    vs = ctx.video(name)
    img = vs.frame_cover(t_src, size)
    if blur and a is not None:
        scale = max(size[0] / vs.w, size[1] / vs.h)
        nw, nh = vs.w * scale, vs.h * scale
        ox, oy = (nw - size[0]) / 2, (nh - size[1]) / 2
        i = vs.frame_index(t_src)

        def shifted(boxes):
            return [
                [bx[0] - ox / scale, bx[1] - oy / scale, bx[2], bx[3], bx[4]]
                for bx in boxes
            ]

        ptr = ctx.people(name, a, b)
        if ptr:
            k = min(ptr, key=lambda v: abs(v - i))
            S.apply_person_blur(img, shifted(ptr.get(k, [])), scale=scale)
        tr = ctx.track(name, a, b)
        if tr:
            k = min(tr, key=lambda v: abs(v - i))
            S.apply_face_blur(
                img, shifted(tr.get(k, [])), scale=scale, pad=0.6, radius=34
            )
    return img


def grade_real(img, contrast=1.08, sat=0.92, lift=-6):
    arr = np.asarray(img).astype(np.float32)
    rgb = arr[..., :3]
    gray = rgb.mean(axis=2, keepdims=True)
    rgb = gray + (rgb - gray) * sat
    rgb = (rgb - 128) * contrast + 128 + lift
    arr[..., :3] = rgb.clip(0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def label_pill(
    img, xy, txt, t, t0, dur=0.5, color=C.CYAN, size=30, anchor="l", mono=True
):
    a = C.seg(t, t0, t0 + dur)
    if a <= 0:
        return img
    f = C.font("Medium", size, mono=mono)
    return C.pill(
        img,
        (xy[0], xy[1] + (1 - a) * 12),
        txt,
        f,
        fg=C.INK,
        bg=(14, 18, 26, int(215 * a)),
        pad=(20, 10),
        alpha=a,
        anchor=anchor,
        tracking=1,
    )


def lower_third(img, t, t0, t1, title, sub=None, color=C.CYAN, x=96, y=H - 190):
    a = C.fade_in_out(t, t0, t1, 0.45, 0.35)
    if a <= 0:
        return img
    ft = C.font("Bold", 52)
    fs = C.font("Regular", 30)
    slide = (1 - C.seg(t, t0, t0 + 0.6)) * 40
    img = C.hline_reveal(img, y - 18, x, x + 620, C.seg(t, t0, t0 + 0.7), color, 4, a)
    img = C.draw_text(img, (x - slide, y), title, ft, C.INK, 0, "la", a, shadow=(0, 3))
    if sub:
        img = C.draw_text(img, (x - slide, y + 66), sub, fs, C.MUTED, 0, "la", a)
    return img


def speed_badge(img, txt, alpha=1.0):
    f = C.font("Bold", 34, mono=True)
    return C.pill(
        img,
        (W - 96, H - 120),
        txt,
        f,
        fg=C.INK,
        bg=(0, 0, 0, 150),
        pad=(18, 8),
        alpha=alpha,
        anchor="r",
    )


def corner_tag(img, txt, alpha=1.0, color=C.MUTED):
    f = C.font("Medium", 24, mono=True)
    return C.draw_text(img, (96, 64), txt, f, color, 3, "la", alpha)


def section_fade(img, t, dur, fin=0.5, fout=0.5):
    a = C.fade_in_out(t, 0, dur, fin, fout)
    if a >= 1:
        return img
    return C.darken(img, 1 - a)


def big_word(img, t, t0, t1, txt, y, size=120, color=C.INK, glow=None, tracking=-2):
    a = C.fade_in_out(t, t0, t1, 0.35, 0.3)
    if a <= 0:
        return img
    f = C.font("ExtraBold", size)
    drift = (1 - C.seg(t, t0, t1)) * 30 - 15
    if glow:
        return C.glow_text(
            img,
            (W / 2 + drift, y),
            txt,
            f,
            glow,
            radius=26,
            strength=0.9,
            tracking=tracking,
            anchor="ma",
            alpha=a,
        )
    return C.draw_text(img, (W / 2 + drift, y), txt, f, color, tracking, "ma", a)


def fov_overlay(img, t, t0, alpha=1.0):
    """Camera-frame brackets + FOV readout, drawn over a first-person view."""
    a = C.seg(t, t0, t0 + 0.5) * alpha
    if a <= 0:
        return img
    layer = C.transparent()
    d = ImageDraw.Draw(layer)
    m, L = 90, 70
    col = C.rgba(C.CYAN, 230)
    for x, y, sx, sy in (
        (m, m, 1, 1),
        (W - m, m, -1, 1),
        (m, H - m, 1, -1),
        (W - m, H - m, -1, -1),
    ):
        d.line([(x, y), (x + sx * L, y)], fill=col, width=4)
        d.line([(x, y), (x, y + sy * L)], fill=col, width=4)
    d.ellipse((W / 2 - 6, H / 2 - 6, W / 2 + 6, H / 2 + 6), outline=col, width=3)
    d.line([(W / 2 - 30, H / 2), (W / 2 - 12, H / 2)], fill=col, width=2)
    d.line([(W / 2 + 12, H / 2), (W / 2 + 30, H / 2)], fill=col, width=2)
    img = C.alpha_over(img, layer, a)
    f = C.font("Medium", 28, mono=True)
    img = C.draw_text(
        img,
        (W - m, m + 84),
        "HEAD RGB-D  ·  LIMITED FIELD OF VIEW",
        f,
        C.CYAN,
        2,
        "ra",
        a,
    )
    img = C.draw_text(
        img,
        (m, H - m - 110),
        "REC ●",
        f,
        C.CORAL,
        2,
        "la",
        a * (0.55 + 0.45 * math.sin(t * 6) ** 2),
    )
    return img


# --------------------------------------------------------------------------- 1. HOOK
HOOK_D = 15.6


def draw_hook(t, ctx):
    # three real-world beats, cinematic letterbox
    if t < 5.2:
        u = t / 5.2
        img = real_clip(ctx, CLIP_CHAIRS, 57.5 + t * 1.5, 55.0, 86.0)
        img = C.zoom_crop(img, 1.0 + 0.07 * u, (0.42, 0.5))
    elif t < 9.0:
        u = (t - 5.2) / 3.8
        img = real_clip(ctx, CLIP_PERSON, 38.6 + (t - 5.2) * 1.3, 37.0, 50.0)
        img = C.zoom_crop(img, 1.08 - 0.06 * u, (0.55, 0.45))
    else:
        u = (t - 9.0) / (HOOK_D - 9.0)
        vs = ctx.video(os.path.join(FINAL_ASSETS, "first_person_rgb.mp4"))
        img = vs.frame_cover(1.0 + (t - 9.0) * 1.6, (W, H))
        img = C.zoom_crop(img, 1.0 + 0.04 * u, (0.5, 0.5))
    img = grade_real(img, 1.08, 0.9, -8)
    img = C.vignette(img, 0.5)
    img = C.letterbox(img, 0.06)
    img = lower_third(
        img,
        t,
        0.8,
        5.0,
        "Mobile grasping in an unknown, dynamic environment",
        "clutter and layouts that are not in the map",
        y=H - 210,
    )
    img = lower_third(
        img,
        t,
        5.4,
        8.8,
        "People and obstacles cross the planned route",
        "the map changes while the robot is moving",
        y=H - 210,
    )
    img = lower_third(
        img,
        t,
        9.3,
        14.8,
        "A single RGB-D head camera with a limited field of view",
        "the robot must choose where to look while it moves",
        y=H - 210,
    )
    if t >= 9.0:
        img = C.pill(
            img,
            (W - 96, 96),
            "HEAD CAMERA VIEW  ·  SIMULATION",
            C.font("Medium", 24, mono=True),
            C.INK,
            (0, 0, 0, 150),
            (16, 8),
            alpha=C.seg(t, 9.2, 9.7),
            anchor="r",
            tracking=2,
        )
    img = section_fade(img, t, HOOK_D, 0.8, 0.35)
    return img


# --------------------------------------------------------------------------- 2. TITLE
TITLE_D = 6.4


def draw_title(t, ctx):
    hero = studio_frame("hero", int(t * 30))
    # push the robot to the right third
    img = C.new_canvas((6, 8, 12))
    img = C.alpha_over(img, hero, 1.0, (330, 0))
    img = (
        C.gradient_overlay(img, (6, 8, 12, 255), (6, 8, 12, 0), 0.0, 0.42)
        if False
        else img
    )
    # left dark wash for legibility
    g = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(g)
    for x in range(0, 900, 6):
        a = int(210 * (1 - x / 900) ** 1.6)
        gd.rectangle((x, 0, x + 6, H), fill=(6, 8, 12, a))
    img = C.alpha_over(img, g)
    a1 = C.seg(t, 0.2, 1.0)
    a2 = C.seg(t, 0.6, 1.4)
    a3 = C.seg(t, 1.0, 1.8)
    x = 110
    img = C.hline_reveal(img, 300, x, x + 160, C.seg(t, 0.0, 0.8), C.CYAN, 5)
    img = C.draw_text(
        img,
        (x, 250),
        "SUPPLEMENTARY VIDEO",
        C.font("Medium", 26, mono=True),
        C.MUTED,
        4,
        "la",
        C.seg(t, 0.2, 0.9),
    )
    f1 = C.font("ExtraBold", 108)
    f2 = C.font("ExtraBold", 108)
    f3 = C.font("SemiBold", 62)
    img = C.draw_text(
        img, (x - (1 - a1) * 40, 330), "Visibility-Aware", f1, C.INK, -3, "la", a1
    )
    img = C.draw_text(
        img, (x - (1 - a2) * 40, 445), "Mobile Grasping", f2, C.INK, -3, "la", a2
    )
    img = C.draw_text(
        img,
        (x - (1 - a3) * 40, 585),
        "in Dynamic Environments",
        f3,
        C.CYAN,
        -1,
        "la",
        a3,
    )
    a4 = C.seg(t, 1.8, 2.6)
    f4 = C.font("Regular", 30)
    img = C.draw_text(
        img,
        (x, 690),
        "Velocity-aware active perception  ·  hierarchical subgoal policy  ·  real-time whole-body replanning",
        f4,
        C.MUTED,
        0,
        "la",
        a4,
    )
    fade = C.fade_in_out(t, 0, TITLE_D, 0.6, 0.5)
    return C.darken(img, 1 - fade)


# --------------------------------------------------------------------------- 3. CONSTRAINTS
CON_D = 30.8


def card_header(img, t, t0, kicker, title, color, x=96, y=96):
    a = C.seg(t, t0, t0 + 0.6)
    if a <= 0:
        return img
    fk = C.font("Medium", 26, mono=True)
    ft = C.font("Bold", 64)
    img = C.draw_text(img, (x, y - (1 - a) * 10), kicker, fk, color, 3, "la", a)
    img = C.draw_text(img, (x, y + 40), title, ft, C.INK, -1, "la", a, shadow=(0, 3))
    img = C.hline_reveal(
        img, y + 128, x, x + 520, C.seg(t, t0 + 0.2, t0 + 0.9), color, 4, a
    )
    return img


def caption_box(img, t, t0, t1, lines, x=96, y=H - 250, color=C.CYAN, w=920):
    a = C.fade_in_out(t, t0, t1, 0.4, 0.3)
    if a <= 0:
        return img
    f = C.font("Regular", 32)
    lh = 44
    hgt = lh * len(lines) + 40
    img = C.rounded_panel(
        img, (x - 28, y - 22, x + w, y + hgt - 10), (10, 13, 20, 190), 16, alpha=a
    )
    img = C.draw_line(img, (x - 28, y - 22), (x - 28, y + hgt - 10), color, 6, a)
    for i, ln in enumerate(lines):
        img = C.draw_text(
            img, (x, y + i * lh), ln, f, C.INK if i == 0 else C.MUTED, 0, "la", a
        )
    return img


def draw_constraints(t, ctx):
    T1, T2, T3 = 4.0, 16.4, 26.6
    if t < T1:
        img = C.new_canvas()
        img = C.grid_bg(img, 96, (255, 255, 255, 9))
        a = C.seg(t, 0.2, 1.0)
        f = C.font("ExtraBold", 92)
        img = C.draw_text(
            img,
            (W / 2, 300 - (1 - a) * 20),
            "Problem: two visibility constraints",
            f,
            C.INK,
            -2,
            "ma",
            a,
        )
        # two cards
        for k, (title, sub, color, t0) in enumerate(
            (
                (
                    "Collision visibility",
                    "the map must cover the volume the body is about to sweep",
                    C.CYAN,
                    1.0,
                ),
                (
                    "Object visibility",
                    "the grasp configuration must keep the target in view",
                    C.AMBER,
                    1.6,
                ),
            )
        ):
            b = C.seg(t, t0, t0 + 0.7, C.ease_out_back)
            if b <= 0:
                continue
            cx = W / 2 + (-1 if k == 0 else 1) * 400
            y0 = 520 + (1 - b) * 60
            box = (cx - 360, y0, cx + 360, y0 + 300)
            img = C.rounded_panel(
                img,
                box,
                (12, 16, 24, 220),
                22,
                outline=C.rgba(color, 140),
                width=2,
                alpha=b,
            )
            img = C.draw_text(
                img,
                (cx - 320, y0 + 36),
                f"0{k + 1}",
                C.font("Bold", 30, mono=True),
                color,
                4,
                "la",
                b,
            )
            img = C.draw_text(
                img, (cx - 320, y0 + 90), title, C.font("Bold", 52), C.INK, -1, "la", b
            )
            img, _ = C.draw_paragraph(
                img,
                (cx - 320, y0 + 170),
                sub,
                C.font("Regular", 29),
                640,
                C.MUTED,
                38,
                0,
                "l",
                b,
            )
        img = section_fade(img, t, CON_D, 0.5, 0.0)
        return img
    if t < T2:
        u = t - T1
        img = studio_frame("swept", int(u * 330 / (T2 - T1)))
        img = C.vignette(img, 0.35)
        img = card_header(
            img,
            t,
            T1 + 0.1,
            "01  COLLISION VISIBILITY CONSTRAINT",
            "Map coverage of the swept volume",
            C.CYAN,
        )
        img = caption_box(
            img,
            t,
            T1 + 2.2,
            T2 - 0.2,
            [
                "A plan that is collision-free in the map",
                "can still hit what the robot has never observed.",
                "Ghosts: swept volume V(ξ), weighted by velocity and lookahead.",
            ],
        )
        img = C.darken(img, 1 - C.fade_in_out(t, T1, T2, 0.4, 0.3))
        return img
    if t < T3:
        u = t - T2
        img = studio_frame("objvis", int(u * 30))
        img = C.vignette(img, 0.35)
        img = card_header(
            img,
            t,
            T2 + 0.1,
            "02  OBJECT VISIBILITY CONSTRAINT",
            "Target visibility at the grasp configuration",
            C.AMBER,
        )
        img = caption_box(
            img,
            t,
            T2 + 1.8,
            T3 - 0.2,
            [
                "The grasp is underspecified: it is refined by observation.",
                "So the goal configuration must not occlude the target,",
                "not by the map, and not by the robot's own arm.",
            ],
            color=C.AMBER,
        )
        occ_t = objvis_occlusion_window()
        if occ_t and occ_t[0] <= u <= occ_t[1]:
            img = label_pill(
                img,
                (W - 96, 260),
                "SELF-OCCLUSION  ·  REJECTED",
                t,
                T2 + occ_t[0],
                0.4,
                C.CORAL,
                30,
                "r",
            )
        elif occ_t and u > occ_t[1] + 1.2:
            img = label_pill(
                img,
                (W - 96, 260),
                "TARGET VISIBLE  ·  COLLISION-FREE (VAMP)",
                t,
                T2 + occ_t[1] + 1.2,
                0.4,
                C.MINT,
                30,
                "r",
            )
        img = C.darken(img, 1 - C.fade_in_out(t, T2, T3, 0.4, 0.3))
        return img
    # compete for one camera: split screen + sliding gaze token
    u = t - T3
    img = C.new_canvas()
    left = studio_frame("swept", int(150 + u * 30))
    right = studio_frame("objvis", int(0 + u * 30))
    half = (W // 2 - 8, H)
    left = C.fit_cover(left, half)
    right = C.fit_cover(right, half)
    img = C.alpha_over(img, left, 1.0, (0, 0))
    img = C.alpha_over(img, right, 1.0, (W // 2 + 8, 0))
    img = C.vignette(img, 0.45)
    a = C.seg(t, T3 + 0.2, T3 + 0.9)
    f = C.font("ExtraBold", 74)
    img = C.draw_text(
        img,
        (W / 2, 96 - (1 - a) * 20),
        "One camera, two competing demands",
        f,
        C.INK,
        -2,
        "ma",
        a,
        shadow=(0, 3),
    )
    fl = C.font("Bold", 36)
    img = C.draw_text(img, (W / 4, H - 150), "swept volume", fl, C.CYAN, 1, "ma", a)
    img = C.draw_text(
        img, (3 * W / 4, H - 150), "target object", fl, C.AMBER, 1, "ma", a
    )
    # gaze token slides
    p = 0.5 - 0.5 * math.cos(u * 1.9)
    p = C.ease_in_out(p)
    x = W / 4 + (W / 2) * p
    y = H - 230
    layer = C.transparent()
    d = ImageDraw.Draw(layer)
    d.ellipse((x - 26, y - 26, x + 26, y + 26), fill=C.rgba(C.INK, 240))
    d.ellipse((x - 11, y - 11, x + 11, y + 11), fill=C.rgba((10, 12, 18), 255))
    glow = layer.filter(ImageFilter.GaussianBlur(18))
    img = C.alpha_over(img, glow, a)
    img = C.alpha_over(img, layer, a)
    img = C.draw_text(
        img, (x, y - 70), "GAZE", C.font("Medium", 24, mono=True), C.INK, 4, "ma", a
    )
    img = C.darken(img, 1 - C.fade_in_out(t, T3, CON_D, 0.4, 0.5))
    return img


# --------------------------------------------------------------------------- 4. METHOD
MET_D = 37.2
EP_PATTERN = os.path.join(OUT, "episode2", "f%04d.png")


def loop_diagram(img, t, t0, active=None, cx=W / 2, cy=H / 2 + 110, r=270, alpha=1.0):
    a = C.seg(t, t0, t0 + 0.8) * alpha
    if a <= 0:
        return img
    nodes = [
        ("ACTIVE PERCEPTION", "π_v  ·  gaze", C.CYAN, -90),
        ("SUBGOAL POLICY", "π_g  ·  goals", C.AMBER, 30),
        ("WHOLE-BODY PLANNER", "π_r  ·  trajectory", C.MINT, 150),
    ]
    pts = [
        (cx + r * math.cos(math.radians(ang)), cy + r * math.sin(math.radians(ang)))
        for _, _, _, ang in nodes
    ]
    layer = C.transparent()
    d = ImageDraw.Draw(layer)
    # ring arcs with arrows
    for i in range(3):
        a0 = nodes[i][3] + 22
        a1 = nodes[(i + 1) % 3][3] - 22
        d.arc(
            (cx - r, cy - r, cx + r, cy + r), a0, a1, fill=C.rgba(C.MUTED, 160), width=4
        )
        # arrow head at a1
        ex, ey = cx + r * math.cos(math.radians(a1)), cy + r * math.sin(
            math.radians(a1)
        )
        tx, ty = -math.sin(math.radians(a1)), math.cos(math.radians(a1))
        d.polygon(
            [
                (ex, ey),
                (
                    ex - 22 * tx + 10 * math.cos(math.radians(a1)),
                    ey - 22 * ty + 10 * math.sin(math.radians(a1)),
                ),
                (
                    ex - 22 * tx - 10 * math.cos(math.radians(a1)),
                    ey - 22 * ty - 10 * math.sin(math.radians(a1)),
                ),
            ],
            fill=C.rgba(C.MUTED, 200),
        )
    img = C.alpha_over(img, layer, a)
    # centre: belief summary
    fm = C.font("Medium", 34, mono=True)
    img = C.rounded_panel(
        img,
        (cx - 190, cy - 64, cx + 190, cy + 64),
        (12, 16, 24, 210),
        18,
        outline=C.rgba(C.MUTED, 80),
        alpha=a,
    )
    img = C.draw_text(
        img, (cx, cy - 32), "b_t = (q_t, M_t, g_t)", fm, C.INK, 0, "ma", a
    )
    img = C.draw_text(
        img,
        (cx, cy + 12),
        "state · map · subgoal",
        C.font("Regular", 24),
        C.MUTED,
        0,
        "ma",
        a,
    )
    for i, ((name, sub, color, ang), (x, y)) in enumerate(zip(nodes, pts)):
        on = active == i
        rr = 26 if not on else 34
        layer = C.transparent()
        d = ImageDraw.Draw(layer)
        d.ellipse(
            (x - rr, y - rr, x + rr, y + rr), fill=C.rgba(color, 255 if on else 170)
        )
        if on:
            g = layer.filter(ImageFilter.GaussianBlur(22))
            img = C.alpha_over(img, g, a)
        img = C.alpha_over(img, layer, a)
        ty = y - 118 if ang < 0 else y + 52
        img = C.draw_text(
            img,
            (x, ty),
            name,
            C.font("Bold", 30, mono=True),
            color if on else C.INK,
            3,
            "ma",
            a,
        )
        img = C.draw_text(
            img, (x, ty + 40), sub, C.font("Regular", 26), C.MUTED, 0, "ma", a
        )
    return img


def formula(img, xy, txt, t, t0, size=40, color=C.INK, alpha=1.0):
    a = C.seg(t, t0, t0 + 0.5) * alpha
    if a <= 0:
        return img
    f = C.font("Medium", size, mono=True)
    return C.pill(
        img,
        xy,
        txt,
        f,
        fg=color,
        bg=(10, 13, 20, 205),
        pad=(24, 14),
        radius=14,
        alpha=a,
        anchor="l",
    )


def draw_method(t, ctx):
    T1, T2, T3 = 5.6, 16.6, 27.4
    if t < T1:
        # loop diagram on a dim episode background
        bg = episode_frame(int(400 + t * 20))
        img = C.darken(C.fit_cover(bg, (W, H)), 0.78)
        img = C.grid_bg(img, 96, (255, 255, 255, 7))
        a = C.seg(t, 0.2, 0.9)
        img = C.draw_text(
            img,
            (W / 2, 70 - (1 - a) * 20),
            "Method: a receding-horizon system",
            C.font("ExtraBold", 80),
            C.INK,
            -2,
            "ma",
            a,
            shadow=(0, 3),
        )
        img = C.draw_text(
            img,
            (W / 2, 160),
            "every ~100 ms: update the map, check the subgoal and trajectory, act",
            C.font("Regular", 30),
            C.MUTED,
            0,
            "ma",
            C.seg(t, 0.8, 1.4),
        )
        act = None if t < 2.4 else int(((t - 2.4) / 1.0)) % 3
        img = loop_diagram(img, t, 0.9, active=act)
        return section_fade(img, t, MET_D, 0.5, 0.0)
    if t < T2:
        u = t - T1
        # velocity-aware gaze: episode nav phase, ghosts visible (frames 200-500 at 1.5x)
        src = 190 + u * 30
        src = int(src / 2) * 2
        img = episode_frame(src)
        img = C.vignette(img, 0.35)
        img = card_header(
            img,
            t,
            T1 + 0.1,
            "01  ACTIVE PERCEPTION POLICY  π_v",
            "Velocity-aware gaze over the swept volume",
            C.CYAN,
        )
        img = formula(img, (96, 300), "a_v* = argmax Σ Φ(x)·V(x, a_v)", t, T1 + 1.2, 36)
        img = formula(
            img,
            (96, 380),
            "w_safety(x) = Σ γ^i · w_d(x, q^(i)) · ‖q̇^(i)‖",
            t,
            T1 + 2.0,
            36,
            C.CYAN,
        )
        img = caption_box(
            img,
            t,
            T1 + 3.2,
            T2 - 0.2,
            [
                "Gaze maximises visibility of an importance field:",
                "the target area while planning, the swept volume while moving.",
                "Imminent and fast-moving parts of the trajectory come first.",
            ],
            y=H - 250,
        )
        # legend for the ghosts
        a = C.seg(t, T1 + 4.0, T1 + 4.6)
        if a > 0:
            layer = C.transparent()
            d = ImageDraw.Draw(layer)
            x0, y0 = W - 96 - 320, 120
            for i in range(320):
                w = i / 319
                if w < 0.5:
                    tt = w / 0.5
                    c = (
                        int(255 * (0.05 + 0.15 * tt)),
                        int(255 * (0.15 + 0.65 * tt)),
                        int(255 * (0.9 + 0.1 * tt)),
                    )
                else:
                    tt = (w - 0.5) / 0.5
                    c = (
                        int(255 * (0.2 + 0.75 * tt)),
                        int(255 * (0.8 - 0.45 * tt)),
                        int(255 * (1.0 - 0.95 * tt)),
                    )
                d.line([(x0 + i, y0), (x0 + i, y0 + 16)], fill=(*c, 255))
            img = C.alpha_over(img, layer, a)
            fm = C.font("Medium", 22, mono=True)
            img = C.draw_text(
                img, (x0, y0 + 26), "later · slow", fm, C.MUTED, 1, "la", a
            )
            img = C.draw_text(
                img, (x0 + 320, y0 + 26), "imminent · fast", fm, C.INK, 1, "ra", a
            )
            img = C.draw_text(
                img, (x0 + 320, y0 - 34), "SWEPT-VOLUME WEIGHT", fm, C.CYAN, 3, "ra", a
            )
        img = C.darken(img, 1 - C.fade_in_out(t, T1, T2, 0.4, 0.3))
        return img
    if t < T3:
        u = t - T2
        img = studio_frame("subgoals", int(u * 30))
        img = C.vignette(img, 0.3)
        img = card_header(
            img,
            t,
            T2 + 0.1,
            "02  HIERARCHICAL SUBGOAL POLICY  π_g",
            "Subgoal hierarchy: grasp, pre-grasp, observe",
            C.AMBER,
        )
        try:
            anc = json.load(open(os.path.join(OUT, "studio_subgoals_anchors.json")))
        except Exception:
            anc = {
                "grasp": [0.58, 0.26],
                "pregrasp": [0.47, 0.51],
                "observe": [0.32, 0.19],
            }
        labels = [
            (
                "grasp",
                "① GRASP IN PLACE",
                "sample-then-verify: GraspNet → capability map → IK + collision",
                C.MINT,
                2.0,
            ),
            (
                "pregrasp",
                "② PRE-GRASP",
                "factored sampling: base pose × torso height × EE pose, target visible",
                C.CYAN,
                5.0,
            ),
            (
                "observe",
                "③ OBSERVE",
                "standoff pose that keeps the target in view when manipulation is blocked",
                C.AMBER,
                8.0,
            ),
        ]
        for key, title, sub, color, t0 in labels:
            a = C.seg(t, T2 + t0, T2 + t0 + 0.5)
            if a <= 0:
                continue
            ax, ay = anc[key][0] * W, anc[key][1] * H
            # leader line to a label box on the right/left
            lx = W - 96 if key != "observe" else 96
            ly = {"grasp": 300, "pregrasp": 560, "observe": 300}[key]
            layer = C.transparent()
            d = ImageDraw.Draw(layer)
            d.ellipse((ax - 9, ay - 9, ax + 9, ay + 9), outline=C.rgba(color), width=3)
            ex = lx - 40 if key != "observe" else lx + 40
            d.line([(ax, ay), (ex, ly + 20)], fill=C.rgba(color, 200), width=2)
            img = C.alpha_over(img, layer, a)
            anchor = "r" if key != "observe" else "l"
            img = C.draw_text(
                img,
                (lx, ly),
                title,
                C.font("Bold", 34, mono=True),
                color,
                2,
                anchor + "a",
                a,
            )
            f = C.font("Regular", 26)
            lines = C.wrap_lines(sub, f, 520)
            for i, ln in enumerate(lines):
                img = C.draw_text(
                    img, (lx, ly + 46 + i * 32), ln, f, C.MUTED, 0, anchor + "a", a
                )
        img = caption_box(
            img,
            t,
            T2 + 1.0,
            T3 - 0.2,
            [
                "A behaviour tree escalates from the most direct level",
                "to a more exploratory one when the current level fails,",
                "so the robot recovers from failures at runtime.",
            ],
            color=C.AMBER,
        )
        img = C.darken(img, 1 - C.fade_in_out(t, T2, T3, 0.4, 0.3))
        return img
    # whole-body planner: replanning moment at 0.6x
    u = t - T3
    src = 30 + u * 13.0
    img = episode_frame(int(src))
    img = C.vignette(img, 0.35)
    img = card_header(
        img,
        t,
        T3 + 0.1,
        "03  WHOLE-BODY MOTION POLICY  π_r",
        "Whole-body replanning within tens of milliseconds",
        C.MINT,
    )
    img = caption_box(
        img,
        t,
        T3 + 1.5,
        MET_D - 0.4,
        [
            "Hybrid A* + Reeds-Shepp base motion, RRT-Connect arm motion,",
            "synchronised and validated with SIMD collision checks (VAMP).",
            "A 20 Hz whole-body tracker follows the trajectory and accepts updates.",
        ],
        color=C.MINT,
    )
    replan_t = T3 + (68 - 30) / 13.0
    if replan_t <= t <= replan_t + 3.0:
        img = label_pill(
            img,
            (W - 96, 260),
            "MAP CHANGED  →  PLAN INVALID",
            t,
            replan_t,
            0.3,
            C.CORAL,
            30,
            "r",
        )
    if t >= replan_t + 1.0:
        img = label_pill(
            img,
            (W - 96, 330),
            "REPLANNED  ·  50–80 ms",
            t,
            replan_t + 1.0,
            0.3,
            C.MINT,
            30,
            "r",
        )
    img = speed_badge(img, "0.65×", C.seg(t, T3 + 0.3, T3 + 0.8))
    img = C.darken(img, 1 - C.fade_in_out(t, T3, MET_D, 0.4, 0.6))
    return img


# --------------------------------------------------------------------------- 5. SIMULATION
SIM_D = 27.0
SIM_SEGS = [
    (0, 150, 20.0),
    (150, 1071, 120.0),
    (1071, 1587, 50.0),
]  # (src_a, src_b, src fps)


def sim_src(u):
    """Map local time to a source frame index of the episode; returns (src, speed)."""
    t = u
    for a, b, rate in SIM_SEGS:
        dur = (b - a) / rate
        if t <= dur:
            return a + t * rate, rate / 20.0
        t -= dur
    return 1586, SIM_SEGS[-1][2] / 20.0


SIM_TOTAL = sum((b - a) / r for a, b, r in SIM_SEGS)
PHASES = [
    (0, 16, "OBSERVE", C.AMBER),
    (16, 68, "PLAN + MOVE", C.CYAN),
    (68, 150, "REPLAN", C.CORAL),
    (150, 1071, "NAVIGATE  ·  GAZE ON SWEPT VOLUME", C.CYAN),
    (1071, 1300, "PRE-GRASP REACHED  ·  GRASP DETECTION", C.AMBER),
    (1300, 1405, "GRASP", C.MINT),
    (1405, 1587, "LIFT", C.MINT),
]


def draw_sim(t, ctx):
    hold = t > SIM_TOTAL
    src, speed = sim_src(min(t, SIM_TOTAL))
    src = int(src)
    if src >= 150:
        src = int(src / 2) * 2 + (1 if src >= 1071 else 0)
    img = episode_frame(src)
    img = C.vignette(img, 0.3)
    # insets: first-person RGB, depth, belief map
    a = C.seg(t, 0.6, 1.3)
    iw, ih = 400, 300
    x0 = W - 96 - iw
    y0 = 96
    if a > 0:
        fp = C.fit_cover(headcam_frame("rgb_", src), (iw, ih))
        dp = C.fit_cover(headcam_frame("depthc_", src), (iw, ih))
        bm = C.fit_cover(headcam_frame("belief_", src), (iw, ih))
        for k, (im, lab) in enumerate(
            (
                (fp, "HEAD CAMERA  RGB"),
                (dp, "HEAD CAMERA  DEPTH"),
                (bm, "BELIEF MAP  M_t  ·  OBSERVED SO FAR"),
            )
        ):
            yy = y0 + k * (ih + 22)
            img = C.rounded_panel(
                img,
                (x0 - 4, yy - 4, x0 + iw + 4, yy + ih + 4),
                (8, 10, 16, 230),
                10,
                alpha=a,
            )
            img = C.alpha_over(img, im, a, (x0, yy))
            img = C.pill(
                img,
                (x0 + 12, yy + 10),
                lab,
                C.font("Medium", 20, mono=True),
                C.INK,
                (0, 0, 0, 150),
                (12, 5),
                alpha=a,
                tracking=2,
            )
    # phase label + progress bar
    ph = [p for p in PHASES if p[0] <= src < p[1]]
    name, color = (ph[0][2], ph[0][3]) if ph else ("", C.INK)
    b = C.seg(t, 0.3, 0.9)
    bx0, bx1, by = 96, W - 96 - iw - 60, H - 96
    layer = C.transparent()
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((bx0, by - 4, bx1, by + 4), 4, fill=(255, 255, 255, 50))
    for pa, pb, pn, pc in PHASES:
        xa = bx0 + (bx1 - bx0) * pa / 1587
        xb = bx0 + (bx1 - bx0) * pb / 1587
        d.rounded_rectangle((xa + 2, by - 4, xb - 2, by + 4), 4, fill=C.rgba(pc, 70))
    px = bx0 + (bx1 - bx0) * src / 1587
    d.rounded_rectangle((bx0, by - 4, px, by + 4), 4, fill=C.rgba(color, 230))
    d.ellipse((px - 10, by - 10, px + 10, by + 10), fill=C.rgba(C.INK, 255))
    img = C.alpha_over(img, layer, b)
    img = C.draw_text(
        img, (bx0, by - 70), name, C.font("Bold", 34, mono=True), color, 3, "la", b
    )
    img = C.draw_text(
        img,
        (bx1, by - 62),
        f"t = {src / 20.0:5.1f} s",
        C.font("Medium", 26, mono=True),
        C.MUTED,
        1,
        "ra",
        b,
    )
    if speed > 1.05 and not hold:
        eff = speed / TIME_SCALE.get("sim", 1.0)
        img = speed_badge(
            img, f"{eff:.0f}×" if abs(eff - round(eff)) < 0.15 else f"{eff:.1f}×"
        )
    # event callouts
    if 60 <= src <= 150:
        img = label_pill(
            img,
            (96, 112),
            "OBSTACLE ENTERS THE ROUTE  ·  MAP UPDATED  ·  TRAJECTORY INVALID  ·  REPLAN",
            t,
            sim_src_time(60),
            0.3,
            C.CORAL,
            28,
            "l",
        )
    if 0 <= src < 40:
        img = label_pill(
            img, (96, 112), "START  ·  NO PRIOR MAP", t, 0.2, 0.4, C.AMBER, 28, "l"
        )
    if hold:
        a = C.seg(t, SIM_TOTAL + 0.1, SIM_TOTAL + 0.6, C.ease_out_back)
        img = label_pill(
            img,
            (96, 112),
            "GRASP SUCCEEDED  ·  OBJECT LIFTED AND HELD",
            t,
            SIM_TOTAL + 0.1,
            0.4,
            C.MINT,
            30,
            "l",
        )
    img = corner_tag(
        img,
        "SIMULATION  ·  MANISKILL3 + REPLICACAD  ·  FETCH",
        C.seg(t, 0.4, 1.0) * (0.0 if 0 <= src < 40 else 1.0),
    )
    return section_fade(img, t, SIM_D, 0.5, 0.5)


def sim_src_time(src):
    """Inverse of sim_src: local time at which a source frame is shown."""
    t = 0.0
    for a, b, rate in SIM_SEGS:
        if src <= b:
            return t + (src - a) / rate
        t += (b - a) / rate
    return t


# --------------------------------------------------------------------------- 6. REAL WORLD
REAL_D = 28.6
# each clip: list of (src_start, src_end, speed) segments played back to back; the last segment ends with the lift
# coffee table: person sits down and blocks, robot repositions, grasps and lifts
CLIP_BLOCKED = "Video 2026-1-21, 14 39 32.mov"
CLIP_LIFT_TABLE = "Video 2026-1-26, 09 25 35.mov"
CLIP_LIFT_SOFA = "Video 2026-1-26, 11 20 02.mov"
CLIP_LIFT_COFFEE = "Video 2026-1-26, 11 25 19.mov"
CLIP_LIFT_COFFEE2 = "Video 2026-1-21, 14 28 26.mov"
# each clip: (src_start, src_end, speed) segments played back to back; the last segment ends with the completed lift
REAL_CLIPS = [
    (
        CLIP_PERSON,
        [(38.6, 50.0, 2.8), (86.0, 97.0, 3.2)],
        "A person steps into the route",
        "the robot re-plans, then grasps and lifts the object  ·  sofa",
        (37.0, 50.0),
        (84.0, 98.0),
    ),
    (
        CLIP_CHAIRS,
        [(57.0, 80.0, 5.5), (96.0, 105.0, 3.2)],
        "Tightly spaced chairs",
        "whole-body path between the chairs, then grasp and lift  ·  workstation",
        (55.0, 86.0),
        (93.0, 106.0),
    ),
    (
        CLIP_BLOCKED,
        [(20.0, 66.0, 7.0), (78.0, 93.0, 3.6)],
        "Manipulation blocked, pre-grasp re-sampled",
        "a person blocks the approach; the subgoal policy falls back to a new base placement  ·  coffee table",
        (19.0, 94.0),
        (19.0, 94.0),
    ),
]


def clip_len(segs):
    return sum((b - a) / sp for a, b, sp in segs)


REAL_T = []
_t = 0.0
for _c in REAL_CLIPS:
    REAL_T.append((_t, _t + clip_len(_c[1])))
    _t += clip_len(_c[1])
REAL_GRID_T0 = REAL_T[-1][1]


def clip_time(segs, u):
    """Local time u -> (src time, speed, segment index)."""
    for k, (a, b, sp) in enumerate(segs):
        d = (b - a) / sp
        if u <= d or k == len(segs) - 1:
            return min(a + u * sp, b), sp, k
        u -= d
    return segs[-1][1], segs[-1][2], len(segs) - 1


def draw_real(t, ctx):
    if t < REAL_GRID_T0:
        for (clip, segs, title, sub, tr1, tr2), (ta, tb) in zip(REAL_CLIPS, REAL_T):
            if ta <= t < tb:
                u = t - ta
                ts, sp, k = clip_time(segs, u)
                a, b = tr1 if k == 0 else tr2
                img = real_clip(ctx, clip, ts, a, b, blur=True)
                img = grade_real(img, 1.06, 0.95, -4)
                img = C.vignette(img, 0.4)
                img = lower_third(img, t, ta + 0.3, tb - 0.1, title, sub)
                eff = sp / TIME_SCALE.get("real", 1.0)
                img = speed_badge(
                    img,
                    f"{eff:.0f}×" if eff >= 2 else f"{eff:.1f}×",
                    C.seg(t, ta + 0.2, ta + 0.6),
                )
                # jump-cut marker between navigation and grasp segments
                dseg = (segs[0][1] - segs[0][0]) / segs[0][2]
                if k == 1:
                    img = label_pill(
                        img,
                        (W - 96, 96),
                        "GRASP  ·  LIFT",
                        t,
                        ta + dseg,
                        0.35,
                        C.MINT,
                        26,
                        "r",
                    )
                img = corner_tag(
                    img,
                    "REAL FETCH ROBOT  ·  FULLY AUTONOMOUS  ·  NO PRIOR OBSTACLE MAP  ·  PEOPLE BLURRED",
                    C.seg(t, 0.3, 0.9),
                )
                # brief dip to black at the jump cut
                dip = 1.0 - 0.85 * C.fade_in_out(
                    t, ta + dseg - 0.12, ta + dseg + 0.12, 0.12, 0.12
                )
                img = C.darken(img, 1 - dip) if dip < 1 else img
                img = C.darken(img, 1 - C.fade_in_out(t, ta, tb, 0.3, 0.2))
                return img
    # 2x2 grid finale with stats
    u = t - REAL_GRID_T0
    img = C.new_canvas()
    grid = [  # four more completed grasps (lift on screen)
        (CLIP_LIFT_TABLE, 117.0, 3.0, (114.0, 131.0)),
        (CLIP_LIFT_SOFA, 137.0, 3.0, (134.0, 149.0)),
        (CLIP_LIFT_COFFEE, 142.0, 3.0, (140.0, 152.0)),
        (CLIP_LIFT_COFFEE2, 138.0, 3.0, (136.0, 149.0)),
    ]
    gw, gh = W // 2 - 6, H // 2 - 6
    for k, (clip, s0, sp, tr) in enumerate(grid):
        a, b = tr
        im = real_clip(ctx, clip, s0 + u * sp, a, b, blur=True, size=(gw, gh))
        im = grade_real(im, 1.05, 0.9, -8)
        x = (k % 2) * (gw + 12)
        y = (k // 2) * (gh + 12)
        aa = C.seg(t, REAL_GRID_T0 + 0.15 * k, REAL_GRID_T0 + 0.15 * k + 0.4)
        img = C.alpha_over(img, im, aa, (x, y))
    img = C.darken(img, 0.45)
    a = C.seg(t, REAL_GRID_T0 + 0.6, REAL_GRID_T0 + 1.3)
    img = C.draw_text(
        img,
        (W / 2, 140 - (1 - a) * 20),
        "Real-world deployment: 40 trials, 5 locations",
        C.font("ExtraBold", 74),
        C.INK,
        -2,
        "ma",
        a,
        shadow=(0, 3),
    )
    img = C.draw_text(
        img,
        (W / 2, 240),
        "dining table · kitchen counter · workstation · coffee table · sofa",
        C.font("Regular", 30),
        C.MUTED,
        0,
        "ma",
        a,
    )
    for k, (lab, val, ref, color, t0) in enumerate(
        (
            ("UNKNOWN STATIC", 65.0, "sim 70.4", C.CYAN, 1.0),
            ("DYNAMIC", 55.0, "sim 57.9", C.AMBER, 1.6),
        )
    ):
        b = C.seg(t, REAL_GRID_T0 + t0, REAL_GRID_T0 + t0 + 1.2)
        if b <= 0:
            continue
        cx = W / 2 + (-1 if k == 0 else 1) * 330
        v = val * b
        img = C.glow_text(
            img,
            (cx, 430),
            f"{v:.0f}%",
            C.font("ExtraBold", 150),
            color,
            30,
            0.8,
            -4,
            "ma",
            b,
        )
        img = C.draw_text(
            img, (cx, 610), lab, C.font("Bold", 30, mono=True), C.INK, 4, "ma", b
        )
        img = C.draw_text(
            img,
            (cx, 656),
            f"success  ·  {ref}% in simulation",
            C.font("Regular", 28),
            C.MUTED,
            0,
            "ma",
            b,
        )
    img = C.draw_text(
        img,
        (W / 2, H - 150),
        "comparable to simulation despite depth noise, localisation drift and execution noise",
        C.font("Regular", 30),
        C.MUTED,
        0,
        "ma",
        C.seg(t, REAL_GRID_T0 + 2.6, REAL_GRID_T0 + 3.2),
    )
    img = C.darken(img, 1 - C.fade_in_out(t, REAL_GRID_T0, REAL_D, 0.3, 0.6))
    return img


# --------------------------------------------------------------------------- 7. RESULTS
RES_D = 39.0
METHODS = [
    # name, succ static, succ dyn, coll static, coll dyn, note, CI half-widths (succ s, succ d, coll s, coll d)
    ("SG-VLA", 1.6, 1.2, 78.8, 80.5, "(easy init)", (1.2, 0.8, 1.9, 2.1)),
    ("RL (MS-HAB)", 26.9, 24.1, 42.7, 44.9, "(easy init)", (2.7, 3.0, 3.1, 2.4)),
    ("Direct Grasping", 27.2, 16.1, 2.5, 14.8, "", (2.5, 1.1, 0.8, 1.9)),
    ("CapMap Placement", 53.0, 49.7, 9.7, 8.2, "", (2.4, 0.9, 1.0, 0.8)),
    ("Nav-and-Manip", 54.0, 48.6, 7.7, 7.5, "", (2.4, 3.4, 1.1, 1.7)),
    ("Ours w/o Vel. Weighting", 62.2, 51.5, 12.3, 18.1, "", (1.8, 1.6, 2.3, 2.4)),
    ("Ours w/o Obs. Stage", 68.3, 56.3, 5.8, 13.2, "", (3.7, 3.4, 2.4, 4.4)),
    ("Ours (full)", 70.3, 57.9, 5.0, 13.9, "", (1.1, 1.5, 1.5, 3.4)),
]


def bar_chart(img, t, t0, mode="success", cx0=96, top=250, bottom=H - 150, alpha=1.0):
    n = len(METHODS)
    x0, x1 = cx0 + 300, W - 96
    row_h = (bottom - top) / n
    vmax = 100.0 if mode == "collision" else 80.0
    fl = C.font("SemiBold", 28)
    fv = C.font("Medium", 26, mono=True)
    # axis
    layer = C.transparent()
    d = ImageDraw.Draw(layer)
    for v in range(0, int(vmax) + 1, 20):
        x = x0 + (x1 - x0) * v / vmax
        d.line([(x, top - 10), (x, bottom)], fill=(255, 255, 255, 22), width=1)
    img = C.alpha_over(img, layer, alpha)
    for v in range(0, int(vmax) + 1, 20):
        x = x0 + (x1 - x0) * v / vmax
        img = C.draw_text(
            img,
            (x, bottom + 12),
            f"{v}",
            C.font("Regular", 22, mono=True),
            C.MUTED,
            0,
            "ma",
            alpha,
        )
    for i, (name, ss, sd, cs, cd, note, ci) in enumerate(METHODS):
        y = top + i * row_h
        ours = name.startswith("Ours (")
        a = C.seg(t, t0 + 0.12 * i, t0 + 0.12 * i + 0.9) * alpha
        if a <= 0:
            continue
        col_name = C.INK if ours else C.MUTED
        img = C.draw_text(
            img,
            (x0 - 24, y + row_h / 2 - 16),
            name,
            C.font("Bold", 28) if ours else fl,
            col_name,
            0,
            "ra",
            a,
        )
        if note:
            img = C.draw_text(
                img,
                (x0 - 24, y + row_h / 2 + 16),
                note,
                C.font("Regular", 20),
                C.MUTED,
                0,
                "ra",
                a,
            )
        vals = (ss, sd) if mode == "success" else (cs, cd)
        cis = ci[:2] if mode == "success" else ci[2:]
        cols = (C.CYAN, C.AMBER) if mode == "success" else (C.CORAL, (255, 140, 120))
        bh = row_h * 0.30
        for k, (v, colr, cv) in enumerate(zip(vals, cols, cis)):
            vv = v * C.ease_out_cubic(a)
            yy = y + row_h * 0.15 + k * (bh + 4)
            xe = x0 + (x1 - x0) * vv / vmax
            layer = C.transparent()
            d = ImageDraw.Draw(layer)
            d.rounded_rectangle(
                (x0, yy, max(x0 + 4, xe), yy + bh),
                5,
                fill=C.rgba(colr, 235 if ours else 130),
            )
            if ours:
                g = layer.filter(ImageFilter.GaussianBlur(12))
                img = C.alpha_over(img, g, a * 0.8)
            img = C.alpha_over(img, layer, a)
            img = C.draw_text(
                img,
                (xe + 14, yy + bh / 2 - 15),
                f"{vv:.1f}",
                fv,
                C.INK if ours else C.MUTED,
                0,
                "la",
                a,
            )
            if a >= 0.999:
                img = C.draw_text(
                    img,
                    (xe + 14 + C.text_size(f"{vv:.1f}", fv)[0] + 6, yy + bh / 2 - 9),
                    f"±{cv:.1f}",
                    C.font("Regular", 19, mono=True),
                    C.MUTED,
                    0,
                    "la",
                    a,
                )
    return img


def draw_results(t, ctx):
    T2 = 21.0
    img = C.new_canvas()
    img = C.grid_bg(img, 96, (255, 255, 255, 7))
    if t < T2:
        a = C.seg(t, 0.2, 0.9)
        img = C.draw_text(
            img,
            (96, 70 - (1 - a) * 20),
            "Simulation results: 400 scenarios, 5 runs",
            C.font("ExtraBold", 66),
            C.INK,
            -2,
            "la",
            a,
        )
        img = C.draw_text(
            img,
            (96, 150),
            "20 ReplicaCAD scenes × 20 YCB objects  ·  mean ± 95% CI over 5 runs  ·  success = lifted 10 cm and held 2 s without collision",  # noqa: E501
            C.font("Regular", 27),
            C.MUTED,
            0,
            "la",
            C.seg(t, 0.6, 1.2),
        )
        # legend
        for k, (lab, colr) in enumerate(
            (("unknown static", C.CYAN), ("dynamic", C.AMBER))
        ):
            b = C.seg(t, 1.0, 1.5)
            x = W - 96 - 420 + k * 230
            layer = C.transparent()
            ImageDraw.Draw(layer).rounded_rectangle(
                (x, 84, x + 26, 108), 5, fill=C.rgba(colr)
            )
            img = C.alpha_over(img, layer, b)
            img = C.draw_text(
                img, (x + 40, 82), lab, C.font("Medium", 26), C.INK, 0, "la", b
            )
        img = C.draw_text(
            img,
            (W - 96, 208),
            "SUCCESS RATE (%)  ·  HIGHER IS BETTER",
            C.font("Medium", 24, mono=True),
            C.MUTED,
            3,
            "ra",
            C.seg(t, 1.0, 1.5),
        )
        img = bar_chart(img, t, 1.2, "success")
        # delta callout
        b = C.seg(t, 8.5, 9.2, C.ease_out_back)
        if b > 0:
            img = C.rounded_panel(
                img,
                (W - 96 - 470, 300, W - 96 - 20, 300 + 150),
                (12, 16, 24, 225),
                18,
                outline=C.rgba(C.CYAN, 160),
                width=2,
                alpha=b,
            )
            img = C.draw_text(
                img,
                (W - 96 - 245, 322),
                "+16.3 / +9.3 percentage points",
                C.font("Bold", 40),
                C.CYAN,
                -1,
                "ma",
                b,
            )
            img = C.draw_text(
                img,
                (W - 96 - 245, 388),
                "over navigation-and-manipulation (static / dynamic)",
                C.font("Regular", 24),
                C.MUTED,
                0,
                "ma",
                b,
            )
        img = C.darken(img, 1 - C.fade_in_out(t, 0, T2, 0.5, 0.4))
        return img
    a = C.seg(t, T2 + 0.2, T2 + 0.9)
    img = C.draw_text(
        img,
        (96, 70 - (1 - a) * 20),
        "Failure analysis: collision rate",
        C.font("ExtraBold", 66),
        C.INK,
        -2,
        "la",
        a,
    )
    img = C.draw_text(
        img,
        (96, 150),
        "collision failures in percent of trials, mean ± 95% CI  ·  learning-based policies act on visual input without an explicit collision model",  # noqa: E501
        C.font("Regular", 25),
        C.MUTED,
        0,
        "la",
        C.seg(t, T2 + 0.6, T2 + 1.2),
    )
    for k, (lab, colr) in enumerate(
        (("unknown static", C.CORAL), ("dynamic", (255, 140, 120)))
    ):
        b = C.seg(t, T2 + 1.0, T2 + 1.5)
        x = W - 96 - 420 + k * 230
        layer = C.transparent()
        ImageDraw.Draw(layer).rounded_rectangle(
            (x, 84, x + 26, 108), 5, fill=C.rgba(colr)
        )
        img = C.alpha_over(img, layer, b)
        img = C.draw_text(
            img, (x + 40, 82), lab, C.font("Medium", 26), C.INK, 0, "la", b
        )
    img = C.draw_text(
        img,
        (W - 96, 208),
        "COLLISION RATE (%)  ·  LOWER IS BETTER",
        C.font("Medium", 24, mono=True),
        C.MUTED,
        3,
        "ra",
        C.seg(t, T2 + 1.0, T2 + 1.5),
    )
    img = bar_chart(img, t, T2 + 1.0, "collision")
    b = C.seg(t, T2 + 8.5, T2 + 9.2, C.ease_out_back)
    if b > 0:
        img = C.rounded_panel(
            img,
            (W - 96 - 560, 560, W - 96 - 20, 560 + 150),
            (12, 16, 24, 225),
            18,
            outline=C.rgba(C.CORAL, 160),
            width=2,
            alpha=b,
        )
        img = C.draw_text(
            img,
            (W - 96 - 290, 582),
            "12.3% vs 5.0% collisions",
            C.font("Bold", 40),
            C.CORAL,
            -1,
            "ma",
            b,
        )
        img = C.draw_text(
            img,
            (W - 96 - 290, 648),
            "without vs. with velocity weighting (static scenes)",
            C.font("Regular", 24),
            C.MUTED,
            0,
            "ma",
            b,
        )
    img = C.darken(img, 1 - C.fade_in_out(t, T2, RES_D, 0.4, 2.0))
    return img


# --------------------------------------------------------------------------- 8. OUTRO
OUT_D = 13.2


def draw_outro(t, ctx):
    hero = studio_frame("hero", int(300 + t * 30))
    img = C.new_canvas((6, 8, 12))
    img = C.alpha_over(img, hero, 1.0, (330, 0))
    g = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(g)
    for x in range(0, 900, 6):
        a = int(210 * (1 - x / 900) ** 1.6)
        gd.rectangle((x, 0, x + 6, H), fill=(6, 8, 12, a))
    img = C.alpha_over(img, g)
    x = 110
    a1 = C.seg(t, 0.3, 1.1)
    img = C.hline_reveal(img, 250, x, x + 160, C.seg(t, 0.1, 0.9), C.CYAN, 5)
    img = C.draw_text(
        img,
        (x - (1 - a1) * 40, 280),
        "Visibility-Aware",
        C.font("ExtraBold", 86),
        C.INK,
        -3,
        "la",
        a1,
    )
    img = C.draw_text(
        img,
        (x - (1 - a1) * 40, 372),
        "Mobile Grasping",
        C.font("ExtraBold", 86),
        C.INK,
        -3,
        "la",
        a1,
    )
    rows = [
        (
            2.2,
            "Velocity-aware active perception",
            "gaze follows the swept volume",
            C.CYAN,
        ),
        (
            3.2,
            "Hierarchical subgoal policy",
            "grasp → pre-grasp → observe, target kept in view",
            C.AMBER,
        ),
        (
            4.2,
            "Real-time whole-body replanning",
            "tens of milliseconds as the map updates",
            C.MINT,
        ),
    ]
    for k, (t0, title, sub, colr) in enumerate(rows):
        b = C.seg(t, t0, t0 + 0.6)
        if b <= 0:
            continue
        y = 520 + k * 96
        layer = C.transparent()
        ImageDraw.Draw(layer).rounded_rectangle(
            (x, y + 10, x + 10, y + 58), 4, fill=C.rgba(colr)
        )
        img = C.alpha_over(img, layer, b)
        img = C.draw_text(
            img,
            (x + 34 - (1 - b) * 20, y),
            title,
            C.font("Bold", 36),
            C.INK,
            0,
            "la",
            b,
        )
        img = C.draw_text(
            img,
            (x + 34 - (1 - b) * 20, y + 44),
            sub,
            C.font("Regular", 26),
            C.MUTED,
            0,
            "la",
            b,
        )
    b = C.seg(t, 5.6, 6.4)
    if b > 0:
        img = C.pill(
            img,
            (x, 850),
            "INTERLEAVED WHOLE-BODY PLANNING AND ACTIVE PERCEPTION",
            C.font("Medium", 28, mono=True),
            C.INK,
            (14, 18, 26, 220),
            (24, 14),
            alpha=b,
            tracking=1,
        )
    img = C.draw_text(
        img,
        (x, 940),
        "Simulation: ManiSkill3 + ReplicaCAD.  Real robot: Fetch, fully autonomous.  People blurred for privacy.",
        C.font("Regular", 22),
        C.MUTED,
        0,
        "la",
        C.seg(t, 6.6, 7.2),
    )
    return C.darken(img, 1 - C.fade_in_out(t, 0, OUT_D, 0.6, 1.4))


# --------------------------------------------------------------------------- timeline
# Base timeline (tuned to the narration generated at rate +2%). The actual timeline is
# scaled per section by the ratio of the current narration length to that base, so the
# visual cues stay aligned when the voice-over speed changes.
BASE_NARR = {
    "01_hook": 14.83,
    "02_title": 5.76,
    "03_constraints": 29.52,
    "04_method": 36.00,
    "05_sim": 24.55,
    "06_real": 27.43,
    "07_results": 36.02,
}
BASE_SECTIONS = [
    ("hook", HOOK_D, draw_hook, "01_hook", 0.5),
    ("title", TITLE_D, draw_title, "02_title", 0.3),
    ("constraints", CON_D, draw_constraints, "03_constraints", 0.5),
    ("method", MET_D, draw_method, "04_method", 0.5),
    ("sim", SIM_D, draw_sim, "05_sim", 0.8),
    ("real", REAL_D, draw_real, "06_real", 0.5),
    ("results", RES_D, draw_results, "07_results", 0.5),
]
TIME_SCALE = {}


def _load_scales():
    try:
        cur = json.load(open(os.path.join(C.ROOT, "narration", "durations.json")))
    except Exception:
        cur = {}
    for name, dur, fn, narr, off in BASE_SECTIONS:
        TIME_SCALE[name] = min(1.0, cur.get(narr, BASE_NARR[narr]) / BASE_NARR[narr])


def _scaled(fn, k):
    return lambda t, ctx: fn(t / k, ctx)


_load_scales()
SECTIONS = [
    (
        name,
        dur * TIME_SCALE[name],
        _scaled(fn, TIME_SCALE[name]),
        narr,
        off * TIME_SCALE[name],
    )
    for name, dur, fn, narr, off in BASE_SECTIONS
]
