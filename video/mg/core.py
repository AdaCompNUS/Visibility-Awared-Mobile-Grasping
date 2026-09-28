"""2D motion-graphics primitives on top of Pillow/numpy.

All shots draw onto RGBA PIL images at the output resolution (W x H). Helpers
here provide fonts, eased animation values, text with tracking, glows, panels
and image fitting.
"""
import math
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, "assets", "fonts")
W, H = 1920, 1080
FPS = 30

# palette -------------------------------------------------------------------
BG = (7, 9, 14)
INK = (238, 241, 247)
MUTED = (150, 158, 172)
CYAN = (56, 214, 255)
AMBER = (255, 176, 46)
CORAL = (255, 92, 92)
MINT = (94, 232, 176)
VIOLET = (170, 130, 255)


def rgba(c, a=255):
    return (c[0], c[1], c[2], int(a))


@lru_cache(maxsize=None)
def font(weight="SemiBold", size=48, mono=False):
    name = f"JetBrainsMono-{weight}.ttf" if mono else f"Inter-{weight}.ttf"
    return ImageFont.truetype(os.path.join(FONT_DIR, name), int(size))


# easing --------------------------------------------------------------------
def clamp01(t):
    return 0.0 if t < 0 else 1.0 if t > 1 else float(t)


def ease_out_cubic(t):
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def ease_out_expo(t):
    t = clamp01(t)
    return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)


def ease_out_back(t, s=1.4):
    t = clamp01(t)
    return 1 + (s + 1) * (t - 1) ** 3 + s * (t - 1) ** 2


def seg(t, t0, t1, fn=ease_out_cubic):
    """Animation progress of a segment starting at t0 lasting (t1-t0)."""
    if t1 <= t0:
        return 1.0 if t >= t0 else 0.0
    return fn((t - t0) / (t1 - t0))


def fade_in_out(t, t0, t1, fin=0.4, fout=0.4):
    """1 inside [t0+fin, t1-fout], fading at both ends."""
    if t < t0 or t > t1:
        return 0.0
    a = 1.0
    if fin > 0:
        a = min(a, clamp01((t - t0) / fin))
    if fout > 0:
        a = min(a, clamp01((t1 - t) / fout))
    return ease_in_out(a)


def lerp(a, b, t):
    return a + (b - a) * t


# canvas --------------------------------------------------------------------
def new_canvas(color=BG, alpha=255):
    return Image.new("RGBA", (W, H), rgba(color, alpha))


def transparent():
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def to_np(img):
    return np.asarray(img.convert("RGB"))


def from_np(arr):
    return Image.fromarray(np.ascontiguousarray(arr[..., :3]), "RGB").convert("RGBA")


def alpha_over(base, layer, alpha=1.0, pos=(0, 0)):
    """Composite `layer` (RGBA) over base at pos with an extra opacity."""
    if alpha <= 0:
        return base
    if alpha < 1:
        a = layer.getchannel("A").point(lambda v: int(v * alpha))
        layer = layer.copy()
        layer.putalpha(a)
    if pos == (0, 0) and layer.size == base.size:
        return Image.alpha_composite(base, layer)
    tmp = Image.new("RGBA", base.size, (0, 0, 0, 0))
    tmp.paste(layer, (int(pos[0]), int(pos[1])), layer)
    return Image.alpha_composite(base, tmp)


def fit_cover(img, size):
    """Resize+crop to fill `size` preserving aspect."""
    w, h = img.size
    tw, th = size
    s = max(tw / w, th / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.LANCZOS if s < 1 else Image.BICUBIC)
    x0, y0 = (nw - tw) // 2, (nh - th) // 2
    return img.crop((x0, y0, x0 + tw, y0 + th))


def fit_contain(img, size):
    w, h = img.size
    tw, th = size
    s = min(tw / w, th / h)
    return img.resize(
        (max(1, round(w * s)), max(1, round(h * s))),
        Image.LANCZOS if s < 1 else Image.BICUBIC,
    )


def zoom_crop(img, zoom=1.0, center=(0.5, 0.5)):
    """Ken-Burns style zoom into an image (keeps size)."""
    if zoom <= 1.0001:
        return img
    w, h = img.size
    cw, ch = w / zoom, h / zoom
    cx, cy = center[0] * w, center[1] * h
    x0 = min(max(0, cx - cw / 2), w - cw)
    y0 = min(max(0, cy - ch / 2), h - ch)
    return img.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize(
        (w, h), Image.BICUBIC
    )


def darken(img, amount=0.5):
    arr = np.asarray(img).astype(np.float32)
    arr[..., :3] *= 1 - amount
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), img.mode)


def vignette(img, strength=0.55, power=2.2):
    mask = _vignette_mask(img.size, strength, power)
    arr = np.asarray(img).astype(np.float32)
    arr[..., :3] *= mask[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), img.mode)


@lru_cache(maxsize=8)
def _vignette_mask(size, strength, power):
    w, h = size
    y, x = np.mgrid[0:h, 0:w]
    nx = (x - w / 2) / (w / 2)
    ny = (y - h / 2) / (h / 2)
    r = np.sqrt(nx**2 + ny**2) / math.sqrt(2)
    return (1 - strength * r**power).astype(np.float32)


def gradient_overlay(img, top=(0, 0, 0, 0), bottom=(0, 0, 0, 200), y0=0.5, y1=1.0):
    """Vertical gradient layer for legible lower-thirds."""
    w, h = img.size
    g = Image.new("RGBA", (1, h), (0, 0, 0, 0))
    px = g.load()
    for y in range(h):
        t = clamp01((y / h - y0) / max(1e-6, (y1 - y0)))
        t = ease_in_out(t)
        px[0, y] = tuple(int(lerp(top[i], bottom[i], t)) for i in range(4))
    return Image.alpha_composite(img, g.resize((w, h)))


def blur_region(img, box, radius=28):
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, x1), min(img.height, y1)
    if x1 <= x0 or y1 <= y0:
        return img
    region = img.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(radius))
    # soft elliptical mask
    m = Image.new("L", region.size, 0)
    ImageDraw.Draw(m).ellipse((0, 0, region.width - 1, region.height - 1), fill=255)
    m = m.filter(ImageFilter.GaussianBlur(max(2, radius // 3)))
    img.paste(region, (x0, y0), m)
    return img


# text ----------------------------------------------------------------------
def text_size(txt, fnt, tracking=0):
    if not txt:
        return 0, 0
    bbox = fnt.getbbox(txt)
    w = bbox[2] + tracking * max(0, len(txt) - 1)
    asc, desc = fnt.getmetrics()
    return w, asc + desc


def draw_text(
    img, xy, txt, fnt, fill=INK, tracking=0, anchor="la", alpha=1.0, shadow=None
):
    """Draw text with letter tracking. anchor: l/m/r + a/m/d (PIL-like, simplified)."""
    if not txt or alpha <= 0:
        return img
    w, h = text_size(txt, fnt, tracking)
    x, y = xy
    if anchor[0] == "m":
        x -= w / 2
    elif anchor[0] == "r":
        x -= w
    if anchor[1] == "m":
        y -= h / 2
    elif anchor[1] == "d":
        y -= h
    layer = Image.new("RGBA", (int(w) + 8, int(h) + 8), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = rgba(fill, 255 * alpha)
    if tracking == 0:
        if shadow:
            d.text(
                (2 + shadow[0], 2 + shadow[1]),
                txt,
                font=fnt,
                fill=rgba((0, 0, 0), 160 * alpha),
            )
        d.text((2, 2), txt, font=fnt, fill=col)
    else:
        cx = 2
        for ch in txt:
            if shadow:
                d.text(
                    (cx + shadow[0], 2 + shadow[1]),
                    ch,
                    font=fnt,
                    fill=rgba((0, 0, 0), 160 * alpha),
                )
            d.text((cx, 2), ch, font=fnt, fill=col)
            cx += fnt.getlength(ch) + tracking
    return alpha_over(img, layer, 1.0, (x - 2, y - 2))


def wrap_lines(txt, fnt, max_w, tracking=0):
    words = txt.split()
    lines, cur = [], ""
    for wd in words:
        cand = (cur + " " + wd).strip()
        if text_size(cand, fnt, tracking)[0] <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def draw_paragraph(
    img, xy, txt, fnt, max_w, fill=INK, line_h=None, tracking=0, anchor="l", alpha=1.0
):
    lines = wrap_lines(txt, fnt, max_w, tracking)
    lh = line_h or int(fnt.size * 1.25)
    x, y = xy
    for i, ln in enumerate(lines):
        img = draw_text(
            img, (x, y + i * lh), ln, fnt, fill, tracking, anchor + "a", alpha
        )
    return img, len(lines) * lh


def glow_text(
    img,
    xy,
    txt,
    fnt,
    fill=CYAN,
    radius=18,
    strength=1.0,
    tracking=0,
    anchor="la",
    alpha=1.0,
):
    w, h = text_size(txt, fnt, tracking)
    pad = radius * 3
    layer = Image.new("RGBA", (int(w) + pad * 2, int(h) + pad * 2), (0, 0, 0, 0))
    layer = draw_text(layer, (pad, pad), txt, fnt, fill, tracking, "la", 1.0)
    glow = layer.filter(ImageFilter.GaussianBlur(radius))
    ga = glow.getchannel("A").point(lambda v: min(255, int(v * 1.6 * strength)))
    glow.putalpha(ga)
    layer = Image.alpha_composite(glow, layer)
    x, y = xy
    if anchor[0] == "m":
        x -= w / 2
    elif anchor[0] == "r":
        x -= w
    if anchor[1] == "m":
        y -= h / 2
    elif anchor[1] == "d":
        y -= h
    return alpha_over(img, layer, alpha, (x - pad, y - pad))


# shapes --------------------------------------------------------------------
def rounded_panel(
    img, box, fill=(12, 16, 24, 200), radius=18, outline=None, width=2, alpha=1.0
):
    x0, y0, x1, y1 = [int(v) for v in box]
    layer = Image.new("RGBA", (x1 - x0 + 4, y1 - y0 + 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle(
        (2, 2, x1 - x0 + 1, y1 - y0 + 1),
        radius=radius,
        fill=fill,
        outline=outline,
        width=width,
    )
    return alpha_over(img, layer, alpha, (x0 - 2, y0 - 2))


def glow_rect(img, box, color=CYAN, radius=14, width=3, blur=16, alpha=1.0):
    x0, y0, x1, y1 = [int(v) for v in box]
    pad = blur * 3
    layer = Image.new("RGBA", (x1 - x0 + pad * 2, y1 - y0 + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle(
        (pad, pad, pad + x1 - x0, pad + y1 - y0),
        radius=radius,
        outline=rgba(color),
        width=width,
    )
    glow = layer.filter(ImageFilter.GaussianBlur(blur))
    ga = glow.getchannel("A").point(lambda v: min(255, int(v * 2.0)))
    glow.putalpha(ga)
    layer = Image.alpha_composite(glow, layer)
    return alpha_over(img, layer, alpha, (x0 - pad, y0 - pad))


def draw_line(img, p0, p1, color=INK, width=3, alpha=1.0):
    layer = transparent()
    ImageDraw.Draw(layer).line([p0, p1], fill=rgba(color), width=width)
    return alpha_over(img, layer, alpha)


def hline_reveal(img, y, x0, x1, t, color=CYAN, width=3, alpha=1.0):
    """A horizontal rule growing from x0 to x1 with progress t."""
    xe = x0 + (x1 - x0) * ease_out_cubic(t)
    if xe <= x0:
        return img
    return draw_line(img, (x0, y), (xe, y), color, width, alpha)


def pill(
    img,
    xy,
    txt,
    fnt,
    fg=INK,
    bg=(20, 26, 36, 220),
    pad=(18, 8),
    radius=999,
    alpha=1.0,
    anchor="l",
    tracking=0,
):
    w, h = text_size(txt, fnt, tracking)
    bw, bh = w + pad[0] * 2, h + pad[1] * 2
    x, y = xy
    if anchor == "m":
        x -= bw / 2
    elif anchor == "r":
        x -= bw
    layer = Image.new("RGBA", (int(bw) + 4, int(bh) + 4), (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle(
        (2, 2, bw + 1, bh + 1), radius=min(radius, bh / 2), fill=bg
    )
    layer = draw_text(layer, (2 + pad[0], 2 + pad[1]), txt, fnt, fg, tracking)
    return alpha_over(img, layer, alpha, (x - 2, y - 2))


def grid_bg(img, spacing=80, color=(255, 255, 255, 10), alpha=1.0):
    layer = transparent()
    d = ImageDraw.Draw(layer)
    for x in range(0, W, spacing):
        d.line([(x, 0), (x, H)], fill=color, width=1)
    for y in range(0, H, spacing):
        d.line([(0, y), (W, y)], fill=color, width=1)
    return alpha_over(img, layer, alpha)


def noise_layer(seed, amount=6):
    rng = np.random.default_rng(seed)
    n = rng.integers(0, amount, size=(H, W, 1), dtype=np.uint8)
    arr = np.repeat(n, 3, axis=2)
    return Image.fromarray(arr, "RGB").convert("RGBA")


def add_grain(img, seed, amount=6):
    return ImageChops.add(img, noise_layer(seed, amount))


def letterbox(img, bar=0.0, color=BG):
    if bar <= 0:
        return img
    h = int(H * bar)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, h), fill=rgba(color))
    d.rectangle((0, H - h, W, H), fill=rgba(color))
    return img


def counter_text(value, decimals=1, suffix="%"):
    return f"{value:.{decimals}f}{suffix}"
