"""Frame sources: video files (cv2), image sequences and stills, plus face blurring."""
import glob
import json
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

cv2.setNumThreads(4)

from . import core

ROOT = core.ROOT
DROPBOX = os.path.join(ROOT, "assets", "dropbox")
FACE_MODEL = os.path.join(ROOT, "assets", "models", "face_detection_yunet_2023mar.onnx")
FACE_CACHE_DIR = os.path.join(ROOT, "out", "face_cache")


class VideoSource:
    """Random-access frames from a video (sequential reads are cached to be fast)."""

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.n / self.fps
        self._last_idx = -10
        self._last = None

    def frame_index(self, t):
        return int(min(max(0, round(t * self.fps)), self.n - 1))

    def frame_np(self, t):
        idx = self.frame_index(t)
        if idx == self._last_idx and self._last is not None:
            return self._last
        if idx != self._last_idx + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = self.cap.read()
        if not ok:
            if self._last is not None:
                return self._last
            fr = np.zeros((self.h, self.w, 3), np.uint8)
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        self._last_idx, self._last = idx, fr
        return fr

    def frame(self, t, size=None):
        fr = self.frame_np(t)
        if size is not None and (fr.shape[1], fr.shape[0]) != tuple(size):
            fr = cv2.resize(
                fr,
                tuple(size),
                interpolation=cv2.INTER_AREA
                if size[0] < fr.shape[1]
                else cv2.INTER_LINEAR,
            )
        return Image.fromarray(fr, "RGB").convert("RGBA")

    def frame_cover(self, t, size=(core.W, core.H)):
        fr = self.frame_np(t)
        h, w = fr.shape[:2]
        tw, th = size
        s = max(tw / w, th / h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        fr = cv2.resize(
            fr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        )
        x0, y0 = (nw - tw) // 2, (nh - th) // 2
        return Image.fromarray(
            np.ascontiguousarray(fr[y0 : y0 + th, x0 : x0 + tw]), "RGB"
        ).convert("RGBA")


class ImageSequence:
    def __init__(self, pattern, fps=30.0):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(pattern)
        self.fps = fps
        self.n = len(self.files)
        self.duration = self.n / fps

    def frame_at_index(self, i):
        i = int(min(max(0, i), self.n - 1))
        return Image.open(self.files[i]).convert("RGBA")

    def frame(self, t):
        return self.frame_at_index(round(t * self.fps))


@lru_cache(maxsize=64)
def still(path):
    return Image.open(path).convert("RGBA")


# ------------------------------------------------------------------ faces
_detector = None


def _det():
    global _detector
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(
            FACE_MODEL, "", (320, 320), 0.5, 0.3, 5000
        )
    return _detector


def detect_faces(frame_rgb, det_w=960):
    h, w = frame_rgb.shape[:2]
    s = det_w / w
    small = cv2.resize(frame_rgb, (det_w, int(h * s)))
    d = _det()
    d.setInputSize((small.shape[1], small.shape[0]))
    _, faces = d.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    out = []
    if faces is not None:
        for f in faces:
            x, y, fw, fh, sc = f[0] / s, f[1] / s, f[2] / s, f[3] / s, float(f[-1])
            out.append([float(x), float(y), float(fw), float(fh), sc])
    return out


def face_track(video_path, t0, t1, step=1, cache=True):
    """Detect faces over [t0, t1] (every `step` frames); returns {frame_idx: [boxes]}.

    Boxes are dilated temporally (kept for +-6 frames) so short misses don't unblur.
    """
    os.makedirs(FACE_CACHE_DIR, exist_ok=True)
    key = f"{os.path.basename(video_path)}_{t0:.2f}_{t1:.2f}_{step}.json".replace(
        " ", "_"
    ).replace(",", "")
    cp = os.path.join(FACE_CACHE_DIR, key)
    if cache and os.path.exists(cp):
        return {int(k): v for k, v in json.load(open(cp)).items()}
    vs = VideoSource(video_path)
    i0, i1 = vs.frame_index(t0), vs.frame_index(t1)
    raw = {}
    for i in range(i0, i1 + 1, step):
        fr = vs.frame_np(i / vs.fps)
        raw[i] = detect_faces(fr)
    # temporal dilation
    out = {}
    keys = sorted(raw)
    for i in keys:
        boxes = []
        for j in keys:
            if abs(j - i) <= 6 * step:
                boxes.extend(raw[j])
        out[i] = boxes
    if cache:
        json.dump(out, open(cp, "w"))
    return out


def merge_boxes(boxes, min_conf=0.6):
    """Cluster overlapping detections (from temporal dilation) into one box each."""
    boxes = [b for b in boxes if b[4] >= min_conf]
    clusters = []
    for b in sorted(boxes, key=lambda v: -v[4]):
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        for c in clusters:
            ccx, ccy = c["cx"], c["cy"]
            if abs(cx - ccx) < 0.9 * max(b[2], c["w"]) and abs(cy - ccy) < 0.9 * max(
                b[3], c["h"]
            ):
                n = c["n"]
                c["cx"] = (ccx * n + cx) / (n + 1)
                c["cy"] = (ccy * n + cy) / (n + 1)
                c["w"] = max(c["w"], b[2])
                c["h"] = max(c["h"], b[3])
                c["n"] = n + 1
                break
        else:
            clusters.append(
                {"cx": cx, "cy": cy, "w": b[2], "h": b[3], "n": 1, "conf": b[4]}
            )
    return [
        [c["cx"] - c["w"] / 2, c["cy"] - c["h"] / 2, c["w"], c["h"], c["conf"]]
        for c in clusters
    ]


def apply_face_blur(img, boxes, scale=1.0, pad=0.45, radius=30):
    """Blur each face box (in source pixels) on img (already scaled by `scale`)."""
    boxes = merge_boxes(boxes)
    for b in boxes:
        x, y, w, h, _ = b
        cx, cy = (x + w / 2) * scale, (y + h / 2) * scale
        rw, rh = w * scale * (1 + pad), h * scale * (1 + pad) * 1.15
        core.blur_region(
            img,
            (cx - rw / 2, cy - rh / 2 - rh * 0.08, cx + rw / 2, cy + rh / 2),
            radius,
        )
    return img


# ------------------------------------------------------------------ people (whole body)
PERSON_MODEL = os.path.join(
    ROOT, "assets", "models", "object_detection_yolox_2022nov.onnx"
)
_person_net = None


def _yolox():
    global _person_net
    if _person_net is None:
        _person_net = cv2.dnn.readNet(PERSON_MODEL)
    return _person_net


def detect_people(frame_rgb, conf=0.3):
    net = _yolox()
    fr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = fr.shape[:2]
    S = 640
    r = min(S / w, S / h)
    nw, nh = int(w * r), int(h * r)
    pad = np.full((S, S, 3), 114, np.uint8)
    pad[:nh, :nw] = cv2.resize(fr, (nw, nh))
    net.setInput(cv2.dnn.blobFromImage(pad, 1.0, (S, S), swapRB=False, crop=False))
    out = net.forward()[0]
    grids, exp = [], []
    for s in (8, 16, 32):
        g = S // s
        yv, xv = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        grids.append(np.stack([xv, yv], -1).reshape(-1, 2))
        exp.append(np.full((g * g, 1), s))
    grids = np.concatenate(grids)
    exp = np.concatenate(exp)
    xy = (out[:, :2] + grids) * exp
    wh = np.exp(out[:, 2:4]) * exp
    scores = out[:, 4] * out[:, 5:].max(1)
    cid = out[:, 5:].argmax(1)
    m = (scores > conf) & (cid == 0)
    if not m.any():
        return []
    boxes = np.concatenate([xy[m] - wh[m] / 2, wh[m]], 1) / r
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores[m].tolist(), conf, 0.5)
    return [
        [float(v) for v in boxes[i]] + [float(scores[m][i])]
        for i in np.array(idx).reshape(-1)
    ]


def person_track(video_path, t0, t1, step=2, cache=True):
    """People boxes over [t0, t1]; boxes are kept for +-8 frames around each detection."""
    os.makedirs(FACE_CACHE_DIR, exist_ok=True)
    key = (
        f"people_{os.path.basename(video_path)}_{t0:.2f}_{t1:.2f}_{step}.json".replace(
            " ", "_"
        ).replace(",", "")
    )
    cp = os.path.join(FACE_CACHE_DIR, key)
    if cache and os.path.exists(cp):
        return {int(k): v for k, v in json.load(open(cp)).items()}
    vs = VideoSource(video_path)
    i0, i1 = vs.frame_index(t0), vs.frame_index(t1)
    raw = {}
    for i in range(i0, i1 + 1, step):
        raw[i] = detect_people(vs.frame_np(i / vs.fps))
    out = {}
    keys = sorted(raw)
    for i in keys:
        boxes = []
        for j in keys:
            if abs(j - i) <= 8:
                boxes.extend(raw[j])
        out[i] = merge_boxes(boxes, min_conf=0.3)
    if cache:
        json.dump(out, open(cp, "w"))
    return out


def anonymize_region(img, box, radius=42, pixel=26):
    """Heavy pixelation + blur inside a soft rounded box (whole person)."""
    from PIL import ImageDraw, ImageFilter

    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, x1), min(img.height, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return img
    region = img.crop((x0, y0, x1, y1))
    small = region.resize(
        (max(1, region.width // pixel), max(1, region.height // pixel)), Image.BILINEAR
    )
    region = small.resize(region.size, Image.NEAREST).filter(
        ImageFilter.GaussianBlur(radius * 0.5)
    )
    m = Image.new("L", region.size, 0)
    ImageDraw.Draw(m).rounded_rectangle(
        (0, 0, region.width - 1, region.height - 1),
        radius=min(60, region.width // 3),
        fill=255,
    )
    m = m.filter(ImageFilter.GaussianBlur(14))
    img.paste(region, (x0, y0), m)
    return img


def apply_person_blur(img, boxes, scale=1.0, pad=0.12):
    for b in boxes:
        x, y, w, h, _ = b
        cx, cy = (x + w / 2) * scale, (y + h / 2) * scale
        rw, rh = w * scale * (1 + pad), h * scale * (1 + pad * 0.6)
        anonymize_region(img, (cx - rw / 2, cy - rh / 2, cx + rw / 2, cy + rh / 2))
    return img
