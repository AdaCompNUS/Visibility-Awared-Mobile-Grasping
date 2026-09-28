"""Build per-frame inset images from the head-camera render:
   - depth_NNNN.png : viridis-coloured z-depth
   - belief_NNNN.png: accumulated point-cloud map (what the robot has observed so far),
                      drawn from a fixed 3/4 view with the executed trail and robot position.
Run after blender/headcam.py:  pixi run python -m mg.belief
"""
import glob
import json
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import core as C

OUT = os.path.join(C.ROOT, "out")
HC = os.path.join(OUT, os.environ.get("HEADCAM_DIR", "headcam2"))
DATA = os.path.join(C.ROOT, "assets/dropbox/replanning_data/replanning_data")
W_IN, H_IN = 400, 300


def viridis(x):
    x = np.clip(x, 0, 1)
    lut = np.array(
        [
            [68, 1, 84],
            [72, 40, 120],
            [62, 74, 137],
            [49, 104, 142],
            [38, 130, 142],
            [31, 158, 137],
            [53, 183, 121],
            [109, 205, 89],
            [180, 222, 44],
            [253, 231, 37],
        ],
        np.float32,
    )
    idx = x * (len(lut) - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, len(lut) - 1)
    t = (idx - i0)[..., None]
    return (lut[i0] * (1 - t) + lut[i1] * t).astype(np.uint8)


def read_aux(path):
    """Multilayer EXR written by the File Output node: channels depth.V and index.V."""
    import OpenEXR

    ex = OpenEXR.File(path)
    chans = {}
    for part in ex.parts:
        for k, c in part.channels.items():
            chans[k] = np.asarray(c.pixels, dtype=np.float32)
    D = chans["depth.V"]
    I = chans.get("index.V", np.zeros_like(D))
    return D, I


def main():
    cam = json.load(open(os.path.join(HC, "camera.json")))
    fovy = math.radians(cam["fovy_deg"])
    w, h = cam["res"]
    fy = (h / 2) / math.tan(fovy / 2)
    fx = fy
    cx, cy = w / 2, h / 2
    Hs = np.load(os.path.join(DATA, "full_trajectory/head_camera_poses.npy"))
    T = np.load(os.path.join(DATA, "full_trajectory/executed_trajectory.npy"))
    frames = sorted(
        int(os.path.basename(p)[4:8]) for p in glob.glob(os.path.join(HC, "aux_*.exr"))
    )
    print("frames with depth:", len(frames))
    # pixel grid (subsampled)
    step = 3
    vs, us = np.mgrid[0:h:step, 0:w:step]
    us = us.astype(np.float32)
    vs = vs.astype(np.float32)
    VOX = 0.04
    occupied = {}
    pts_all = np.zeros((0, 3), np.float32)
    # fixed map view camera: 3/4 top-down over the route area
    look_from = np.array([-0.5, -8.5, 7.5])
    look_at = np.array([1.6, -2.6, 0.4])
    fwd = look_at - look_from
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0, 0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    fmap = 1.35 * W_IN / 2
    zmin, zmax = -0.05, 2.0

    def project(P):
        d = P - look_from
        x = d @ right
        y = d @ up
        z = d @ fwd
        ok = z > 0.2
        u = W_IN / 2 + fmap * x / np.maximum(z, 1e-3)
        v = H_IN / 2 - fmap * y / np.maximum(z, 1e-3)
        return u, v, z, ok

    for k, f in enumerate(frames):
        D, I = read_aux(os.path.join(HC, f"aux_{f:04d}.exr"))
        # ---- depth inset
        valid = (D > 0.05) & (D < 1e5)
        dn = np.zeros_like(D)
        dn[valid] = np.clip((D[valid] - 0.3) / (5.0 - 0.3), 0, 1)
        img = viridis(1 - dn)
        img[~valid] = (20, 20, 30)
        img[I > 0.5] = (img[I > 0.5] * 0.55).astype(np.uint8)
        Image.fromarray(img).resize((W_IN, H_IN), Image.BILINEAR).save(
            os.path.join(HC, f"depthc_{f:04d}.png")
        )
        # ---- back-project into the world (skip robot pixels + far/invalid)
        d = D[::step, ::step]
        ii = I[::step, ::step]
        m = (d > 0.15) & (d < 6.0) & (ii < 0.5)
        if m.any():
            zc = d[m]
            xc = (us[m] - cx) / fx * zc
            yc = (vs[m] - cy) / fy * zc
            Pc = np.stack([xc, yc, zc, np.ones_like(zc)], 1)
            Pw = (Hs[f] @ Pc.T).T[:, :3]
            Pw = Pw[(Pw[:, 2] > 0.03) & (Pw[:, 2] < 2.3)]  # drop floor + ceiling
            keys = np.floor(Pw / VOX).astype(np.int32)
            new = []
            for key, p in zip(map(tuple, keys), Pw):
                if key not in occupied:
                    occupied[key] = True
                    new.append(p)
            if new:
                pts_all = np.concatenate([pts_all, np.array(new, np.float32)], 0)
        # ---- belief map inset
        canvas = np.full((H_IN, W_IN, 3), (8, 10, 16), np.uint8)
        if len(pts_all):
            u, v, z, ok = project(pts_all)
            order = np.argsort(-z)
            u, v, z, ok, hz = u[order], v[order], z[order], ok[order], pts_all[order, 2]
            col = viridis((hz - zmin) / (zmax - zmin))
            ui = np.round(u).astype(int)
            vi = np.round(v).astype(int)
            inb = ok & (ui >= 0) & (ui < W_IN) & (vi >= 0) & (vi < H_IN)
            canvas[vi[inb], ui[inb]] = col[inb]
            # thicken slightly
            canvas = cv2.dilate(canvas, np.ones((2, 2), np.uint8))
        pil = Image.fromarray(canvas)
        dr = ImageDraw.Draw(pil)
        # trail + robot
        trail = T[: f + 1 : 4, :2]
        if len(trail) > 1:
            P3 = np.concatenate([trail, np.full((len(trail), 1), 0.02)], 1)
            u, v, z, ok = project(P3)
            dr.line(list(zip(u.tolist(), v.tolist())), fill=(255, 200, 90), width=2)
        u, v, z, ok = project(np.array([[T[f, 0], T[f, 1], 0.1]]))
        dr.ellipse(
            (u[0] - 5, v[0] - 5, u[0] + 5, v[0] + 5),
            fill=(255, 255, 255),
            outline=(56, 214, 255),
            width=2,
        )
        pil.save(os.path.join(HC, f"belief_{f:04d}.png"))
        if k % 100 == 0:
            print(f"frame {f}: {len(pts_all)} map points", flush=True)
    print("done")


if __name__ == "__main__":
    main()
