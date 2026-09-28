"""Generate collision-free configurations and arm motions for the studio shots with VAMP.

Runs in the MAIN repo environment (it needs the vendored `vamp` module):
    pixi run --manifest-path ../pixi.toml python blender/plan_studio.py
Writes out/studio_plan.json which blender/studio.py consumes.

Geometry is expressed in the robot base frame (base_link at origin, x forward), which is
what the fixed-base VAMP Fetch model (torso + 7 arm joints) expects.
"""
import json
import math
import os

import numpy as np
import vamp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "studio_plan.json")
f = vamp.fetch
rng = np.random.default_rng(3)

REST = [1.32, 1.40, -0.20, 1.72, 0.0, 1.66, 0.0]
PREPOSE_ARM = [1.29, -0.811, -1.127, 1.524, -0.495, 0.892, 1.554]
GRASP_ARM = [-0.214, -0.276, 0.752, 0.694, -0.69, 1.384, 0.758]
LIM = np.array(
    [
        [-1.6056, 1.6056],
        [-1.221, 1.518],
        [-3.14, 3.14],
        [-2.251, 2.251],
        [-3.14, 3.14],
        [-2.16, 2.16],
        [-3.14, 3.14],
    ]
)


def base_T(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]])


def to_base(p, x, y, yaw):
    T = np.linalg.inv(base_T(x, y, yaw))
    return (T @ np.array([p[0], p[1], p[2], 1.0]))[:3]


def table_env(center, size, base=(0, 0, 0), extra=()):
    """Table (top + 4 legs) and extra cuboids [(center, half), ...] in the base frame."""
    env = vamp.Environment()
    x, y, yaw = base
    cx, cy = center
    w, d, h = size
    top = to_base((cx, cy, h - 0.02), x, y, yaw)
    env.add_cuboid(vamp.Cuboid(list(top), [0, 0, -yaw], [w / 2, d / 2, 0.02]))
    for sx in (-1, 1):
        for sy in (-1, 1):
            leg = to_base(
                (cx + sx * (w / 2 - 0.06), cy + sy * (d / 2 - 0.06), (h - 0.04) / 2),
                x,
                y,
                yaw,
            )
            env.add_cuboid(
                vamp.Cuboid(list(leg), [0, 0, -yaw], [0.025, 0.025, (h - 0.04) / 2])
            )
    for c, half in extra:
        env.add_cuboid(
            vamp.Cuboid(list(to_base(c, x, y, yaw)), [0, 0, -yaw], list(half))
        )
    return env


def valid(q8, env):
    return bool(f.validate([float(v) for v in q8], env))


def ee(q8):
    return np.array(f.eefk([float(v) for v in q8])[0])


def seg_point_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(1e-9, np.dot(ab, ab)), 0, 1)
    return np.linalg.norm(p - (a + t * ab)), t


def occludes(q8, cam, target, margin=0.0):
    """True if any robot sphere crosses the camera->target segment."""
    for s in f.fk([float(v) for v in q8]):
        d, t = seg_point_dist(np.array([s.x, s.y, s.z]), cam, target)
        if 0.08 < t < 0.95 and d < s.r + margin:
            return True
    return False


def sample_near(arm0, env, torso, score, n=4000, sigma=0.55, require=None):
    best, best_s = None, 1e9
    for k in range(n):
        arm = np.array(arm0) + rng.normal(0, sigma, 7) * (0.35 if k < n // 4 else 1.0)
        arm = np.clip(arm, LIM[:, 0], LIM[:, 1])
        q = [torso] + list(arm)
        if not valid(q, env):
            continue
        if require is not None and not require(q):
            continue
        s = score(q)
        if s < best_s:
            best, best_s = q, s
    return best, best_s


def make_rng(seed):
    for ctor in (
        lambda: f.halton(seed),
        lambda: f.halton(),
        lambda: f.xorshift(seed),
        lambda: f.xorshift(),
    ):
        try:
            return ctor()
        except Exception:
            continue
    raise RuntimeError("no rng")


def plan(q0, q1, env, seed=1):
    st = vamp.RRTCSettings()
    st.max_iterations = 200000
    st.max_samples = 200000
    r = f.rrtc([float(v) for v in q0], [float(v) for v in q1], env, st, make_rng(seed))
    if not r.solved:
        print("  rrtc failed")
        return None
    ss = vamp.SimplifySettings()
    try:
        sr = f.simplify(r.path, env, ss, make_rng(seed + 7))
        path = sr.path
    except Exception as e:
        print("  simplify err", e)
        path = r.path
    for name in ("interpolate_to_resolution", "interpolate"):
        if hasattr(path, name):
            try:
                getattr(path, name)(f.resolution())
                break
            except Exception as e:
                print("  interp err", name, e)
    pts = []
    for i in range(len(path)):
        c = path[i]
        pts.append(list(map(float, c.to_list() if hasattr(c, "to_list") else list(c))))
    return pts


out = {}

# ----------------------------------------------------------------- OBJVIS
TORSO = 0.25
CAN = np.array([0.95, 0.05, 0.79])
CAM = np.array([0.164, 0.0, 0.377 + TORSO + 0.603 + 0.058 + 0.0225])
env = table_env(
    (1.05, 0.0), (1.0, 0.7, 0.74), extra=[((0.95, 0.05, 0.79), (0.035, 0.035, 0.05))]
)
print("objvis: camera", CAM.round(3), "can", CAN)
# visible pre-grasp: end effector ~0.28 m in front-above of the can, not crossing the camera ray
goal_vis = CAN + np.array([-0.28, 0.0, 0.12])
q_vis, s = sample_near(
    [0.25, 0.35, 0.0, 1.35, 0.0, 0.95, 0.0],
    env,
    TORSO,
    lambda q: np.linalg.norm(ee(q) - goal_vis),
    require=lambda q: not occludes(q, CAM, CAN, 0.03),
)
print("q_vis", np.round(q_vis, 3), "ee err", round(s, 3))
# occluding pose: forearm/wrist crossing the camera->can ray about 40% of the way, away from the table
ray_pt = CAM + 0.42 * (CAN - CAM)
q_occ, s = sample_near(
    [0.95, -0.25, 0.0, 0.85, 0.0, 0.9, 0.0],
    env,
    TORSO,
    lambda q: np.linalg.norm(ee(q) - (ray_pt + np.array([0.0, 0.0, -0.05]))),
    require=lambda q: occludes(q, CAM, CAN, -0.01),
)
print("q_occ", np.round(q_occ, 3), "ee err", round(s, 3))
# second visible pre-grasp on the other side (approach from the robot's right), also not occluding
goal_vis2 = CAN + np.array([-0.22, -0.2, 0.14])
q_vis2, s = sample_near(
    [-0.35, 0.55, 0.3, 1.25, -0.2, 0.75, 0.2],
    env,
    TORSO,
    lambda q: np.linalg.norm(ee(q) - goal_vis2),
    require=lambda q: not occludes(q, CAM, CAN, 0.03),
)
print("q_vis2", np.round(q_vis2, 3), "ee err", round(s, 3))
p1 = plan(q_vis, q_occ, env, 1)
p2 = plan(q_occ, q_vis2, env, 2)
print("paths", None if p1 is None else len(p1), None if p2 is None else len(p2))
out["objvis"] = {
    "torso": TORSO,
    "q_vis": q_vis,
    "q_occ": q_occ,
    "q_vis2": q_vis2,
    "path1": p1,
    "path2": p2,
    "cam": CAM.tolist(),
    "can": CAN.tolist(),
    "occl_along_path1": [occludes(q, CAM, CAN, -0.01) for q in (p1 or [])],
    "occl_along_path2": [occludes(q, CAM, CAN, -0.01) for q in (p2 or [])],
}

# ----------------------------------------------------------------- SUBGOALS
TBL_C, TBL_S = (1.6, 0.0), (1.1, 0.8, 0.74)
CAN_W = np.array([1.45, 0.1, 0.79])
cands = {}
# grasp in place: base right at the table edge, gripper reaching the can from the front
for base in [(0.85, 0.05, 0.0), (0.8, 0.0, 0.0), (0.82, 0.15, -0.1)]:
    e = table_env(TBL_C, TBL_S, base, extra=[(tuple(CAN_W), (0.035, 0.035, 0.05))])
    can_b = to_base(CAN_W, *base)
    q, s = sample_near(
        GRASP_ARM,
        e,
        0.3,
        lambda q: np.linalg.norm(ee(q) - (can_b + np.array([-0.16, 0.0, 0.03]))),
        n=6000,
    )
    if q is not None and s < 0.06:
        cands["grasp"] = {"base": list(base), "q": q, "ee_err": s}
        break
print("grasp", cands.get("grasp"))
# pre-grasp: base ~0.9 m from the can at an angle, torso 0.2, arm in a mid reach that keeps the can visible
for base in [(0.55, -0.95, 0.75), (0.5, -1.0, 0.8), (0.45, -0.9, 0.7)]:
    e = table_env(TBL_C, TBL_S, base, extra=[(tuple(CAN_W), (0.035, 0.035, 0.05))])
    can_b = to_base(CAN_W, *base)
    cam_b = np.array([0.164, 0.0, 0.377 + 0.2 + 0.603 + 0.058 + 0.0225])
    q, s = sample_near(
        PREPOSE_ARM,
        e,
        0.2,
        lambda q: np.linalg.norm(ee(q) - (can_b + np.array([-0.45, 0.0, 0.22]))),
        n=8000,
        sigma=0.7,
        require=lambda q: not occludes(q, cam_b, can_b, 0.03),
    )
    print("  pregrasp try", base, None if q is None else round(s, 3))
    if q is not None and s < 0.35:
        cands["pregrasp"] = {"base": list(base), "q": q, "ee_err": s}
        break
print("pregrasp", cands.get("pregrasp"))
# observe: standoff pose, arm tucked (REST), torso low
base = (-0.35, 1.15, -0.6)
e = table_env(TBL_C, TBL_S, base)
q = [0.05] + REST
cands["observe"] = {"base": list(base), "q": q, "valid": valid(q, e)}
print("observe", cands["observe"])
out["subgoals"] = cands
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT)
