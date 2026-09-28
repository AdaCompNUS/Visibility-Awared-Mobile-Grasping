"""Cinematic re-render of the recorded dynamic-obstacle episode (replanning_data) in Blender.

Modes (env):
  PREVIEW=1  -> render a handful of frames at low res to out/episode_preview/
  FRAMES=a:b:step, RES=1920x1080, SAMPLES=64, OUTDIR=...
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
import render_common as rc
import replica as rp
from fetch_rig import FetchRig, ghost_material
from mathutils import Matrix
from scipy.ndimage import gaussian_filter1d

DATA = os.path.join(rc.ROOT, "assets/dropbox/replanning_data/replanning_data")
PREVIEW = os.environ.get("PREVIEW", "0") == "1"
RES = tuple(
    int(v)
    for v in os.environ.get("RES", "960x540" if PREVIEW else "1920x1080").split("x")
)
SAMPLES = int(os.environ.get("SAMPLES", "32" if PREVIEW else "64"))
OUTDIR = os.environ.get(
    "OUTDIR", os.path.join(rc.OUT, "episode_preview" if PREVIEW else "episode")
)
FRAMES = os.environ.get("FRAMES", "0:1587:1")
fa, fb, fstep = (int(v) for v in FRAMES.split(":"))

T = np.load(os.path.join(DATA, "full_trajectory/executed_trajectory.npy"))
H = np.load(os.path.join(DATA, "full_trajectory/head_camera_poses.npy"))
E = np.load(os.path.join(DATA, "full_trajectory/ee_poses.npy"))
N = len(T)
blocked = np.load(
    os.path.join(DATA, "stage_01_replanning_moment/trajectories/blocked_path.npy")
)
replan = np.load(
    os.path.join(DATA, "stage_01_replanning_moment/trajectories/replan_path.npy")
)
meta = json.load(open(os.path.join(DATA, "metadata.json")))
REPLAN_FRAME = 68
CHAIR_IN = (44, 66)
GRASP_CLOSE = 1405  # ee closest to object
LIFT_START = 1272

sc = rc.reset_scene()
rc.setup_cycles(
    samples=SAMPLES,
    res=RES,
    fps=30,
    motion_blur=False,
    look="AgX - Medium High Contrast",
)
rc.world_background((0.55, 0.65, 0.85), 0.35)
col = bpy.data.collections.new("scene")
sc.collection.children.link(col)

# ------------------------------------------------------------------ scene
stage_root, stage_objs, cfg = rp.import_stage(
    os.path.join(
        rp.DATASET,
        "configs/scenes",
        os.environ.get("SCENE", "apt_2") + ".scene_instance.json",
    ),
    col,
)
objs = rp.import_objects(cfg, col)
arts = rp.import_articulated(cfg, col)
rp.cut_ceiling([o for o in bpy.data.objects if o.type == "MESH"], 2.45)
can_root, can_objs = rp.import_ycb(
    meta["object_id"], meta["object_position"], meta["object_orientation"], col
)
chair_root, chair_objs = rp.import_replica_object(
    "frl_apartment_chair_01", col, "chair_obstacle"
)
mn, mx = rp.bounds(chair_objs)
op = meta["obstacle_pose"]
chair_final = np.array([op["translation"][0], op["translation"][1], -mn.z])
chair_start = chair_final + np.array([0.0, -2.2, 0.0])
chair_root.rotation_mode = "QUATERNION"
chair_root.rotation_quaternion = (
    op["rotation"][0],
    op["rotation"][1],
    op["rotation"][2],
    op["rotation"][3],
)
for f, p in (
    (0, chair_start),
    (CHAIR_IN[0], chair_start),
    (CHAIR_IN[1], chair_final),
    (N, chair_final),
):
    chair_root.location = p
    chair_root.keyframe_insert("location", frame=f)
rc.smooth_fcurves(chair_root)

# lights: warm sun + soft sky, plus a few practicals
rc.add_sun(
    "sun",
    (math.radians(48), 0, math.radians(-35)),
    energy=2.2,
    color=(1, 0.95, 0.88),
    collection=col,
)
rc.add_sun(
    "sun2",
    (math.radians(60), 0, math.radians(140)),
    energy=0.6,
    color=(0.85, 0.9, 1.0),
    collection=col,
)

# ------------------------------------------------------------------ robot
rig = FetchRig("fetch", col)
rig.add_frustum(
    depth=2.2, hfov_deg=57, vfov_deg=45, alpha=0.07, strength=1.2, edge_radius=0.005
)
for f_, hid in ((0, False), (1100, False), (1101, True)):
    rig.frustum.hide_render = hid
    rig.frustum.keyframe_insert("hide_render", frame=f_)
for i in range(N):
    rig.set_config(T[i])
    pan, tilt = rig.head_from_camera_pose(H[i], T[i, 2])
    rig.set_head(pan, tilt)
    g = 0.05 if i < GRASP_CLOSE else 0.02
    rig.set_gripper(g)
    rig.keyframe(i)
for l in list(rig.links.values()) + [rig.root]:
    rc.linear_fcurves(l)

# object follows the gripper after the grasp
gl = rig.links["gripper_link"]
can_root.animation_data_create()
for f in range(N):
    if f < GRASP_CLOSE:
        can_root.matrix_world = rp.pose_matrix(
            meta["object_position"], meta["object_orientation"]
        )
    else:
        sc.frame_set(f)
        # keep the object rigidly attached: object pose = ee_pose(f) * inv(ee_pose(grasp)) * object_pose0
        Eg = Matrix([list(r) for r in E[GRASP_CLOSE]])
        Ef = Matrix([list(r) for r in E[f]])
        can_root.matrix_world = (
            Ef
            @ Eg.inverted()
            @ rp.pose_matrix(meta["object_position"], meta["object_orientation"])
        )
    can_root.keyframe_insert("location", frame=f)
    can_root.keyframe_insert("rotation_euler", frame=f)
rc.linear_fcurves(can_root)

# ------------------------------------------------------------------ swept-volume ghosts (velocity-aware weights)
NG, DG, GAMMA = 5, 14, 0.8
ghosts, gmats = [], []
for k in range(NG):
    m = ghost_material(f"ghost_mat_{k}", (0.2, 0.8, 1.0, 1.0), alpha=0.08, strength=0.9)
    g = FetchRig(
        f"ghost{k}", col, share=rig, material_override=m, with_camera_frame=False
    )
    ghosts.append(g)
    gmats.append(m)
qdot = (
    np.linalg.norm(np.gradient(T[:, :2], axis=0) * 20, axis=1) * 1.0
    + np.linalg.norm(np.gradient(T[:, 3:], axis=0) * 20, axis=1) * 0.35
)
qdot = gaussian_filter1d(qdot, 3)
speed_ref = np.percentile(qdot[qdot > 0.02], 85) if (qdot > 0.02).any() else 1.0


def heat_color(w):
    # 0 -> deep blue, 0.5 -> cyan, 1 -> hot orange
    w = float(max(0.0, min(1.0, w)))
    if w < 0.5:
        t = w / 0.5
        return (0.05 + 0.15 * t, 0.15 + 0.65 * t, 0.9 + 0.1 * t, 1.0)
    t = (w - 0.5) / 0.5
    return (0.2 + 0.8 * t, 0.8 - 0.35 * t, 1.0 - 0.9 * t, 1.0)


for f in range(N):
    for k, (g, m) in enumerate(zip(ghosts, gmats)):
        j = f + (k + 1) * DG
        emis = [n for n in m.node_tree.nodes if n.type == "EMISSION"][0]
        mixn = [n for n in m.node_tree.nodes if n.type == "MIX_SHADER"][0]
        if j >= N or f >= 1071:
            vis = 0.0
            g.set_config(T[min(j, N - 1)])
        else:
            g.set_config(T[j])
            vis = (GAMMA**k) * min(1.0, qdot[j] / (0.45 * speed_ref))
        w = vis
        emis.inputs["Color"].default_value = heat_color(w)
        emis.inputs["Strength"].default_value = 0.3 + 1.2 * w
        for ob in g.meshes.values():
            ob.hide_render = vis < 0.03
            ob.keyframe_insert("hide_render", frame=f)
        emis.inputs["Color"].keyframe_insert("default_value", frame=f)
        emis.inputs["Strength"].keyframe_insert("default_value", frame=f)
        g.keyframe(f, gripper=False, head=False)
for g in ghosts:
    for l in list(g.links.values()) + [g.root]:
        rc.linear_fcurves(l)


# ------------------------------------------------------------------ planned paths on the floor
def path_tube(name, xy, color, radius=0.012, z=0.012):
    me = bpy.data.meshes.new(name)
    verts = [(float(x), float(y), z) for x, y in xy]
    edges = [(i, i + 1) for i in range(len(verts) - 1)]
    me.from_pydata(verts, edges, [])
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    ob.modifiers.new("skin", "SKIN")
    for v in ob.data.skin_vertices[0].data:
        v.radius = (radius, radius)
    mat = bpy.data.materials.new(name + "_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix = nt.nodes.new("ShaderNodeMixShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = color
    em.inputs["Strength"].default_value = 6.0
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    mix.inputs["Fac"].default_value = 1.0
    ob.data.materials.append(mat)
    ob.visible_shadow = False
    return ob, mix.inputs["Fac"], em.inputs["Color"]


def sub(xy, n=60):
    idx = np.linspace(0, len(xy) - 1, n).astype(int)
    return xy[idx]


p_block, f_block, c_block = path_tube(
    "path_blocked", sub(blocked[:, :2]), (0.2, 0.85, 1.0, 1)
)
p_re, f_re, c_re = path_tube("path_replan", sub(replan[:, :2]), (0.2, 0.85, 1.0, 1))
# keyframes: initial plan cyan from frame 8; turns red at REPLAN_FRAME; fades by +30;
# replan appears at +6, fades out at 400
for f, v in (
    (0, 0.0),
    (8, 0.0),
    (20, 1.0),
    (REPLAN_FRAME + 20, 1.0),
    (REPLAN_FRAME + 50, 0.0),
):
    f_block.default_value = v
    f_block.keyframe_insert("default_value", frame=f)
for f, cval in (
    (REPLAN_FRAME - 1, (0.2, 0.85, 1.0, 1)),
    (REPLAN_FRAME + 4, (1.0, 0.15, 0.1, 1)),
):
    c_block.default_value = cval
    c_block.keyframe_insert("default_value", frame=f)
for f, v in (
    (0, 0.0),
    (REPLAN_FRAME + 6, 0.0),
    (REPLAN_FRAME + 18, 1.0),
    (380, 1.0),
    (450, 0.0),
):
    f_re.default_value = v
    f_re.keyframe_insert("default_value", frame=f)

# ------------------------------------------------------------------ camera (3 shots: wide, chase, orbit)
base = T[:, :2].copy()
yaw = np.unwrap(T[:, 2])
bs = gaussian_filter1d(base, 25, axis=0)
ys = gaussian_filter1d(yaw, 40)
SHOT_B, SHOT_C = 150, 1071
cam = rc.add_camera(
    "cam", (0, 0, 2), None, lens=30, collection=col, dof_dist=4.0, fstop=4.0
)
tgt = rc.add_target("cam_tgt", (0, 0, 0.8), col)
rc.track_to(cam, tgt)
cam_pos = np.zeros((N, 3))
tgt_pos = np.zeros((N, 3))
for f in range(N):
    if f < SHOT_B:  # wide: dolly in from the living room towards the start area
        a = f / SHOT_B
        cam_pos[f] = (
            np.array([2.7, -3.3, 3.1]) * (1 - a) + np.array([1.9, -2.5, 2.7]) * a
        )
        tgt_pos[f] = (
            np.array([-0.4, -0.6, 0.5]) * (1 - a) + np.array([0.0, -0.7, 0.6]) * a
        )
    elif f < SHOT_C:  # chase cam over the walls
        y = ys[f]
        c, s_ = math.cos(y), math.sin(y)
        off = np.array([-2.3, 0.9, 3.1])
        wo = np.array([c * off[0] - s_ * off[1], s_ * off[0] + c * off[1], off[2]])
        p = np.array([bs[f, 0], bs[f, 1], 0]) + wo
        p[0] = min(max(p[0], -1.5), 4.1)
        p[1] = min(max(p[1], -6.0), -0.15)
        cam_pos[f] = p
        tgt_pos[f] = np.array([bs[f, 0] + c * 0.6, bs[f, 1] + s_ * 0.6, 0.85])
    else:  # slow push-in from the north-east onto the grasp
        a = (f - SHOT_C) / (N - SHOT_C)
        keys_c = json.loads(
            os.environ.get(
                "CAM_C_KEYS",
                "[[0.0, [3.2, -2.2, 3.0], [3.5, -4.3, 0.9]], [0.45, [3.45, -2.8, 2.15], [3.6, -4.45, 0.85]], [1.0, [3.35, -3.15, 1.65], [3.7, -4.5, 0.8]]]",  # noqa: E501
            )
        )
        for (a0, c0, t0_), (a1, c1, t1_) in zip(keys_c[:-1], keys_c[1:]):
            if a0 <= a <= a1:
                u = (a - a0) / (a1 - a0)
                cam_pos[f] = np.array(c0) * (1 - u) + np.array(c1) * u
                tgt_pos[f] = np.array(t0_) * (1 - u) + np.array(t1_) * u
# smooth only within shots (avoid blending across cuts)
for lo, hi in ((0, SHOT_B), (SHOT_B, SHOT_C), (SHOT_C, N)):
    cam_pos[lo:hi] = gaussian_filter1d(cam_pos[lo:hi], 14, axis=0, mode="nearest")
    tgt_pos[lo:hi] = gaussian_filter1d(tgt_pos[lo:hi], 14, axis=0, mode="nearest")
for f in range(N):
    cam.location = cam_pos[f]
    cam.keyframe_insert("location", frame=f)
    tgt.location = tgt_pos[f]
    tgt.keyframe_insert("location", frame=f)
    d = float(np.linalg.norm(cam_pos[f] - tgt_pos[f]))
    cam.data.dof.focus_distance = d
    cam.data.dof.keyframe_insert("focus_distance", frame=f)
    cam.data.lens = 30.0 if f < SHOT_C else 32.0 + 8.0 * (f - SHOT_C) / (N - SHOT_C)
    cam.data.keyframe_insert("lens", frame=f)
rc.linear_fcurves(cam)
rc.linear_fcurves(tgt)

# ------------------------------------------------------------------ render
os.makedirs(OUTDIR, exist_ok=True)
if PREVIEW and os.environ.get("STILLS", "0") == "1":
    for f in [0, 60, 80, 120, 300, 600, 900, 1071, 1250, 1450, 1586]:
        if fa <= f < fb:
            t0 = time.time()
            rc.render_still(os.path.join(OUTDIR, f"f{f:05d}.png"), frame=f)
            print("frame", f, round(time.time() - t0, 1), "s")
else:
    sc.frame_start, sc.frame_end, sc.frame_step = fa, fb - 1, fstep
    sc.render.filepath = os.path.join(OUTDIR, "f")
    sc.render.use_overwrite = False
    sc.render.use_placeholder = True
    t0 = time.time()
    bpy.ops.render.render(animation=True)
    print("animation done", round(time.time() - t0, 1), "s")
