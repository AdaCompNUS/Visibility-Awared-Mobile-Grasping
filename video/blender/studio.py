"""Studio (dark-background) Blender shots used by the motion-graphics sections.

SHOT=hero      : slow turntable of Fetch with a glowing camera frustum (title/outro)
SHOT=swept     : robot drives along a curve; velocity-weighted swept-volume ghosts +
                 the frustum sweeping over them (collision-visibility constraint)
SHOT=objvis    : robot at a pre-grasp pose in front of a table; frustum on the target;
                 arm swings into a self-occluding pose and the frustum turns red
SHOT=subgoals  : three candidate whole-body configurations appear one by one
                 (grasp in place / pre-grasp / observation pose)
Env: PREVIEW=1 (640x360, few samples), FRAMES=a:b:step, OUTDIR
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
from fetch_rig import FetchRig, ghost_material, plain_material
from mathutils import Vector

SHOT = os.environ.get("SHOT", "hero")
PREVIEW = os.environ.get("PREVIEW", "0") == "1"
RES = tuple(
    int(v)
    for v in os.environ.get("RES", "640x360" if PREVIEW else "1920x1080").split("x")
)
SAMPLES = int(os.environ.get("SAMPLES", "16" if PREVIEW else "48"))
OUTDIR = os.environ.get(
    "OUTDIR", os.path.join(rc.OUT, f"studio_{SHOT}" + ("_preview" if PREVIEW else ""))
)
FPS = 30

sc = rc.reset_scene()
rc.setup_cycles(samples=SAMPLES, res=RES, fps=FPS, transparent=False)
rc.world_gradient(top=(0.012, 0.016, 0.028), bottom=(0.0, 0.0, 0.0), strength=1.0)
col = bpy.data.collections.new("studio")
sc.collection.children.link(col)
floor = rc.add_floor(
    size=60, rgba=(0.012, 0.013, 0.017, 1), roughness=0.4, collection=col
)

REST = [1.32, 1.40, -0.20, 1.72, 0.0, 1.66, 0.0]
PREPOSE_ARM = [1.29, -0.811, -1.127, 1.524, -0.495, 0.892, 1.554]
GRASP_ARM = [-0.214, -0.276, 0.752, 0.694, -0.69, 1.384, 0.758]
CYAN = (0.2, 0.8, 1.0, 1.0)
AMBER = (1.0, 0.62, 0.15, 1.0)
RED = (1.0, 0.18, 0.12, 1.0)
MINT = (0.35, 0.95, 0.65, 1.0)


def lights(target, key_energy=900, rim_energy=800, fill_energy=180):
    rc.add_area_light(
        "key",
        (2.8, -2.6, 3.2),
        target,
        size=2.5,
        energy=key_energy,
        color=(1.0, 0.95, 0.9),
        collection=col,
    )
    rc.add_area_light(
        "rim",
        (-3.0, 2.0, 2.6),
        target,
        size=1.5,
        energy=rim_energy,
        color=(0.5, 0.78, 1.0),
        collection=col,
    )
    rc.add_area_light(
        "fill",
        (0.5, 3.5, 1.6),
        target,
        size=3.5,
        energy=fill_energy,
        color=(0.8, 0.85, 1.0),
        collection=col,
    )


def heat_color(w):
    w = float(max(0.0, min(1.0, w)))
    if w < 0.5:
        t = w / 0.5
        return (0.05 + 0.15 * t, 0.15 + 0.65 * t, 0.9 + 0.1 * t, 1.0)
    t = (w - 0.5) / 0.5
    return (0.2 + 0.75 * t, 0.8 - 0.45 * t, 1.0 - 0.95 * t, 1.0)


def key_visibility(objs, frame, hidden):
    for ob in objs:
        ob.hide_render = hidden
        ob.keyframe_insert("hide_render", frame=frame)


def emissive_ring(name, center, radius=0.6, color=CYAN, strength=6.0, width=0.012):
    bpy.ops.mesh.primitive_torus_add(
        location=center,
        major_radius=radius,
        minor_radius=width,
        major_segments=96,
        minor_segments=8,
    )
    ob = bpy.context.object
    ob.name = name
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    col.objects.link(ob)
    ob.data.materials.append(
        plain_material(name + "_m", (0, 0, 0, 1), 0.5, 0.0, color, strength)
    )
    ob.visible_shadow = False
    return ob


def table(center, size=(1.2, 0.7, 0.75)):
    bpy.ops.mesh.primitive_cube_add(location=(center[0], center[1], size[2] - 0.02))
    top = bpy.context.object
    top.scale = (size[0] / 2, size[1] / 2, 0.02)
    top.name = "table_top"
    for c in list(top.users_collection):
        c.objects.unlink(top)
    col.objects.link(top)
    top.data.materials.append(plain_material("table_m", (0.16, 0.12, 0.09, 1), 0.35))
    legs = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            bpy.ops.mesh.primitive_cylinder_add(
                location=(
                    center[0] + sx * (size[0] / 2 - 0.06),
                    center[1] + sy * (size[1] / 2 - 0.06),
                    (size[2] - 0.04) / 2,
                ),
                radius=0.02,
                depth=size[2] - 0.04,
            )
            l = bpy.context.object
            for c in list(l.users_collection):
                c.objects.unlink(l)
            col.objects.link(l)
            l.data.materials.append(
                plain_material("leg_m", (0.05, 0.05, 0.06, 1), 0.4, 0.6)
            )
            legs.append(l)
    return top, legs


N = 0
# ----------------------------------------------------------------------------- HERO
if SHOT == "hero":
    N = int(os.environ.get("N", "600"))
    rig = FetchRig("fetch", col)
    rig.set_torso(0.18)
    rig.set_head(0.15, 0.32)
    rig.set_arm([1.05, 0.9, -0.6, 1.5, 0.3, 1.2, 0.2])
    rig.add_frustum(depth=1.4, alpha=0.035, strength=1.0, edge_radius=0.004)
    ring = emissive_ring("ring", (0, 0, 0.004), radius=0.55, strength=4.0)
    tgt = rc.add_target("t", (0.05, 0, 0.78), col)
    cam = rc.add_camera(
        "cam", (3.0, -2.6, 1.35), tgt, lens=42, collection=col, dof_dist=3.9, fstop=3.2
    )
    lights(tgt)
    for f in range(N + 1):
        a = 2 * math.pi * f / N
        rig.root.rotation_euler = (0, 0, a * 0.35)  # slow turntable
        rig.root.keyframe_insert("rotation_euler", frame=f)
        # slow head scan
        rig.set_head(0.35 * math.sin(a * 2), 0.32 + 0.1 * math.sin(a * 3 + 1))
        rig.keyframe(f)
        cam.location = (
            3.0 + 0.25 * math.sin(a),
            -2.6 + 0.2 * math.cos(a),
            1.35 + 0.08 * math.sin(a * 2),
        )
        cam.keyframe_insert("location", frame=f)
    for l in list(rig.links.values()) + [rig.root]:
        rc.smooth_fcurves(l)

# ----------------------------------------------------------------------------- SWEPT VOLUME
elif SHOT == "swept":
    N = int(os.environ.get("N", "330"))
    # S-curve path with a speed profile: slow, accelerate through the bend, slow near the end
    M = 400
    s = np.linspace(0, 1, M)
    xy = np.stack(
        [-2.4 + 4.8 * s, 0.9 * np.sin(s * math.pi * 1.4) * (1 - 0.3 * s)], axis=1
    )
    yaw = np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0]))
    # timing: cumulative "time" so that speed peaks mid-path
    speed = 0.35 + 1.0 * np.exp(-(((s - 0.55) / 0.22) ** 2))
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = seg / speed[1:]
    tcum = np.concatenate([[0], np.cumsum(dt)])
    tcum /= tcum[-1]
    # robot travels the path over frames [30, N-40]
    f0, f1 = 30, N - 40

    def state_at(f):
        u = (f - f0) / (f1 - f0)
        u = min(max(u, 0.0), 1.0)
        i = np.searchsorted(tcum, u)
        i = min(max(i, 1), M - 1)
        w = (u - tcum[i - 1]) / max(1e-9, tcum[i] - tcum[i - 1])
        p = xy[i - 1] * (1 - w) + xy[i] * w
        y = yaw[i - 1] * (1 - w) + yaw[i] * w
        return p, y, speed[i]

    rig = FetchRig("fetch", col)
    rig.set_torso(0.1)
    rig.set_arm(REST)
    rig.add_frustum(depth=2.0, alpha=0.04, strength=1.0, edge_radius=0.004)
    NG = 9
    ghosts = []
    for k in range(NG):
        m = ghost_material(f"gm{k}", CYAN, alpha=0.13, strength=1.3)
        g = FetchRig(f"g{k}", col, share=rig, material_override=m)
        g.set_torso(0.1)
        g.set_arm(REST)
        ghosts.append((g, m))
    # path tube on the floor
    verts = [(float(x), float(y), 0.01) for x, y in xy[::4]]
    me = bpy.data.meshes.new("path")
    me.from_pydata(verts, [(i, i + 1) for i in range(len(verts) - 1)], [])
    pob = bpy.data.objects.new("path", me)
    col.objects.link(pob)
    mod = pob.modifiers.new("skin", "SKIN")
    for v in pob.data.skin_vertices[0].data:
        v.radius = (0.01, 0.01)
    pob.data.materials.append(
        plain_material("path_m", (0, 0, 0, 1), 0.5, 0.0, CYAN, 5.0)
    )
    pob.visible_shadow = False
    tgt = rc.add_target("t", (0.0, 0.0, 0.6), col)
    cam = rc.add_camera(
        "cam", (0.3, -4.6, 2.6), tgt, lens=40, collection=col, dof_dist=5.4, fstop=5.6
    )
    lights(tgt, key_energy=1200, rim_energy=900, fill_energy=250)
    LOOK = 22  # frames ahead per ghost
    GAMMA = 0.82
    vmax = speed.max()
    for f in range(N + 1):
        p, y, v = state_at(f)
        rig.set_base(p[0], p[1], y)
        # gaze: look toward the fastest upcoming ghost
        best_k, best_w = 0, -1
        for k in range(NG):
            pk, yk, vk = state_at(f + (k + 1) * LOOK)
            w = (GAMMA**k) * (vk / vmax)
            if w > best_w:
                best_w, best_k = w, k
        pk, yk, vk = state_at(f + (best_k + 1) * LOOK)
        d = np.array([pk[0] - p[0], pk[1] - p[1]])
        c, s_ = math.cos(-y), math.sin(-y)
        db = np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1]])
        pan = math.atan2(db[1], db[0]) if np.linalg.norm(db) > 0.05 else 0.0
        pan = max(-1.3, min(1.3, pan))
        tilt = 0.55 if np.linalg.norm(db) < 1.2 else 0.35
        rig.set_head(pan, tilt)
        rig.keyframe(f)
        for k, (g, m) in enumerate(ghosts):
            pk, yk, vk = state_at(f + (k + 1) * LOOK)
            g.set_base(pk[0], pk[1], yk)
            w = (GAMMA**k) * (vk / vmax)
            emis = [n for n in m.node_tree.nodes if n.type == "EMISSION"][0]
            emis.inputs["Color"].default_value = heat_color(w)
            emis.inputs["Strength"].default_value = 0.3 + 1.4 * w
            emis.inputs["Color"].keyframe_insert("default_value", frame=f)
            emis.inputs["Strength"].keyframe_insert("default_value", frame=f)
            hidden = (f < 12) or (f + (k + 1) * LOOK > f1 + 5)
            for ob in g.meshes.values():
                ob.hide_render = hidden
                ob.keyframe_insert("hide_render", frame=f)
            g.keyframe(f, gripper=False, head=False)
        cam.location = (0.3 + 1.4 * (f / N), -4.6, 2.6 - 0.3 * (f / N))
        cam.keyframe_insert("location", frame=f)
    for g, _ in ghosts:
        for l in list(g.links.values()) + [g.root]:
            rc.linear_fcurves(l)
    for l in list(rig.links.values()) + [rig.root]:
        rc.smooth_fcurves(l)

# ----------------------------------------------------------------------------- OBJECT VISIBILITY
elif SHOT == "objvis":
    N = int(os.environ.get("N", "320"))
    top, legs = table((1.05, 0.0), size=(1.0, 0.7, 0.74))
    meta_obj = "005_tomato_soup_can"
    can_root, can_objs = rp.import_ycb(
        meta_obj, (0.95, 0.05, 0.74), (1, 0, 0, 0), col, "can"
    )
    ring = emissive_ring(
        "ring", (0.95, 0.05, 0.745), radius=0.12, strength=6.0, width=0.006
    )
    rig = FetchRig("fetch", col)
    rig.set_torso(0.25)
    rig.set_base(0.0, 0.0, 0.0)
    fr, fe = rig.add_frustum(
        depth=1.35,
        hfov_deg=57,
        vfov_deg=45,
        alpha=0.045,
        strength=1.1,
        edge_radius=0.004,
    )
    fr_mat = fr.data.materials[0]
    fe_mat = fe.data.materials[0]
    fr_em = [n for n in fr_mat.node_tree.nodes if n.type == "EMISSION"][0]
    fe_b = fe_mat.node_tree.nodes["Principled BSDF"]
    tgt = rc.add_target("t", (0.55, 0.0, 0.8), col)
    cam = rc.add_camera(
        "cam", (2.3, -3.1, 1.7), tgt, lens=42, collection=col, dof_dist=3.9, fstop=4.0
    )
    lights(tgt)
    # arm motion from VAMP (collision-checked against the table + can, see plan_studio.py)
    plan = json.load(open(os.path.join(rc.OUT, "studio_plan.json")))["objvis"]
    rig.set_torso(plan["torso"])
    p1, p2 = plan["path1"], plan["path2"]
    o1, o2 = plan["occl_along_path1"], plan["occl_along_path2"]
    HOLD0, HOLD1, HOLD2 = (
        75,
        60,
        60,
    )  # frames: hold visible, hold occluded, hold visible again
    D1, D2 = 55, 70  # frames for each motion

    def path_at(path, occl, u):
        i = min(len(path) - 1, max(0, int(round(u * (len(path) - 1)))))
        return path[i], occl[i]

    timing = []
    for f in range(N + 1):
        if f < HOLD0:
            q, occ = p1[0], o1[0]
        elif f < HOLD0 + D1:
            q, occ = path_at(p1, o1, (f - HOLD0) / D1)
        elif f < HOLD0 + D1 + HOLD1:
            q, occ = p1[-1], o1[-1]
        elif f < HOLD0 + D1 + HOLD1 + D2:
            q, occ = path_at(p2, o2, (f - HOLD0 - D1 - HOLD1) / D2)
        else:
            q, occ = p2[-1], o2[-1]
        rig.set_torso(q[0])
        rig.set_arm(q[1:8])
        rig.set_head(0.05, 0.42)
        rig.keyframe(f)
        occf = 1.0 if occ else 0.0
        colr = tuple(CYAN[i] * (1 - occf) + RED[i] * occf for i in range(4))
        fr_em.inputs["Color"].default_value = colr
        fr_em.inputs["Color"].keyframe_insert("default_value", frame=f)
        fe_b.inputs["Emission Color"].default_value = colr
        fe_b.inputs["Emission Color"].keyframe_insert("default_value", frame=f)
        cam.location = (2.3 - 0.5 * f / N, -3.1 + 0.3 * f / N, 1.7 - 0.1 * f / N)
        cam.keyframe_insert("location", frame=f)
        timing.append(bool(occ))
    json.dump(
        {"occluded": timing, "fps": FPS},
        open(os.path.join(rc.OUT, "studio_objvis_timing.json"), "w"),
    )
    N_needed = HOLD0 + D1 + HOLD1 + D2 + HOLD2
    print("objvis frames needed", N_needed, "N", N)
    for l in list(rig.links.values()) + [rig.root]:
        rc.smooth_fcurves(l)

# ----------------------------------------------------------------------------- SUBGOALS
elif SHOT == "subgoals":
    N = int(os.environ.get("N", "360"))
    top, legs = table((1.6, 0.0), size=(1.1, 0.8, 0.74))
    can_root, can_objs = rp.import_ycb(
        "002_master_chef_can", (1.45, 0.1, 0.74), (1, 0, 0, 0), col, "can"
    )
    ring = emissive_ring(
        "ring", (1.45, 0.1, 0.745), radius=0.14, strength=6.0, width=0.006
    )
    rig = FetchRig("fetch", col)
    rig.set_torso(0.1)
    rig.set_arm(REST)
    rig.set_base(-0.9, -0.9, 0.5)
    rig.set_head(0.35, 0.35)
    # three candidates as ghosts (base pose + configuration validated by VAMP, see plan_studio.py)
    plan = json.load(open(os.path.join(rc.OUT, "studio_plan.json")))["subgoals"]
    colors = {"grasp": MINT, "pregrasp": CYAN, "observe": AMBER}
    cands = []
    for name in ("grasp", "pregrasp", "observe"):
        c = plan[name]
        cands.append((name, tuple(c["base"]), c["q"][0], c["q"][1:8], colors[name]))
    ghosts = []
    for name, (x, y, yaw), torso, arm, color in cands:
        m = ghost_material(f"gm_{name}", color, alpha=0.16, strength=1.6)
        g = FetchRig(f"g_{name}", col, share=rig, material_override=m)
        g.set_base(x, y, yaw)
        g.set_torso(torso)
        g.set_arm(arm)
        g.set_head(0.0, 0.4)
        g.add_frustum(
            depth=0.9, alpha=0.03, strength=0.8, edge_radius=0.003, color=color
        )
        ghosts.append((g, m))
    tgt = rc.add_target("t", (0.6, 0.0, 0.6), col)
    cam = rc.add_camera(
        "cam", (0.4, -4.4, 3.6), tgt, lens=38, collection=col, dof_dist=5.6, fstop=5.6
    )
    lights(tgt, key_energy=1100, rim_energy=900, fill_energy=220)
    appear = [60, 150, 240]
    for f in range(N + 1):
        for k, (g, m) in enumerate(ghosts):
            hidden = f < appear[k]
            for ob in list(g.meshes.values()) + [g.frustum, g.frustum_edges]:
                ob.hide_render = hidden
                ob.keyframe_insert("hide_render", frame=f)
            emis = [n for n in m.node_tree.nodes if n.type == "EMISSION"][0]
            u = min(1.0, max(0.0, (f - appear[k]) / 20.0))
            emis.inputs["Strength"].default_value = 1.6 * (0.2 + 0.8 * u) + (
                1.5 if u < 1 else 0.0
            ) * (1 - u)
            emis.inputs["Strength"].keyframe_insert("default_value", frame=f)
        rig.keyframe(f)
    # export screen-space anchors of the candidates for 2D labels
    from bpy_extras.object_utils import world_to_camera_view

    bpy.context.view_layer.update()
    anchors = {}
    for (name, (x, y, yaw), torso, arm, color), (g, m) in zip(cands, ghosts):
        head = g.world_matrix("head_tilt_link").translation
        v = world_to_camera_view(sc, cam, head)
        anchors[name] = [float(v.x), float(1 - v.y)]
    v = world_to_camera_view(sc, cam, Vector((1.45, 0.1, 0.8)))
    anchors["target"] = [float(v.x), float(1 - v.y)]
    v = world_to_camera_view(sc, cam, rig.world_matrix("head_tilt_link").translation)
    anchors["robot"] = [float(v.x), float(1 - v.y)]
    json.dump(
        anchors,
        open(os.path.join(rc.OUT, "studio_subgoals_anchors.json"), "w"),
        indent=1,
    )
    print("anchors", anchors)
else:
    raise SystemExit(f"unknown SHOT {SHOT}")

# ----------------------------------------------------------------------------- render
os.makedirs(OUTDIR, exist_ok=True)
fa, fb, fstep = (int(v) for v in os.environ.get("FRAMES", f"0:{N + 1}:1").split(":"))
sc.frame_start, sc.frame_end, sc.frame_step = fa, fb - 1, fstep
sc.render.filepath = os.path.join(OUTDIR, "f")
sc.render.use_overwrite = False
sc.render.use_placeholder = True
t0 = time.time()
bpy.ops.render.render(animation=True)
print("animation done", round(time.time() - t0, 1), "s", SHOT)
