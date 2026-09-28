"""Import a ReplicaCAD scene (as ManiSkill builds it) into Blender with the same world frame."""
import json
import math
import os
import xml.etree.ElementTree as ET

import bpy
from mathutils import Matrix, Quaternion, Vector

DATASET = os.path.expanduser("~/.maniskill/data/scene_datasets/replica_cad_dataset")
YCB = os.path.expanduser("~/.maniskill/data/assets/mani_skill2_ycb/models")

# ManiSkill rotates every ReplicaCAD asset by +90 deg about X (Y-up -> Z-up).
RQ = Matrix.Rotation(math.radians(90), 4, "X")
RQ_INV = RQ.inverted()

SKIP_SUBSTR = ()


def pose_matrix(pos, quat_wxyz):
    q = Quaternion((quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]))
    m = q.to_matrix().to_4x4()
    m.translation = Vector(pos)
    return m


def _import_glb(path, collection, name):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    root = bpy.data.objects.new(name, None)
    collection.objects.link(root)
    for o in new:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        collection.objects.link(o)
        if o.parent is None:
            mw = o.matrix_world.copy()
            o.parent = root
            o.matrix_parent_inverse = Matrix.Identity(4)
            o.matrix_world = mw
        if o.type == "MESH":
            for p in o.data.polygons:
                p.use_smooth = True
    return root, new


def import_stage(scene_json, collection, hide_names=()):
    cfg = json.load(open(scene_json))
    stage = os.path.basename(cfg["stage_instance"]["template_name"])
    root, objs = _import_glb(
        os.path.join(DATASET, "stages", stage + ".glb"), collection, "stage"
    )
    root.matrix_world = Matrix.Identity(4)  # P = RQ ; blender = P * RQ^-1 = I
    for o in objs:
        if any(h.lower() in o.name.lower() for h in hide_names):
            o.hide_render = True
            o.hide_viewport = True
    return root, objs, cfg


def import_objects(cfg, collection, skip_substr=SKIP_SUBSTR):
    roots = []
    for i, meta in enumerate(cfg["object_instances"]):
        tname = os.path.basename(meta["template_name"])
        if any(s in tname for s in skip_substr):
            continue
        ocfg = json.load(
            open(
                os.path.join(
                    DATASET, "configs", "objects", tname + ".object_config.json"
                )
            )
        )
        glb = os.path.normpath(
            os.path.join(DATASET, "configs", "objects", ocfg["render_asset"])
        )
        root, _ = _import_glb(glb, collection, f"obj.{tname}.{i}")
        P = RQ @ pose_matrix(meta["translation"], meta["rotation"])
        root.matrix_world = P @ RQ_INV
        roots.append(root)
    return roots


def _urdf_visuals(urdf_path):
    """Return list of (mesh_path, world_from_mesh 4x4 in the URDF root frame) at zero joint values."""
    tree = ET.parse(urdf_path)
    r = tree.getroot()
    d = os.path.dirname(urdf_path)
    parent_of = {}
    joint_T = {}
    for j in r.findall("joint"):
        c = j.find("child").get("link")
        p = j.find("parent").get("link")
        o = j.find("origin")
        xyz = [
            float(v)
            for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()
        ]
        rpy = [
            float(v)
            for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()
        ]
        T = (
            Matrix.Translation(xyz)
            @ Matrix.Rotation(rpy[2], 4, "Z")
            @ Matrix.Rotation(rpy[1], 4, "Y")
            @ Matrix.Rotation(rpy[0], 4, "X")
        )
        parent_of[c] = p
        joint_T[c] = T

    def link_T(name):
        T = Matrix.Identity(4)
        while name in parent_of:
            T = joint_T[name] @ T
            name = parent_of[name]
        return T

    out = []
    for l in r.findall("link"):
        LT = link_T(l.get("name"))
        for v in l.findall("visual"):
            g = v.find("geometry")
            m = g.find("mesh") if g is not None else None
            if m is None:
                continue
            o = v.find("origin")
            xyz = [
                float(x)
                for x in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()
            ]
            rpy = [
                float(x)
                for x in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()
            ]
            VT = (
                Matrix.Translation(xyz)
                @ Matrix.Rotation(rpy[2], 4, "Z")
                @ Matrix.Rotation(rpy[1], 4, "Y")
                @ Matrix.Rotation(rpy[0], 4, "X")
            )
            sc = m.get("scale")
            S = Matrix.Identity(4)
            if sc:
                s = [float(x) for x in sc.split()]
                S = Matrix.Diagonal((s[0], s[1], s[2], 1.0))
            out.append(
                (os.path.normpath(os.path.join(d, m.get("filename"))), LT @ VT @ S)
            )
    return out


def import_articulated(cfg, collection):
    roots = []
    for i, meta in enumerate(cfg.get("articulated_object_instances", [])):
        t = meta["template_name"]
        if "door" in t:
            continue
        urdf = os.path.join(DATASET, "urdf", t, t + ".urdf")
        if not os.path.exists(urdf):
            print("missing urdf", urdf)
            continue
        base = RQ @ pose_matrix(meta["translation"], meta["rotation"])
        for k, (mesh, T) in enumerate(_urdf_visuals(urdf)):
            if not os.path.exists(mesh):
                print("missing mesh", mesh)
                continue
            root, _ = _import_glb(mesh, collection, f"art.{t}.{i}.{k}")
            root.matrix_world = base @ T @ RQ_INV
            roots.append(root)
    return roots


def import_ycb(model_id, pos, quat_wxyz, collection, name=None):
    obj_path = os.path.join(YCB, model_id, "textured.obj")
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=obj_path, forward_axis="Y", up_axis="Z")
    new = [o for o in bpy.data.objects if o not in before]
    root = bpy.data.objects.new(name or f"ycb.{model_id}", None)
    collection.objects.link(root)
    for o in new:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        collection.objects.link(o)
        o.parent = root
        o.matrix_parent_inverse = Matrix.Identity(4)
        for p in o.data.polygons:
            p.use_smooth = True
    root.matrix_world = pose_matrix(pos, quat_wxyz)
    return root, new


def import_replica_object(template, collection, name):
    """A single ReplicaCAD object (e.g. a chair) placed by a SAPIEN world pose later."""
    glb = os.path.join(DATASET, "objects", template + ".glb")
    root, objs = _import_glb(glb, collection, name)
    return root, objs


def bounds(objs):
    mn = Vector((1e9, 1e9, 1e9))
    mx = Vector((-1e9, -1e9, -1e9))
    bpy.context.view_layer.update()
    for o in objs:
        if o.type != "MESH":
            continue
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            mn = Vector((min(mn.x, w.x), min(mn.y, w.y), min(mn.z, w.z)))
            mx = Vector((max(mx.x, w.x), max(mx.y, w.y), max(mx.z, w.z)))
    return mn, mx


def cut_ceiling(objs, z_cut=2.6):
    """Delete faces lying above z_cut (ceiling / upper walls) so cameras can look in from above."""
    import bmesh

    removed = 0
    for o in objs:
        if o.type != "MESH":
            continue
        bm = bmesh.new()
        bm.from_mesh(o.data)
        mw = o.matrix_world
        kill = [f for f in bm.faces if all((mw @ v.co).z > z_cut for v in f.verts)]
        if kill:
            bmesh.ops.delete(bm, geom=kill, context="FACES")
            removed += len(kill)
            bm.to_mesh(o.data)
        bm.free()
    return removed
