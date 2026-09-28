"""Fetch mobile manipulator rig for Blender (bpy), built from the URDF joint tree.

Usage (inside bpy):
    rig = FetchRig("fetch")
    rig.set_config(q11)             # [x, y, yaw, torso, 7 arm joints]
    rig.set_head(pan, tilt)
    rig.keyframe(frame)
"""
import math
import os

import bpy
import numpy as np
from mathutils import Euler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLB_DIR = os.path.join(ROOT, "assets", "fetch_glb")
TEX_DIR = os.path.join(
    ROOT,
    "..",
    ".pixi/envs/default/lib/python3.11/site-packages/mani_skill/assets/robots/fetch/fetch_description/meshes",
)

# child, parent, xyz, rpy, axis, type
JOINTS = [
    (
        "torso_lift_link",
        "base_link",
        (-0.086875, 0, 0.37743),
        (0, 0, 0),
        "z",
        "prismatic",
    ),
    (
        "torso_fixed_link",
        "base_link",
        (-0.086875, 0, 0.377425),
        (0, 0, 0),
        None,
        "fixed",
    ),
    ("bellows_link", "torso_lift_link", (0, 0, 0), (0, 0, 0), None, "fixed"),
    (
        "head_pan_link",
        "torso_lift_link",
        (0.053125, 0, 0.603001),
        (0, 0, 0),
        "z",
        "revolute",
    ),
    (
        "head_tilt_link",
        "head_pan_link",
        (0.14253, 0, 0.057999),
        (0, 0, 0),
        "y",
        "revolute",
    ),
    (
        "head_camera_link",
        "head_tilt_link",
        (0.055, 0, 0.0225),
        (0, 0, 0),
        None,
        "fixed",
    ),
    (
        "head_camera_optical",
        "head_camera_link",
        (0, 0.02, 0),
        (-math.pi / 2, 0, -math.pi / 2),
        None,
        "fixed",
    ),
    (
        "shoulder_pan_link",
        "torso_lift_link",
        (0.119525, 0, 0.34858),
        (0, 0, 0),
        "z",
        "revolute",
    ),
    (
        "shoulder_lift_link",
        "shoulder_pan_link",
        (0.117, 0, 0.06),
        (0, 0, 0),
        "y",
        "revolute",
    ),
    (
        "upperarm_roll_link",
        "shoulder_lift_link",
        (0.219, 0, 0),
        (0, 0, 0),
        "x",
        "revolute",
    ),
    (
        "elbow_flex_link",
        "upperarm_roll_link",
        (0.133, 0, 0),
        (0, 0, 0),
        "y",
        "revolute",
    ),
    ("forearm_roll_link", "elbow_flex_link", (0.197, 0, 0), (0, 0, 0), "x", "revolute"),
    (
        "wrist_flex_link",
        "forearm_roll_link",
        (0.1245, 0, 0),
        (0, 0, 0),
        "y",
        "revolute",
    ),
    ("wrist_roll_link", "wrist_flex_link", (0.1385, 0, 0), (0, 0, 0), "x", "revolute"),
    ("gripper_link", "wrist_roll_link", (0.16645, 0, 0), (0, 0, 0), None, "fixed"),
    (
        "r_gripper_finger_link",
        "gripper_link",
        (0, 0.015425, 0),
        (0, 0, 0),
        "y",
        "prismatic",
    ),
    (
        "l_gripper_finger_link",
        "gripper_link",
        (0, -0.015425, 0),
        (0, 0, 0),
        "-y",
        "prismatic",
    ),
    (
        "estop_link",
        "base_link",
        (-0.12465, 0.23892, 0.31127),
        (1.5708, 0, 0),
        None,
        "fixed",
    ),
    ("laser_link", "base_link", (0.235, 0, 0.2878), (3.14159, 0, 0), None, "fixed"),
    (
        "r_wheel_link",
        "base_link",
        (0.0012914, -0.18738, 0.055325),
        (0, 0, 0),
        None,
        "fixed",
    ),
    (
        "l_wheel_link",
        "base_link",
        (0.0012914, 0.18738, 0.055325),
        (0, 0, 0),
        None,
        "fixed",
    ),
]
VISUAL_OFFSET = {
    "r_gripper_finger_link": (0, 0.101425, 0),
    "l_gripper_finger_link": (0, -0.101425, 0),
}
ARM_JOINTS = [
    "shoulder_pan_link",
    "shoulder_lift_link",
    "upperarm_roll_link",
    "elbow_flex_link",
    "forearm_roll_link",
    "wrist_flex_link",
    "wrist_roll_link",
]
TEXTURED = [
    "base_link",
    "torso_lift_link",
    "torso_fixed_link",
    "head_pan_link",
    "head_tilt_link",
    "shoulder_pan_link",
    "shoulder_lift_link",
    "upperarm_roll_link",
    "elbow_flex_link",
    "forearm_roll_link",
    "wrist_flex_link",
    "wrist_roll_link",
    "gripper_link",
]
REST_ARM = [1.32, 1.40, -0.20, 1.72, 0.0, 1.66, 0.0]


def _link_to_collection(obj, collection):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    collection.objects.link(obj)


def new_empty(name, collection, size=0.05):
    e = bpy.data.objects.new(name, None)
    e.empty_display_size = size
    collection.objects.link(e)
    return e


def texture_path(link):
    for cand in (f"{link}_uv.png", link.replace("_link", "") + "_uv.png"):
        p = os.path.join(TEX_DIR, cand)
        if os.path.exists(p):
            return p
    return None


_mat_cache = {}


def textured_material(link):
    key = f"fetch_tex_{link}"
    if key in bpy.data.materials:
        return bpy.data.materials[key]
    mat = bpy.data.materials.new(key)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.42
    bsdf.inputs["Metallic"].default_value = 0.0
    p = texture_path(link)
    if p:
        img = bpy.data.images.load(p, check_existing=True)
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.location = (-400, 0)
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (0.85, 0.85, 0.87, 1)
    return mat


def plain_material(
    name, rgba, roughness=0.5, metallic=0.0, emission=None, emission_strength=0.0
):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = emission
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat


def ghost_material(name, rgba, alpha=0.25, strength=2.0):
    """Translucent emissive 'hologram' material (works in Cycles)."""
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix = nt.nodes.new("ShaderNodeMixShader")
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    emis = nt.nodes.new("ShaderNodeEmission")
    fres = nt.nodes.new("ShaderNodeLayerWeight")
    fres.inputs["Blend"].default_value = 0.6
    emis.inputs["Color"].default_value = rgba
    emis.inputs["Strength"].default_value = strength
    # fresnel-boosted alpha: edges brighter
    inv = nt.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    nt.links.new(fres.outputs["Facing"], inv.inputs[1])
    mathn = nt.nodes.new("ShaderNodeMath")
    mathn.operation = "MULTIPLY_ADD"
    mathn.inputs[1].default_value = min(1.0, alpha * 2.5)
    mathn.inputs[2].default_value = alpha
    nt.links.new(inv.outputs[0], mathn.inputs[0])
    nt.links.new(mathn.outputs[0], mix.inputs["Fac"])
    nt.links.new(transp.outputs[0], mix.inputs[1])
    nt.links.new(emis.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    mat.blend_method = "BLEND"
    mat.use_backface_culling = False
    return mat


class FetchRig:
    def __init__(
        self,
        name="fetch",
        collection=None,
        share=None,
        material_override=None,
        with_camera_frame=True,
    ):
        self.name = name
        self.collection = collection or bpy.context.scene.collection
        self.links = {}
        self.joint_axis = {}
        self.meshes = {}
        self.root = new_empty(f"{name}.base_link", self.collection, 0.2)
        self.links["base_link"] = self.root
        for child, parent, xyz, rpy, axis, jtype in JOINTS:
            j = new_empty(f"{name}.J.{child}", self.collection)
            j.parent = self.links[parent]
            j.location = xyz
            j.rotation_euler = Euler(rpy, "XYZ")
            l = new_empty(f"{name}.{child}", self.collection)
            l.parent = j
            self.links[child] = l
            self.joint_axis[child] = (axis, jtype)
        for link in list(self.links.keys()):
            glb = os.path.join(GLB_DIR, link + ".glb")
            if not os.path.exists(glb):
                continue
            obj = self._load_mesh(link, share)
            obj.parent = self.links[link]
            obj.location = VISUAL_OFFSET.get(link, (0, 0, 0))
            obj.rotation_euler = (0, 0, 0)
            if material_override is not None:
                obj.data.materials.clear() if share is None else None
                obj.material_slots  # ensure exists
                if len(obj.data.materials) == 0:
                    obj.data.materials.append(material_override)
                for i in range(len(obj.material_slots)):
                    obj.material_slots[i].link = "OBJECT"
                    obj.material_slots[i].material = material_override
            self.meshes[link] = obj
        self.set_arm(REST_ARM)
        self.set_gripper(0.05)
        self.set_torso(0.0)

    # ---------------------------------------------------------------- meshes
    def _load_mesh(self, link, share):
        if share is not None and link in share.meshes:
            src = share.meshes[link]
            obj = bpy.data.objects.new(f"{self.name}.mesh.{link}", src.data)
            self.collection.objects.link(obj)
            return obj
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=os.path.join(GLB_DIR, link + ".glb"))
        new = [o for o in bpy.data.objects if o not in before]
        meshes = [o for o in new if o.type == "MESH"]
        for o in meshes:
            mw = o.matrix_world.copy()
            o.parent = None
            o.matrix_world = mw
        for o in new:
            if o.type != "MESH":
                bpy.data.objects.remove(o, do_unlink=True)
        if len(meshes) > 1:
            bpy.ops.object.select_all(action="DESELECT")
            for o in meshes:
                o.select_set(True)
            bpy.context.view_layer.objects.active = meshes[0]
            bpy.ops.object.join()
        obj = meshes[0]
        obj.name = f"{self.name}.mesh.{link}"
        obj.data.name = f"fetch_mesh_{link}"
        _link_to_collection(obj, self.collection)
        # smooth shading
        for p in obj.data.polygons:
            p.use_smooth = True
        obj.data.materials.clear()
        if link in TEXTURED:
            obj.data.materials.append(textured_material(link))
        elif "finger" in link:
            obj.data.materials.append(
                plain_material("fetch_finger", (0.05, 0.05, 0.05, 1), 0.45)
            )
        elif "wheel" in link:
            obj.data.materials.append(
                plain_material("fetch_wheel", (0.03, 0.03, 0.03, 1), 0.7)
            )
        elif "bellows" in link:
            obj.data.materials.append(
                plain_material("fetch_bellows", (0.04, 0.04, 0.045, 1), 0.8)
            )
        elif "laser" in link:
            obj.data.materials.append(
                plain_material("fetch_laser", (0.02, 0.02, 0.02, 1), 0.35)
            )
        else:
            obj.data.materials.append(
                plain_material("fetch_estop", (0.7, 0.05, 0.05, 1), 0.4)
            )
        return obj

    # ---------------------------------------------------------------- posing
    def set_joint(self, link, value):
        axis, jtype = self.joint_axis[link]
        l = self.links[link]
        sign = -1.0 if axis.startswith("-") else 1.0
        ax = axis[-1]
        if jtype == "revolute":
            e = [0.0, 0.0, 0.0]
            e["xyz".index(ax)] = sign * value
            l.rotation_euler = Euler(e, "XYZ")
        elif jtype == "prismatic":
            v = [0.0, 0.0, 0.0]
            v["xyz".index(ax)] = sign * value
            l.location = v

    def set_base(self, x, y, yaw):
        self.root.location = (x, y, 0.0)
        self.root.rotation_euler = (0.0, 0.0, yaw)

    def set_torso(self, h):
        self.set_joint("torso_lift_link", float(h))

    def set_arm(self, q7):
        for link, v in zip(ARM_JOINTS, q7):
            self.set_joint(link, float(v))

    def set_head(self, pan, tilt):
        self.set_joint("head_pan_link", float(pan))
        self.set_joint("head_tilt_link", float(tilt))

    def set_gripper(self, opening_each=0.05):
        self.set_joint("r_gripper_finger_link", float(opening_each))
        self.set_joint("l_gripper_finger_link", float(opening_each))

    def set_config(self, q11):
        q = [float(v) for v in q11]
        self.set_base(q[0], q[1], q[2])
        self.set_torso(q[3])
        self.set_arm(q[4:11])

    def head_from_camera_pose(self, T_cam_world, yaw):
        """Derive (pan, tilt) from an OpenCV-convention camera world pose (z forward)."""
        f = np.asarray(T_cam_world)[:3, 2]
        c, s = math.cos(-yaw), math.sin(-yaw)
        fb = np.array([c * f[0] - s * f[1], s * f[0] + c * f[1], f[2]])
        pan = math.atan2(fb[1], fb[0])
        tilt = math.atan2(-fb[2], math.hypot(fb[0], fb[1]))
        return pan, tilt

    def keyframe(self, frame, gripper=True, head=True):
        self.root.keyframe_insert("location", frame=frame)
        self.root.keyframe_insert("rotation_euler", frame=frame)
        self.links["torso_lift_link"].keyframe_insert("location", frame=frame)
        for link in ARM_JOINTS:
            self.links[link].keyframe_insert("rotation_euler", frame=frame)
        if head:
            self.links["head_pan_link"].keyframe_insert("rotation_euler", frame=frame)
            self.links["head_tilt_link"].keyframe_insert("rotation_euler", frame=frame)
        if gripper:
            self.links["r_gripper_finger_link"].keyframe_insert("location", frame=frame)
            self.links["l_gripper_finger_link"].keyframe_insert("location", frame=frame)

    def world_matrix(self, link):
        bpy.context.view_layer.update()
        return self.links[link].matrix_world.copy()

    # ---------------------------------------------------------------- extras
    def add_frustum(
        self,
        depth=2.0,
        hfov_deg=57.0,
        vfov_deg=45.0,
        color=(0.1, 0.8, 1.0, 1.0),
        alpha=0.12,
        strength=3.0,
        edge_radius=0.006,
    ):
        """Camera view frustum attached to the head camera optical frame."""
        tx = math.tan(math.radians(hfov_deg / 2)) * depth
        ty = math.tan(math.radians(vfov_deg / 2)) * depth
        verts = [
            (0, 0, 0),
            (-tx, -ty, depth),
            (tx, -ty, depth),
            (tx, ty, depth),
            (-tx, ty, depth),
        ]
        faces = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1)]
        me = bpy.data.meshes.new(f"{self.name}_frustum")
        me.from_pydata(verts, [], faces)
        ob = bpy.data.objects.new(f"{self.name}.frustum", me)
        self.collection.objects.link(ob)
        ob.parent = self.links["head_camera_optical"]
        ob.data.materials.append(
            ghost_material(
                f"{self.name}_frustum_mat", color, alpha=alpha, strength=strength
            )
        )
        ob.visible_shadow = False
        # edges
        me2 = bpy.data.meshes.new(f"{self.name}_frustum_edges")
        me2.from_pydata(
            verts, [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)], []
        )
        ob2 = bpy.data.objects.new(f"{self.name}.frustum_edges", me2)
        self.collection.objects.link(ob2)
        ob2.parent = self.links["head_camera_optical"]
        ob2.modifiers.new("skin", "SKIN")
        for v in ob2.data.skin_vertices[0].data:
            v.radius = (edge_radius, edge_radius)
        ob2.data.materials.append(
            plain_material(
                f"{self.name}_frustum_edge_mat", color, 0.4, 0.0, color, strength * 3
            )
        )
        ob2.visible_shadow = False
        self.frustum = ob
        self.frustum_edges = ob2
        return ob, ob2

    def hide(self, hide=True):
        for o in self.meshes.values():
            o.hide_render = hide
            o.hide_viewport = hide
