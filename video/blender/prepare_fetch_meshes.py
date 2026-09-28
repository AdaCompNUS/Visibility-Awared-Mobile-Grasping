"""Convert the Fetch link meshes (DAE/STL) into GLB files Blender 5 can import.

Blender's glTF importer rotates Y-up glTF data into Blender's Z-up frame
(v_blender = Rx(+90) v_gltf). The URDF meshes are already Z-up, so we pre-rotate
them by Rx(-90) here; the importer's rotation then restores the URDF frame.
"""
import os

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS_FETCH = os.path.join(
    ROOT,
    "..",
    ".pixi/envs/default/lib/python3.11/site-packages/mani_skill/assets/robots/fetch/fetch_description/meshes",
)
OUT = os.path.join(ROOT, "assets/fetch_glb")
os.makedirs(OUT, exist_ok=True)

PRE = trimesh.transformations.rotation_matrix(np.deg2rad(-90), [1, 0, 0])

DAE = [
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
    "estop_link",
]
STL = [
    "r_gripper_finger_link",
    "l_gripper_finger_link",
    "bellows_link",
    "l_wheel_link",
    "r_wheel_link",
    "laser_link",
]

for n in DAE:
    s = trimesh.load(os.path.join(MS_FETCH, n + ".dae"), force="scene")
    m = (
        trimesh.util.concatenate([g for g in s.dump()])
        if len(s.geometry) > 1
        else list(s.geometry.values())[0]
    )
    # keep UVs; drop material (textures are assigned in Blender by link name)
    uv = m.visual.uv if hasattr(m.visual, "uv") else None
    m2 = trimesh.Trimesh(
        vertices=m.vertices.copy(), faces=m.faces.copy(), process=False
    )
    if uv is not None:
        m2.visual = trimesh.visual.TextureVisuals(uv=uv)
    m2.apply_transform(PRE)
    m2.export(os.path.join(OUT, n + ".glb"))
    print(n, m2.vertices.shape[0], "uv" if uv is not None else "nouv")
for n in STL:
    m = trimesh.load(os.path.join(MS_FETCH, n + ".STL"))
    m.apply_transform(PRE)
    m.export(os.path.join(OUT, n + ".glb"))
    print(n, m.vertices.shape[0])
