"""Head-camera view of the recorded episode: RGB + z-depth + robot mask per frame.

Same scene as episode.py (apartment, chair obstacle, can, robot) but the camera is
placed at the recorded head-camera pose (OpenCV convention) for every frame.
Outputs:  out/headcam/rgb_NNNN.png, depth_NNNN.exr (Z pass), index_NNNN.exr (robot=1)
Env: FRAMES=a:b:step, FOVY_DEG (default 75), RES (default 640x480), SAMPLES
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
from fetch_rig import FetchRig
from mathutils import Matrix

DATA = os.path.join(rc.ROOT, "assets/dropbox/replanning_data/replanning_data")
OUTDIR = os.environ.get("OUTDIR", os.path.join(rc.OUT, "headcam"))
RES = tuple(int(v) for v in os.environ.get("RES", "640x480").split("x"))
SAMPLES = int(os.environ.get("SAMPLES", "24"))
FOVY = math.radians(float(os.environ.get("FOVY_DEG", "75")))
fa, fb, fstep = (int(v) for v in os.environ.get("FRAMES", "0:1587:1").split(":"))

T = np.load(os.path.join(DATA, "full_trajectory/executed_trajectory.npy"))
H = np.load(os.path.join(DATA, "full_trajectory/head_camera_poses.npy"))
E = np.load(os.path.join(DATA, "full_trajectory/ee_poses.npy"))
N = len(T)
meta = json.load(open(os.path.join(DATA, "metadata.json")))
CHAIR_IN = (44, 66)
GRASP_CLOSE = 1405

sc = rc.reset_scene()
rc.setup_cycles(samples=SAMPLES, res=RES, fps=30)
sc.cycles.use_denoising = True
rc.world_background((0.55, 0.65, 0.85), 0.35)
col = bpy.data.collections.new("scene")
sc.collection.children.link(col)

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

rig = FetchRig("fetch", col)
for ob in rig.meshes.values():
    ob.pass_index = 1
for i in range(N):
    rig.set_config(T[i])
    pan, tilt = rig.head_from_camera_pose(H[i], T[i, 2])
    rig.set_head(pan, tilt)
    rig.set_gripper(0.05 if i < GRASP_CLOSE else 0.02)
    rig.keyframe(i)
for l in list(rig.links.values()) + [rig.root]:
    rc.linear_fcurves(l)
for f in range(N):
    if f < GRASP_CLOSE:
        can_root.matrix_world = rp.pose_matrix(
            meta["object_position"], meta["object_orientation"]
        )
    else:
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

# camera at the recorded head pose (OpenCV -> Blender: flip y and z axes)
cam = rc.add_camera("headcam", (0, 0, 1), None, lens=35, collection=col)
cam.data.sensor_fit = "VERTICAL"
cam.data.sensor_height = 24.0
cam.data.lens = (cam.data.sensor_height / 2) / math.tan(FOVY / 2)
cam.data.clip_start = 0.05
FLIP = Matrix.Diagonal((1, -1, -1, 1))
for f in range(N):
    M = Matrix([list(r) for r in H[f]]) @ FLIP
    cam.matrix_world = M
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_euler", frame=f)
rc.linear_fcurves(cam)

# passes + compositor file outputs (Blender 5 node-group compositor)
vl = sc.view_layers[0]
vl.use_pass_z = True
vl.use_pass_object_index = True
ng = bpy.data.node_groups.new("headcam_comp", "CompositorNodeTree")
sc.compositing_node_group = ng
rl = ng.nodes.new("CompositorNodeRLayers")
bpy.context.view_layer.update()


def out_sock(node, *names):
    for o in node.outputs:
        if o.name in names or o.identifier in names:
            return o
    raise KeyError(names)


os.makedirs(OUTDIR, exist_ok=True)
fo = ng.nodes.new("CompositorNodeOutputFile")
fo.directory = OUTDIR
fo.file_name = "aux_"
fo.format.color_depth = "32"
fo.file_output_items.new("FLOAT", "depth")
fo.file_output_items.new("FLOAT", "index")
ng.links.new(out_sock(rl, "Depth"), fo.inputs["depth"])
ng.links.new(out_sock(rl, "Object Index", "IndexOB"), fo.inputs["index"])
go = ng.nodes.new("NodeGroupOutput")
ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
ng.links.new(rl.outputs["Image"], go.inputs[0])

sc.frame_start, sc.frame_end, sc.frame_step = fa, fb - 1, fstep
sc.render.filepath = os.path.join(OUTDIR, "rgb_")
sc.render.use_overwrite = False
sc.render.use_placeholder = True
t0 = time.time()
bpy.ops.render.render(animation=True)
print(
    "headcam done",
    round(time.time() - t0, 1),
    "s",
    "fovy",
    math.degrees(FOVY),
    "lens",
    cam.data.lens,
)
json.dump(
    {"fovy_deg": math.degrees(FOVY), "res": RES},
    open(os.path.join(OUTDIR, "camera.json"), "w"),
)
