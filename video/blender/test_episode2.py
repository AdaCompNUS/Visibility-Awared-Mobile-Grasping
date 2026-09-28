import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
import render_common as rc
import replica as rp
from fetch_rig import FetchRig

DATA = os.path.join(rc.ROOT, "assets/dropbox/replanning_data/replanning_data")
sc = rc.reset_scene()
rc.setup_cycles(samples=48, res=(960, 540))
rc.world_background((0.7, 0.78, 0.9), 0.5)
col = bpy.data.collections.new("scene")
sc.collection.children.link(col)
stage_root, stage_objs, cfg = rp.import_stage(
    os.path.join(rp.DATASET, "configs/scenes/apt_0.scene_instance.json"), col
)
objs = rp.import_objects(cfg, col)
arts = rp.import_articulated(cfg, col)
allmesh = [o for o in bpy.data.objects if o.type == "MESH"]
print("cut faces:", rp.cut_ceiling(allmesh, 2.4))
meta = json.load(open(os.path.join(DATA, "metadata.json")))
can_root, can_objs = rp.import_ycb(
    meta["object_id"], meta["object_position"], meta["object_orientation"], col
)
chair_root, chair_objs = rp.import_replica_object(
    "frl_apartment_chair_01", col, "chair_obstacle"
)
mn, mx = rp.bounds(chair_objs)
op = meta["obstacle_pose"]
chair_root.matrix_world = rp.pose_matrix(
    [op["translation"][0], op["translation"][1], -mn.z], op["rotation"]
)
rig = FetchRig("fetch", col)
T = np.load(os.path.join(DATA, "full_trajectory/executed_trajectory.npy"))
H = np.load(os.path.join(DATA, "full_trajectory/head_camera_poses.npy"))
FR = 300
rig.set_config(T[FR])
pan, tilt = rig.head_from_camera_pose(H[FR], T[FR, 2])
rig.set_head(pan, tilt)
rig.add_frustum(depth=2.5, alpha=0.05, strength=1.0)
# sun + sky
rc.add_sun(
    "sun",
    (math.radians(50), 0, math.radians(-30)),
    energy=2.5,
    color=(1, 0.97, 0.92),
    collection=col,
)
# top-down ortho
cam = rc.add_camera("top", (1.0, -2.0, 25.0), None, lens=50, collection=col)
cam.data.type = "ORTHO"
cam.data.ortho_scale = 14.0
cam.rotation_euler = (0, 0, 0)
rc.render_still(os.path.join(rc.OUT, "tests", "apt_topdown.png"))
# perspective from inside, high angle towards the robot's route
tgt = rc.add_target("t", (1.4, -2.3, 0.6), col)
cam2 = rc.add_camera("persp", (-1.8, 1.8, 3.6), tgt, lens=24, collection=col)
rc.render_still(os.path.join(rc.OUT, "tests", "apt_persp.png"))
print("done")
