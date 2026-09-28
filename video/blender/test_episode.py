import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
import render_common as rc
import replica as rp
from fetch_rig import FetchRig

DATA = os.path.join(rc.ROOT, "assets/dropbox/replanning_data/replanning_data")
sc = rc.reset_scene()
rc.setup_cycles(samples=48, res=(960, 540))
rc.world_background((0.6, 0.7, 0.85), 0.6)
col = bpy.data.collections.new("scene")
sc.collection.children.link(col)
t0 = time.time()
stage_root, stage_objs, cfg = rp.import_stage(
    os.path.join(rp.DATASET, "configs/scenes/apt_0.scene_instance.json"), col
)
print("stage objects:", [o.name for o in stage_objs])
objs = rp.import_objects(cfg, col)
arts = rp.import_articulated(cfg, col)
print(
    "imported",
    len(objs),
    "objects,",
    len(arts),
    "articulated parts in",
    round(time.time() - t0, 1),
    "s",
)
meta = json.load(open(os.path.join(DATA, "metadata.json")))
can_root, can_objs = rp.import_ycb(
    meta["object_id"], meta["object_position"], meta["object_orientation"], col
)
print("can bounds", rp.bounds(can_objs))
chair_root, chair_objs = rp.import_replica_object(
    "frl_apartment_chair_01", col, "chair_obstacle"
)
mn, mx = rp.bounds(chair_objs)
print("chair bounds raw", mn, mx)
op = meta["obstacle_pose"]
chair_root.matrix_world = rp.pose_matrix(
    [op["translation"][0], op["translation"][1], -mn.z], op["rotation"]
)
rig = FetchRig("fetch", col)
T = np.load(os.path.join(DATA, "full_trajectory/executed_trajectory.npy"))
H = np.load(os.path.join(DATA, "full_trajectory/head_camera_poses.npy"))
FR = int(os.environ.get("FRAME", "300"))
rig.set_config(T[FR])
pan, tilt = rig.head_from_camera_pose(H[FR], T[FR, 2])
rig.set_head(pan, tilt)
rig.add_frustum(depth=2.5, alpha=0.06, strength=1.2)
# camera: high 3/4 view like the original render
tgt = rc.add_target("t", (1.5, -3.5, 0.6), col)
cam = rc.add_camera("cam", (-3.5, -9.5, 6.0), tgt, lens=28, collection=col)
# lights: ceiling grid
for x in np.linspace(-1.5, 3.5, 3):
    for y in np.linspace(-7, 3.5, 4):
        rc.add_area_light(
            f"L{x:.0f}{y:.0f}",
            (x, y, 2.85),
            rc.add_target(f"tt{x:.0f}{y:.0f}", (x, y, 0), col),
            size=1.2,
            energy=150,
            color=(1, 0.96, 0.9),
            collection=col,
        )
t0 = time.time()
rc.render_still(os.path.join(rc.OUT, "tests", f"episode_test_{FR}.png"))
print("render took", round(time.time() - t0, 1), "s")
