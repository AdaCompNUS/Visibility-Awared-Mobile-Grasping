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
rc.setup_cycles(samples=24, res=(1300, 730))
rc.world_background((0.8, 0.85, 0.95), 0.8)
col = bpy.data.collections.new("scene")
sc.collection.children.link(col)
_, stage_objs, cfg = rp.import_stage(
    os.path.join(
        rp.DATASET,
        "configs/scenes",
        os.environ.get("SCENE", "apt_2") + ".scene_instance.json",
    ),
    col,
)
rp.import_objects(cfg, col)
rp.import_articulated(cfg, col)
rp.cut_ceiling([o for o in bpy.data.objects if o.type == "MESH"], 2.45)
meta = json.load(open(os.path.join(DATA, "metadata.json")))
rp.import_ycb(
    meta["object_id"], meta["object_position"], meta["object_orientation"], col
)
T = np.load(os.path.join(DATA, "full_trajectory/executed_trajectory.npy"))
for f in [0, 300, 700, 1000, 1300]:
    r = FetchRig(f"r{f}", col)
    r.set_config(T[f])
rc.add_sun("sun", (math.radians(60), 0, math.radians(-35)), energy=2.0, collection=col)
cam = rc.add_camera("top", (1.0, -3.0, 30.0), None, lens=50, collection=col)
cam.data.type = "ORTHO"
cam.data.ortho_scale = 13.0
cam.rotation_euler = (0, 0, 0)
rc.render_still(os.path.join(rc.OUT, "tests", "apt_map.png"))
