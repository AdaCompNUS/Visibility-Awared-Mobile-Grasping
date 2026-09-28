import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_common as rc
from fetch_rig import FetchRig

sc = rc.reset_scene()
rc.setup_cycles(samples=64, res=(960, 540))
rc.world_gradient()
floor = rc.add_floor(rgba=(0.015, 0.016, 0.02, 1), roughness=0.18)
rig = FetchRig("fetch")
rig.set_torso(0.2)
rig.set_head(0.25, 0.35)
rig.add_frustum(depth=1.6)
t = rc.add_target("cam_target", (0.1, 0, 0.75))
cam = rc.add_camera("cam", (2.6, -2.4, 1.3), t, lens=40, dof_dist=3.4, fstop=4.0)
rc.add_area_light(
    "key", (2.5, -2.0, 3.0), t, size=2.5, energy=900, color=(1.0, 0.95, 0.9)
)
rc.add_area_light(
    "rim", (-2.5, 1.5, 2.2), t, size=1.5, energy=700, color=(0.55, 0.8, 1.0)
)
rc.add_area_light(
    "fill", (0.5, 3.0, 1.5), t, size=3.0, energy=200, color=(0.8, 0.85, 1.0)
)
t0 = time.time()
rc.render_still(os.path.join(rc.OUT, "tests", "hero_test.png"))
print("render took", round(time.time() - t0, 1), "s")
