# ManiSkill ↔ RoboMesh interactive demo

Wraps the ManiSkill mobile-grasping scene as a backend "node" for
[robomesh-node-server](https://github.com/NUS-SSI/robomesh-node-server), so a browser user
can **click an object to select it** (then say *"grasp it"*) and **type chat commands to change the view**.

## How it maps to robomesh

robomesh delivers two browser→backend events (see its `core/webrtc/messaging.go`), which the
Flask bridge (`interfaces/ros_interface.py`) turns into ROS topics:
| browser action | robomesh | ROS topic | what our node does |
|---|---|---|---|
| **click** on the video | `/point` | `/user_point` (x,y ∈ [0,1]) | **select** the object under the click (ring marker); does NOT grasp |
| **chat** text (typed or the **grasp button**) | `/chat` | `/user_instruction` (String) | `grasp it` / `pick it` grasps the selected object; also switch view, orbit, `grasp <object>` |
| — | — | `/robot_feedback` (String) | status messages; `'end'` = task done — the webapp locks its chat input ("working…") from every chat command until it arrives, so the node ends **every** command with it |
| video | RTP :5004 | `sensor_msgs/Image` | our node publishes the composite HUD; `ros_to_ffmpeg.py` streams it |

The node: `experiments/robomesh/maniskill_robomesh_node.py`.

### Layout (composite HUD)
The stream is a single **composite**, by default **960×720** (`--stream-width`/`--stream-height`);
the main pane is rendered **natively at that resolution** (the `render_camera` is sized to match, so
it is crisp rather than upscaled from ManiSkill's small default — sharper than before at the same
bandwidth). Raising it (e.g. 1280×960) is sharper still but streams noticeably slower over WebRTC.
All three panes appear in every frame:
- **MAIN (full frame)** — a third-person view of the room; this is the **interactive** pane. Click
  an object to **select** it (a green ring + label appear on it); the click does **not** start a
  grasp — say `grasp it` / `pick it` (or press the grasp button) to pick the selected object. It
  defaults to a tuned **angled overview**; chat `scene view` / `overview` returns to it, and
  `left/right/up/down` + `zoom in/out` **orbit** it around the object centroid. `top down` /
  `overhead` reframes the main pane straight down (every object visible). Clicks project object
  centers into this camera and pick the nearest.
- **TOP-RIGHT inset — `ROBOT VIEW`** — the robot's onboard **head camera** (`fetch_head`): what it
  actually sees, so you can watch a grasp up close.
- **BOTTOM-RIGHT inset — `WHAT THE ROBOT KNOWS`** — the collision cloud the planner actually plans
  against (`robot.scene.current_environment()`), height-colored (blue low → red high) over a
  dimmed copy of the main view, rendered from the **same camera pose as the main pane**. It
  updates **live** as the robot perceives (the ray-casting scene manager fuses new points).

The main pane and the point-cloud inset share **one render pass** (same pose), so the cloud always
mirrors the main view. Chat `robot view` / `point cloud` just point you at the corresponding inset
(both are always shown); `reset view` reframes the main pane to the angled overview. Clicks that
land on a HUD inset are **ignored**, so you can't accidentally grasp something hidden behind a panel.

### Scene & objects (randomized every round)
The apartment is **`scene_7`** (`--scene`), the ReplicaCAD build config with the most benchmark
coverage — its seed (7) selects the layout, and it is held **fixed** across resets so the room
never changes under the user (a single ManiSkill env reconfigures on every `reset()`, so the seed
*is* the apartment).

ReplicaCAD ships **no graspable objects**, so the node spawns them. On startup and on **every
reset** it draws `--num-objects` (default 6) fresh entries at random from a curated pool,
`resources/robomesh_easy_objects.json` — so no two rounds of the demo look alike.

How the pool was derived (see the file's own `_meta` for the full audit trail): scenes
7/8/11/12/14/17 all share this apartment, giving **120 benchmark experiments** in it; each was run
**5 independent times**, and a run counts as passed only if `success ∧ hold_success ∧ ¬collision`.
Of the 66 tasks that passed **≥ 4 of 5** runs, a solo-render visibility audit removed 7 that were
hidden by apartment geometry, then the four-per-model diversity cap removed one extra 4/5 entry.
The resulting pool has **58 entries across 34 YCB models** (34 tasks at 5/5, 24 at 4/5), each with
the exact position/orientation it was benchmarked at, its tier, and — for the 4/5 ones — the failure
reason of the single bad run. Every retained entry is **clickable in the tuned overview** (in frame,
unoccluded, clear of the HUD insets), which is why the pool and `SCENE_OVERVIEW_P/Q` must ship
together. Each entry was benchmarked *alone*, so the sampler rejection-samples until all 6 are ≥
`min_separation_m` (0.3 m) apart, which provably prevents interpenetration.

- `--no-random-objects` — old deterministic behavior: the scene's first N `grasp_tasks`.
- `--object-seed N` — reproducible draws (default: OS entropy). It does **not** touch the sim seed.

### Auto-reset after each grasp
After **every** grasp finishes (success or failure), the node automatically resets: it releases
the held object, sends the robot home, **draws a new random set of objects** from the pool, clears
the perceived collision map back to the static baseline, and clears the click selection. The
**camera view is left unchanged** — the demo stays in the view mode you were in so you can
immediately pick the next object (the tuned overview frames every pool object; the orbit and
top-down framings re-center on the new objects). Chat `reset` / `start over` triggers the same
reset manually — the chat input shows *working…* until the reset finishes (`'end'`); if a grasp is
in flight it instead replies that the grasp will auto-reset, and unlocks immediately. `reset view`
explicitly reframes the main pane to the angled overview. `render_frame` never raises — a failure
falls back to the last good frame instead of freezing the stream.

## Quick self-test (no ROS)

```bash
# add --no-grasp to render the composite HUD without a grasp (no grasp server needed)
pixi run python experiments/robomesh/maniskill_robomesh_node.py --selftest --no-grasp --num-objects 6
# builds the scene, saves the composite HUD in each main-pane framing to
# debug/robomesh/node_view_composite_{scene,orbit,topdown}.png, reports which objects the head
# camera sees, then resets once (-> node_view_composite_after_reset.png) and prints the object set
# before/after -- it must CHANGE, that is the per-reset randomization. Run it twice and you get two
# different sets. Drop --no-grasp to also run one grasp (needs the Contact-GraspNet server on :4003).
```

## Local test with ROS (no Go server / no RoboMesh account needed)

Run each in its own terminal, from this repo unless noted. All the python processes use this
repo's pixi env (the robomesh interface scripts only need rospy + flask, which the env has).

```bash
# 1) ROS master
pixi run roscore

# 2) Contact-GraspNet grasp server (:4003) — from third_party/perception_services (see its README)
cd third_party/perception_services && pixi run -e grasp grasp-server

# 3) the ManiSkill demo node (publishes /maniskill/scene/image_raw, subscribes point+instruction)
pixi run robomesh

# 4) the robomesh Flask bridge (:11111 -> /user_point, /user_instruction; /robot_feedback -> TCP :8080)
#    clone https://github.com/NUS-SSI/robomesh-node-server next to this repo first
pixi run python ../robomesh-node-server/interfaces/ros_interface.py

# 5) simulate the browser (instead of the Go server / webapp):
bash experiments/robomesh/simulate.sh chat  "top down"
bash experiments/robomesh/simulate.sh point 0.5 0.5     # click center -> SELECT nearest object
bash experiments/robomesh/simulate.sh chat  "grasp it"  # then grasp the selected object
# check the video topic is live:
pixi run rostopic hz /maniskill/scene/image_raw
```

## Full stack to a live browser (adds Go + ffmpeg + a RoboMesh node)

```bash
# video encoder: ROS image -> H.264 RTP :5004  (needs ffmpeg on PATH)
cd ../robomesh-node-server && python ros_to_ffmpeg.py /maniskill/scene/image_raw
# Go WebRTC server (needs Go 1.22+, and NODE_ID/NODE_TOKEN in .env from robomesh.ssilabs.org)
cd ../robomesh-node-server && go run main.go
```
Then connect to the node from the RoboMesh webapp; click objects and chat to drive the demo.

## Notes / tuning
- **Spawn pose** (`--spawn-yaw-deg`, default −35 = turned right): on startup **and every reset**
  the robot is re-posed with the base yawed and the arm in the navigation **TUCK**
  (`TUCK_JOINTS`) instead of ManiSkill's `rest` keyframe. The head camera's first view seeds the
  planner's collision map before the first plan of a round, so it should contain the furniture
  between the robot and the objects — at −35° it sees both tables and the floor it will cross —
  and ManiSkill's rest keyframe folds the arm right in front of the camera, occluding exactly
  that view (the tuck holds it down and out of frame, and is gravity-stable). `0` restores the
  ManiSkill default heading.
- **Head-cam FOV**: ManiSkill's Fetch `fetch_head` camera defaults to `fov=2` rad (~115°),
  which makes objects tiny (~20–40 px → sparse point clouds → grasp perception fails). The node
  builds `ManiSkillEnv(..., camera_fov=1.0)` (~57°) so a targeted object fills the frame.
  `camera_fov` is opt-in on `ManiSkillEnv` (default `None` → unchanged, so the benchmark is
  not affected).
- **Grasp server**: the node's grasp mode is `DEPTH_SEGMENTATION` → it calls the server's
  `/sample_grasp` on `:4003`. If grasps never come back, check that
  `contact_grasp_estimator.predict_scene_grasps_from_depth_K_and_2d_seg` unpacks **three**
  values from `extract_point_clouds` (`pc_full, pc_segments, pc_colors`) — a 2-value unpack
  there makes the server return 0 grasps for every request.
- **Scene/objects**: `--scene scene_7 --num-objects 6` (see *Scene & objects* above). The object
  pool's positions are furniture-relative, so it is only valid for **scene_7's apartment**
  (= scenes 7/8/11/12/14/17); pointing `--scene` at any other scene logs a warning and falls back
  to that scene's own `grasp_tasks` from `resources/grasp_benchmark.json`. Objects are metres from
  the robot spawn, so every grasp needs a base reposition first; they can still fail at
  **navigation/planning**, which is separate from perception.
- **Overview camera**: `SCENE_OVERVIEW_P/Q` is a high "dollhouse" angle tuned for this apartment.
  Measured by rendering each pool entry on its own and reading the GT segmentation: **all 58** are
  in frame, unoccluded and clear of the HUD insets under this pose — so whichever 6 get drawn, the
  user can click every one of them. The pool's `uv_overview` fields are the projections under this
  exact pose, so **changing it invalidates them** — re-run the visibility check if you move it.
