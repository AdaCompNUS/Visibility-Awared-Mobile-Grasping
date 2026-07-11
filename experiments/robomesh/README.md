# ManiSkill ↔ RoboMesh interactive demo

Wraps the ManiSkill mobile-grasping scene as a backend "node" for
[robomesh-node-server](https://github.com/NUS-SSI/robomesh-node-server), so a browser user
can **click an object to grasp it** and **type chat commands to change the view**.

## How it maps to robomesh

robomesh delivers two browser→backend events (see its `core/webrtc/messaging.go`), which the
Flask bridge (`interfaces/ros_interface.py`) turns into ROS topics:
| browser action | robomesh | ROS topic | what our node does |
|---|---|---|---|
| **click** on the video | `/point` | `/user_point` (x,y ∈ [0,1]) | resolve the object under the click → grasp it |
| **chat** text | `/chat` | `/user_instruction` (String) | switch view, orbit, or `grasp <object>` (see below) |
| — | — | `/robot_feedback` (String) | status messages (`'end'` = task done) |
| video | RTP :5004 | `sensor_msgs/Image` | our node publishes the composite HUD; `ros_to_ffmpeg.py` streams it |

The node: `experiments/robomesh/maniskill_robomesh_node.py`.

### Layout (composite HUD)
The stream is a single **composite** at a constant 960×720 — all three panes in every frame:
- **MAIN (full frame)** — a third-person view of the room; this is the **interactive** pane, so
  **click an object here to grasp it**. It defaults to a tuned **angled overview**; chat
  `scene view` / `overview` returns to it, and `left/right/up/down` + `zoom in/out` **orbit** it
  around the object centroid. `top down` / `overhead` reframes the main pane straight down (every
  object visible — best for freely picking any object). Clicks project object centers into this
  camera and pick the nearest.
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

### Auto-reset after each grasp
After **every** grasp finishes (success or failure), the node automatically resets: it releases
the held object, sends the robot home, restores every object to its initial pose, clears the
perceived collision map back to the static baseline, and reframes the main pane to the **angled
overview** so you can immediately pick the next object. Chat `reset` / `start over` triggers the
same reset manually (queued until the current grasp finishes); `reset view` only reframes the
main pane. `render_frame` never raises — a failure falls back to the last good frame instead of
freezing the stream.

## Quick self-test (no ROS)

```bash
# add --no-grasp to render the composite HUD without a grasp (no grasp server needed)
pixi run python experiments/robomesh/maniskill_robomesh_node.py --selftest --no-grasp --num-objects 6
# builds the scene, saves the composite HUD in each main-pane framing to
# debug/robomesh/node_view_composite_{scene,orbit,topdown}.png, and reports which objects the
# head camera sees. Drop --no-grasp to also run one grasp (needs the Contact-GraspNet server on :4003).
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
bash experiments/robomesh/simulate.sh point 0.5 0.5     # click center -> grasp nearest object
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
- **Scene/objects**: `--scene scene_0 --num-objects N` picks a benchmark scene and how many YCB
  objects to spawn (from `resources/grasp_benchmark.json`). Far objects (scene_0 has some
  ~7–9 m away) can still fail at **navigation/planning**, which is separate from perception.
