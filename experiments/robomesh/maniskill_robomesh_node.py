#!/usr/bin/env python3
"""Interactive ManiSkill demo node for the RoboMesh platform.

Bridges the ManiSkill mobile-grasping scene to robomesh-node-server so a browser user
can (1) CLICK an object to grasp it and (2) type CHAT commands to change the view.

RoboMesh contract (via robomesh-node-server/interfaces/ros_interface.py):
  subscribes  /user_point       (geometry_msgs/Point32)  -- click, x,y normalized [0,1] (y down, 0=top)
  subscribes  /user_instruction (std_msgs/String)        -- chat text
  publishes   /robot_feedback   (std_msgs/String)        -- status; 'end' marks task done
  publishes   <image_topic>     (sensor_msgs/Image, rgb8)-- the streamed scene view
                                                            (feed to ros_to_ffmpeg.py)

Streamed as a single composite HUD (default 960x720, --stream-width/-height; the main pane is
rendered natively at that resolution -- crisp, not upscaled). See experiments/robomesh/README.md:
  - MAIN (full frame)   third-person view of the room -- the INTERACTIVE pane; click an object
                        to SELECT it, then say 'grasp it'. Defaults to the tuned angled overview;
                        chat left/right/up/down + zoom orbit it, 'top down' makes it overhead.
  - TOP-RIGHT inset     the robot's first-person head camera (fetch_head) -- what it sees.
  - BOTTOM-RIGHT inset  the collision map the planner plans against
                        (robot.scene.current_environment()), height-colored, rendered from the
                        SAME pose as the main pane -- what the robot currently knows (live).
Click -> SELECT: a click on the main pane selects the nearest object (ringed with a marker); it
does NOT start a grasp. Say "grasp it" / "pick it" (or press the RoboMesh grasp button, which
sends that chat text) to grasp the selected object; "grasp <object>" also works by name. Clicks
on the HUD insets are ignored. After every grasp the scene auto-resets (objects + robot) but
KEEPS the current camera view.

Run (needs roscore + the Contact-GraspNet server on :4003):
  pixi run python experiments/robomesh/maniskill_robomesh_node.py
Local self-test (no ROS; builds scene, saves the composite HUD frames, grasps one object):
  pixi run python experiments/robomesh/maniskill_robomesh_node.py --selftest
"""
import argparse
import json
import os
import threading
import uuid

import numpy as np
import sapien
import torch
import yaml
from mani_skill.utils.building import actors
from mani_skill.utils.sapien_utils import look_at

from grasp_anywhere.core.scheduler import Scheduler
from grasp_anywhere.envs.maniskill.maniskill_env_mpc import ManiSkillEnv
from grasp_anywhere.robot.fetch import Fetch
from grasp_anywhere.utils.logger import log

CONFIG_PATH = "grasp_anywhere/configs/maniskill_fetch.yaml"
BENCHMARK_PATH = "resources/grasp_benchmark.json"


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _orbit_pose(target, yaw, pitch, radius):
    tx, ty, tz = target
    eye = [
        tx + radius * np.cos(pitch) * np.cos(yaw),
        ty + radius * np.cos(pitch) * np.sin(yaw),
        tz + radius * np.sin(pitch),
    ]
    return look_at(eye, target)


# Tuned third-person overview pose (a good "whole room" framing). p = world position
# (x,y,z); q = SAPIEN quaternion (w,x,y,z). Used as the 'scene' view default.
SCENE_OVERVIEW_P = [3.364771604537964, 0.5355691909790039, 3.3463778495788574]
SCENE_OVERVIEW_Q = [
    0.39639514684677124,
    0.34094130992889404,
    0.16146793961524963,
    -0.8369933366775513,
]

# Composite HUD layout (on the constant stream_w x stream_h canvas): the two informational
# insets are drawn in the top-right (first-person) and bottom-right (point cloud) corners.
INSET_W, INSET_H = 300, 225  # 4:3 thumbnails (match the main pane's aspect)
INSET_MARGIN = 16  # gap from the canvas edges
INSET_BORDER = 3  # colored frame around each inset
PANE_LABELS = ["ROBOT VIEW", "WHAT THE ROBOT KNOWS"]
PANE_COLORS = [
    (80, 200, 255),
    (255, 170, 70),
]  # cyan = first person, amber = collision map
SELECT_COLOR = (60, 255, 120)  # marker drawn on the click-selected object


class ManiSkillRoboMeshNode:
    def __init__(
        self,
        scene_id="scene_0",
        num_objects=6,
        image_topic="/maniskill/scene/image_raw",
        stream_w=960,
        stream_h=720,
        fps=12,
        head_w=1024,
        head_h=768,
        head_fov=1.0,
        max_attempts=5,
    ):
        self.image_topic = image_topic
        # The third-person main pane is rendered at the stream resolution natively (the
        # render_camera is sized to match, see _build_scene), so it is crisp at 960x720 rather
        # than upscaled from ManiSkill's small default -- a resolution win with no extra bandwidth.
        self.stream_w, self.stream_h, self.fps = stream_w, stream_h, fps
        self.head_w, self.head_h = head_w, head_h
        # Every scene_0 object is >0.96 m from the robot spawn, so EVERY grasp needs the mobile
        # base to reposition first; each grasp_anywhere attempt is one stochastic base-reposition
        # cycle. Match the benchmark's budget (5) so far/cluttered objects get enough tries.
        self.max_attempts = max_attempts
        # Narrow the head-cam FOV from ManiSkill's default ~115deg to ~57deg so targeted
        # objects fill the frame -> dense point cloud -> the grasp perception succeeds.
        self.head_fov = head_fov

        # --- view state ---
        # The stream is a single composite HUD: a large third-person MAIN pane (interactive)
        # plus two insets (first-person head cam; perceived collision map). main_view selects
        # the main pane's framing: "scene" = angled third-person overview (tuned/orbit),
        # "topdown" = overhead. The point-cloud inset always mirrors the main pane's pose.
        self.main_view = "scene"
        # scene sub-mode: "tuned" = the tuned overview pose; "orbit" = orbit around the
        # object centroid (engaged the moment the user rotates/zooms).
        self._scene_sub = "tuned"
        self.view_yaw = 2.4
        self.view_pitch = 0.5
        self.view_radius = 2.6
        self._default_view = (self.view_yaw, self.view_pitch, self.view_radius)

        # inset rectangles (top-left x,y + w,h in stream pixels): top-right = first person,
        # bottom-right = point cloud. Used both to draw the HUD and to reject clicks on it.
        inset_x0 = stream_w - INSET_MARGIN - INSET_W
        self._inset_rects = [
            (inset_x0, INSET_MARGIN, INSET_W, INSET_H),  # first person
            (
                inset_x0,
                stream_h - INSET_MARGIN - INSET_H,
                INSET_W,
                INSET_H,
            ),  # point cloud
        ]

        # click-selected target: a click SELECTS the object under it; the actual grasp is
        # triggered separately by a "grasp it" / "pick it" chat command (or the RoboMesh
        # grasp button, which sends the same chat text). None = nothing selected.
        self._selected = None  # (name, info) or None

        # --- grasp concurrency (only one grasp at a time) ---
        self._busy = threading.Lock()

        self._build_scene(scene_id, num_objects)

    # ------------------------------------------------------------------ scene
    def _build_scene(self, scene_id, num_objects):
        with open(CONFIG_PATH) as f:
            self.config = yaml.safe_load(f)
        with open(BENCHMARK_PATH) as f:
            benchmark = json.load(f)
        scene_data = benchmark[scene_id]
        seed = int(scene_data.get("seed", 0))
        self._seed = seed
        self._tasks = scene_data["grasp_tasks"][:num_objects]

        log.info(
            f"[robomesh] building scene {scene_id} (seed={seed}) with {len(self._tasks)} objects"
        )
        self.sim_env = ManiSkillEnv(
            env_id="ReplicaCAD_SceneManipulation-v1",
            robot_uids="fetch",
            render_mode="rgb_array",
            camera_width=self.head_w,
            camera_height=self.head_h,  # first-person head cam res
            camera_fov=self.head_fov,  # realistic FOV (see __init__)
            # render the third-person main pane at the stream resolution (no upscaling)
            render_camera_size=(self.stream_w, self.stream_h),
        )
        self.sim_env.reset(seed=seed)
        self.scene = self.sim_env.env.unwrapped.scene

        # place the objects (rebuilt after every reset -- see _spawn_objects / reset_scene)
        self._spawn_objects()

        # orbit target = centroid of the objects
        positions = [
            np.asarray(t["position"], dtype=np.float32).reshape(-1, 3)[0]
            for t in self._tasks
        ]
        pos_arr = np.asarray(positions) if positions else np.array([[0, 0, 0.7]])
        self.view_target = np.mean(pos_arr, axis=0)
        self.view_target[2] = max(0.5, float(self.view_target[2]))

        # top-down framing: center over the object region + a height that fits its extent.
        # Used by the 'topdown' view and the 'pointcloud' map.
        lo, hi = pos_arr.min(0), pos_arr.max(0)
        self._region_center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.4])
        span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1.5))
        rc_fov = float(
            getattr(self.scene.human_render_cameras["render_camera"].config, "fov", 1.0)
        )
        self._topdown_height = (span / 2) / np.tan(rc_fov / 2) * 1.25 + 1.0

        # benchmark dynamic challenges off; build robot + scheduler with the canonical collision map
        self.sim_env.benchmark_manager.enabled = False
        canonical = f"resources/benchmark/canonical_maps/{scene_id}.ply"
        static_pcds = [canonical] if os.path.exists(canonical) else []
        self.robot = Fetch(
            config_path=CONFIG_PATH,
            robot_env=self.sim_env,
            static_pcd_paths=static_pcds,
        )
        self.scheduler = Scheduler(robot=self.robot, config_path=CONFIG_PATH)

        # static collision map points (fallback for the 'pointcloud' view before any perception)
        self._static_map = None
        if os.path.exists(canonical):
            try:
                import open3d as o3d

                self._static_map = np.asarray(
                    o3d.io.read_point_cloud(canonical).points, dtype=np.float32
                )
            except Exception as e:  # noqa: BLE001
                log.warning(f"[robomesh] could not load static map {canonical}: {e}")

        # the third-person render camera we re-pose
        self._rc = self.scene.human_render_cameras["render_camera"]
        self._rc_w = int(getattr(self._rc.config, "width", 512))
        self._rc_h = int(getattr(self._rc.config, "height", self._rc_w))
        log.info(
            f"[robomesh] scene ready. render_camera width={self._rc_w}. objects: "
            f"{[o['model_id'] for o in self.objects.values()]}"
        )

    def _spawn_objects(self):
        """(Re)build the YCB objects on the CURRENT live scene and (re)populate the name/seg
        maps. env.reset() drops actors added after construction, so this runs at build time AND
        after every reset to keep the objects in the scene."""
        self.objects = {}  # actor_name -> {"actor", "model_id", "init_pose"}
        self._segid_to_name = {}  # head-cam GT segmentation id -> actor_name
        for t in self._tasks:
            model_id = t["model_id"]
            pos = np.asarray(t["position"], dtype=np.float32).reshape(-1, 3)[0]
            quat = np.asarray(t["orientation"], dtype=np.float32).reshape(-1, 4)[0]
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
            builder.initial_pose = sapien.Pose(p=pos, q=quat)
            name = f"ycb_{model_id}_{uuid.uuid4().hex[:8]}"
            actor = builder.build(name=name)
            self.objects[name] = {
                "actor": actor,
                "model_id": model_id,
                "init_pose": (pos.copy(), quat.copy()),
            }
            try:
                self._segid_to_name[int(actor._objs[0].per_scene_id)] = name
            except Exception:  # noqa: BLE001  (falls back to projection-based clicking)
                pass

    # ------------------------------------------------------------------ render
    def _topdown_sapien_pose(self):
        """Approximately top-down pose over the object region (slight -x tilt for depth cues)."""
        c = self._region_center
        eye = [
            float(c[0]) - 0.20 * self._topdown_height,
            float(c[1]),
            float(c[2]) + self._topdown_height,
        ]
        pose = look_at(eye, [float(c[0]), float(c[1]), float(c[2])])
        p = _np(pose.raw_pose)[0]
        return sapien.Pose(p[:3], p[3:])

    def _apply_view_pose(self, mode):
        """Pose the third-person render_camera for scene / topdown / pointcloud modes."""
        if mode in ("topdown", "pointcloud"):
            sp = self._topdown_sapien_pose()
        elif self._scene_sub == "tuned":
            sp = sapien.Pose(SCENE_OVERVIEW_P, SCENE_OVERVIEW_Q)
        else:  # orbit around the object centroid
            pose = _orbit_pose(
                self.view_target.tolist(),
                self.view_yaw,
                self.view_pitch,
                self.view_radius,
            )
            p = _np(pose.raw_pose)[0]
            sp = sapien.Pose(p[:3], p[3:])
        for c in self._rc.camera._render_cameras:
            c.set_local_pose(sp)

    def _resize(self, rgb):
        from PIL import Image as PILImage

        if (rgb.shape[1], rgb.shape[0]) != (self.stream_w, self.stream_h):
            rgb = np.asarray(
                PILImage.fromarray(rgb).resize((self.stream_w, self.stream_h))
            )
        return rgb

    def render_frame(self):
        """Return the current view as an (stream_h, stream_w, 3) uint8 RGB array.

        Never raises: a failure in one view must not kill the streaming loop (which would
        freeze the browser on the last frame). Falls back to the last good frame."""
        try:
            rgb = self._compose_frame()
            self._last_frame = rgb
            return rgb
        except Exception as e:  # noqa: BLE001
            if not getattr(self, "_render_err_logged", False):
                log.error(
                    f"[robomesh] render_frame failed (main_view '{self.main_view}'): {e}"
                )
                self._render_err_logged = True
            if getattr(self, "_last_frame", None) is not None:
                return self._last_frame
            return np.zeros((self.stream_h, self.stream_w, 3), dtype=np.uint8)

    def _compose_frame(self):
        """Build the streamed HUD: the large third-person MAIN pane (full stream canvas) with the
        robot's first-person view (top-right) and the perceived collision map (bottom-right) as
        insets. The main pane and the point-cloud pane share ONE render pass (same camera pose).
        """
        with self.sim_env._env_lock:
            self._apply_view_pose(self.main_view)
            self.scene.update_render()
            full = _np(self.sim_env.env.render())[0]  # (H, 2*W, 3)
            params = self._rc.get_params()
        scene_native = full[:, : self._rc_w, :].astype(
            np.uint8
        )  # left tile = render_camera
        cloud_native = self._pointcloud_overlay(
            scene_native, params
        )  # same pose as the main pane
        main_rgb = self._resize(scene_native)  # -> full stream canvas
        marker = self._selected_marker(params)  # highlight the click-selected object
        return self._compose(main_rgb, [self._head_rgb(), cloud_native], marker)

    def _selected_marker(self, params):
        """(sx, sy, model_id) of the click-selected object in stream pixels, or None."""
        if not self._selected:
            return None
        _, info = self._selected
        K = _np(params["intrinsic_cv"])
        K = K[0] if K.ndim == 3 else K
        ext = _np(params["extrinsic_cv"])
        ext = ext[0] if ext.ndim == 3 else ext
        pw = _np(info["actor"].pose.p).reshape(-1)[:3]
        cam = ext[:3, :3] @ pw + ext[:3, 3]
        if cam[2] <= 1e-6:
            return None
        uv = K @ (cam / cam[2])
        sx = float(uv[0]) * self.stream_w / self._rc_w
        sy = float(uv[1]) * self.stream_h / self._rc_h
        return sx, sy, info["model_id"]

    def _head_rgb(self):
        """The robot's first-person head-camera RGB, cached so a momentary miss won't blank it."""
        snap = self.sim_env.get_sensor_snapshot()
        if snap is not None and snap.get("rgb") is not None:
            self._last_head_rgb = _np(snap["rgb"]).astype(np.uint8)[..., :3]
        if getattr(self, "_last_head_rgb", None) is not None:
            return self._last_head_rgb
        return np.zeros((self.head_h, self.head_w, 3), dtype=np.uint8)

    def _compose(self, main_rgb, panes, marker=None):
        """Draw `panes` = [first_person_rgb, pointcloud_rgb] into the HUD insets over main_rgb;
        if `marker` = (sx, sy, label) is set, ring the click-selected object first."""
        from PIL import Image as PILImage
        from PIL import ImageDraw

        canvas = PILImage.fromarray(np.ascontiguousarray(main_rgb))
        draw = ImageDraw.Draw(canvas)
        if marker is not None:
            sx, sy, mlabel = marker
            r = 22
            draw.ellipse(
                [sx - r, sy - r, sx + r, sy + r], outline=SELECT_COLOR, width=3
            )
            draw.line([sx - r - 7, sy, sx + r + 7, sy], fill=SELECT_COLOR, width=1)
            draw.line([sx, sy - r - 7, sx, sy + r + 7], fill=SELECT_COLOR, width=1)
            draw.text(
                (sx + r + 5, sy - r), f"{mlabel}  (say 'grasp it')", fill=SELECT_COLOR
            )
        for img, (x0, y0, w, h), label, color in zip(
            panes, self._inset_rects, PANE_LABELS, PANE_COLORS
        ):
            thumb = PILImage.fromarray(np.ascontiguousarray(img)).resize((w, h))
            draw.rectangle(
                [
                    x0 - INSET_BORDER,
                    y0 - INSET_BORDER,
                    x0 + w + INSET_BORDER - 1,
                    y0 + h + INSET_BORDER - 1,
                ],
                fill=color,
            )
            canvas.paste(thumb, (x0, y0))
            draw.rectangle(
                [x0, y0, x0 + w - 1, y0 + 15], fill=(0, 0, 0)
            )  # caption strip
            draw.text((x0 + 5, y0 + 4), label, fill=color)
        return np.asarray(canvas, dtype=np.uint8)

    # -------------------------------------------------------------- pointcloud map
    def _known_points(self):
        """The collision cloud the robot currently plans against (dynamically updated),
        falling back to the static canonical map before any perception has happened."""
        pts = None
        try:
            env = self.robot.scene.current_environment()
            pts = _np(env).reshape(-1, 3).astype(np.float32)
        except Exception:  # noqa: BLE001
            pts = None
        if pts is None or pts.shape[0] == 0:
            pts = self._static_map
        return pts

    @staticmethod
    def _height_color(z):
        """Color points by height z (blue low -> green -> red high)."""
        t = np.clip(z / 1.4, 0.0, 1.0)
        r = np.clip(1.6 * t, 0, 1)
        g = np.clip(1.0 - np.abs(t - 0.5) * 2.0, 0, 1)
        b = np.clip(1.6 * (1.0 - t), 0, 1)
        return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)

    def _pointcloud_overlay(self, scene_rgb, params):
        """The robot's known collision map splatted into an already-rendered scene image, using
        that render's camera params -- i.e. from the SAME pose as the main pane. Returns a dimmed
        copy of scene_rgb with height-colored known points drawn on top."""
        img = (scene_rgb.astype(np.float32) * 0.28).astype(np.uint8)  # dim room
        H, W = img.shape[:2]
        K = _np(params["intrinsic_cv"])
        K = K[0] if K.ndim == 3 else K
        ext = _np(params["extrinsic_cv"])
        ext = ext[0] if ext.ndim == 3 else ext

        pts = self._known_points()
        if pts is not None and pts.shape[0] > 0:
            if pts.shape[0] > 120000:  # subsample very large clouds for speed
                pts = pts[np.linspace(0, pts.shape[0] - 1, 120000).astype(int)]
            cam = (ext[:3, :3] @ pts.T).T + ext[:3, 3]  # world -> cam
            z = cam[:, 2]
            front = z > 1e-6
            cam, world_z = cam[front], pts[front, 2]
            uv = (K @ (cam / cam[:, 2:3]).T).T
            u = np.round(uv[:, 0]).astype(int)
            v = np.round(uv[:, 1]).astype(int)
            inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u, v, dz, wz = u[inb], v[inb], cam[inb, 2], world_z[inb]
            order = np.argsort(-dz)  # far first, near on top
            u, v, cols = u[order], v[order], self._height_color(wz[order])
            for du, dv in [(0, 0), (1, 0), (0, 1), (1, 1)]:  # 2px splat
                img[np.clip(v + dv, 0, H - 1), np.clip(u + du, 0, W - 1)] = cols
        return img

    def _obj_pixel(self, actor):
        """Project actor world center -> (u,v) pixel in the render camera (native _rc_w square)."""
        params = self._rc.get_params()
        K = _np(params["intrinsic_cv"])
        K = K[0] if K.ndim == 3 else K
        ext = _np(params["extrinsic_cv"])
        ext = ext[0] if ext.ndim == 3 else ext
        pw = _np(actor.pose.p).reshape(-1)[:3]
        cam = ext[:3, :3] @ pw + ext[:3, 3]
        if cam[2] <= 1e-6:
            return None
        uv = K @ (cam / cam[2])
        return float(uv[0]), float(uv[1])

    def _resolve_click_scene(self, x_norm, y_norm, max_dist_frac=0.12):
        """scene mode: project each object center into render_camera, pick nearest to the click.

        The click is normalized [0,1] with y DOWN (0 = top), matching the robomesh bridge and
        image-pixel rows -- so it maps straight to render pixels with no vertical flip.
        """
        u, v = x_norm * self._rc_w, y_norm * self._rc_h
        best, bestd = None, 1e18
        for name, info in self.objects.items():
            uv = self._obj_pixel(info["actor"])
            if uv is None:
                continue
            d = (uv[0] - u) ** 2 + (uv[1] - v) ** 2
            if d < bestd:
                bestd, best = d, name
        if best is None or bestd**0.5 > max_dist_frac * self._rc_w:
            return None, None
        return best, self.objects[best]

    def _in_inset(self, x_norm, y_norm):
        """True if a normalized click [0,1] (y DOWN, 0 = top) falls on a HUD inset."""
        px, py = x_norm * self.stream_w, y_norm * self.stream_h
        for ix, iy, iw, ih in self._inset_rects:
            if ix <= px <= ix + iw and iy <= py <= iy + ih:
                return True
        return False

    def resolve_click(self, x_norm, y_norm):
        """Normalized click [0,1] (y down, 0=top) -> (name, info) of the clicked object, or (None, None).

        Clicks map into the main third-person pane (the full canvas); clicks that land on a HUD
        inset are ignored so the user can't grasp something hidden behind a panel."""
        if self._in_inset(x_norm, y_norm):
            return None, None
        # ensure the render camera is at the main pane's pose so get_params() reflects it,
        # then project object centers and pick the nearest to the click.
        with self.sim_env._env_lock:
            self._apply_view_pose(self.main_view)
        return self._resolve_click_scene(x_norm, y_norm)

    # ------------------------------------------------------------------ grasp
    def grasp_object(self, name, info):
        if not self._busy.acquire(blocking=False):
            self._feedback("Busy with another grasp — please wait.")
            return
        try:
            model_id = info["model_id"]
            pos = _np(info["actor"].pose.p).reshape(-1)[:3].astype(np.float32)
            self._feedback(f"Grasping the {model_id} ...")
            log.info(
                f"[robomesh] grasp_anywhere target={name} pos={pos.tolist()} "
                f"max_attempts={self.max_attempts}"
            )
            success, message = self.scheduler.grasp_anywhere(
                pos.reshape(1, 3), max_attempts=self.max_attempts, target_model_id=name
            )
            self._feedback(
                f"{'Grasped' if success else 'Could not grasp'} the {model_id}. ({message})"
            )
        except Exception as e:  # noqa: BLE001
            self._feedback(f"Grasp error: {e}")
            log.error(f"[robomesh] grasp error: {e}")
        finally:
            # Always reset after a grasp finishes (success or not) so the demo returns to a
            # clean, ready state for the next object.
            self.reset_scene()
            self._feedback("end")
            self._busy.release()

    def reset_scene(self):
        """Return the sim to its start state: release the held object, robot home, objects
        back to their initial poses, clear the perceived collision map. The camera view is
        left UNCHANGED — the demo keeps the same scene and view the user was on."""
        self._feedback("Resetting ...")
        try:
            try:
                self.robot.detach_objects_from_eef()  # drop the planning attachment
            except Exception:  # noqa: BLE001
                pass
            # reset robot (home) + episode state. env.reset() rebuilds the scene from the env's
            # registered actors, which DROPS our dynamically-spawned YCB objects and re-poses/
            # recreates the render camera. So we re-capture the live scene + camera and RE-SPAWN
            # the objects at their initial poses; the camera view (main_view) is left unchanged.
            self.sim_env.reset(seed=self._seed)
            self.scene = self.sim_env.env.unwrapped.scene
            self._rc = self.scene.human_render_cameras["render_camera"]
            with self.sim_env._env_lock:
                self._spawn_objects()  # env.reset dropped them; rebuild at their initial poses
                self.scene.update_render()
            self.robot.clear_pointclouds()  # clear the VAMP planning cloud
            try:
                self.robot.scene.clear_observations()  # fused perception map -> static baseline
            except Exception:  # noqa: BLE001
                pass
            log.info("[robomesh] scene reset to initial state")
        except Exception as e:  # noqa: BLE001
            log.error(f"[robomesh] reset_scene failed: {e}")
        # keep the current camera view; just clear the click selection for the next pick
        self._selected = None
        self._feedback("Ready — click an object, then say 'grasp it'.")

    # ------------------------------------------------------------------ chat
    def handle_instruction(self, text):
        t = text.strip().lower()
        dyaw, dpitch, dr = 0.4, 0.25, 0.4
        # --- main-pane framing (the big interactive third-person view) ---
        if any(
            w in t
            for w in [
                "scene view",
                "overview",
                "third person",
                "third-person",
                "wide view",
                "whole scene",
                "show scene",
                "angle view",
                "angled view",
                "reset view",
                "reset camera",
            ]
        ):
            self.main_view = "scene"
            self._scene_sub = "tuned"
            self.view_yaw, self.view_pitch, self.view_radius = self._default_view
            self._feedback("Main view: angled third-person overview.")
        elif any(
            w in t
            for w in [
                "top down",
                "top-down",
                "topdown",
                "top view",
                "overhead",
                "birds",
                "bird's",
                "map view",
            ]
        ):
            self.main_view = "topdown"
            self._feedback(
                "Main view: top-down overhead — click any object to grasp it."
            )
        # --- the two HUD insets are always shown; these just point the user at them ---
        elif any(
            w in t
            for w in [
                "robot view",
                "first person",
                "first-person",
                "robot cam",
                "robot camera",
                "onboard",
                "head cam",
            ]
        ):
            self._feedback("The robot's first-person view is the top-right panel.")
        elif any(
            w in t
            for w in [
                "point cloud",
                "pointcloud",
                "collision map",
                "known map",
                "what the robot knows",
                "what robot knows",
                "robot knows",
                "belief map",
            ]
        ):
            self._feedback(
                "What the robot knows (its collision map) is the bottom-right panel."
            )
        # --- reset the whole scene ---
        elif any(
            w in t
            for w in ["reset", "start over", "reset scene", "reset all", "restart"]
        ):
            if self._busy.locked():
                self._feedback(
                    "Busy with a grasp — it will reset automatically when it finishes."
                )
            else:
                threading.Thread(target=self.reset_scene, daemon=True).start()
        # --- orbit controls for the main pane ---
        elif "left" in t:
            self.main_view = "scene"
            self._scene_sub = "orbit"
            self.view_yaw += dyaw
            self._feedback("Rotated left.")
        elif "right" in t:
            self.main_view = "scene"
            self._scene_sub = "orbit"
            self.view_yaw -= dyaw
            self._feedback("Rotated right.")
        elif "up" in t:
            self.main_view = "scene"
            self._scene_sub = "orbit"
            self.view_pitch = min(1.4, self.view_pitch + dpitch)
            self._feedback("Tilted up.")
        elif "down" in t:
            self.main_view = "scene"
            self._scene_sub = "orbit"
            self.view_pitch = max(-0.2, self.view_pitch - dpitch)
            self._feedback("Tilted down.")
        elif any(w in t for w in ["zoom in", "closer", "nearer"]):
            self.main_view = "scene"
            self._scene_sub = "orbit"
            self.view_radius = max(1.0, self.view_radius - dr)
            self._feedback("Zoomed in.")
        elif any(w in t for w in ["zoom out", "farther", "further", "back"]):
            self.main_view = "scene"
            self._scene_sub = "orbit"
            self.view_radius += dr
            self._feedback("Zoomed out.")
        elif t.startswith(("grasp", "pick", "grab", "take")):
            # "grasp <object>" targets that object by name; a bare "grasp it" / "pick it"
            # (or the RoboMesh grasp button) grasps the currently click-selected object.
            target = self._find_object_by_name(t) or self._selected
            if target is None:
                self._feedback(
                    "Click an object first to select it, then say 'grasp it'."
                )
            else:
                threading.Thread(
                    target=self.grasp_object, args=target, daemon=True
                ).start()
        else:
            self._feedback(
                "Sorry, that's not something this demo can do — it's a mobile-grasping scene, "
                "not a general chat assistant. You can: click an object to select it and say "
                "'grasp it' (or 'grasp <object>') to pick it up; 'scene view' / 'top down' to "
                "reframe the view; left/right/up/down or zoom in/out to move it; 'reset' to start over."
            )

    def _find_object_by_name(self, text):
        for name, info in self.objects.items():
            token = info["model_id"].split("_", 1)[-1].replace("-", " ")
            if token in text or info["model_id"] in text:
                return name, info
        return None

    # ------------------------------------------------------------------ ROS glue
    def _feedback(self, text):
        if getattr(self, "_fb_pub", None) is not None:
            from std_msgs.msg import String

            self._fb_pub.publish(String(data=text))
        log.info(f"[robomesh feedback] {text}")

    def _on_point(self, msg):
        # A click only SELECTS the object under it; it does NOT start a grasp. The grasp is
        # triggered by a "grasp it" / "pick it" chat command (or the RoboMesh grasp button,
        # which sends that chat text).
        name, info = self.resolve_click(float(msg.x), float(msg.y))
        if name is None:
            self._feedback(
                "No object there — click directly on an object to select it."
            )
            return
        self._selected = (name, info)
        self._feedback(
            f"Selected the {info['model_id']}. Say 'grasp it' (or press the grasp button) to pick it up."
        )

    def _on_instruction(self, msg):
        self.handle_instruction(msg.data)

    def run_ros(self):
        import rospy
        from geometry_msgs.msg import Point32
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        rospy.init_node("maniskill_robomesh_node", anonymous=False)
        self._fb_pub = rospy.Publisher("/robot_feedback", String, queue_size=10)
        self._img_pub = rospy.Publisher(self.image_topic, Image, queue_size=1)
        rospy.Subscriber("/user_point", Point32, self._on_point, queue_size=5)
        rospy.Subscriber(
            "/user_instruction", String, self._on_instruction, queue_size=5
        )
        log.info(
            f"[robomesh] ROS node up. streaming {self.image_topic}; "
            f"feed it to:  python ros_to_ffmpeg.py {self.image_topic}"
        )

        rate = rospy.Rate(self.fps)
        while not rospy.is_shutdown():
            rgb = self.render_frame()
            m = Image()
            m.header.stamp = rospy.Time.now()
            m.height, m.width = rgb.shape[0], rgb.shape[1]
            m.encoding = "rgb8"
            m.is_bigendian = 0
            m.step = rgb.shape[1] * 3
            m.data = rgb.tobytes()
            self._img_pub.publish(m)
            rate.sleep()

    # ------------------------------------------------------------------ self test
    def selftest(self, do_grasp=True):
        from PIL import Image as PILImage

        os.makedirs("debug/robomesh", exist_ok=True)

        def save(tag):
            frame = self.render_frame()  # the actual streamed composite HUD
            PILImage.fromarray(frame).save(f"debug/robomesh/node_view_{tag}.png")
            print(f"[selftest] {tag}: {frame.shape} mean={int(frame.mean())}")

        # The streamed composite in its two main-pane framings (each frame includes all three
        # panes: big third-person main + first-person inset + collision-map inset).
        self.main_view = "scene"
        self._scene_sub = "tuned"
        save("composite_scene")
        self._scene_sub = "orbit"
        save("composite_orbit")
        self.main_view = "topdown"
        save("composite_topdown")
        # back to the interactive default
        self.main_view = "scene"
        self._scene_sub = "tuned"

        # simulate a click-selection to verify the selection marker renders on the main pane
        self._selected = next(iter(self.objects.items()))
        save("composite_selected")
        print(f"[selftest] selected marker on {self._selected[1]['model_id']}")
        self._selected = None

        kp = self._known_points()
        print(f"[selftest] pointcloud map: {0 if kp is None else len(kp)} known points")

        # which objects does the robot's head camera actually see? (top-right inset content)
        snap = self.sim_env.get_sensor_snapshot()
        if snap is not None and snap.get("segmentation") is not None:
            seg = _np(snap["segmentation"])
            seg = seg[..., 0] if seg.ndim == 3 else seg
            present = {int(s) for s in np.unique(seg)}
            visible = [
                self.objects[n]["model_id"]
                for sid, n in self._segid_to_name.items()
                if sid in present
            ]
            print(
                f"[selftest] head-cam sees objects: {visible or '(none — robot facing away)'}"
            )

        if do_grasp:
            name, info = next(iter(self.objects.items()))
            uv = self._obj_pixel(info["actor"])
            print(f"[selftest] first object {info['model_id']} projects to {uv}")
            print("[selftest] running one grasp (blocking; needs GraspNet :4003)...")
            self.grasp_object(name, info)
        self.sim_env.close()
        print("SELFTEST_DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene_0")
    ap.add_argument("--num-objects", type=int, default=6)
    ap.add_argument("--image-topic", default="/maniskill/scene/image_raw")
    ap.add_argument(
        "--stream-width",
        type=int,
        default=960,
        help="streamed HUD width; the third-person main pane renders natively at this size "
        "(higher = sharper but more GPU/WebRTC bandwidth -- 1280 streamed noticeably slower)",
    )
    ap.add_argument(
        "--stream-height", type=int, default=720, help="streamed HUD height"
    )
    ap.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="grasp_anywhere base-reposition attempts per grasp (benchmark uses 5; "
        "lower = snappier but fewer tries on far/cluttered objects)",
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="build scene + save frames + one grasp (no ROS)",
    )
    ap.add_argument(
        "--no-grasp",
        action="store_true",
        help="selftest: only render the views, skip the grasp",
    )
    args = ap.parse_args()

    node = ManiSkillRoboMeshNode(
        scene_id=args.scene,
        num_objects=args.num_objects,
        image_topic=args.image_topic,
        stream_w=args.stream_width,
        stream_h=args.stream_height,
        max_attempts=args.max_attempts,
    )
    if args.selftest:
        node.selftest(do_grasp=not args.no_grasp)
    else:
        node.run_ros()


if __name__ == "__main__":
    main()
