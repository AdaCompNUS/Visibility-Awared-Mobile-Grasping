# Paper video — Visibility-Aware Mobile Grasping in Dynamic Environments

A ~3:06 explainer video built entirely from scripts: Blender (Cycles, GPU) re-renders
of the recorded simulation episode and a dark-studio Fetch robot, motion graphics
drawn with Pillow, real-robot footage with automatic face blurring, neural TTS
narration and a procedural ambient music bed.

```
video/
├── pixi.toml            own pixi env: python 3.11, bpy (Blender 5 as a module), ffmpeg, opencv, edge-tts ...
├── blender/
│   ├── prepare_fetch_meshes.py   Fetch DAE/STL -> GLB (Blender 5 has no Collada importer)
│   ├── fetch_rig.py              URDF joint tree rig, ghost/hologram materials, camera frustum
│   ├── replica.py                ReplicaCAD scene import with ManiSkill's world frame
│   ├── render_common.py          Cycles/OptiX setup, lights, cameras, floors
│   ├── episode.py                cinematic re-render of assets/dropbox/replanning_data (3 shots, SCENE=apt_2)
│   ├── headcam.py                same episode from the recorded head-camera pose (RGB + depth + robot mask)
│   ├── plan_studio.py            VAMP: collision-free configurations + arm motions for the studio shots
│   └── studio.py                 SHOT=hero|swept|objvis|subgoals studio shots
├── mg/
│   ├── core.py                   drawing primitives (fonts, easing, glow text, panels)
│   ├── sources.py                video/image sources + YuNet face tracking & blur
│   ├── belief.py                 accumulates the head-camera depth into the belief-map inset
│   ├── scenes.py                 the 7 sections (hook, title, constraints, method, sim, real, results)
│   ├── render.py                 preview / full / stills; sections in parallel -> concat -> audio mux
│   └── music.py                  procedural ambient bed
├── narration/                    script.json + edge-tts mp3s + durations.json
├── assets/                       fonts, YuNet model, fetch_glb (generated), dropbox mirror (gitignored)
└── out/                          renders, sections, final/ (gitignored)
```

## Regenerate

```bash
cd video
pixi install
# 1. mirror the Dropbox assets (the stored rclone client id is broken; override it)
RCLONE_DROPBOX_CLIENT_ID= rclone copy "dropbox-antares:paper/Visibility-aware Mobile grasping/video" assets/dropbox \
    --include "final assets/**" --include "real robot demo/all demos/*.mov" --include "replanning_data/**"
# 1b. YOLOX person detector for anonymisation (gitignored, 35 MB; YuNet face model is committed)
wget -P assets/models https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx
# 2. robot meshes + narration
pixi run python blender/prepare_fetch_meshes.py
pixi run python narration/tts.py en-US-AndrewMultilingualNeural "-2%"
# 3. collision-free studio configurations/motions with the vendored VAMP planner (main repo env)
pixi run --manifest-path ../pixi.toml python blender/plan_studio.py      # -> out/studio_plan.json
# 4. Blender renders (GPU 2, ~3 h total). The recorded episode is benchmark seed 0 = ReplicaCAD apt_2.
for R in 0:150:1 150:1071:2 1071:1587:2; do SCENE=apt_2 FRAMES=$R OUTDIR=out/episode2 pixi run python blender/episode.py; done
for R in 0:150:1 150:1071:2 1071:1587:2; do SCENE=apt_2 FRAMES=$R OUTDIR=out/headcam2 pixi run python blender/headcam.py; done
for S in hero swept objvis subgoals; do SHOT=$S pixi run python blender/studio.py; done
HEADCAM_DIR=headcam2 pixi run python -m mg.belief                      # depth + belief-map insets from the head-camera render
# 5. composite (preview = 1/3 res, every 3rd frame; full = 1080p30)
pixi run python -m mg.render preview
pixi run python -m mg.render full      # -> out/final/visibility_aware_mobile_grasping_final.mp4 (crf 16 master, ~130 MB)
# 6. size-targeted deliverable (<20 MB): two-pass H.264 at 760 kbps + AAC 96k (also 720p and HEVC variants)
./out/encode/run.sh                    # -> out/encode/v5_1080p_x264.mp4
```

Runtime: the narration speed sets the length. `narration/tts.py <voice> "+8%"` gives 2:55; the section
timings in `mg/scenes.py` are scaled automatically from `narration/durations.json` (see `TIME_SCALE`).

Edit `mg/scenes.py` to change timing/text (each section is a `draw(t, ctx)` function and the
`SECTIONS` table at the bottom sets durations and which narration segment starts when).
Edit `narration/script.json` and rerun `tts.py` to change the voice-over.

Notes
- The title card carries no author names because the paper is under anonymous review.
- People in real footage are anonymised automatically (YOLOX whole-body boxes, pixelated + blurred,
  plus a YuNet face pass); check `out/stills` before publishing. Every real clip ends with the lift.
- Thread caps (`OMP_NUM_THREADS=4`, `cv2.setNumThreads(4)`) are set because the box is shared.
