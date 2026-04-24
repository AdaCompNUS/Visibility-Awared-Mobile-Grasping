import glob
import json
import os
import time
from pathlib import Path

import numpy as np

import grasp_anywhere.utils.motion_utils as motion_utils


class VizDataCollector:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.viz_log: list[dict] = []
        self.current_stage: str = "INIT"
        self._apply_patch()

    def _apply_patch(self) -> None:
        if not self.enabled:
            return

        self._original_save_traj = motion_utils.save_whole_body_trajectory

        def patched_save_traj(
            arm_path: list,
            base_configs: list,
            filename: str = "debug/whole_body_plan.npy",
        ) -> None:
            # 1. Call original to write file (and handle conversion logic)
            self._original_save_traj(arm_path, base_configs, filename)
            # 2. Immediately log it
            self.log_trajectory_file(self.current_stage, filename)

        motion_utils.save_whole_body_trajectory = patched_save_traj

    def set_stage(self, stage_name: str) -> None:
        if not self.enabled:
            return
        self.current_stage = stage_name
        self.log_stage_change(stage_name)

    def reset(self) -> None:
        if not self.enabled:
            return
        self.viz_log = []
        # Clear old debug files
        for f in glob.glob("debug/initial_config_plan.npy") + glob.glob(
            "debug/replan_config_*.npy"
        ):
            os.remove(f)

    def log_stage_change(self, stage_name: str) -> None:
        if not self.enabled:
            return
        self.viz_log.append(
            {"timestamp": time.time(), "type": "stage_change", "stage": stage_name}
        )

    def record_rgbd_path(self, path: str) -> None:
        if not self.enabled:
            return
        self.viz_log.append(
            {"timestamp": time.time(), "type": "rgbd_source", "path": str(path)}
        )

    def log_trajectory_file(self, stage_name: str, filename: str) -> None:
        """Reads a specific trajectory file and appends it to viz_log."""
        if not self.enabled:
            return
        # No error handling
        data = np.load(filename)
        self.viz_log.append(
            {
                "timestamp": time.time(),
                "type": "trajectory",
                "stage": stage_name,
                "data": data.tolist(),
            }
        )

    def collect_trajectory_logs(self, stage_name: str) -> None:
        """Helper to collect and clean up trajectory logs from debug folder."""
        if not self.enabled:
            return
        traj_files = glob.glob("debug/initial_config_plan.npy") + glob.glob(
            "debug/replan_config_*.npy"
        )
        for f in traj_files:
            # No error handling as requested
            data = np.load(f)
            self.viz_log.append(
                {
                    "timestamp": time.time(),
                    "type": "trajectory",
                    "stage": stage_name,
                    "data": data.tolist(),
                }
            )
            os.remove(f)

    def save(self, run_name: str) -> None:
        """Helper to save the collected visualization logs."""
        if not self.enabled:
            return
        timestamp = int(time.time())
        # Ensure debug/visualization directory exists
        save_dir = Path("debug/visualization")
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = save_dir / f"{run_name}_{timestamp}_viz.json"

        with open(filename, "w") as f:
            json.dump(self.viz_log, f, indent=2)
