from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
BUNDLE_DIR = REPO_ROOT / "run_app_acconeer_exptool_a2-v0_22_1-presence_script_v7"
WHEEL_GLOB = "acconeer_exptool-*.whl"
VENDOR_DIR = REPO_ROOT / ".vendor" / "acconeer_exptool"
ALGO_GROUP_NAME = "wall_mount_presence"
PROCESSOR_CONFIG_DATASET_NAME = "presence_processor_config"
BASE_STEP_LENGTH_M = 2.5e-3
HEATMAP_VISUALIZATION_PATH = (
    BUNDLE_DIR / "examples" / "a2" / "algo" / "heatmap" / "heatmap_visualization.py"
)


def _ensure_bundled_exptool() -> None:
    if (VENDOR_DIR / "acconeer" / "exptool" / "__init__.py").exists():
        sys.path.insert(0, str(VENDOR_DIR))
        return

    wheel_paths = sorted(BUNDLE_DIR.glob(WHEEL_GLOB))
    if not wheel_paths:
        raise RuntimeError(f"Bundled Acconeer wheel not found under {BUNDLE_DIR}")

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel_paths[0]) as wheel:
        wheel.extractall(VENDOR_DIR)

    sys.path.insert(0, str(VENDOR_DIR))


def _load_heatmap_visualization_module() -> Any:
    if not HEATMAP_VISUALIZATION_PATH.exists():
        raise RuntimeError(f"Missing visualization module: {HEATMAP_VISUALIZATION_PATH}")

    spec = importlib.util.spec_from_file_location(
        "bundled_heatmap_visualization",
        HEATMAP_VISUALIZATION_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load visualization module: {HEATMAP_VISUALIZATION_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_heatmap_view_session_config(session_config: Any, subsweep_config: Any) -> Any:
    return SimpleNamespace(
        update_rate=session_config.update_rate,
        sensor_config=SimpleNamespace(subsweep=subsweep_config),
    )


def _import_runtime(enable_visualization: bool, enable_camera: bool) -> dict[str, Any]:
    _ensure_bundled_exptool()

    missing: list[str] = []
    imported: dict[str, Any] = {}

    for module_name in ["numpy", "h5py", "scipy", "attrs"]:
        try:
            imported[module_name] = __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if enable_visualization or enable_camera:
        try:
            imported["cv2"] = __import__("cv2")
        except ImportError:
            if enable_camera:
                missing.append("opencv-python")
            imported["cv2"] = None
    else:
        imported["cv2"] = None

    if enable_camera:
        try:
            from ultralytics import YOLO
        except ImportError:
            missing.append("ultralytics")
        else:
            imported["YOLO"] = YOLO

    try:
        import acconeer.exptool as et
        from acconeer.exptool import a2
        from acconeer.exptool.a2.algo.heatmap.conventional import (
            HeatmapConfig as ConventionalHeatmapConfig,
        )
        from acconeer.exptool.a2.algo.heatmap.conventional import (
            HeatmapProcessor as ConventionalHeatmapProcessor,
        )
        from acconeer.exptool.a2.algo.noise_normalization import (
            NoiseNormalizationRecording,
            load_noise_normalization_recording,
            save_noise_normalization_recording,
        )
        from acconeer.exptool.a2.algo.presence.wall_mount import (
            BeamformingMethod,
            DetectorConfig,
            MultiObsHandling,
            ObjectTrackerConfig,
            ThresholdMethod,
            TrackingProcessorConfig,
        )
        from acconeer.exptool.a2.algo.presence.wall_mount.processor import TrackingProcessor
    except ImportError as exc:
        missing.append(f"acconeer_exptool ({exc})")
    else:
        imported.update(
            {
                "et": et,
                "a2": a2,
                "ConventionalHeatmapConfig": ConventionalHeatmapConfig,
                "ConventionalHeatmapProcessor": ConventionalHeatmapProcessor,
                "NoiseNormalizationRecording": NoiseNormalizationRecording,
                "load_noise_normalization_recording": load_noise_normalization_recording,
                "save_noise_normalization_recording": save_noise_normalization_recording,
                "BeamformingMethod": BeamformingMethod,
                "DetectorConfig": DetectorConfig,
                "MultiObsHandling": MultiObsHandling,
                "ObjectTrackerConfig": ObjectTrackerConfig,
                "ThresholdMethod": ThresholdMethod,
                "TrackingProcessorConfig": TrackingProcessorConfig,
                "TrackingProcessor": TrackingProcessor,
            }
        )

        if enable_visualization:
            try:
                imported["heatmap_visualization"] = _load_heatmap_visualization_module()
            except Exception:
                imported["heatmap_visualization"] = None

    if missing:
        deps = ", ".join(missing)
        raise RuntimeError(
            "Missing runtime dependencies: "
            f"{deps}. Install them in the Python environment used for `python main.py`."
        )

    return imported


class HeatmapViewer:
    def update_result(
        self,
        raw_heatmap: Any,
        suppressed_heatmap: Any,
        summary: dict[str, Any],
        np: Any,
    ) -> None:
        raise NotImplementedError

    def set_live(self, enabled: bool) -> bool:
        del enabled
        return False

    def close(self) -> None:
        pass


class NullViewer(HeatmapViewer):
    def update_result(
        self,
        raw_heatmap: Any,
        suppressed_heatmap: Any,
        summary: dict[str, Any],
        np: Any,
    ) -> None:
        del raw_heatmap, suppressed_heatmap, summary, np


class LiveDisplayController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._viewer: HeatmapViewer | None = None

    def set_viewer(self, viewer: HeatmapViewer | None) -> None:
        with self._lock:
            self._viewer = viewer

    def set_enabled(self, enabled: bool) -> bool:
        with self._lock:
            if self._viewer is None:
                return False
            return self._viewer.set_live(enabled)


def _default_radar_status() -> dict[str, Any]:
    return {
        "presence": False,
        "distance_m": None,
        "angle_deg": {
            "azimuth": None,
            "elevation": None,
        },
        "energy_score": None,
    }


def _default_camera_status(source: str = "/dev/video2") -> dict[str, Any]:
    return {
        "person_detected": False,
        "person_count": 0,
        "max_confidence": None,
        "source": source,
    }


def _fuse_person_status(radar_status: dict[str, Any], camera_status: dict[str, Any]) -> str:
    radar_presence = bool(radar_status.get("presence", False))
    camera_presence = bool(camera_status.get("person_detected", False))
    if radar_presence and camera_presence:
        return "radar and camera on person"
    if radar_presence:
        return "radar only person candidate"
    if camera_presence:
        return "camera only person candidate"
    return "no person detected"


class CombinedStatusStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {
            "radar": _default_radar_status(),
            "camera": _default_camera_status(),
            "fusion": {"person_status": "no person detected"},
        }

    def set_camera_source(self, source: str) -> None:
        with self._lock:
            self._status["camera"]["source"] = source
            self._status["fusion"]["person_status"] = _fuse_person_status(
                self._status["radar"], self._status["camera"]
            )

    def update_radar(self, radar_status: dict[str, Any]) -> None:
        with self._lock:
            self._status["radar"] = json.loads(json.dumps(radar_status))
            self._status["fusion"]["person_status"] = _fuse_person_status(
                self._status["radar"], self._status["camera"]
            )

    def update_camera(self, camera_status: dict[str, Any]) -> None:
        with self._lock:
            self._status["camera"] = json.loads(json.dumps(camera_status))
            self._status["fusion"]["person_status"] = _fuse_person_status(
                self._status["radar"], self._status["camera"]
            )

    def get(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._status))


class ConsoleCoordinator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prompt_active = False

    def start_prompt(self) -> None:
        with self._lock:
            self._prompt_active = True
            sys.stdout.write("\nQuestion> ")
            sys.stdout.flush()

    def finish_prompt(self) -> None:
        with self._lock:
            self._prompt_active = False

    def emit_lines(self, lines: list[str]) -> None:
        with self._lock:
            sys.stdout.write("\n")
            for line in lines:
                sys.stdout.write(f"{line}\n")
            if self._prompt_active:
                sys.stdout.write("\nQuestion> ")
            sys.stdout.flush()


class OpenCvHeatmapViewer(HeatmapViewer):
    _SUPPRESSED_PANEL_WIDTH = 860
    _SUPPRESSED_PANEL_HEIGHT = 520
    _TEXT_PANEL_WIDTH = 360
    _CAMERA_WINDOW_HEIGHT = 720

    def __init__(
        self,
        cv2_module: Any,
        enabled: bool,
        camera_frame_getter: Any = None,
    ) -> None:
        self._cv2 = cv2_module
        self._enabled = enabled and cv2_module is not None
        self._camera_frame_getter = camera_frame_getter
        self._radar_window_name = "Radar Display"
        self._camera_window_name = "Camera Live"
        self._radar_window_ready = False
        self._camera_window_ready = False

    def set_live(self, enabled: bool) -> bool:
        if self._cv2 is None:
            return False
        self._enabled = enabled
        if not enabled:
            self.close()
        return True

    def update_result(
        self,
        raw_heatmap: Any,
        suppressed_heatmap: Any,
        summary: dict[str, Any],
        np: Any,
    ) -> None:
        if not self._enabled:
            return

        cv2 = self._cv2
        try:
            suppressed_frame = self._heatmap_to_frame(
                suppressed_heatmap,
                np,
                cv2,
                "Suppressed Heatmap",
                self._SUPPRESSED_PANEL_WIDTH,
                self._SUPPRESSED_PANEL_HEIGHT,
            )
            if suppressed_frame is None:
                return

            frame = cv2.hconcat(
                [
                    suppressed_frame,
                    self._build_text_panel(suppressed_frame.shape[0], summary, cv2),
                ]
            )

            if not self._radar_window_ready:
                cv2.namedWindow(self._radar_window_name, cv2.WINDOW_NORMAL)
                self._radar_window_ready = True

            cv2.imshow(self._radar_window_name, frame)
            cv2.resizeWindow(self._radar_window_name, frame.shape[1], frame.shape[0])

            camera_frame = self._get_camera_panel(self._CAMERA_WINDOW_HEIGHT, cv2)
            if camera_frame is not None:
                if not self._camera_window_ready:
                    cv2.namedWindow(self._camera_window_name, cv2.WINDOW_NORMAL)
                    self._camera_window_ready = True
                cv2.imshow(self._camera_window_name, camera_frame)
                cv2.resizeWindow(self._camera_window_name, camera_frame.shape[1], camera_frame.shape[0])
            cv2.waitKey(1)
        except Exception:
            self._enabled = False

    def close(self) -> None:
        destroyed = False
        if self._radar_window_ready:
            try:
                self._cv2.destroyWindow(self._radar_window_name)
                destroyed = True
            except Exception:
                pass
            self._radar_window_ready = False
        if self._camera_window_ready:
            try:
                self._cv2.destroyWindow(self._camera_window_name)
                destroyed = True
            except Exception:
                pass
            self._camera_window_ready = False
        if destroyed:
            with suppress(Exception):
                self._cv2.waitKey(1)
            with suppress(Exception):
                self._cv2.waitKey(1)

    def _heatmap_to_frame(
        self,
        heatmap: Any,
        np: Any,
        cv2: Any,
        title: str,
        target_width: int,
        target_height: int,
    ) -> Any:
        heatmap_2d = _heatmap_to_2d(heatmap, np)
        if heatmap_2d is None:
            return None

        frame = cv2.normalize(heatmap_2d.T, None, 0, 255, cv2.NORM_MINMAX)
        frame = frame.astype("uint8")
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        frame = cv2.resize(
            frame,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
        cv2.putText(
            frame,
            title,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Azimuth",
            (max(12, frame.shape[1] // 2 - 45), frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Distance",
            (12, max(40, frame.shape[0] // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _get_camera_panel(self, target_height: int, cv2: Any) -> Any:
        if self._camera_frame_getter is None:
            return None
        try:
            frame = self._camera_frame_getter()
        except Exception:
            return None
        if frame is None:
            return None

        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            return None

        target_width = max(1, int(width * (target_height / float(height))))
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        cv2.putText(
            frame,
            "Camera",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _build_text_panel(self, height: int, summary: dict[str, Any], cv2: Any) -> Any:
        panel = cv2.cvtColor(
            cv2.resize(
                cv2.UMat(1, 1, cv2.CV_8UC1, 24).get(),
                (self._TEXT_PANEL_WIDTH, height),
                interpolation=cv2.INTER_NEAREST,
            ),
            cv2.COLOR_GRAY2BGR,
        )

        cv2.putText(
            panel,
            "Stable Target",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        payload = _summary_to_normalized_radar_status(summary)
        lines = [
            f"presence: {str(payload['presence']).lower()}",
            f"distance_m: {payload['distance_m']}",
            f"azimuth_deg: {payload['angle_deg']['azimuth']}",
            f"elevation_deg: {payload['angle_deg']['elevation']}",
            f"energy_score: {payload['energy_score']}",
        ]
        for idx, line in enumerate(lines):
            cv2.putText(
                panel,
                line,
                (18, 78 + idx * 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

        return panel


class PyqtgraphHeatmapViewer(HeatmapViewer):
    def __init__(self, rt: dict[str, Any], session_config: Any, heatmap_config: Any) -> None:
        module = rt["heatmap_visualization"]
        self._pg_process = rt["et"].PGProcess(module.PGUpdater(session_config, heatmap_config))
        self._started = False
        self._alive = True
        self._pg_process.start()
        self._started = True

    def update_result(self, processor_result: Any, np: Any) -> None:
        del np
        if not self._alive or getattr(processor_result, "heatmap", None) is None:
            return

        try:
            self._pg_process.put_data(processor_result)
        except Exception:
            self._alive = False

    def close(self) -> None:
        if self._started:
            try:
                self._pg_process.close()
            except Exception:
                pass


def _build_viewer(
    args: argparse.Namespace,
    rt: dict[str, Any],
    session_config: Any,
    heatmap_config: Any,
    subsweep_config: Any,
    camera_frame_getter: Any = None,
) -> HeatmapViewer:
    del session_config, heatmap_config, subsweep_config
    if rt["cv2"] is None:
        return NullViewer()
    return OpenCvHeatmapViewer(
        rt["cv2"],
        enabled=args.visualization,
        camera_frame_getter=camera_frame_getter,
    )


def _build_tracking_setup(rt: dict[str, Any]) -> tuple[Any, Any]:
    a2 = rt["a2"]
    BeamformingMethod = rt["BeamformingMethod"]
    DetectorConfig = rt["DetectorConfig"]
    MultiObsHandling = rt["MultiObsHandling"]
    ObjectTrackerConfig = rt["ObjectTrackerConfig"]
    ThresholdMethod = rt["ThresholdMethod"]
    TrackingProcessorConfig = rt["TrackingProcessorConfig"]

    presence_config = TrackingProcessorConfig(
        num_az_angles=60,
        az_angle_range=(-60, 60),
        num_el_angles=21,
        el_angle_range=(-45, 45),
        covariance_matrix_time_constant_s=0.15,
        high_pass_filter_time_constant_s=0.5,
        antenna_array=a2.AntennaArray.AP212_R2,
        threshold_sigma_level=9,
        az_offset=-90,
        beamforming_method=BeamformingMethod.ROBUST_CAPON_EXTENDED,
        threshold_method=ThresholdMethod.CFAR,
        static_reflector_vibration_tol_um=0,
        hide_heatmap=False,
        histogram_view=False,
        tracking_engineering_mode=False,
        room_coords_mount_position_m=(0.0, 1.5, 0.0),
        room_dimensions_m=(4.0, 3.0, 3.0),
        el_mount_offset=0.0,
        az_mount_offset=0.0,
        mirror_az=False,
        object_tracker_config=ObjectTrackerConfig(
            dbscan_eps_m=0.30,
            dbscan_MinPts=3,
            model_to_measurement_noise_rel=0.8,
            kalman_uncertainty_radius=(0.45, 0.45, 1.0),
            non_human_assignment_penalty_factor=10,
            sample_horizon_s=1.5,
            max_num_clusters=20,
            multi_obs_handling=MultiObsHandling.SINGLE_NN,
            tacker_kill_timeout_s=5.0,
            classifier_horizon_s=1.5,
            max_static_std=0.15,
            human_min_horizon_density=0.9,
            min_avg_directional_vel=0.5,
            classification_time_s=1.0,
        ),
    )

    detector_config = DetectorConfig(
        start_m=0.5,
        signal_quality_break_point_m=2.5,
        end_m=4.0,
        profile=a2.Profile.PROFILE_6,
        max_SNR_drop_dB=0.8,
        signal_quality_close_dB=35,
        signal_quality_far_dB=29,
        hwaas_min=2,
        num_sweeps=4,
        frame_rate=8,
    )

    session_config = detector_config.get_session_config()
    for subsweep in session_config.sensor_config.subsweeps:
        subsweep.enable_tx = True

    return presence_config, session_config


def _record_update_rate(record: Any) -> float:
    session_index = 0
    tick_period = record.session(session_index).tick_period
    if tick_period and tick_period > 0:
        return record.server_info.ticks_per_second / tick_period
    return float(record.session_config.update_rate or 10.0)


def _iter_record_results(record: Any) -> Any:
    if hasattr(record, "results"):
        for result in record.results:
            yield result
        return

    if hasattr(record, "group_results"):
        for group_result in record.group_results:
            yield getattr(group_result, "result", group_result)
        return

    raise RuntimeError("Unsupported record object: no iterable radar results found.")


def _heatmap_to_2d(heatmap: Any, np: Any) -> Any:
    if heatmap is None:
        return None
    if heatmap.ndim == 2:
        return np.nan_to_num(heatmap)
    if heatmap.ndim == 3:
        return np.nan_to_num(heatmap[:, heatmap.shape[1] // 2, :])
    raise RuntimeError(f"Unsupported heatmap shape: {heatmap.shape}")


def _heatmap_to_angle_view(heatmap: Any, np: Any) -> Any:
    if heatmap is None:
        return None
    if heatmap.ndim == 2:
        return np.nan_to_num(heatmap.T)
    if heatmap.ndim != 3:
        raise RuntimeError(f"Unsupported heatmap shape: {heatmap.shape}")

    distance_projection = np.max(heatmap, axis=(0, 1))
    distance_index = int(np.argmax(distance_projection))
    return np.nan_to_num(heatmap[:, :, distance_index].T)


class BackgroundSuppressor:
    def __init__(self, np: Any, alpha: float = 0.02) -> None:
        self._np = np
        self._alpha = alpha
        self._background: Any = None

    def apply(self, heatmap: Any, protected_index: tuple[int, int, int] | None = None) -> Any:
        if heatmap is None:
            return None

        current = self._np.nan_to_num(heatmap.astype(float, copy=False))
        if self._background is None or self._background.shape != current.shape:
            self._background = self._np.zeros_like(current, dtype=float)

        alpha_map = self._np.full(current.shape, self._alpha, dtype=float)
        if protected_index is not None and len(current.shape) == 3:
            az_idx, el_idx, dist_idx = protected_index
            az_slice = slice(max(0, az_idx - 2), min(current.shape[0], az_idx + 3))
            el_slice = slice(max(0, el_idx - 1), min(current.shape[1], el_idx + 2))
            dist_slice = slice(max(0, dist_idx - 2), min(current.shape[2], dist_idx + 3))
            alpha_map[az_slice, el_slice, dist_slice] = self._alpha * 0.1
        elif protected_index is not None and len(current.shape) == 2:
            az_idx, dist_idx = protected_index[0], protected_index[-1]
            az_slice = slice(max(0, az_idx - 2), min(current.shape[0], az_idx + 3))
            dist_slice = slice(max(0, dist_idx - 2), min(current.shape[1], dist_idx + 3))
            alpha_map[az_slice, dist_slice] = self._alpha * 0.1

        self._background = alpha_map * current + (1.0 - alpha_map) * self._background
        suppressed = current - self._background
        suppressed[suppressed < 0.0] = 0.0
        return suppressed


class TargetStabilizer:
    KMEANS_K = 2 
    MIN_CLUSTER_POINTS = 3
    MAX_CLUSTER_RADIUS_M = 0.45
    MIN_CLUSTER_TOTAL_ENERGY = 40.0
    STABLE_TARGET_DISTANCE_WEIGHT = 4.0

    def __init__(
        self,
        distance_smoothing_alpha: float = 0.12,
        angle_smoothing_alpha: float = 0.12,
        energy_threshold: float = 20.0,
        max_distance_jump_m: float = 0.35,
        max_angle_jump_deg: float = 8.0,
        switch_confirm_frames: int = 5,
        presence_on_frames: int = 2,
        presence_off_frames: int = 12,
        target_hold_frames: int = 24,
    ) -> None:
        self.distance_smoothing_alpha = distance_smoothing_alpha
        self.angle_smoothing_alpha = angle_smoothing_alpha
        self.energy_threshold = energy_threshold
        self.max_distance_jump_m = max_distance_jump_m
        self.max_angle_jump_deg = max_angle_jump_deg
        self.switch_confirm_frames = switch_confirm_frames
        self.presence_on_frames = presence_on_frames
        self.presence_off_frames = presence_off_frames
        self.target_hold_frames = target_hold_frames

        self._stable_target: dict[str, Any] | None = None
        self._pending_target: dict[str, Any] | None = None
        self._pending_count = 0
        self._valid_count = 0
        self._missing_count = 0
        self._presence = False

    @property
    def protected_index(self) -> tuple[int, int, int] | None:
        if self._stable_target is None:
            return None
        return self._stable_target.get("_raw_index")

    def update(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        valid_candidates = [
            candidate for candidate in candidates if float(candidate.get("energy", 0.0)) >= self.energy_threshold
        ]

        accepted_candidate = self._select_candidate(valid_candidates)
        if accepted_candidate is not None:
            self._valid_count += 1
            self._missing_count = 0
            self._presence = self._presence or self._valid_count >= self.presence_on_frames
        else:
            self._valid_count = 0
            self._missing_count += 1
            if self._missing_count >= self.presence_off_frames:
                self._presence = False

        if accepted_candidate is not None:
            self._stable_target = self._smooth_target(accepted_candidate)
        elif self._stable_target is not None and self._missing_count < self.target_hold_frames:
            self._stable_target = dict(self._stable_target)
            self._stable_target["presence"] = False

        if self._stable_target is None:
            return {"presence": False, "targets": []}

        output_target = dict(self._stable_target)
        #output_target["presence"] = self._presence and float(output_target.get("energy", 0.0)) >= self.energy_threshold
        output_target["presence"] = self._presence
        if not output_target["presence"] and self._missing_count >= self.target_hold_frames:
            return {"presence": False, "targets": []}

        output_target.pop("_raw_index", None)
        output_target.pop("_xyz", None)
        return {
            "presence": bool(output_target["presence"]),
            "targets": [output_target],
        }

    def _select_candidate(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            self._pending_target = None
            self._pending_count = 0
            return None

        clustered_candidate = self._select_clustered_candidate(candidates)
        fallback_candidate = self._select_fallback_candidate(candidates)
        chosen_candidate = clustered_candidate if clustered_candidate is not None else fallback_candidate
        if chosen_candidate is None:
            self._pending_target = None
            self._pending_count = 0
            return None

        if self._stable_target is None:
            self._pending_target = None
            self._pending_count = 0
            return chosen_candidate

        dist_jump = abs(float(chosen_candidate["distance_m"]) - float(self._stable_target["distance_m"]))
        az_jump = abs(
            float(chosen_candidate["angle_deg"]["horizontal"])
            - float(self._stable_target["angle_deg"]["horizontal"])
        )
        el_prev = self._stable_target["angle_deg"]["elevation"]
        el_curr = chosen_candidate["angle_deg"]["elevation"]
        el_jump = 0.0 if el_prev is None or el_curr is None else abs(float(el_curr) - float(el_prev))

        if (
            dist_jump <= self.max_distance_jump_m
            and az_jump <= self.max_angle_jump_deg
            and el_jump <= self.max_angle_jump_deg
        ):
            self._pending_target = None
            self._pending_count = 0
            return chosen_candidate

        if self._pending_target is not None and self._candidate_cost(chosen_candidate, self._pending_target) < 0.35:
            self._pending_count += 1
        else:
            self._pending_target = chosen_candidate
            self._pending_count = 1

        if self._pending_count >= self.switch_confirm_frames:
            accepted = self._pending_target
            self._pending_target = None
            self._pending_count = 0
            return accepted

        return None

    def _select_clustered_candidate(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if len(candidates) < self.KMEANS_K * self.MIN_CLUSTER_POINTS:
            return None

        clusters = self._kmeans_clusters(candidates)
        valid_clusters = [cluster for cluster in clusters if self._is_valid_cluster(cluster)]
        if not valid_clusters:
            return None

        if self._stable_target is not None:
            selected_cluster = min(
                valid_clusters,
                key=lambda cluster: self._cluster_cost_to_stable_target(cluster, self._stable_target),
            )
        else:
            selected_cluster = max(valid_clusters, key=lambda cluster: float(cluster["total_energy"]))

        return self._cluster_to_target(selected_cluster)

    def _select_fallback_candidate(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            return None
        if self._stable_target is None:
            return max(candidates, key=lambda candidate: float(candidate["energy"]))
        return min(candidates, key=lambda candidate: self._candidate_cost(candidate, self._stable_target))

    def _kmeans_clusters(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        points = [candidate["_xyz"] for candidate in candidates]
        labels = self._run_kmeans(points)
        clusters: list[dict[str, Any]] = []
        for cluster_id in range(self.KMEANS_K):
            cluster_candidates = [
                candidate for candidate, label in zip(candidates, labels) if label == cluster_id
            ]
            if not cluster_candidates:
                continue
            clusters.append(self._build_cluster(cluster_candidates))
        return clusters

    def _run_kmeans(self, points: list[tuple[float, float, float]]) -> list[int]:
        centroids = [points[0], points[-1]]
        for _ in range(12):
            labels = [self._nearest_centroid_index(point, centroids) for point in points]
            new_centroids: list[tuple[float, float, float]] = []
            for cluster_id in range(self.KMEANS_K):
                cluster_points = [point for point, label in zip(points, labels) if label == cluster_id]
                if not cluster_points:
                    new_centroids.append(centroids[cluster_id])
                    continue
                count = float(len(cluster_points))
                new_centroids.append(
                    (
                        sum(point[0] for point in cluster_points) / count,
                        sum(point[1] for point in cluster_points) / count,
                        sum(point[2] for point in cluster_points) / count,
                    )
                )
            if new_centroids == centroids:
                break
            centroids = new_centroids
        return [self._nearest_centroid_index(point, centroids) for point in points]

    def _nearest_centroid_index(
        self,
        point: tuple[float, float, float],
        centroids: list[tuple[float, float, float]],
    ) -> int:
        return min(
            range(len(centroids)),
            key=lambda idx: math.dist(point, centroids[idx]),
        )

    def _build_cluster(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(candidates)
        centroid_x = sum(float(candidate["position_xy"][0]) for candidate in candidates) / count
        centroid_y = sum(float(candidate["position_xy"][1]) for candidate in candidates) / count
        centroid_z = sum(float(candidate["_xyz"][2]) for candidate in candidates) / count
        avg_distance = sum(float(candidate["distance_m"]) for candidate in candidates) / count
        avg_azimuth = sum(float(candidate["angle_deg"]["horizontal"]) for candidate in candidates) / count
        elevation_values = [
            float(candidate["angle_deg"]["elevation"])
            for candidate in candidates
            if candidate["angle_deg"]["elevation"] is not None
        ]
        avg_elevation = (
            sum(elevation_values) / len(elevation_values) if elevation_values else None
        )
        total_energy = sum(float(candidate["energy"]) for candidate in candidates)
        max_energy_candidate = max(candidates, key=lambda candidate: float(candidate["energy"]))
        radius_m = max(
            math.dist(candidate["_xyz"], (centroid_x, centroid_y, centroid_z))
            for candidate in candidates
        )
        return {
            "points": candidates,
            "num_points": count,
            "centroid_xyz": (centroid_x, centroid_y, centroid_z),
            "centroid_xy": (centroid_x, centroid_y),
            "avg_distance": avg_distance,
            "avg_azimuth": avg_azimuth,
            "avg_elevation": avg_elevation,
            "total_energy": total_energy,
            "max_energy": float(max_energy_candidate["energy"]),
            "radius_m": radius_m,
            "representative_raw_index": max_energy_candidate.get("_raw_index"),
        }

    def _is_valid_cluster(self, cluster: dict[str, Any]) -> bool:
        if int(cluster["num_points"]) < self.MIN_CLUSTER_POINTS:
            return False
        if float(cluster["radius_m"]) > self.MAX_CLUSTER_RADIUS_M:
            return False
        if float(cluster["total_energy"]) < self.MIN_CLUSTER_TOTAL_ENERGY:
            return False
        return True

    def _cluster_cost_to_stable_target(self, cluster: dict[str, Any], stable_target: dict[str, Any]) -> float:
        centroid_x, centroid_y = cluster["centroid_xy"]
        stable_x, stable_y = stable_target["position_xy"]
        distance_cost = math.hypot(float(centroid_x) - float(stable_x), float(centroid_y) - float(stable_y))
        azimuth_cost = abs(float(cluster["avg_azimuth"]) - float(stable_target["angle_deg"]["horizontal"])) / 30.0
        elevation_cost = 0.0
        stable_el = stable_target["angle_deg"]["elevation"]
        cluster_el = cluster["avg_elevation"]
        if stable_el is not None and cluster_el is not None:
            elevation_cost = abs(float(cluster_el) - float(stable_el)) / 30.0
        energy_bonus = float(cluster["total_energy"]) / max(float(cluster["max_energy"]), 1.0)
        return (
            self.STABLE_TARGET_DISTANCE_WEIGHT * distance_cost
            + azimuth_cost
            + elevation_cost
            - 0.05 * energy_bonus
        )

    def _cluster_to_target(self, cluster: dict[str, Any]) -> dict[str, Any]:
        centroid_x, centroid_y = cluster["centroid_xy"]
        centroid_z = cluster["centroid_xyz"][2]
        return {
            "presence": True,
            "distance_m": round(float(cluster["avg_distance"]), 3),
            "position_xy": [round(float(centroid_x), 3), round(float(centroid_y), 3)],
            "angle_deg": {
                "horizontal": round(float(cluster["avg_azimuth"]), 3),
                "elevation": (
                    round(float(cluster["avg_elevation"]), 3)
                    if cluster["avg_elevation"] is not None
                    else None
                ),
            },
            "energy": round(float(cluster["total_energy"]), 3),
            "_raw_index": cluster["representative_raw_index"],
            "_xyz": (float(centroid_x), float(centroid_y), float(centroid_z)),
        }

    def _candidate_cost(self, candidate: dict[str, Any], reference: dict[str, Any]) -> float:
        dist = abs(float(candidate["distance_m"]) - float(reference["distance_m"]))
        az = abs(float(candidate["angle_deg"]["horizontal"]) - float(reference["angle_deg"]["horizontal"])) / 30.0
        ref_el = reference["angle_deg"]["elevation"]
        cand_el = candidate["angle_deg"]["elevation"]
        if ref_el is None or cand_el is None:
            el = 0.0
        else:
            el = abs(float(cand_el) - float(ref_el)) / 30.0
        return dist + az + el

    def _smooth_target(self, candidate: dict[str, Any]) -> dict[str, Any]:
        if self._stable_target is None:
            return dict(candidate)

        previous = self._stable_target
        smoothed = dict(candidate)
        smoothed["distance_m"] = _ema(
            float(previous["distance_m"]),
            float(candidate["distance_m"]),
            self.distance_smoothing_alpha,
        )
        smoothed["position_xy"] = [
            _ema(float(previous["position_xy"][0]), float(candidate["position_xy"][0]), self.distance_smoothing_alpha),
            _ema(float(previous["position_xy"][1]), float(candidate["position_xy"][1]), self.distance_smoothing_alpha),
        ]
        smoothed["angle_deg"] = {
            "horizontal": _ema(
                float(previous["angle_deg"]["horizontal"]),
                float(candidate["angle_deg"]["horizontal"]),
                self.angle_smoothing_alpha,
            ),
            "elevation": self._smooth_optional_angle(
                previous["angle_deg"]["elevation"],
                candidate["angle_deg"]["elevation"],
            ),
        }
        smoothed["energy"] = _ema(float(previous["energy"]), float(candidate["energy"]), self.angle_smoothing_alpha)
        smoothed["_raw_index"] = candidate.get("_raw_index")
        candidate_xyz = candidate.get("_xyz")
        previous_xyz = previous.get("_xyz")
        if candidate_xyz is not None and previous_xyz is not None:
            smoothed["_xyz"] = (
                _ema(float(previous_xyz[0]), float(candidate_xyz[0]), self.distance_smoothing_alpha),
                _ema(float(previous_xyz[1]), float(candidate_xyz[1]), self.distance_smoothing_alpha),
                _ema(float(previous_xyz[2]), float(candidate_xyz[2]), self.distance_smoothing_alpha),
            )
        else:
            smoothed["_xyz"] = candidate_xyz
        return smoothed

    def _smooth_optional_angle(self, previous: Any, current: Any) -> Any:
        if current is None:
            return previous
        if previous is None:
            return current
        return _ema(float(previous), float(current), self.angle_smoothing_alpha)


def _ema(previous: float, current: float, alpha: float) -> float:
    return alpha * current + (1.0 - alpha) * previous


def _target_from_cartesian(
    xyz: tuple[float, float, float],
    energy: float,
    is_presence: bool,
    has_elevation: bool,
    raw_index: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    x, y, z = xyz
    xy_norm = math.hypot(x, y)
    return {
        "presence": bool(is_presence),
        "distance_m": round(math.sqrt(x * x + y * y + z * z), 3),
        "position_xy": [round(x, 3), round(y, 3)],
        "angle_deg": {
            "horizontal": round(math.degrees(math.atan2(y, x)), 3),
            "elevation": (
                round(math.degrees(math.atan2(z, max(xy_norm, 1e-9))), 3)
                if has_elevation
                else None
            ),
        },
        "energy": round(float(energy), 3),
        "_raw_index": raw_index,
        "_xyz": (float(x), float(y), float(z)),
    }


def _extract_tracking_candidates(result: Any, suppressed_heatmap: Any, np: Any) -> list[dict[str, Any]]:
    observations = list(getattr(result, "current_obs_m", []))
    has_elevation = False
    heatmap = getattr(result, "heatmap", None)
    if heatmap is not None and getattr(heatmap, "ndim", 0) == 3 and heatmap.shape[1] > 1:
        has_elevation = True

    targets: list[dict[str, Any]] = []
    for observation in observations:
        (obs_x, obs_y, obs_z), raw_inds, obs_energy = observation
        energy = float(obs_energy)
        try:
            if suppressed_heatmap is not None and len(raw_inds) >= 3:
                energy = float(suppressed_heatmap[raw_inds[0], raw_inds[1], raw_inds[2]])
        except Exception:
            pass

        targets.append(
            _target_from_cartesian(
                (float(obs_x), float(obs_y), float(obs_z)),
                energy,
                True,
                has_elevation,
                tuple(int(v) for v in raw_inds[:3]) if len(raw_inds) >= 3 else None,
            )
        )

    return targets


def _build_generic_processor(rt: dict[str, Any], record: Any) -> tuple[Any, Any]:
    a2 = rt["a2"]
    ConventionalHeatmapConfig = rt["ConventionalHeatmapConfig"]
    ConventionalHeatmapProcessor = rt["ConventionalHeatmapProcessor"]

    update_rate = _record_update_rate(record)
    session_config = record.session_config
    subsweep_config = session_config.sensor_config.subsweep
    heatmap_config = ConventionalHeatmapConfig(
        antenna_array=a2.AntennaArray.AP212,
        heatmap_mode=ConventionalHeatmapConfig.HeatmapMode.MOVEMENT,
        covariance_matrix_time_constant_s=0.15,
        high_pass_filter_time_constant_s=1.0,
        az_angle_range=(-60.0, 60.0),
        el_angle_range=(-45.0, 45.0),
        num_az_angles=25,
        num_el_angles=21,
        normalized_heat_cutoff=0.0,
        max_2d_heat=None,
        log_scale_output=False,
    )
    processor = ConventionalHeatmapProcessor(
        subsweep_config=subsweep_config,
        heatmap_config=heatmap_config,
        update_rate=update_rate,
    )
    return processor, heatmap_config


def _extract_generic_candidates(
    heatmap_2d: Any,
    heatmap_config: Any,
    session_config: Any,
    np: Any,
) -> list[dict[str, Any]]:
    if heatmap_2d is None or heatmap_2d.size == 0:
        return []

    sensor_config = session_config.sensor_config
    subsweep = sensor_config.subsweep
    distances_m = (subsweep.start_point + np.arange(subsweep.num_points) * subsweep.step_length) * BASE_STEP_LENGTH_M
    az_angles_deg = np.linspace(
        float(heatmap_config.az_angle_range[0]),
        float(heatmap_config.az_angle_range[1]),
        int(heatmap_config.num_az_angles),
    )

    max_energy = float(np.max(heatmap_2d))
    if max_energy <= 0:
        return []

    threshold = max(float(np.quantile(heatmap_2d, 0.995)), max_energy * 0.55)
    candidate_peaks: list[tuple[float, int, int]] = []
    az_count, dist_count = heatmap_2d.shape

    for az_idx in range(az_count):
        for dist_idx in range(dist_count):
            value = float(heatmap_2d[az_idx, dist_idx])
            if value < threshold:
                continue

            az_slice = slice(max(0, az_idx - 1), min(az_count, az_idx + 2))
            dist_slice = slice(max(0, dist_idx - 1), min(dist_count, dist_idx + 2))
            if value < float(np.max(heatmap_2d[az_slice, dist_slice])):
                continue

            candidate_peaks.append((value, az_idx, dist_idx))

    candidate_peaks.sort(reverse=True)
    targets: list[dict[str, Any]] = []
    accepted_xy: list[tuple[float, float]] = []

    for value, az_idx, dist_idx in candidate_peaks[:20]:
        distance_m = float(distances_m[min(dist_idx, len(distances_m) - 1)])
        az_deg = float(az_angles_deg[min(az_idx, len(az_angles_deg) - 1)])
        az_rad = math.radians(az_deg)
        x = distance_m * math.cos(az_rad)
        y = distance_m * math.sin(az_rad)

        if any(math.hypot(x - prev_x, y - prev_y) < 0.25 for prev_x, prev_y in accepted_xy):
            continue

        accepted_xy.append((x, y))
        targets.append(
            _target_from_cartesian(
                (x, y, 0.0),
                value,
                True,
                False,
                (int(az_idx), 0, int(dist_idx)),
            )
        )
        if len(targets) >= 10:
            break

    return targets


def _print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2))


def _summary_to_normalized_radar_status(summary: dict[str, Any]) -> dict[str, Any]:
    targets = summary.get("targets", [])
    target = targets[0] if targets else None

    if target is None:
        radar_status = _default_radar_status()
        radar_status["presence"] = bool(summary.get("presence", False))
        return radar_status

    return {
        "presence": bool(summary.get("presence", False)),
        "distance_m": round(float(target["distance_m"]), 3),
        "angle_deg": {
            "azimuth": round(float(target["angle_deg"]["horizontal"]), 3),
            "elevation": (
                round(float(target["angle_deg"]["elevation"]), 2)
                if target["angle_deg"]["elevation"] is not None
                else None
            ),
        },
        "energy_score": round(float(target["energy"]), 3),
    }


class CombinedStatusReporter:
    def __init__(
        self,
        combined_status_store: CombinedStatusStore,
        model: str,
        console: ConsoleCoordinator,
        period_s: float = 10.0,
    ) -> None:
        self._combined_status_store = combined_status_store
        self._model = model
        self._console = console
        self._period_s = period_s
        self._enabled = True
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled

    def _run(self) -> None:
        while not self._stop_event.wait(self._period_s):
            if not self._enabled:
                continue
            report_question = (
                "Summarize the current radar and camera person-monitoring status. "
                "State whether a person is detected, whether radar and camera agree, "
                "and mention distance/angle only if radar has them."
            )
            report_time = datetime.now().strftime("%H:%M:%S")
            status = self._combined_status_store.get()
            prompt = _build_combined_status_prompt(
                question=report_question,
                status_json=status,
            )
            response = _query_ollama(self._model, prompt)
            lines = [
                f"Start LLM Generation: {report_time}",
            ]
            if response:
                lines.append(response)
            self._console.emit_lines(lines)


class CameraPersonDetector:
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        combined_status_store: CombinedStatusStore,
        cv2_module: Any,
        yolo_class: Any,
        source: str,
        model_name: str,
        confidence_threshold: float,
        enabled: bool,
    ) -> None:
        self._combined_status_store = combined_status_store
        self._cv2 = cv2_module
        self._yolo_class = yolo_class
        self._source = source
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._enabled = enabled
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_lock = threading.Lock()
        self._latest_frame: Any = None

    def start(self) -> None:
        self._combined_status_store.set_camera_source(self._source)
        self._combined_status_store.update_camera(_default_camera_status(self._source))
        if not self._enabled:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        capture = self._cv2.VideoCapture(self._source)
        if not capture.isOpened():
            print(f"[camera error] Could not open camera source {self._source}", file=sys.stderr)
            return

        try:
            model = self._yolo_class(self._model_name)
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    time.sleep(0.1)
                    continue
                with self._frame_lock:
                    self._latest_frame = frame.copy()

                status = self._detect_persons(model, frame)
                self._combined_status_store.update_camera(status)
        except Exception as exc:
            print(f"[camera error] {exc}", file=sys.stderr)
        finally:
            capture.release()

    def _detect_persons(self, model: Any, frame: Any) -> dict[str, Any]:
        result = model.predict(
            source=frame,
            verbose=False,
            conf=self._confidence_threshold,
            classes=[self.PERSON_CLASS_ID],
        )[0]
        person_confidences: list[float] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            confidences = getattr(boxes, "conf", None)
            classes = getattr(boxes, "cls", None)
            if confidences is not None and classes is not None:
                conf_values = confidences.tolist()
                class_values = classes.tolist()
                for cls_id, confidence in zip(class_values, conf_values):
                    if int(cls_id) == self.PERSON_CLASS_ID:
                        person_confidences.append(float(confidence))

        return {
            "person_detected": bool(person_confidences),
            "person_count": len(person_confidences),
            "max_confidence": round(max(person_confidences), 3) if person_confidences else None,
            "source": self._source,
        }

    def show_latest_image(self) -> tuple[bool, str]:
        if not self._enabled or self._cv2 is None:
            return False, "Camera display is unavailable."
        with self._frame_lock:
            if self._latest_frame is None:
                return False, "No camera image available yet."
            frame = self._latest_frame.copy()
        try:
            self._cv2.namedWindow("Camera Still Image", self._cv2.WINDOW_NORMAL)
            self._cv2.imshow("Camera Still Image", frame)
            while True:
                key = self._cv2.waitKey(100)
                if key != -1:
                    break
                visible = self._cv2.getWindowProperty("Camera Still Image", self._cv2.WND_PROP_VISIBLE)
                if visible < 1:
                    break
            self._cv2.destroyWindow("Camera Still Image")
            return True, "Displayed latest camera still image. Press any key in the image window to close it."
        except Exception as exc:
            try:
                output_path = os.path.join(tempfile.gettempdir(), "radar_camera_still.jpg")
                self._cv2.imwrite(output_path, frame)
                return False, f"Could not open image window ({exc}). Saved still image to {output_path}."
            except Exception:
                return False, f"Could not display still image: {exc}"

    def get_latest_frame(self) -> Any:
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()


def _build_combined_status_prompt(question: str, status_json: dict[str, Any]) -> str:
    status_text = json.dumps(status_json, ensure_ascii=False)
    radar = status_json.get("radar", {})
    camera = status_json.get("camera", {})
    fusion = status_json.get("fusion", {})
    radar_presence = bool(radar.get("presence", False))
    camera_presence = bool(camera.get("person_detected", False))
    any_person = radar_presence or camera_presence
    decision_lines = [
        f"radar_presence={str(radar_presence).lower()}",
        f"camera_person_detected={str(camera_presence).lower()}",
        f"any_person_detected={str(any_person).lower()}",
        f"fusion_person_status={fusion.get('person_status', 'unknown')}",
    ]
    return (
        "You are assisting with a fused radar and camera person monitoring console.\n"
        "Use the combined status JSON to answer the user's question.\n"
        "First classify the user's message.\n"
        "- If it is casual conversation, greeting, social talk, or a non-sensor-related request, answer naturally and ignore the sensor values.\n"
        "- Only use radar/camera/sensor values when the user is explicitly asking about presence, person detection, distance, angle, energy, camera, radar, monitoring status, or related sensor observations.\n"
        "- Do not force sensor information into casual conversation.\n"
        "You must follow the boolean status strictly.\n"
        "Decision rules:\n"
        "- If any_person_detected=true, you must say that a person is currently detected.\n"
        "- If any_person_detected=false, you must say that no person is currently detected.\n"
        "- Do not contradict radar_presence, camera_person_detected, or any_person_detected.\n"
        "Answer only about person presence, distance from the sensor in meters, "
        "position or angle in degrees, camera and radar agreement, and the radar detection energy score.\n"
        "If both radar and camera detect a person, clearly say both radar and camera detect a person.\n"
        "If only radar detects a person, clearly say radar detects a person but camera does not.\n"
        "If only camera detects a person, clearly say camera detects a person but radar does not.\n"
        "If neither detects a person, say no person is currently detected.\n"
        "Never mention non-person YOLO objects.\n"
        "Never say distance is in kilometers or generic units.\n"
        "Never describe the energy score as dB, dBm, watts, or physical power.\n"
        "For energy questions, answer with: The radar detection energy score is X.\n"
        "If the data is uncertain, say so explicitly.\n"
        "Keep the answer concise and directly useful.\n\n"
        f"Derived decision state:\n{chr(10).join(decision_lines)}\n\n"
        f"Combined status JSON:\n{status_text}\n\n"
        f"User question:\n{question}\n"
    )


def _query_ollama(model: str, prompt: str) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return f"[ollama error] {exc}"

    if result.returncode != 0:
        stderr = result.stderr.strip()
        return f"[ollama error] {stderr or f'process exited with code {result.returncode}'}"

    return result.stdout.strip()


class OllamaConsoleAgent:
    def __init__(
        self,
        combined_status_store: CombinedStatusStore,
        model: str,
        console: ConsoleCoordinator,
        reporter: CombinedStatusReporter,
        camera_detector: CameraPersonDetector,
        live_display: LiveDisplayController,
        enabled: bool,
    ) -> None:
        self._combined_status_store = combined_status_store
        self._model = model
        self._console = console
        self._reporter = reporter
        self._camera_detector = camera_detector
        self._live_display = live_display
        self._enabled = enabled
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._enabled:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._console.start_prompt()
                question = sys.stdin.readline()
                self._console.finish_prompt()
                if question == "":
                    self._stop_event.set()
                    break
                question = question.strip()
            except EOFError:
                self._stop_event.set()
                break
            except KeyboardInterrupt:
                self._stop_event.set()
                break

            if not question:
                continue
            if question.lower() in {"quit", "exit"}:
                self._stop_event.set()
                break
            if self._handle_command(question):
                continue

            combined_status = self._combined_status_store.get()
            prompt = _build_combined_status_prompt(question, combined_status)
            response = _query_ollama(self._model, prompt)
            if response:
                self._console.emit_lines([response])

    def _handle_command(self, question: str) -> bool:
        lowered = question.strip().lower()
        if lowered in {"report on", "report start", "start report"}:
            self._reporter.set_enabled(True)
            self._console.emit_lines(["Report enabled."])
            return True
        if lowered in {"report off", "report stop", "stop report"}:
            self._reporter.set_enabled(False)
            self._console.emit_lines(["Report disabled."])
            return True
        if lowered in {"live on", "display on"}:
            enabled = self._live_display.set_enabled(True)
            self._console.emit_lines(
                ["Live display enabled."] if enabled else ["Live display is unavailable."]
            )
            return True
        if lowered in {"live off", "display off"}:
            disabled = self._live_display.set_enabled(False)
            self._console.emit_lines(
                ["Live display disabled."] if disabled else ["Live display is unavailable."]
            )
            return True
        if "image" in lowered:
            shown, message = self._camera_detector.show_latest_image()
            del shown
            self._console.emit_lines([message])
            return True
        return False


def _run_recorded_mode(
    args: argparse.Namespace,
    rt: dict[str, Any],
    combined_status_store: CombinedStatusStore,
    live_display: LiveDisplayController,
    camera_detector: CameraPersonDetector,
) -> dict[str, Any]:
    a2 = rt["a2"]
    TrackingProcessor = rt["TrackingProcessor"]
    load_noise_normalization_recording = rt["load_noise_normalization_recording"]
    np = rt["numpy"]

    record = a2.open_record(args.input_file)
    final_summary = {"presence": False, "targets": []}
    suppressor = BackgroundSuppressor(np=np, alpha=args.background_alpha)
    stabilizer = TargetStabilizer(
        energy_threshold=args.energy_threshold,
        max_distance_jump_m=args.max_distance_jump_m,
        max_angle_jump_deg=args.max_angle_jump_deg,
        switch_confirm_frames=args.switch_confirm_frames,
        presence_on_frames=args.presence_on_frames,
        presence_off_frames=args.presence_off_frames,
        target_hold_frames=args.target_hold_frames,
    )

    tracking_processor = None
    try:
        algo_group = record.file["algo"]
        noise_recording = load_noise_normalization_recording(algo_group)
        try:
            processor_config = rt["TrackingProcessorConfig"].from_json(
                algo_group.get(PROCESSOR_CONFIG_DATASET_NAME, None)[()].decode()
            )
        except Exception:
            processor_config, _ = _build_tracking_setup(rt)

        tracking_processor = TrackingProcessor(
            presence_config=processor_config,
            session_config=record.session_config,
            noise_norm_recording=noise_recording,
            result_ind=0,
        )
    except Exception:
        tracking_processor = None

    if tracking_processor is not None:
        viewer = _build_viewer(
            args,
            rt,
            record.session_config,
            tracking_processor.get_heatmap_config(),
            tracking_processor.concat_subsweep_config,
            camera_frame_getter=camera_detector.get_latest_frame,
        )
        live_display.set_viewer(viewer)
        try:
            for frame_idx, result in enumerate(_iter_record_results(record), start=1):
                processor_result = tracking_processor.process(result)
                raw_heatmap = getattr(processor_result, "heatmap", None)
                suppressed_heatmap = suppressor.apply(raw_heatmap, stabilizer.protected_index)
                candidates = _extract_tracking_candidates(processor_result, suppressed_heatmap, np)
                final_summary = stabilizer.update(candidates)
                combined_status_store.update_radar(_summary_to_normalized_radar_status(final_summary))
                viewer.update_result(raw_heatmap, suppressed_heatmap, final_summary, np)
                if args.max_frames and frame_idx >= args.max_frames:
                    break
        finally:
            viewer.close()
            live_display.set_viewer(None)
        return final_summary

    generic_processor, heatmap_config = _build_generic_processor(rt, record)
    viewer = _build_viewer(
        args,
        rt,
        record.session_config,
        heatmap_config,
        record.session_config.sensor_config.subsweep,
        camera_frame_getter=camera_detector.get_latest_frame,
    )
    live_display.set_viewer(viewer)
    try:
        for frame_idx, result in enumerate(_iter_record_results(record), start=1):
            heatmap_result = generic_processor.process(result)
            raw_heatmap = getattr(heatmap_result, "heatmap", None)
            suppressed_heatmap = suppressor.apply(raw_heatmap, stabilizer.protected_index)
            heatmap_2d = _heatmap_to_2d(suppressed_heatmap, np)
            candidates = _extract_generic_candidates(
                heatmap_2d=heatmap_2d,
                heatmap_config=heatmap_config,
                session_config=record.session_config,
                np=np,
            )
            final_summary = stabilizer.update(candidates)
            combined_status_store.update_radar(_summary_to_normalized_radar_status(final_summary))
            viewer.update_result(raw_heatmap, suppressed_heatmap, final_summary, np)
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        viewer.close()
        live_display.set_viewer(None)

    return final_summary


def _store_live_algo_metadata(recorder: Any, presence_config: Any, noise_recording: Any, rt: dict[str, Any]) -> None:
    algo_group = recorder.require_algo_group(ALGO_GROUP_NAME)
    rt["save_noise_normalization_recording"](algo_group, noise_recording)
    algo_group.create_dataset(PROCESSOR_CONFIG_DATASET_NAME, data=presence_config.to_json())


def _open_client(a2: Any, prefer_mock: bool) -> Any:
    if prefer_mock:
        return a2.Client.open(mock=True)

    try:
        return a2.Client.open(usb_device=True)
    except Exception:
        return a2.Client.open(mock=True)


def _run_live_mode(
    args: argparse.Namespace,
    rt: dict[str, Any],
    combined_status_store: CombinedStatusStore,
    live_display: LiveDisplayController,
    camera_detector: CameraPersonDetector,
) -> dict[str, Any]:
    a2 = rt["a2"]
    NoiseNormalizationRecording = rt["NoiseNormalizationRecording"]
    TrackingProcessor = rt["TrackingProcessor"]
    np = rt["numpy"]

    presence_config, session_config = _build_tracking_setup(rt)
    final_summary = {"presence": False, "targets": []}
    suppressor = BackgroundSuppressor(np=np, alpha=args.background_alpha)
    stabilizer = TargetStabilizer(
        energy_threshold=args.energy_threshold,
        max_distance_jump_m=args.max_distance_jump_m,
        max_angle_jump_deg=args.max_angle_jump_deg,
        switch_confirm_frames=args.switch_confirm_frames,
        presence_on_frames=args.presence_on_frames,
        presence_off_frames=args.presence_off_frames,
        target_hold_frames=args.target_hold_frames,
    )

    with _open_client(a2, args.mock) as client:
        noise_recording = NoiseNormalizationRecording.create(client, session_config)
        processor = TrackingProcessor(
            presence_config=presence_config,
            session_config=session_config,
            noise_norm_recording=noise_recording,
            result_ind=0,
        )
        viewer = _build_viewer(
            args,
            rt,
            session_config,
            processor.get_heatmap_config(),
            processor.concat_subsweep_config,
            camera_frame_getter=camera_detector.get_latest_frame,
        )
        live_display.set_viewer(viewer)

        recorder = None
        if args.output_file:
            recorder = a2.H5Recorder(args.output_file, mode="w")
            client.attach_recorder(recorder)
            _store_live_algo_metadata(recorder, presence_config, noise_recording, rt)

        try:
            client.setup_session(session_config)
            client.start_session()

            frame_idx = 0
            interrupt_handler = rt["et"].utils.ExampleInterruptHandler()
            while not interrupt_handler.got_signal:
                group_result = client.get_next(format=a2.Client.ResultFormat.AUTO)
                processor_result = processor.process(group_result)
                raw_heatmap = getattr(processor_result, "heatmap", None)
                suppressed_heatmap = suppressor.apply(raw_heatmap, stabilizer.protected_index)
                candidates = _extract_tracking_candidates(processor_result, suppressed_heatmap, np)
                final_summary = stabilizer.update(candidates)
                combined_status_store.update_radar(_summary_to_normalized_radar_status(final_summary))
                viewer.update_result(raw_heatmap, suppressed_heatmap, final_summary, np)
                frame_idx += 1
                if args.max_frames and frame_idx >= args.max_frames:
                    break
        finally:
            viewer.close()
            live_display.set_viewer(None)
            client.stop_session()
            if recorder is not None:
                attached = client.detach_recorder()
                attached.close()

    return final_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Radar-based presence detection entrypoint.")
    parser.add_argument(
        "--input-file",
        help="Replay a recorded H5 radar file instead of running a live session.",
    )
    parser.add_argument(
        "--output-file",
        help="Optional H5 file path for storing a live session.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force Acconeer mock mode for live execution.",
    )
    parser.add_argument(
        "--visualization",
        action="store_true",
        help="Enable OpenCV heatmap rendering.",
    )
    parser.add_argument(
        "--background-alpha",
        type=float,
        default=0.005,
        help="EMA alpha for background clutter estimation. Smaller is slower.",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=8.0,
        help="Reject detections below this energy.",
    )
    parser.add_argument(
        "--max-distance-jump-m",
        type=float,
        default=0.9,
        help="Immediate target switches larger than this are rejected until confirmed.",
    )
    parser.add_argument(
        "--max-angle-jump-deg",
        type=float,
        default=15.0,
        help="Immediate azimuth/elevation jumps larger than this are rejected until confirmed.",
    )
    parser.add_argument(
        "--switch-confirm-frames",
        type=int,
        default=3,
        help="Frames required before accepting a far target switch.",
    )
    parser.add_argument(
        "--presence-on-frames",
        type=int,
        default=2,
        help="Consecutive valid detections required to assert presence.",
    )
    parser.add_argument(
        "--presence-off-frames",
        type=int,
        default=8,
        help="Consecutive missing frames required to clear presence.",
    )
    parser.add_argument(
        "--target-hold-frames",
        type=int,
        default=20,
        help="Keep the last stable target geometry this many missing frames before returning nulls.",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2:3b",
        help="Ollama model name used for console Q&A.",
    )
    parser.add_argument(
        "--no-ollama-console",
        action="store_true",
        help="Disable console question answering via Ollama.",
    )
    parser.add_argument(
        "--camera-device",
        default="/dev/video2",
        help="Camera device path or index used for person detection.",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolo11n.pt",
        help="YOLOv11 model path or name used for person detection.",
    )
    parser.add_argument(
        "--camera-confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum YOLO confidence required for person detections.",
    )
    parser.add_argument(
        "--report-period-s",
        type=float,
        default=30.0,
        help="Seconds between combined radar and camera status reports.",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Disable camera-based person detection.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        rt = _import_runtime(
            enable_visualization=args.visualization,
            enable_camera=not args.no_camera,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    combined_status_store = CombinedStatusStore()
    combined_status_store.update_radar(_default_radar_status())
    combined_status_store.update_camera(_default_camera_status(args.camera_device))
    console = ConsoleCoordinator()
    live_display = LiveDisplayController()
    camera_detector = CameraPersonDetector(
        combined_status_store=combined_status_store,
        cv2_module=rt["cv2"],
        yolo_class=rt.get("YOLO"),
        source=args.camera_device,
        model_name=args.yolo_model,
        confidence_threshold=args.camera_confidence_threshold,
        enabled=not args.no_camera,
    )
    reporter = CombinedStatusReporter(
        combined_status_store=combined_status_store,
        model=args.ollama_model,
        console=console,
        period_s=args.report_period_s,
    )
    console_agent = OllamaConsoleAgent(
        combined_status_store=combined_status_store,
        model=args.ollama_model,
        console=console,
        reporter=reporter,
        camera_detector=camera_detector,
        live_display=live_display,
        enabled=not args.no_ollama_console,
    )
    camera_detector.start()
    reporter.start()
    console_agent.start()

    try:
        if args.input_file:
            _run_recorded_mode(args, rt, combined_status_store, live_display, camera_detector)
        else:
            _run_live_mode(args, rt, combined_status_store, live_display, camera_detector)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        camera_detector.stop()
        reporter.stop()
        console_agent.stop()
        return 2

    camera_detector.stop()
    reporter.stop()
    console_agent.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
