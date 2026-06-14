from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import zipfile
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


def _import_runtime(enable_visualization: bool) -> dict[str, Any]:
    _ensure_bundled_exptool()

    missing: list[str] = []
    imported: dict[str, Any] = {}

    for module_name in ["numpy", "h5py", "scipy", "attrs"]:
        try:
            imported[module_name] = __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if enable_visualization:
        try:
            imported["cv2"] = __import__("cv2")
        except ImportError:
            imported["cv2"] = None
    else:
        imported["cv2"] = None

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
    def update_result(self, processor_result: Any, np: Any) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class OpenCvHeatmapViewer(HeatmapViewer):
    def __init__(self, cv2_module: Any, enabled: bool) -> None:
        self._cv2 = cv2_module
        self._enabled = enabled and cv2_module is not None
        self._window_name = "Radar Heatmap"
        self._window_ready = False

    def update_result(self, processor_result: Any, np: Any) -> None:
        heatmap_2d = _heatmap_to_2d(getattr(processor_result, "heatmap", None), np)
        if not self._enabled or heatmap_2d is None:
            return

        cv2 = self._cv2
        try:
            frame = heatmap_2d.T
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame = frame.astype("uint8")
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            frame = cv2.resize(frame, None, fx=8, fy=4, interpolation=cv2.INTER_NEAREST)

            if not self._window_ready:
                cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                self._window_ready = True

            cv2.imshow(self._window_name, frame)
            cv2.waitKey(1)
        except Exception:
            self._enabled = False

    def close(self) -> None:
        if self._enabled and self._window_ready:
            try:
                self._cv2.destroyWindow(self._window_name)
            except Exception:
                pass


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
) -> HeatmapViewer:
    if args.no_visualization:
        return OpenCvHeatmapViewer(rt["cv2"], enabled=False)

    if rt.get("heatmap_visualization") is not None:
        try:
            view_session_config = _make_heatmap_view_session_config(session_config, subsweep_config)
            return PyqtgraphHeatmapViewer(rt, view_session_config, heatmap_config)
        except Exception:
            pass

    return OpenCvHeatmapViewer(rt["cv2"], enabled=True)


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
        num_el_angles=1,
        el_angle_range=(-20, 20),
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
        room_dimensions_m=(4.0, 3.0, 0.0),
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


def _target_from_cartesian(
    xyz: tuple[float, float, float],
    energy: float,
    energy_scale: float,
    is_presence: bool,
) -> dict[str, Any]:
    x, y, z = xyz
    xy_norm = math.hypot(x, y)
    return {
        "presence": bool(is_presence),
        "distance_m": round(math.sqrt(x * x + y * y + z * z), 3),
        "position_xy": [round(x, 3), round(y, 3)],
        "angle_deg": {
            "horizontal": round(math.degrees(math.atan2(y, x)), 3),
            "elevation": round(math.degrees(math.atan2(z, max(xy_norm, 1e-9))), 3),
        },
        "energy": round(float(energy / max(energy_scale, 1e-9)), 3),
    }


def _summarize_tracking_result(result: Any) -> dict[str, Any]:
    observations = list(getattr(result, "current_obs_m", []))
    tracker_positions = [tuple(map(float, pos)) for pos in getattr(result, "tracker_positions_m", [])]
    max_energy = max((float(obs[2]) for obs in observations), default=1.0)

    unmatched = set(range(len(observations)))
    targets: list[dict[str, Any]] = []

    for tracker_x, tracker_y in tracker_positions:
        matched_idx = None
        matched_dist = float("inf")
        for obs_idx in unmatched:
            (obs_x, obs_y, obs_z), _, _ = observations[obs_idx]
            dist = math.hypot(obs_x - tracker_x, obs_y - tracker_y)
            if dist < matched_dist:
                matched_dist = dist
                matched_idx = obs_idx

        if matched_idx is None:
            targets.append(
                _target_from_cartesian((tracker_x, tracker_y, 0.0), max_energy, max_energy, True)
            )
            continue

        unmatched.remove(matched_idx)
        (obs_x, obs_y, obs_z), _, obs_energy = observations[matched_idx]
        targets.append(
            _target_from_cartesian(
                (tracker_x, tracker_y, float(obs_z)),
                float(obs_energy),
                max_energy,
                True,
            )
        )

    for obs_idx in sorted(unmatched):
        (obs_x, obs_y, obs_z), _, obs_energy = observations[obs_idx]
        targets.append(
            _target_from_cartesian(
                (float(obs_x), float(obs_y), float(obs_z)),
                float(obs_energy),
                max_energy,
                False,
            )
        )

    return {
        "presence": any(target["presence"] for target in targets) or bool(getattr(result, "num_people", 0)),
        "targets": targets,
    }


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
        el_angle_range=(0.0, 0.0),
        num_az_angles=25,
        num_el_angles=1,
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


def _detect_generic_targets(
    heatmap_2d: Any,
    heatmap_config: Any,
    session_config: Any,
    np: Any,
) -> dict[str, Any]:
    if heatmap_2d is None or heatmap_2d.size == 0:
        return {"presence": False, "targets": []}

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
        return {"presence": False, "targets": []}

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
        targets.append(_target_from_cartesian((x, y, 0.0), value, max_energy, True))
        if len(targets) >= 10:
            break

    return {
        "presence": bool(targets),
        "targets": targets,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2))


def _run_recorded_mode(args: argparse.Namespace, rt: dict[str, Any]) -> dict[str, Any]:
    a2 = rt["a2"]
    TrackingProcessor = rt["TrackingProcessor"]
    load_noise_normalization_recording = rt["load_noise_normalization_recording"]

    record = a2.open_record(args.input_file)
    final_summary = {"presence": False, "targets": []}

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
        )
        try:
            for frame_idx, result in enumerate(_iter_record_results(record), start=1):
                processor_result = tracking_processor.process(result)
                viewer.update_result(processor_result, rt["numpy"])
                final_summary = _summarize_tracking_result(processor_result)
                if args.max_frames and frame_idx >= args.max_frames:
                    break
        finally:
            viewer.close()
        return final_summary

    generic_processor, heatmap_config = _build_generic_processor(rt, record)
    viewer = _build_viewer(
        args,
        rt,
        record.session_config,
        heatmap_config,
        record.session_config.sensor_config.subsweep,
    )
    try:
        for frame_idx, result in enumerate(_iter_record_results(record), start=1):
            heatmap_result = generic_processor.process(result)
            heatmap_2d = _heatmap_to_2d(heatmap_result.heatmap, rt["numpy"])
            viewer.update_result(heatmap_result, rt["numpy"])
            final_summary = _detect_generic_targets(
                heatmap_2d=heatmap_2d,
                heatmap_config=heatmap_config,
                session_config=record.session_config,
                np=rt["numpy"],
            )
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        viewer.close()

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


def _run_live_mode(args: argparse.Namespace, rt: dict[str, Any]) -> dict[str, Any]:
    a2 = rt["a2"]
    NoiseNormalizationRecording = rt["NoiseNormalizationRecording"]
    TrackingProcessor = rt["TrackingProcessor"]

    presence_config, session_config = _build_tracking_setup(rt)
    final_summary = {"presence": False, "targets": []}

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
        )

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
                viewer.update_result(processor_result, rt["numpy"])
                final_summary = _summarize_tracking_result(processor_result)
                frame_idx += 1
                if args.max_frames and frame_idx >= args.max_frames:
                    break
        finally:
            viewer.close()
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
        "--no-visualization",
        action="store_true",
        help="Disable OpenCV heatmap rendering.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        rt = _import_runtime(enable_visualization=not args.no_visualization)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        if args.input_file:
            summary = _run_recorded_mode(args, rt)
        else:
            summary = _run_live_mode(args, rt)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
