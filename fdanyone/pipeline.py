"""Top-level inference orchestration."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from fdanyone.assets import (
    CHECKPOINT,
    HF_REPO_ID,
    HF_REVISION,
    TURBO_LORA,
    TURBO_LORA_NAME,
    TURBO_LORA_SHA256,
    resolve_base_assets,
    resolve_checkpoint,
    resolve_foreground_model,
    resolve_turbo_lora,
)
from fdanyone.config import BASE24, INFERENCE, RANK64_DELTA4
from fdanyone.device import CUDA_ALLOCATOR_CONF, select_cuda_devices
from fdanyone.errors import ConfigurationError
from fdanyone.io import remove_tree, resolve_output_path, write_json
from fdanyone.motion.result import MotionResult
from fdanyone.output_directory import OutputDirectory
from fdanyone.run_request import save_run_request
from fdanyone.video import (
    decode_canonical_clip,
    validate_clip_options,
    validate_required_video_codecs,
    verify_lossless_video,
    write_working_video,
)
from fdanyone.views import ViewPlan, resolve_view_plan

LOGGER = logging.getLogger("fdanyone")
PROGRESS = logging.getLogger("fdanyone.progress")
PROGRESS.addHandler(logging.NullHandler())
PROGRESS.propagate = False


def _resolve_output_dir(output_dir: str | None, video_path: str) -> Path:
    """Resolve the output directory without following its final path component."""

    target = Path("data/fdanyone") / Path(video_path).stem if output_dir is None else Path(output_dir)
    return resolve_output_path(target)


def _discard_scratch(path: Path) -> None:
    """Best-effort cleanup that can never invalidate a published result.

    Some network filesystems keep an open, hidden tombstone after a file is
    unlinked.  Such a tombstone may remain ``EBUSY`` until this process exits,
    so cleanup must not be part of the atomic publication transaction.
    """

    try:
        remove_tree(path)
    except OSError as exc:
        LOGGER.warning(
            "Could not remove temporary files at %s (%s). "
            "The result is unaffected; the hidden scratch directory can be removed after this process exits.",
            path,
            exc,
        )


def _worker_environment() -> dict[str, str]:
    """Give the short-lived workers this checkout and stable CUDA flags."""

    environment = os.environ.copy()
    environment.update(
        {
            "TORCH_CUDNN_V8_API_DISABLED": "1",
            "CUDNN_FRONTEND_DISABLE": "1",
            "CUDNN_LOGINFO_DBG": "0",
            "CUDNN_LOGDEST_DBG": "stderr",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NVIDIA_TF32_OVERRIDE": "0",
        }
    )
    environment.pop("PYTHONHOME", None)
    # The allocator split policy is specific to the long-lived DiT process; these
    # short-lived preprocessing workers use unrelated allocation shapes.
    environment.pop(CUDA_ALLOCATOR_CONF, None)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    return environment



def _build_conditioning(
    *,
    working_video: Path,
    clip_metadata: Path,
    motion_result_dir: Path,
    view_plan: ViewPlan,
    output_dir: Path,
    foreground_model: Path,
    device: str,
    worker_python: str,
    sam_keypoints: Path | None = None,
):
    from fdanyone.skeleton.pipeline import Conditioning

    request_path = output_dir.parent / ".skeleton-worker-request.json"
    write_json(
        request_path,
        {
            "working_video": str(working_video),
            "clip_metadata": str(clip_metadata),
            "motion_result_dir": str(motion_result_dir),
            "foreground_model_path": str(foreground_model),
            "output_dir": str(output_dir),
            "device": device,
            "view_plan": view_plan.to_dict(),
            # The tools/sam3d_mhr70.py npz holding this clip's MHR70 body pose.
            "sam_keypoints": str(sam_keypoints) if sam_keypoints else None,
        },
    )
    try:
        subprocess.run(
            [
                worker_python,
                "-m",
                "fdanyone.skeleton.worker",
                str(request_path),
            ],
            check=True,
            env=_worker_environment(),
        )
    finally:
        request_path.unlink(missing_ok=True)
    return Conditioning.load(output_dir)



SAM3D_KEYS = ("keypoints_incam", "vertices", "cam_t", "keypoints_2d", "intrinsics",
              "image_size")


def _accept_supplied_pose(supplied: Path, output_npz: Path, frame_count: int) -> Path:
    """Take a pose npz estimated by our caller and park it in this clip's cache.

    A caller that already has SAM 3D Body loaded, such as the ComfyUI node running inside a
    ComfyUI that ships the model, can estimate the pose itself and hand it over. That saves
    a second copy of a 2.83 GB model in a second process, and it is the reason the estimator
    no longer has to be configured with an interpreter and a checkout.

    The handover is checked rather than trusted: a caller that estimated pose on the wrong
    frames would misalign the whole conditioning stage silently.
    """
    import numpy as np

    if not supplied.is_file():
        raise ConfigurationError(f"--sam3d_npz does not exist: {supplied}")
    with np.load(supplied) as data:
        missing = [key for key in SAM3D_KEYS if key not in data]
        if missing:
            raise ConfigurationError(
                f"--sam3d_npz is missing {', '.join(missing)}: {supplied}")
        frames = int(data["keypoints_incam"].shape[0])
        joints = int(data["keypoints_incam"].shape[1])
    if joints != 70:
        raise ConfigurationError(
            f"--sam3d_npz holds {joints} keypoints, expected the 70 of MHR70: {supplied}")
    if frames != frame_count:
        raise ConfigurationError(
            f"--sam3d_npz covers {frames} frames but this clip has {frame_count}. It was "
            "estimated on different frames, so it cannot be used for this run.")
    supplied = supplied.resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    if supplied != output_npz.resolve():
        shutil.copyfile(supplied, output_npz)
    LOGGER.info("Using SAM 3D Body pose supplied by the caller (%d frames)", frames)
    return output_npz


def _run_sam3d(*, working_video: Path, output_npz: Path, device: str,
               supplied_npz: Path | None = None, frame_count: int = 0) -> Path:
    """Estimate MHR70 body pose with SAM 3D Body, replacing the GVHMR motion stage.

    Three ways in, in order of preference: a pose the caller already estimated, a cached one
    from an earlier run of this clip, or our own run in a ComfyUI interpreter. The cache
    means re-running a clip with different view settings does not re-estimate pose.
    """
    from fdanyone.config import SAM3D

    if supplied_npz is not None:
        return _accept_supplied_pose(supplied_npz, output_npz, frame_count)

    if output_npz.is_file():
        LOGGER.info("Reusing SAM 3D Body pose at %s", output_npz)
        return output_npz

    comfy_root, python = SAM3D.comfy_root, SAM3D.python
    weights = SAM3D.resolved_weights()
    missing = [name for name, value in (("FDANYONE_SAM3D_COMFY_ROOT", comfy_root),
                                        ("FDANYONE_SAM3D_PYTHON", python)) if not value]
    if missing:
        raise ConfigurationError(
            "SAM 3D Body is not configured. Set " + " and ".join(missing) + ". "
            "The model ships inside ComfyUI, so these point at a ComfyUI checkout and its "
            "interpreter; weights default to models/detection/ inside that checkout and can "
            "be overridden with FDANYONE_SAM3D_WEIGHTS. A caller that has the model loaded "
            "already can skip all of this by passing --sam3d_npz instead."
        )
    for label, value in (("interpreter", python), ("weights", weights)):
        if not Path(value).is_file():
            raise ConfigurationError(f"SAM 3D Body {label} not found: {value}")

    script = Path(__file__).resolve().parent.parent / "tools" / "sam3d_mhr70.py"
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Estimating body pose with SAM 3D Body")
    subprocess.run(
        [python, str(script), str(working_video), str(output_npz),
         "--comfy-root", comfy_root, "--weights", weights,
         "--batch-size", str(SAM3D.batch_size), "--fov", str(SAM3D.fov_degrees)],
        check=True,
    )
    if not output_npz.is_file():
        raise ConfigurationError(f"SAM 3D Body produced no output at {output_npz}")
    return output_npz


def prepare_clip_only(
    *,
    video_path: str,
    output_dir: str,
    start_time: float,
    target_fps: str | int | float,
    pad_short: bool = False,
) -> dict:
    """Decode this clip's canonical frames into ``output_dir``, then stop.

    The pose stage does not run on the source file but on the canonical clip: the frames
    selected by start_time, the target frame rate and the 121-frame contract. A caller that
    wants to estimate pose itself therefore needs that exact clip, which used to exist only
    inside a scratch directory for the length of a run.

    Everything here is cheap: a decode and a lossless re-encode, no models and no GPU. A
    clip already in ``output_dir`` is reused only when it matches the source file's identity
    *and* the requested start time and frame rate, so neither a re-encode of the input nor a
    changed window can be served a stale clip. This does not publish a 4DAnyone result, so it
    does not use the locked output protocol.
    """
    import json
    from fractions import Fraction

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    out = Path(output_dir).expanduser()
    clip_video = out / "canonical_clip.mp4"
    clip_metadata = out / "canonical_clip.json"

    canonical_fps = None if str(target_fps).lower() == "auto" else target_fps
    if clip_video.is_file() and clip_metadata.is_file():
        source = Path(video_path).expanduser().resolve()
        try:
            cached = json.loads(clip_metadata.read_text())
            stat = source.stat()
            same_start = abs(
                float(Fraction(cached["start_time_num"], cached["start_time_den"])) - float(start_time)
            ) < 1e-6
            same_rate = canonical_fps is None or abs(
                float(Fraction(cached["fps_num"], cached["fps_den"])) - float(Fraction(str(canonical_fps)))
            ) < 1e-6
            fresh = (
                cached.get("source_path") == source.name
                and cached.get("source_size_bytes") == stat.st_size
                and cached.get("source_mtime_ns") == stat.st_mtime_ns
                and cached.get("num_frames") == INFERENCE.num_frames
                and same_start
                and same_rate
            )
        except (OSError, ValueError, KeyError, ZeroDivisionError):
            fresh = False
        if fresh:
            LOGGER.info("Reusing canonical clip at %s", clip_video)
            return {"working_video": str(clip_video), "clip_metadata": str(clip_metadata),
                    "num_frames": int(cached["num_frames"]), "reused": True}

    validate_required_video_codecs()
    if not Path(video_path).is_file():
        raise ConfigurationError(f"Input video does not exist: {video_path}")
    clip = decode_canonical_clip(
        video_path,
        num_frames=INFERENCE.num_frames,
        start_time=start_time,
        fps=canonical_fps,
        pad_short=pad_short,
    )
    out.mkdir(parents=True, exist_ok=True)
    clip.write_metadata(clip_metadata)
    write_working_video(clip, clip_video)
    LOGGER.info("Canonical clip written to %s (%d frames)", clip_video, len(clip.frames))
    return {"working_video": str(clip_video), "clip_metadata": str(clip_metadata),
            "num_frames": len(clip.frames), "reused": False}


def run_pipeline(
    *,
    video_path: str,
    output_dir: str | None,
    model_dir: str,
    checkpoint_path: str | None,
    gpu_ids: list[int] | None,
    attention_backend: str,
    start_time: float,
    target_fps: str | int | float,
    seed: int,
    views_per_layer: int,
    layer_pitches: list[int],
    start_yaw: int,
    yaw_span: int,
    enable_rcp: bool,
    enable_tcr: bool,
    enable_turbo: bool,
    sam3d_npz: str | None = None,
    pad_short: bool = False,
    vae_path: str | None = None,
    prompt_context_path: str | None = None,
    foreground_model_dir: str | None = None,
    turbo_lora_path: str | None = None,
) -> dict:
    """Execute inference for one clip. Body pose comes from SAM 3D Body (``sam3d_npz``)."""

    request_options = locals().copy()
    pipeline_started = time.monotonic()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    PROGRESS.info("Checking input and settings", extra={"fraction": 0.02})
    if seed < 0:
        raise ConfigurationError(f"seed must be non-negative, got {seed}.")
    if not isinstance(enable_turbo, bool):
        raise ConfigurationError(f"enable_turbo must be True or False, got {enable_turbo!r}.")
    denoising_profile = RANK64_DELTA4 if enable_turbo else BASE24
    view_plan = resolve_view_plan(
        views_per_layer=views_per_layer,
        layer_pitches=layer_pitches,
        start_yaw=start_yaw,
        yaw_span=yaw_span,
        enable_rcp=enable_rcp,
        enable_tcr=enable_tcr,
    )
    canonical_fps = validate_clip_options(
        start_time=start_time,
        fps=None if str(target_fps).lower() == "auto" else target_fps,
    )
    checkpoint = resolve_checkpoint(checkpoint_path, model_dir=model_dir) if checkpoint_path is not None else None
    destination = _resolve_output_dir(output_dir, video_path)
    clip_name = Path(video_path).stem
    output = OutputDirectory(destination)
    # Check before downloads and decoding, then again when locking the output.
    output.validate_available()
    validate_required_video_codecs()
    devices = select_cuda_devices(gpu_ids)
    device = devices[0]

    from fdanyone.vendor.diffsynth.models.wan_video_dit import get_attention_backend

    # Resolve once, before downloading assets or preparing conditioning. Every
    # DiT, including spawned replicas, receives this concrete backend.
    attention_backend = get_attention_backend(attention_backend)
    LOGGER.info("Using attention backend: %s", attention_backend)

    PROGRESS.info("Preparing model assets", extra={"fraction": 0.05})
    if not Path(video_path).is_file():
        raise ConfigurationError(f"Input video does not exist: {video_path}")
    turbo_lora = resolve_turbo_lora(model_dir, path=turbo_lora_path) if enable_turbo else None
    worker_python = os.path.abspath(sys.executable)

    if checkpoint is None:
        checkpoint = resolve_checkpoint(model_dir=model_dir)
    base_assets = resolve_base_assets(model_dir, vae_path=vae_path, prompt_context_path=prompt_context_path)
    foreground_model = resolve_foreground_model(model_dir, path=foreground_model_dir)
    PROGRESS.info("Preparing the 121-frame clip", extra={"fraction": 0.10})
    clip = decode_canonical_clip(
        video_path,
        num_frames=INFERENCE.num_frames,
        start_time=start_time,
        fps=canonical_fps,
        pad_short=pad_short,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".{clip_name}.scratch-", dir=destination.parent))
    try:
        clip_metadata = scratch / "canonical_clip.json"
        clip.write_metadata(clip_metadata)
        working_video = write_working_video(clip, scratch / "canonical_clip.mp4")

        with output.stage() as work:
            PROGRESS.info("Reading SAM 3D Body pose", extra={"fraction": 0.15})
            sam_keypoints = _run_sam3d(
                working_video=working_video,
                output_npz=scratch / "sam3d_mhr70.npz",
                device=device,
                supplied_npz=Path(sam3d_npz) if sam3d_npz else None,
                frame_count=len(clip.frames),
            )
            # Cheap to derive from the pose, so it is rebuilt every run rather than
            # trusted from a previous one: the pose npz is the reusable artifact.
            import numpy as _np

            from fdanyone.skeleton.sam3d import motion_from_sam

            motion = motion_from_sam(_np.load(sam_keypoints), len(clip.frames))
            motion.validate_against_clip(clip)
            save_run_request(destination, request_options)
            if output.motion_dir.exists():
                LOGGER.info("Keeping the motion record already at %s", output.motion_dir)
            else:
                output.save_motion(motion)

            # Record the published identity only for the published checkpoint; an
            # explicit override must not claim the frozen Hugging Face coordinates.
            if checkpoint_path is None:
                model_identity = {"checkpoint": CHECKPOINT, "repo_id": HF_REPO_ID, "revision": HF_REVISION}
            else:
                model_identity = {"checkpoint": checkpoint.name, "source": "local_override"}
            if turbo_lora is not None:
                model_identity["turbo_lora"] = {
                    "name": TURBO_LORA_NAME,
                    "file": TURBO_LORA,
                    "sha256": TURBO_LORA_SHA256,
                }

            # Heavy rendering and generation are imported only after the motion
            # contract has been materialized, keeping CLI/help and CPU tests light.
            from fdanyone.model.inference import generate_views
            from fdanyone.output_writer import write_output

            PROGRESS.info("Building foreground masks and skeletons", extra={"fraction": 0.30})
            conditioning = _build_conditioning(
                working_video=working_video,
                clip_metadata=clip_metadata,
                motion_result_dir=output.motion_dir,
                view_plan=view_plan,
                output_dir=scratch / "conditioning",
                foreground_model=foreground_model,
                device=device,
                worker_python=worker_python,
                sam_keypoints=sam_keypoints,
            )
            if conditioning.num_frames != len(clip.frames) or (
                conditioning.fps_num,
                conditioning.fps_den,
            ) != (
                clip.fps_num,
                clip.fps_den,
            ):
                raise ConfigurationError("Skeleton conditioning does not match the canonical clip timeline.")
            # Re-decode the worker-produced source before it becomes a model tensor.
            verify_lossless_video(clip, conditioning.source_video)
            # Generation reads the verified working video. Only lightweight
            # provenance is needed from the original full-resolution clip now.
            clip_info = clip.info
            del clip
            PROGRESS.info("Generating target-view videos", extra={"fraction": 0.45})
            generated = generate_views(
                conditioning=conditioning,
                checkpoint_path=checkpoint,
                turbo_lora_path=turbo_lora,
                denoising_profile=denoising_profile,
                attention_backend=attention_backend,
                assets=base_assets,
                output_dir=scratch / "generation",
                devices=devices,
                seed=seed,
            )
            PROGRESS.info("Saving and validating generated videos", extra={"fraction": 0.92})
            summary = write_output(
                clip=clip_info,
                conditioning=conditioning,
                generated=generated,
                destination=work,
                motion=motion,
                model_identity=model_identity,
                pipeline_started=pipeline_started,
            )
    finally:
        _discard_scratch(scratch)
    summary["output_dir"] = str(destination)
    PROGRESS.info("Inference complete", extra={"fraction": 1.0})
    return summary
