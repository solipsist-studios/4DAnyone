"""Body pose from SAM 3D Body: the one place the model is loaded and run.

4DAnyone conditions its generator on a 70-keypoint MHR body pose, which is what SAM 3D Body
emits. ComfyUI ships the model, so this module runs in ComfyUI's interpreter and has two
callers that share ``estimate_pose``: the comfyui-cumuli nodes import it in-process, and
standalone inference runs ``python -m fdanyone.pose`` under ComfyUI's interpreter. The
result is the small npz that ``--sam3d_npz`` takes.

Adapted from ``core/four_d_anyone/pose.py`` in ComfyUI-SplatKit by mickmumpitz, which is
MIT-licensed (see ``third_party/licenses/SPLATKIT_LICENSE``).

The loader is ours for one reason that is not a preference: ComfyUI's own loader picks
float16 on any card that supports it and installs no manual cast, and with hand
refinement on the decoder feeds a float32 intermediate into those half weights and dies
on ``mat1 and mat2 must have the same dtype``. Building in float32 avoids that, at 5.6 GB
of weights instead of 2.8 GB. Everything else is the core loader, line for line.

``comfy`` is imported inside the functions that need it, so this module imports cleanly
in an interpreter without ComfyUI. It depends only on numpy, av and ``fdanyone.errors``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from fdanyone.errors import FourDAnyoneError

LOGGER = logging.getLogger("fdanyone")

#: The npz contract the generator validates (``fdanyone.pipeline.SAM3D_KEYS``).
NPZ_KEYS = ("keypoints_incam", "vertices", "cam_t", "keypoints_2d", "intrinsics", "image_size")


class PoseError(FourDAnyoneError):
    """Raised when the body pose cannot be estimated, with what to do about it."""


def core_has_sam3d() -> bool:
    """Whether this ComfyUI ships the SAM 3D Body nodes (it arrived in 0.34)."""

    try:
        import comfy_extras.nodes_sam3d_body  # noqa: F401
    except Exception:
        return False
    return True


def host_problem() -> str | None:
    """Why this interpreter cannot estimate the pose, or ``None`` when it can."""

    if not core_has_sam3d():
        return (
            "this ComfyUI does not ship the SAM 3D Body nodes (they arrived in 0.34). Update ComfyUI; "
            "the body pose cannot be estimated without them."
        )
    return None


def _read_frames(video: Path) -> np.ndarray:
    import av

    frames = []
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise PoseError(f"No frames could be decoded from {video}.")
    return np.stack(frames)


def load_model(model_file: str):
    """The core's SAM3DBody_Loader with the dtype pinned to float32 (see the module docstring)."""

    import comfy.model_management
    import comfy.model_patcher
    import comfy.ops
    import comfy.utils
    import torch
    from comfy.ldm.sam3d_body.model.model import SAM3DBody

    state_dict = comfy.utils.load_torch_file(model_file, safe_load=True)
    state_dict = {key.replace(".layers.0.0.", ".layers.0."): value for key, value in state_dict.items()}

    load_device = comfy.model_management.get_torch_device()
    dtype = torch.float32
    quant_config = comfy.utils.detect_layer_quantization(state_dict, "")
    if quant_config is not None:
        operations = comfy.ops.mixed_precision_ops(quant_config, dtype)
    else:
        operations = comfy.ops.pick_operations(dtype, None, load_device=load_device, disable_fast_fp8=True)

    model = SAM3DBody(dtype=dtype, operations=operations)
    state_dict.pop("hand_cls_embed.weight", None)
    state_dict.pop("hand_cls_embed.bias", None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise PoseError(
            f"SAM 3D Body checkpoint key mismatch: missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]}. "
            f"{model_file} is not the weights this ComfyUI's SAM 3D Body expects."
        )
    model.backbone_dtype = dtype
    return comfy.model_patcher.CoreModelPatcher(
        model,
        load_device=load_device,
        offload_device=comfy.model_management.unet_offload_device(),
        size=comfy.model_management.module_size(model),
    )


def _default_intrinsics(height: int, width: int) -> np.ndarray:
    """What the model assumes when no field of view is given: a diagonal focal length and a
    centred principal point. Recorded explicitly, because everything downstream projects with it."""

    focal = float(np.hypot(width, height))
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _is_oom(exc: BaseException) -> bool:
    """Whether ``exc`` is a CUDA out-of-memory error, in whichever form torch raised it."""

    try:
        import torch

        if isinstance(exc, torch.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    return "out of memory" in str(exc).lower()


def _release_cached_memory() -> None:
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except Exception:  # noqa: BLE001 - best effort between attempts
        LOGGER.debug("Could not empty the CUDA cache between attempts", exc_info=True)


def _under_inference_mode(fn):
    """Run ``fn`` with autograd off. ComfyUI's executor already does this around a node,
    which is why the core SAM 3D Body node never says so; called from anywhere else (a
    script, a test) the model builds a graph, keeps every activation alive, and then fails
    on ``.numpy()`` of a tensor that requires grad. Not relying on the caller is the point."""

    def wrapped(*args, **kwargs):
        import torch

        with torch.inference_mode():
            return fn(*args, **kwargs)

    return wrapped


def run_with_batch_backoff(run, start: int, *, is_oom=_is_oom, release=_release_cached_memory):
    """Call ``run(batch_size)``, halving the batch each time it runs out of memory.

    SAM 3D Body's memory grows with the crops processed together (about 29 GB at 16 crops
    on a 1088x1600 clip), and what fits depends on the clip and the card, so a fixed
    batch size is either wasteful or a crash. Only memory errors are retried, and only
    down to a batch of one; anything else, and an out-of-memory at one, is raised as it is.
    Returns ``(result, batch_size_that_worked)``.
    """

    size = max(1, int(start))
    while True:
        try:
            return run(size), size
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is a memory error we can retry
            if size == 1 or not is_oom(exc):
                raise
            next_size = max(1, size // 2)
            LOGGER.warning("SAM 3D Body ran out of memory at batch size %d; retrying at %d", size, next_size)
            size = next_size
            release()


def estimate_pose(
    video: Path,
    out_npz: Path,
    *,
    weights: Path,
    fov: float = 0.0,
    batch_size: int = 8,
    hands: bool = True,
) -> dict:
    """Run SAM 3D Body over the canonical clip and write the generator's pose npz.

    No person detector is run: the core node estimates one body from the whole frame,
    which is the pipeline's contract (exactly one person). Two people in frame give one
    blended estimate and a poor generation, not an error here.
    """

    problem = host_problem()
    if problem:
        raise PoseError(problem)
    import torch
    from comfy_extras.nodes_sam3d_body import SAM3DBody_Predict

    frames = _read_frames(Path(video))
    count, height, width, _ = frames.shape
    LOGGER.info("Estimating body pose with SAM 3D Body (%d frames at %dx%d)", count, width, height)

    patcher = load_model(str(weights))
    image = torch.from_numpy(frames).float().div_(255.0)  # ComfyUI's IMAGE convention
    pose, used_batch = run_with_batch_backoff(
        _under_inference_mode(
            lambda size: SAM3DBody_Predict.execute(
                sam3d_body_model=patcher,
                image=image,
                run_hand_refinement=hands,
                fov=float(fov),
                batch_size=size,
            ).result[0]
        ),
        batch_size,
    )
    LOGGER.info("SAM 3D Body finished at batch size %d", used_batch)

    keypoints, vertices, cam_t, keypoints_2d = [], [], [], []
    for index, people in enumerate(pose["frames"]):
        if not people:
            raise PoseError(
                f"SAM 3D Body found no body in frame {index} of the clip. The subject must stay in frame "
                "for the whole 121-frame window; pick a start_time where they do."
            )
        person = people[0]
        keypoints.append(np.asarray(person["pred_keypoints_3d"], dtype=np.float32))
        vertices.append(np.asarray(person["pred_vertices"], dtype=np.float32))
        cam_t.append(np.asarray(person["pred_cam_t"], dtype=np.float32))
        keypoints_2d.append(np.asarray(person["pred_keypoints_2d"], dtype=np.float32))

    if fov:
        from comfy_extras.sam3d_body.utils import cam_int_from_fov

        cam_int = cam_int_from_fov(int(height), int(width), float(fov))
        intrinsics = np.asarray(cam_int[0] if cam_int.ndim == 3 else cam_int, dtype=np.float64)
    else:
        intrinsics = _default_intrinsics(height, width)

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    # Written aside and moved into place: this file is a cache, and a half-written npz from an
    # interrupted run would be taken for a finished pose by the next one.
    staged = out_npz.with_name(out_npz.name + ".partial.npz")
    np.savez_compressed(
        staged,
        keypoints_incam=np.stack(keypoints),
        vertices=np.stack(vertices),
        cam_t=np.stack(cam_t),
        keypoints_2d=np.stack(keypoints_2d),
        intrinsics=intrinsics,
        image_size=np.asarray((int(height), int(width)), dtype=np.int64),
    )
    os.replace(staged, out_npz)
    # Give the card back before the generator starts; it needs nearly all of it.
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:  # noqa: BLE001 - freeing memory is best effort
        LOGGER.debug("Could not release ComfyUI models after pose estimation", exc_info=True)
    return {
        "path": str(out_npz),
        "frames": count,
        "weights": str(weights),
        "batch_size": used_batch,
        "keypoints": int(np.stack(keypoints).shape[1]),
    }


def main(argv: list[str] | None = None) -> None:
    """Standalone entry: ``python -m fdanyone.pose VIDEO OUT.npz --comfy-root DIR --weights FILE``.

    Run it with ComfyUI's interpreter. It puts the checkout on ``sys.path`` so ``comfy`` imports.
    """

    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="python -m fdanyone.pose", description=main.__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("out_npz", type=Path)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fov", type=float, default=0.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    sys.path.insert(0, str(args.comfy_root))
    try:
        estimate_pose(args.video, args.out_npz, weights=args.weights, fov=args.fov, batch_size=args.batch_size)
    except FourDAnyoneError as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
