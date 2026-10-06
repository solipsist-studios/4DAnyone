"""Run 4DAnyone inference from a monocular video."""

from __future__ import annotations

import sys

from fdanyone.attention import validate_attention_backend
from fdanyone.device import configure_inference_cuda_allocator, has_low_memory_gpu
from fdanyone.errors import ConfigurationError, FourDAnyoneError


def inference(
    video_path: str,
    output_dir: str | None = None,
    views_per_layer: int = 24,
    layer_pitches: list[int] = [15],  # noqa: B006 - normalized without mutation
    start_yaw: int = 0,
    yaw_span: int = 360,
    enable_rcp: bool = True,
    enable_tcr: bool = True,
    enable_turbo: bool = True,
    model_dir: str = "models",
    checkpoint_path: str | None = None,
    gpu_ids: list[int] | None = None,
    attention_backend: str = "auto",
    target_fps: str | int | float = "auto",
    start_time: float = 0.0,
    seed: int = 42,
    sam3d_npz: str | None = None,
    prepare_only: bool = False,
    pad_short: bool = False,
    vae_path: str | None = None,
    prompt_context_path: str | None = None,
    foreground_model_dir: str | None = None,
    turbo_lora_path: str | None = None,
) -> dict:
    """Generate synchronized target-view videos from one monocular video.

    Args:
        video_path: Input video; it must contain at least 121 usable frames.
        output_dir: Output directory for this clip. Defaults to data/fdanyone/<clip>.
        views_per_layer: Number of evenly spaced yaw views at each pitch.
            Total views (this number times the layer count) must be divisible by 6.
        layer_pitches: Camera pitch for each layer in degrees, for example
            [-10,15,35]. Positive values place the camera above the subject;
            each value must be between -15 and 45.
        start_yaw: First yaw in every layer, in degrees; 0 faces the person.
        yaw_span: Angular range sampled by each layer, from 1 to 360 degrees.
            The end angle is excluded so a full ring never duplicates a view.
        enable_rcp: Use six proposal views before generating more than six targets.
        enable_tcr: Shift view groups cyclically between denoising steps.
        enable_turbo: Whether to use 4DAnyone-Turbo for accelerated denoising.
            Disable it to use the base 4DAnyone model.
        model_dir: Local model root. Missing weights must be installed manually.
        checkpoint_path: Local 4DAnyone checkpoint override.
        gpu_ids: GPU IDs used for parallel pose/VAE view stages and target
            denoising. Omit to use all visible GPUs.
        attention_backend: auto uses SDPA when any selected GPU has at most
            24 GiB; otherwise it prefers FlashAttention-3, then SageAttention,
            then SDPA. Override with sdpa, sageattention, or flash_attn_3.
        target_fps: auto preserves the input clock unless it divides evenly
            to 24, 25, or 30 FPS; a positive number requests an explicit FPS.
        start_time: Clip start time on the input timeline, in seconds.
        seed: Random seed shared by proposal and target generation.
        sam3d_npz: Body pose (SAM 3D Body, MHR70) for this clip. It must cover exactly
            the canonical clip's frames, and it is checked. A caller that already has
            SAM 3D Body loaded, such as a ComfyUI node inside a ComfyUI that ships the
            model, estimates the pose itself and passes the result here.
        prepare_only: Decode this clip's canonical frames, write them into
            ``output_dir`` (``canonical_clip.mp4`` and ``canonical_clip.json``), and
            stop. This is how a caller gets the exact frames the pose must be
            estimated on; it needs no models and no GPU, and it does not create a
            4DAnyone result.
        pad_short: Hold the last frame to reach the frame contract instead of
            refusing an input that is too short.
        vae_path, prompt_context_path, foreground_model_dir, turbo_lora_path:
            Explicit local model files, instead of the layout under ``model_dir``.
    """

    # This must run before the first model/PyTorch import. It protects the
    # reusable 5--6 GiB DiT FFN allocation from allocator fragmentation.
    validate_attention_backend(attention_backend)
    low_memory = has_low_memory_gpu(gpu_ids)
    configure_inference_cuda_allocator(use_expandable_segments=low_memory)
    if attention_backend == "auto" and low_memory:
        attention_backend = "sdpa"
    # Keep model imports out of module scope so ``--help`` stays lightweight.
    from fdanyone.pipeline import prepare_clip_only, run_pipeline

    if prepare_only:
        # Before anything heavy: this is how a caller gets the canonical clip without
        # paying for model resolution or a GPU.
        if output_dir is None:
            raise ConfigurationError("--prepare_only needs --output_dir: it is where the canonical clip is written.")
        return prepare_clip_only(
            video_path=video_path,
            output_dir=output_dir,
            start_time=start_time,
            target_fps=target_fps,
            pad_short=pad_short,
        )

    return run_pipeline(
        video_path=video_path,
        output_dir=output_dir,
        views_per_layer=views_per_layer,
        layer_pitches=layer_pitches,
        start_yaw=start_yaw,
        yaw_span=yaw_span,
        enable_rcp=enable_rcp,
        enable_tcr=enable_tcr,
        enable_turbo=enable_turbo,
        model_dir=model_dir,
        checkpoint_path=checkpoint_path,
        vae_path=vae_path,
        prompt_context_path=prompt_context_path,
        foreground_model_dir=foreground_model_dir,
        turbo_lora_path=turbo_lora_path,
        gpu_ids=gpu_ids,
        attention_backend=attention_backend,
        target_fps=target_fps,
        start_time=start_time,
        seed=seed,
        sam3d_npz=sam3d_npz,
        pad_short=pad_short,
    )


def main() -> None:
    """Bootstrap the CLI without importing Fire or PyTorch at module import."""

    configure_inference_cuda_allocator()
    from fire import Fire

    try:
        Fire(inference)
    except FourDAnyoneError as exc:
        message = " ".join(line.strip() for line in str(exc).splitlines())
        print(f"error: {message}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
