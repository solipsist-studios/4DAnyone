import pytest

from fdanyone import pose


def test_the_pose_batch_halves_on_out_of_memory_until_it_fits():
    class FakeOOM(RuntimeError):
        pass

    attempts, released = [], []

    def run(size):
        attempts.append(size)
        if size > 3:
            raise FakeOOM("CUDA out of memory")
        return f"done at {size}"

    result, used = pose.run_with_batch_backoff(
        run, 16, is_oom=lambda exc: isinstance(exc, FakeOOM), release=lambda: released.append(1)
    )
    assert (result, used) == ("done at 2", 2) and attempts == [16, 8, 4, 2] and len(released) == 3


def test_the_pose_backoff_stops_at_one_and_never_retries_other_errors():
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        pose.run_with_batch_backoff(
            lambda size: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")), 4, release=lambda: None
        )
    calls = []

    def broken(size):
        calls.append(size)
        raise ValueError("a real bug")

    with pytest.raises(ValueError, match="a real bug"):
        pose.run_with_batch_backoff(broken, 16, release=lambda: None)
    assert calls == [16]  # not retried: it is not a memory error


def test_the_pose_model_runs_with_autograd_off_whoever_calls_it():
    torch = pytest.importorskip("torch")

    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    assert pose._under_inference_mode(lambda: torch.is_inference_mode_enabled())() is True
