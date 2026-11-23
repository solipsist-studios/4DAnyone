import pytest
import torch
from safetensors.torch import save_file

from fdanyone import assets
from fdanyone.errors import AssetError
from fdanyone.model import loader
from fdanyone.weights_io import load_state_dict

STATE = {
    "pose_encoder.a.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    "blocks.0.weight": torch.ones(2, 2, dtype=torch.float32),
    "head.bias": torch.zeros(3, dtype=torch.float32),
}


@pytest.fixture
def both(tmp_path):
    torch.save(STATE, tmp_path / "w.pth")
    save_file(STATE, str(tmp_path / "w.safetensors"), metadata={"format": "pt"})
    return tmp_path / "w.pth", tmp_path / "w.safetensors"


def _assert_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert left[key].dtype == right[key].dtype and torch.equal(left[key], right[key])


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, set(STATE)),
        ({"include_prefix": "pose_encoder."}, {"pose_encoder.a.weight"}),
        ({"exclude_prefixes": ("pose_encoder.",)}, {"blocks.0.weight", "head.bias"}),
        ({"include_prefix": "blocks.", "exclude_prefixes": ("blocks.0.",)}, set()),
    ],
)
def test_both_formats_return_equal_tensors(both, kwargs, expected):
    pth, st = both
    from_pth, meta_pth = load_state_dict(pth, **kwargs)
    from_st, meta_st = load_state_dict(st, **kwargs)
    assert set(from_pth) == expected
    _assert_equal(from_pth, from_st)
    assert meta_pth == {} and meta_st == {"format": "pt"}


def test_model_state_wrapper_is_unwrapped(tmp_path):
    torch.save({"model_state": STATE, "epoch": 3}, tmp_path / "w.pt")
    state, _ = load_state_dict(tmp_path / "w.pt")
    _assert_equal(state, STATE)


def test_unknown_extension_and_missing_file_raise_asset_error(tmp_path):
    (tmp_path / "w.ckpt").write_bytes(b"x")
    with pytest.raises(AssetError, match=r"\.safetensors.*\.pth"):
        load_state_dict(tmp_path / "w.ckpt")
    with pytest.raises(AssetError, match="does not exist"):
        load_state_dict(tmp_path / "missing.safetensors")


def test_pickles_that_need_code_are_refused(tmp_path):
    class Evil:
        def __reduce__(self):
            return (print, ("run",))

    torch.save({"x": Evil()}, tmp_path / "w.pth", pickle_protocol=2)
    with pytest.raises(AssetError, match="Could not read weights"):
        load_state_dict(tmp_path / "w.pth")


def test_resolve_base_assets_finds_the_vae_in_either_format(tmp_path):
    root = tmp_path / "4danyone"
    root.mkdir()
    (root / "prompt_context.safetensors").write_bytes(b"x")
    with pytest.raises(AssetError, match=r"Wan2\.2_VAE\.pth or Wan2\.2_VAE\.safetensors"):
        assets.resolve_base_assets(tmp_path)
    (root / assets.WAN_VAE_SAFETENSORS).write_bytes(b"x")
    assert assets.resolve_base_assets(tmp_path).vae.name == assets.WAN_VAE_SAFETENSORS
    (root / assets.WAN_VAE).write_bytes(b"x")  # both present: the .pth wins
    assert assets.resolve_base_assets(tmp_path).vae.name == assets.WAN_VAE


def test_an_explicit_vae_path_may_be_either_format_but_not_another(tmp_path):
    (tmp_path / "prompt_context.safetensors").write_bytes(b"x")
    (tmp_path / "comfy.safetensors").write_bytes(b"x")
    (tmp_path / "vae.zip").write_bytes(b"x")
    found = assets.resolve_base_assets(tmp_path, vae_path=tmp_path / "comfy.safetensors",
                                       prompt_context_path=tmp_path / "prompt_context.safetensors")
    assert found.vae.name == "comfy.safetensors"
    with pytest.raises(AssetError, match="must end in"):
        assets.resolve_base_assets(tmp_path, vae_path=tmp_path / "vae.zip")


def test_load_vae_accepts_both_formats(both, monkeypatch):
    from fdanyone.vendor.diffsynth.models import wan_video_vae as vae_module

    seen = []
    converter = vae_module.WanVideoVAE38.state_dict_converter()

    class TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.ModuleDict({"head": torch.nn.Linear(3, 2, bias=False)})

        def materialize_normalization(self, device):
            pass

        @staticmethod
        def state_dict_converter():
            return converter

    weights = {"head.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    pth, st = both
    torch.save(weights, pth)
    save_file(weights, str(st))
    monkeypatch.setattr(vae_module, "WanVideoVAE38", TinyVAE)
    for path in (pth, st):
        vae = loader._load_vae(path, torch.float32)
        seen.append(vae.model["head"].weight)
    assert torch.equal(seen[0], seen[1]) and torch.equal(seen[0], weights["head.weight"])
