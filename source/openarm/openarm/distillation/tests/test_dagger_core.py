"""dagger 코어 로직 정적 테스트 (isaaclab/rl_games 없이 스텁으로 로드)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_DAGGER_PATH = Path(__file__).resolve().parents[1] / "dagger.py"


def _install_stubs() -> None:
    """dagger.py의 module-level import를 만족시키는 최소 스텁."""
    warp = types.ModuleType("warp")
    warp.set_device = lambda *_a, **_k: None
    warp.init = lambda: None
    sys.modules.setdefault("warp", warp)

    tbx = types.ModuleType("tensorboardX")
    tbx.SummaryWriter = object
    sys.modules.setdefault("tensorboardX", tbx)

    torchvision = sys.modules.get("torchvision") or types.ModuleType("torchvision")
    tv_utils = types.ModuleType("torchvision.utils")
    tv_utils.make_grid = lambda *_a, **_k: None
    torchvision.utils = tv_utils
    sys.modules.setdefault("torchvision", torchvision)
    sys.modules.setdefault("torchvision.utils", tv_utils)

    rl_games = types.ModuleType("rl_games")
    algos = types.ModuleType("rl_games.algos_torch")
    torch_ext = types.ModuleType("rl_games.algos_torch.torch_ext")

    def apply_masks(losses, mask):
        loss = losses[0]
        return [loss.mean()], None

    torch_ext.apply_masks = apply_masks
    torch_ext.AverageMeter = object
    torch_ext.load_checkpoint = lambda *_a, **_k: {}
    torch_ext.save_checkpoint = lambda *_a, **_k: None
    algos.torch_ext = torch_ext

    model_builder = types.ModuleType("rl_games.algos_torch.model_builder")
    model_builder.ModelBuilder = object
    running_mean_std = types.ModuleType("rl_games.algos_torch.running_mean_std")
    running_mean_std.RunningMeanStd = object

    sys.modules.setdefault("rl_games", rl_games)
    sys.modules.setdefault("rl_games.algos_torch", algos)
    sys.modules.setdefault("rl_games.algos_torch.torch_ext", torch_ext)
    sys.modules.setdefault("rl_games.algos_torch.model_builder", model_builder)
    sys.modules.setdefault("rl_games.algos_torch.running_mean_std", running_mean_std)

    depth_augs = types.ModuleType("openarm.distillation.depth_augs")
    depth_augs.DepthAug = object
    rgb_augs = types.ModuleType("openarm.distillation.rgb_augs")
    rgb_augs.RgbAug = object
    pkg = types.ModuleType("openarm.distillation")
    pkg.__path__ = [str(_DAGGER_PATH.parent)]
    openarm = types.ModuleType("openarm")
    openarm.__path__ = [str(_DAGGER_PATH.parents[1])]
    sys.modules.setdefault("openarm", openarm)
    sys.modules.setdefault("openarm.distillation", pkg)
    sys.modules.setdefault("openarm.distillation.depth_augs", depth_augs)
    sys.modules.setdefault("openarm.distillation.rgb_augs", rgb_augs)


def _load_dagger():
    _install_stubs()
    spec = importlib.util.spec_from_file_location("_dagger_under_test", _DAGGER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dagger = _load_dagger()


def test_l2_is_zero_for_identical_tensors():
    x = torch.randn(4, 11)
    assert torch.allclose(dagger.l2(x, x), torch.zeros(4), atol=1e-6)


def test_weighted_l2_upweights_confident_teacher_axes():
    # teacher sigma가 작은 축(=가중치 큰 축)의 오차가 손실을 더 키운다
    student = torch.tensor([[1.0, 0.0]])
    target = torch.zeros(1, 2)

    err_on_confident_axis = dagger.weighted_l2(
        student, target, weights=torch.tensor([[100.0, 1.0]])
    )
    err_on_loose_axis = dagger.weighted_l2(
        student.flip(-1), target, weights=torch.tensor([[100.0, 1.0]])
    )

    assert err_on_confident_axis > err_on_loose_axis


def test_adjust_state_dict_keys_inserts_orig_mod():
    ckpt = {"a2c_network.actor.0.weight": torch.zeros(1)}
    model = {"a2c_network.actor._orig_mod.0.weight": torch.zeros(1)}

    adjusted = dagger.adjust_state_dict_keys(ckpt, model)

    assert "a2c_network.actor._orig_mod.0.weight" in adjusted


def test_adjust_state_dict_keys_strips_orig_mod():
    ckpt = {"a2c_network.actor._orig_mod.0.weight": torch.zeros(1)}
    model = {"a2c_network.actor.0.weight": torch.zeros(1)}

    adjusted = dagger.adjust_state_dict_keys(ckpt, model)

    assert "a2c_network.actor.0.weight" in adjusted


def test_adjust_state_dict_keys_passes_through_exact_match():
    ckpt = {"running_mean_std.mean": torch.zeros(1)}

    adjusted = dagger.adjust_state_dict_keys(ckpt, dict(ckpt))

    assert list(adjusted) == ["running_mean_std.mean"]


@pytest.mark.parametrize(
    ("beta", "expected"),
    [
        (0.0, "student"),   # DAgger 본 학습: student on-policy rollout
        (1.0, "teacher"),   # beta=1: teacher가 전 env를 스텝
    ],
)
def test_mix_actions_selects_policy_by_beta(beta, expected):
    num_envs = 32
    fake = SimpleNamespace(num_envs=num_envs, device="cpu")
    student = {"actions": torch.ones(num_envs, 11)}
    teacher = {"actions": torch.full((num_envs, 11), 2.0)}

    mixed = dagger.Dagger._mix_actions(fake, student, teacher, beta)

    expected_value = 1.0 if expected == "student" else 2.0
    assert torch.allclose(mixed, torch.full((num_envs, 11), expected_value))


def test_mix_actions_returns_both_policies_at_intermediate_beta():
    num_envs = 512
    fake = SimpleNamespace(num_envs=num_envs, device="cpu")
    student = {"actions": torch.ones(num_envs, 11)}
    teacher = {"actions": torch.full((num_envs, 11), 2.0)}

    mixed = dagger.Dagger._mix_actions(fake, student, teacher, 0.5)

    assert torch.any(mixed == 1.0) and torch.any(mixed == 2.0)


def test_backbone_stays_frozen_until_threshold():
    # 초반 인코더 동결 → 헤드가 먼저 수렴한 뒤 backbone finetune
    assert dagger.BACKBONE_FREEZE_ITERS < dagger.DEFAULT_NUM_ITERS
