import pytest
import torch

from scripts.eval_s2r.providers import LiveProvider, StateFrozenProvider, make_provider


class FakeEnv:
    """object_pos만 갖는 duck-type env."""
    def __init__(self, pos):
        self.object_pos = torch.tensor(pos, dtype=torch.float32)


class TestLiveProvider:
    def test_always_none(self):
        env = FakeEnv([[0.3, 0.0, 0.1]])
        p = LiveProvider()
        p.on_reset(env, torch.tensor([0]))
        assert p.get_override(env) is None


class TestStateFrozenProvider:
    def test_freezes_pose_at_reset(self):
        env = FakeEnv([[0.3, 0.0, 0.1], [0.2, -0.1, 0.1]])
        p = StateFrozenProvider()
        p.on_reset(env, torch.tensor([0, 1]))
        # 물체가 움직여도 override는 reset 시점 값 고정
        env.object_pos = env.object_pos + 1.0
        ov = p.get_override(env)
        assert torch.allclose(ov[0], torch.tensor([0.3, 0.0, 0.1]))

    def test_partial_reset_updates_only_those_envs(self):
        env = FakeEnv([[0.3, 0.0, 0.1], [0.2, -0.1, 0.1]])
        p = StateFrozenProvider()
        p.on_reset(env, torch.tensor([0, 1]))
        env.object_pos = torch.tensor([[9.0, 9.0, 9.0], [0.5, 0.5, 0.5]])
        p.on_reset(env, torch.tensor([1]))  # env1만 재캡처
        ov = p.get_override(env)
        assert torch.allclose(ov[0], torch.tensor([0.3, 0.0, 0.1]))   # 불변
        assert torch.allclose(ov[1], torch.tensor([0.5, 0.5, 0.5]))   # 갱신

    def test_get_override_before_reset_raises(self):
        p = StateFrozenProvider()
        with pytest.raises(RuntimeError):
            p.get_override(FakeEnv([[0.0, 0.0, 0.0]]))

    def test_nonfinite_pose_rejected(self):
        env = FakeEnv([[float("nan"), 0.0, 0.1]])
        p = StateFrozenProvider()
        with pytest.raises(ValueError):
            p.on_reset(env, torch.tensor([0]))

    def test_override_is_detached_copy(self):
        env = FakeEnv([[0.3, 0.0, 0.1]])
        p = StateFrozenProvider()
        p.on_reset(env, torch.tensor([0]))
        env.object_pos[0, 0] = 99.0  # 원본 in-place 변경이 override에 새지 않아야 함
        assert float(p.get_override(env)[0, 0]) == pytest.approx(0.3)


class TestFactory:
    def test_names(self):
        assert isinstance(make_provider("live"), LiveProvider)
        assert isinstance(make_provider("state_frozen"), StateFrozenProvider)

    def test_camera_frozen_not_implemented(self):
        with pytest.raises(NotImplementedError):
            make_provider("camera_frozen")

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            make_provider("bogus")
