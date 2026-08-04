import pytest
import torch

from scripts.eval_s2r.providers import LiveProvider, StateFrozenProvider, make_provider


class FakeEnv:
    """object_init_pos만 갖는 duck-type env."""
    def __init__(self, pos):
        self.object_init_pos = torch.tensor(pos, dtype=torch.float32)


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
        env.object_init_pos = env.object_init_pos + 1.0
        ov = p.get_override(env)
        assert torch.allclose(ov[0], torch.tensor([0.3, 0.0, 0.1]))

    def test_partial_reset_updates_only_those_envs(self):
        env = FakeEnv([[0.3, 0.0, 0.1], [0.2, -0.1, 0.1]])
        p = StateFrozenProvider()
        p.on_reset(env, torch.tensor([0, 1]))
        env.object_init_pos = torch.tensor([[9.0, 9.0, 9.0], [0.5, 0.5, 0.5]])
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
        env.object_init_pos[0, 0] = 99.0  # 원본 in-place 변경이 override에 새지 않아야 함
        assert float(p.get_override(env)[0, 0]) == pytest.approx(0.3)

    def test_partial_first_reset_then_get_override_raises(self):
        """부분 first reset (env_ids 미포함 env가 있음) 후 get_override → RuntimeError."""
        env = FakeEnv([[0.3, 0.0, 0.1], [0.2, -0.1, 0.1]])
        p = StateFrozenProvider()
        p.on_reset(env, torch.tensor([0]))  # env_1만 리셋하지 않음
        with pytest.raises(RuntimeError, match="some envs never reset"):
            p.get_override(env)

    def test_returned_tensor_mutation_does_not_corrupt_buffer(self):
        """반환된 텐서를 in-place 수정해도 내부 buffer는 불변."""
        env = FakeEnv([[0.3, 0.0, 0.1]])
        p = StateFrozenProvider()
        p.on_reset(env, torch.tensor([0]))
        ov = p.get_override(env)
        ov[0, 0] = 123.0  # 반환된 복사본을 in-place 수정
        # 다시 get_override → 여전히 원본값
        ov2 = p.get_override(env)
        assert float(ov2[0, 0]) == pytest.approx(0.3)


class TestFactory:
    def test_names(self):
        assert isinstance(make_provider("live"), LiveProvider)
        assert isinstance(make_provider("state_frozen"), StateFrozenProvider)

    def test_camera_frozen_missing_kwargs_raises(self):
        # SP2 구현 완료: camera_frozen은 이제 NotImplementedError가 아니라 인자 검증 ValueError
        with pytest.raises(ValueError):
            make_provider("camera_frozen")

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            make_provider("bogus")
