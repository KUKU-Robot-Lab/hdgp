import pytest
import torch

from openarm.agnostic.tasks.iker_shoe import place_stage as ps


def _window(n: int, cfg: ps.PlaceRewardCfg) -> torch.Tensor:
    return ps.new_window(n, cfg, "cpu")


def _at_home(n: int) -> torch.Tensor:
    return torch.zeros(n)


def _step(args, window, cfg, home=None):
    n = args[0].shape[0]
    return ps.place_step(*args, window, cfg, arm_home_err=_at_home(n) if home is None else home)


def _good(cfg):
    return (torch.tensor([0.01]), torch.tensor([0.30]), torch.tensor([cfg.rack_top_z]), torch.tensor([0.0]))


def _bad(cfg):
    return (torch.tensor([0.30]), torch.tensor([0.30]), torch.tensor([cfg.rack_top_z]), torch.tensor([0.0]))


def test_all_five_conditions_must_hold():
    cfg = ps.PlaceRewardCfg()
    keypoint_dist = torch.tensor([0.01, 0.20, 0.01, 0.01, 0.01])   # 2번만 멀다
    palm_shoe = torch.tensor([0.30, 0.30, 0.05, 0.30, 0.30])       # 3번만 아직 손에 있다
    bottom_z = torch.full((5,), cfg.rack_top_z)
    speed = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])               # 4번만 움직인다
    home_err = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])             # 5번만 팔이 home 자세가 아니다
    step = ps.place_step(keypoint_dist, palm_shoe, bottom_z, speed, _window(5, cfg), cfg, arm_home_err=home_err)
    assert step.ok.tolist() == [True, False, False, False, False]
    assert step.home.tolist() == [True, True, True, True, False]


def test_home_is_required_keyword():
    cfg = ps.PlaceRewardCfg()
    with pytest.raises(TypeError):
        ps.place_step(*_good(cfg), _window(1, cfg), cfg)


def test_home_joint_tolerance_boundary():
    cfg = ps.PlaceRewardCfg()
    assert cfg.home_joint_tol == pytest.approx(0.035)
    step = ps.place_step(*[torch.cat([t, t]) for t in _good(cfg)], _window(2, cfg), cfg,
                         arm_home_err=torch.tensor([0.034, 0.036]))
    assert step.home.tolist() == [True, False]


def test_window_keeps_only_the_last_window_steps():
    cfg = ps.PlaceRewardCfg()
    window = _window(1, cfg)
    for _ in range(5):
        window = _step(_good(cfg), window, cfg).window
    assert _step(_bad(cfg), window, cfg).stable_count.tolist() == [5.0]
    for _ in range(cfg.window_steps):
        window = _step(_bad(cfg), window, cfg).window
    assert window.sum().item() == 0.0


def test_a_single_break_does_not_reset_the_count():
    cfg = ps.PlaceRewardCfg()
    window = _window(1, cfg)
    for _ in range(10):
        window = _step(_good(cfg), window, cfg).window
    step = _step(_bad(cfg), window, cfg)
    assert step.stable_count.tolist() == [10.0]
    assert not bool(step.success)


def test_success_needs_stable_steps_within_the_window():
    cfg = ps.PlaceRewardCfg()
    window = _window(1, cfg)
    for _ in range(cfg.stable_steps - 1):
        step = _step(_good(cfg), window, cfg)
        window = step.window
        assert not bool(step.success)
    assert bool(_step(_good(cfg), window, cfg).success)


def test_no_success_while_the_arm_is_not_home():
    cfg = ps.PlaceRewardCfg()
    window = _window(1, cfg)
    away = torch.tensor([1.0])
    for _ in range(cfg.window_steps):
        step = _step(_good(cfg), window, cfg, home=away)
        window = step.window
    assert step.stable_count.tolist() == [0.0]
    assert not bool(step.success)


def test_success_survives_scattered_breaks_inside_the_window():
    cfg = ps.PlaceRewardCfg()
    window = _window(1, cfg)
    step = None
    pattern = [True, True, False] * 10
    assert sum(pattern) == cfg.stable_steps and len(pattern) == cfg.window_steps
    for is_good in pattern:
        step = _step(_good(cfg) if is_good else _bad(cfg), window, cfg)
        window = step.window
    assert step.stable_count.tolist() == [float(cfg.stable_steps)]
    assert bool(step.success)


def test_resting_rejects_floating_shoes():
    cfg = ps.PlaceRewardCfg()
    floating = torch.tensor([cfg.rack_top_z + 10 * cfg.resting_tol, cfg.rack_top_z])
    step = ps.place_step(torch.tensor([0.01, 0.01]), torch.tensor([0.30, 0.30]),
                         floating, torch.zeros(2), _window(2, cfg), cfg, arm_home_err=_at_home(2))
    assert step.resting.tolist() == [False, True]


def test_place_tolerance_is_three_centimetres():
    cfg = ps.PlaceRewardCfg()
    assert cfg.place_tolerance == 0.03
    step = ps.place_step(torch.tensor([0.025, 0.045]), torch.tensor([0.30, 0.30]),
                         torch.full((2,), cfg.rack_top_z), torch.zeros(2), _window(2, cfg), cfg,
                         arm_home_err=_at_home(2))
    assert step.placed.tolist() == [True, False]


# ------------------------------------------------------------------ scripted home return


def test_retract_triggers_only_when_placed_resting_still_and_open():
    cfg = ps.PlaceRewardCfg()
    t, f = torch.tensor, None
    placed = t([True, False, True, True, True, True])
    resting = t([True, True, False, True, True, True])
    still = t([True, True, True, False, True, True])
    open_frac = t([0.95, 0.95, 0.95, 0.95, 0.50, 0.95])
    retracting = t([False, False, False, False, False, True])
    start = ps.retract_trigger(placed, resting, still, open_frac, retracting, cfg)
    assert start.tolist() == [True, False, False, False, False, False], "이미 복귀 중이면 다시 트리거하지 않는다"


def test_retract_targets_reach_home_exactly_and_stay_there():
    cfg = ps.PlaceRewardCfg()
    q0 = torch.tensor([[0.5, -0.5, 1.0]])
    home = torch.tensor([0.0, 0.0, 0.0])
    first = ps.retract_targets(q0, home, torch.tensor([0]), cfg)
    assert torch.all((first - q0).abs() < (home - q0).abs()), "첫 스텝부터 home 쪽으로 움직인다"
    last = ps.retract_targets(q0, home, torch.tensor([cfg.retract_steps - 1]), cfg)
    assert torch.allclose(last, home[None]), "retract_steps 번째 스텝에 정확히 home"
    after = ps.retract_targets(q0, home, torch.tensor([cfg.retract_steps + 10]), cfg)
    assert torch.allclose(after, home[None]), "그 뒤에는 home 에 머문다"


def test_retract_targets_are_monotone():
    cfg = ps.PlaceRewardCfg()
    q0 = torch.tensor([[1.0]])
    home = torch.tensor([0.0])
    values = [float(ps.retract_targets(q0, home, torch.tensor([k]), cfg)) for k in range(cfg.retract_steps)]
    assert all(b <= a for a, b in zip(values, values[1:])), "되돌아가거나 넘어서지 않는다"
