import torch

from openarm.agnostic.tasks.iker_shoe import place_stage as ps


def _window(n: int, cfg: ps.PlaceRewardCfg) -> torch.Tensor:
    return ps.new_window(n, cfg, "cpu")


def test_all_four_conditions_must_hold():
    cfg = ps.PlaceRewardCfg()
    keypoint_dist = torch.tensor([0.01, 0.20, 0.01, 0.01])   # 2번만 멀다
    palm_shoe = torch.tensor([0.30, 0.30, 0.05, 0.30])       # 3번만 아직 손에 있다
    bottom_z = torch.full((4,), cfg.rack_top_z)
    speed = torch.tensor([0.0, 0.0, 0.0, 1.0])               # 4번만 움직인다
    step = ps.place_step(keypoint_dist, palm_shoe, bottom_z, speed, _window(4, cfg), cfg)
    assert step.ok.tolist() == [True, False, False, False]


def test_window_keeps_only_the_last_window_steps():
    """창은 최근 window_steps 만 본다 — 그보다 오래된 충족은 빠져나간다."""
    cfg = ps.PlaceRewardCfg()
    good = (torch.tensor([0.01]), torch.tensor([0.30]), torch.tensor([cfg.rack_top_z]), torch.tensor([0.0]))
    bad = (torch.tensor([0.30]), torch.tensor([0.30]), torch.tensor([cfg.rack_top_z]), torch.tensor([0.0]))
    window = _window(1, cfg)
    for _ in range(5):
        window = ps.place_step(*good, window, cfg).window
    assert ps.place_step(*bad, window, cfg).stable_count.tolist() == [5.0], "한 번 깨져도 앞의 5스텝은 남는다"
    for _ in range(cfg.window_steps):
        window = ps.place_step(*bad, window, cfg).window
    assert window.sum().item() == 0.0, "창 길이를 넘어가면 옛 충족은 전부 빠져나간다"


def test_a_single_break_does_not_reset_the_count():
    """연속 방식과의 결정적 차이: 한 스텝 흔들려도 0 으로 돌아가지 않는다(노이즈 내성)."""
    cfg = ps.PlaceRewardCfg()
    good = (torch.tensor([0.01]), torch.tensor([0.30]), torch.tensor([cfg.rack_top_z]), torch.tensor([0.0]))
    bad = (torch.tensor([0.30]), torch.tensor([0.30]), torch.tensor([cfg.rack_top_z]), torch.tensor([0.0]))
    window = _window(1, cfg)
    for _ in range(10):
        window = ps.place_step(*good, window, cfg).window
    step = ps.place_step(*bad, window, cfg)
    assert step.stable_count.tolist() == [10.0]
    assert not bool(step.success)


def test_success_needs_stable_steps_within_the_window():
    cfg = ps.PlaceRewardCfg()
    good = (torch.tensor([0.01]), torch.tensor([0.30]), torch.tensor([cfg.rack_top_z]), torch.tensor([0.0]))
    window = _window(1, cfg)
    for _ in range(cfg.stable_steps - 1):
        step = ps.place_step(*good, window, cfg)
        window = step.window
        assert not bool(step.success)
    assert bool(ps.place_step(*good, window, cfg).success)


def test_success_survives_scattered_breaks_inside_the_window():
    """창 안에서 흐트러져도 충족 스텝이 stable_steps 를 채우면 성공이다 — 연속 방식은 여기서 실패했다."""
    cfg = ps.PlaceRewardCfg()
    good = (torch.tensor([0.01]), torch.tensor([0.30]), torch.tensor([cfg.rack_top_z]), torch.tensor([0.0]))
    bad = (torch.tensor([0.30]), torch.tensor([0.30]), torch.tensor([cfg.rack_top_z]), torch.tensor([0.0]))
    window = _window(1, cfg)
    step = None
    # 30스텝 창에 good 20 · bad 10 을 섞어 넣는다(2 good 마다 1 bad).
    pattern = [True, True, False] * 10
    assert sum(pattern) == cfg.stable_steps and len(pattern) == cfg.window_steps
    for is_good in pattern:
        step = ps.place_step(*(good if is_good else bad), window, cfg)
        window = step.window
    assert step.stable_count.tolist() == [float(cfg.stable_steps)]
    assert bool(step.success), "연속이 아니어도 창 안에서 20스텝을 채우면 성공해야 한다"


def test_resting_rejects_floating_shoes():
    cfg = ps.PlaceRewardCfg()
    floating = torch.tensor([cfg.rack_top_z + 10 * cfg.resting_tol, cfg.rack_top_z])
    step = ps.place_step(torch.tensor([0.01, 0.01]), torch.tensor([0.30, 0.30]),
                         floating, torch.zeros(2), _window(2, cfg), cfg)
    assert step.resting.tolist() == [False, True]


def test_place_tolerance_is_three_centimetres():
    """2026-09-17 사용자 결정 — 설계된 두 신발 틈(2.8 cm)보다 허용오차가 크면 성공이 '붙어 보이지' 않는다."""
    cfg = ps.PlaceRewardCfg()
    assert cfg.place_tolerance == 0.03
    step = ps.place_step(torch.tensor([0.025, 0.045]), torch.tensor([0.30, 0.30]),
                         torch.full((2,), cfg.rack_top_z), torch.zeros(2), _window(2, cfg), cfg)
    assert step.placed.tolist() == [True, False]
