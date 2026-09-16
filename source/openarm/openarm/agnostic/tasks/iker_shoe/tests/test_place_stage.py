import torch

from openarm.agnostic.tasks.iker_shoe import place_stage as ps


def test_all_four_conditions_must_hold():
    cfg = ps.PlaceRewardCfg()
    keypoint_dist = torch.tensor([0.01, 0.20, 0.01, 0.01])   # 2번만 멀다
    palm_shoe = torch.tensor([0.30, 0.30, 0.05, 0.30])       # 3번만 아직 손에 있다
    bottom_z = torch.full((4,), cfg.rack_top_z)
    speed = torch.tensor([0.0, 0.0, 0.0, 1.0])               # 4번만 움직인다
    step = ps.place_step(keypoint_dist, palm_shoe, bottom_z, speed, torch.zeros(4), cfg)
    assert step.ok.tolist() == [True, False, False, False]


def test_counter_is_consecutive_not_cumulative():
    cfg = ps.PlaceRewardCfg()
    good = (torch.tensor([0.01]), torch.tensor([0.30]), torch.tensor([cfg.rack_top_z]), torch.tensor([0.0]))
    bad = (torch.tensor([0.30]), torch.tensor([0.30]), torch.tensor([cfg.rack_top_z]), torch.tensor([0.0]))
    count = torch.zeros(1)
    for _ in range(5):
        count = ps.place_step(*good, count, cfg).stable_count
    assert count.tolist() == [5.0]
    count = ps.place_step(*bad, count, cfg).stable_count
    assert count.tolist() == [0.0], "조건이 깨지면 0 으로 돌아가야 한다(기존 IKER 는 누적이었다)"


def test_success_needs_stable_steps_in_a_row():
    cfg = ps.PlaceRewardCfg()
    good = (torch.tensor([0.01]), torch.tensor([0.30]), torch.tensor([cfg.rack_top_z]), torch.tensor([0.0]))
    count = torch.zeros(1)
    for _ in range(cfg.stable_steps - 1):
        step = ps.place_step(*good, count, cfg)
        count = step.stable_count
        assert not bool(step.success)
    assert bool(ps.place_step(*good, count, cfg).success)


def test_resting_rejects_floating_shoes():
    cfg = ps.PlaceRewardCfg()
    floating = torch.tensor([cfg.rack_top_z + 10 * cfg.resting_tol, cfg.rack_top_z])
    step = ps.place_step(torch.tensor([0.01, 0.01]), torch.tensor([0.30, 0.30]),
                         floating, torch.zeros(2), torch.zeros(2), cfg)
    assert step.resting.tolist() == [False, True]
