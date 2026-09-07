"""progress_reward 단위 테스트 (Isaac 불필요). DESIGN §3·§8 계약."""

from __future__ import annotations

import math

import pytest
import torch

from openarm.agnostic.modules import progress_reward as PR

N, K, ARM, HAND = 4, 5, 7, 20


def _inputs(**over):
    """중립 입력: 리프트 전·센티널·정지·목표 밖·손은 상판 위."""
    base = dict(
        obj_z=torch.full((N,), 0.30),
        settled_z=torch.full((N,), 0.30),
        lifted_prev=torch.zeros(N, dtype=torch.bool),
        ft_dist=torch.full((N, K), 0.20),
        closest_ft=torch.full((N, K), -1.0),
        kp_dist=torch.full((N,), 0.15),
        closest_kp=torch.full((N,), -1.0),
        near_goal=torch.zeros(N, dtype=torch.bool),
        arm_qd=torch.zeros(N, ARM),
        hand_qd=torch.zeros(N, HAND),
        hand_z_min=torch.full((N,), 0.40),
        cmd_rate=torch.zeros(N),
        cfg=PR.ProgressRewardCfg(),
    )
    base.update(over)
    return base


def _run(**over):
    return PR.compute_progress_reward(**_inputs(**over))


# =============================================================================
# 계약: 항 이름·순서·shape
# =============================================================================
def test_terms_names_and_order_match_contract():
    total, terms, out = _run()
    assert tuple(terms) == PR.PROGRESS_REWARD_TERMS
    assert PR.PROGRESS_REWARD_TERMS == (
        "fingertip_progress", "lift", "lift_bonus", "keypoint_progress",
        "goal_bonus", "arm_vel", "hand_vel", "hand_floor", "cmd_rate",
    )
    assert total.shape == (N,)
    for name, t in terms.items():
        assert t.shape == (N,), name
    assert out["lifted"].dtype == torch.bool and out["just_lifted"].dtype == torch.bool
    assert out["closest_ft"].shape == (N, K) and out["closest_kp"].shape == (N,)


def test_cfg_defaults_match_design():
    c = PR.ProgressRewardCfg()
    assert (c.ft_scale, c.lift_scale, c.lift_base, c.lift_clip) == (50.0, 20.0, 0.05, 0.5)
    assert (c.lift_bonus, c.lift_latch_height, c.kp_scale) == (300.0, 0.10, 200.0)
    assert (c.goal_bonus, c.success_steps) == (1000.0, 10)
    assert (c.arm_vel_scale, c.hand_vel_scale) == (0.03, 0.003)
    assert (c.hand_floor_penalty, c.hand_floor_z, c.hand_floor_max) == (10.0, 0.215, 5.0)
    assert c.cmd_rate_scale == 0.1
    assert c.cmd_rate_hold_dz == 0.03


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        _run(kp_dist=torch.zeros(N + 1))
    with pytest.raises(ValueError):
        _run(closest_ft=torch.full((N, K + 1), -1.0))
    with pytest.raises(TypeError):
        _run(near_goal=torch.zeros(N))
    with pytest.raises(ValueError):
        _run(cmd_rate=torch.zeros(N + 1))


# =============================================================================
# 리프트 래치
# =============================================================================
def test_lift_latch_is_sticky_and_bonus_fires_exactly_once():
    cfg = PR.ProgressRewardCfg()
    high = torch.full((N,), 0.30 + cfg.lift_latch_height + 0.01)
    # step 1: 임계 초과 → 최초 진입, 보너스 1회
    _, t1, o1 = _run(obj_z=high)
    assert o1["lifted"].all() and o1["just_lifted"].all()
    assert torch.allclose(t1["lift_bonus"], torch.full((N,), cfg.lift_bonus))
    # step 2: 여전히 높음 → 래치 유지, 보너스 0
    _, t2, o2 = _run(obj_z=high, lifted_prev=o1["lifted"])
    assert o2["lifted"].all() and not o2["just_lifted"].any()
    assert (t2["lift_bonus"] == 0).all()
    # step 3: 다시 내려가도 래치 유지(sticky), 보너스 0
    _, t3, o3 = _run(obj_z=torch.full((N,), 0.30), lifted_prev=o2["lifted"])
    assert o3["lifted"].all() and not o3["just_lifted"].any()
    assert (t3["lift_bonus"] == 0).all()


def test_below_threshold_never_lifts():
    cfg = PR.ProgressRewardCfg()
    _, t, o = _run(obj_z=torch.full((N,), 0.30 + cfg.lift_latch_height))  # 경계값(> 아님)
    assert not o["lifted"].any() and (t["lift_bonus"] == 0).all()


def test_prelift_terms_zero_after_lift_and_keypoint_zero_before():
    closest_ft = torch.full((N, K), 0.30)
    closest_kp = torch.full((N,), 0.30)
    # 리프트 전: ft·lift 살아있고 keypoint 0
    _, pre, _ = _run(closest_ft=closest_ft, closest_kp=closest_kp)
    assert (pre["fingertip_progress"] > 0).all()
    assert (pre["lift"] > 0).all()
    assert (pre["keypoint_progress"] == 0).all()
    # 리프트 후(래치): ft·lift 0, keypoint 살아있음
    _, post, _ = _run(closest_ft=closest_ft, closest_kp=closest_kp,
                      lifted_prev=torch.ones(N, dtype=torch.bool), obj_z=torch.full((N,), 0.45))
    assert (post["fingertip_progress"] == 0).all()
    assert (post["lift"] == 0).all()
    assert (post["keypoint_progress"] > 0).all()


def test_lift_term_value_and_clip():
    cfg = PR.ProgressRewardCfg()
    _, t, _ = _run(obj_z=torch.full((N,), 0.32))  # dz 0.02 < latch
    assert torch.allclose(t["lift"], torch.full((N,), cfg.lift_scale * (cfg.lift_base + 0.02)))
    # dz 음수(파묻힘) → base 로 clamp 하한 0
    _, t2, _ = _run(obj_z=torch.full((N,), 0.10))
    assert (t2["lift"] == 0).all()


# =============================================================================
# 진행 보상(fingertip / keypoint)
# =============================================================================
def test_fingertip_progress_rewards_only_decreases_and_sentinel_first_step_is_zero():
    cfg = PR.ProgressRewardCfg()
    # step 0: 센티널 → 0, closest 는 현재값으로
    _, t0, o0 = _run()
    assert (t0["fingertip_progress"] == 0).all()
    assert torch.allclose(o0["closest_ft"], torch.full((N, K), 0.20))
    # step 1: 접근 0.20 → 0.15 (5개 손끝) → 50·5·0.05
    _, t1, o1 = _run(ft_dist=torch.full((N, K), 0.15), closest_ft=o0["closest_ft"])
    assert torch.allclose(t1["fingertip_progress"], torch.full((N,), cfg.ft_scale * K * 0.05))
    assert torch.allclose(o1["closest_ft"], torch.full((N, K), 0.15))
    # step 2: 후퇴 0.15 → 0.25 → 0, closest 는 최소값 유지
    _, t2, o2 = _run(ft_dist=torch.full((N, K), 0.25), closest_ft=o1["closest_ft"])
    assert (t2["fingertip_progress"] == 0).all()
    assert torch.allclose(o2["closest_ft"], torch.full((N, K), 0.15))
    # step 3: 부분 접근(손끝별) — 최소 아래로 간 손끝만 보상
    ft = o2["closest_ft"].clone()
    ft[:, 0] = 0.10  # 1개만 전진 0.05
    _, t3, _ = _run(ft_dist=ft, closest_ft=o2["closest_ft"])
    assert torch.allclose(t3["fingertip_progress"], torch.full((N,), cfg.ft_scale * 0.05))


def test_keypoint_progress_sentinel_then_progress_then_retreat():
    cfg = PR.ProgressRewardCfg()
    lifted = torch.ones(N, dtype=torch.bool)
    _, t0, o0 = _run(lifted_prev=lifted, kp_dist=torch.full((N,), 0.15))
    assert (t0["keypoint_progress"] == 0).all()
    assert torch.allclose(o0["closest_kp"], torch.full((N,), 0.15))
    _, t1, o1 = _run(lifted_prev=lifted, kp_dist=torch.full((N,), 0.12), closest_kp=o0["closest_kp"])
    assert torch.allclose(t1["keypoint_progress"], torch.full((N,), cfg.kp_scale * 0.03))
    _, t2, o2 = _run(lifted_prev=lifted, kp_dist=torch.full((N,), 0.20), closest_kp=o1["closest_kp"])
    assert (t2["keypoint_progress"] == 0).all()
    assert torch.allclose(o2["closest_kp"], torch.full((N,), 0.12))


def test_private_progress_delta_agrees_with_keypoint_goal():
    """keypoint_goal.progress_delta 가 있으면 사본과 동일해야 한다(드리프트 감시)."""
    KG = pytest.importorskip("openarm.agnostic.modules.keypoint_goal")
    torch.manual_seed(0)
    for shape in [(N,), (N, K)]:
        curr = torch.rand(shape)
        closest = torch.rand(shape)
        closest[..., 0] = -1.0  # 센티널 섞기
        d_a, c_a = PR._progress_delta(curr, closest)
        d_b, c_b = KG.progress_delta(curr, closest)
        assert torch.allclose(d_a, d_b) and torch.allclose(c_a, c_b), shape


# =============================================================================
# 목표 보너스 · 속도 벌점 · hand_floor
# =============================================================================
def test_goal_bonus_per_near_step():
    cfg = PR.ProgressRewardCfg(goal_bonus=1000.0, success_steps=10)
    near = torch.tensor([True, False, True, False])
    _, t, _ = _run(near_goal=near, cfg=cfg)
    expect = torch.where(near, torch.tensor(100.0), torch.tensor(0.0))
    assert torch.allclose(t["goal_bonus"], expect)
    # 10 스텝 누적이면 goal_bonus 전액
    assert math.isclose(10 * (cfg.goal_bonus / cfg.success_steps), cfg.goal_bonus)


def test_velocity_penalties_are_l1():
    cfg = PR.ProgressRewardCfg()
    arm = torch.zeros(N, ARM); arm[:, 0] = 2.0; arm[:, 3] = -1.0     # L1 = 3
    hand = torch.zeros(N, HAND); hand[:, 5] = -0.5; hand[:, 7] = 0.5  # L1 = 1
    _, t, _ = _run(arm_qd=arm, hand_qd=hand)
    assert torch.allclose(t["arm_vel"], torch.full((N,), -cfg.arm_vel_scale * 3.0))
    assert torch.allclose(t["hand_vel"], torch.full((N,), -cfg.hand_vel_scale * 1.0))


def test_hand_floor_zero_above_threshold_and_capped():
    cfg = PR.ProgressRewardCfg()
    z = torch.tensor([cfg.hand_floor_z + 0.05, cfg.hand_floor_z, cfg.hand_floor_z - 0.02, cfg.hand_floor_z - 1.0])
    _, t, _ = _run(hand_z_min=z)
    assert t["hand_floor"][0] == 0.0
    assert t["hand_floor"][1] == 0.0
    assert math.isclose(t["hand_floor"][2].item(), -cfg.hand_floor_penalty * 0.02, rel_tol=1e-5)
    assert t["hand_floor"][3] == -cfg.hand_floor_max


# =============================================================================
# cmd_rate — 리프트 후 지령 변화 벌점 (09.07 A-v/B-v)
# =============================================================================
def test_cmd_rate_penalty_is_lift_gated_linear_and_unclamped():
    cfg = PR.ProgressRewardCfg()
    rate = torch.tensor([0.0, 1.0, 12.5, -3.0])
    # 리프트 전: 0 — 억제 항은 과제가 성립한 뒤에만(suppression-terms-need-task-first)
    _, pre, _ = _run(cmd_rate=rate)
    assert (pre["cmd_rate"] == 0).all()
    # 리프트 후 **들고 있을 때**(dz > hold_dz): −scale·rate, 음수 입력은 0 으로
    lifted = torch.ones(N, dtype=torch.bool)
    hold = torch.full((N,), 0.30 + cfg.cmd_rate_hold_dz + 0.05)
    _, post, _ = _run(cmd_rate=rate, lifted_prev=lifted, obj_z=hold)
    assert torch.allclose(post["cmd_rate"], -cfg.cmd_rate_scale * rate.clamp(min=0.0))
    # 상한 없음 — 작동점(≈10×)에서 clamp 되면 항이 상수가 되어 기울기가 죽는다(reward-clamp-kills-gradient)
    _, big, _ = _run(cmd_rate=torch.full((N,), 30.0), lifted_prev=lifted, obj_z=hold)
    assert torch.allclose(big["cmd_rate"], torch.full((N,), -cfg.cmd_rate_scale * 30.0))


def test_cmd_rate_is_zero_when_latched_but_object_is_back_down():
    """★a7 붕괴 원인: 래치는 sticky 라 튕겨 올라갔다 상판에 놓인 컵도 에피소드 끝까지 lifted 다.
    그 env 에 −1.35/step 이 500 스텝 붙어 "컵을 건드리지 말자"를 배웠다. 들고 있을 때(dz > hold_dz)만 벌한다."""
    cfg = PR.ProgressRewardCfg()
    lifted = torch.ones(N, dtype=torch.bool)
    z = torch.tensor([0.30, 0.30 + cfg.cmd_rate_hold_dz - 1e-3, 0.30 + cfg.cmd_rate_hold_dz + 1e-3, 0.45])
    _, t, o = _run(cmd_rate=torch.full((N,), 10.0), lifted_prev=lifted, obj_z=z)
    assert o["lifted"].all(), "래치는 그대로(sticky)"
    assert t["cmd_rate"][0] == 0.0 and t["cmd_rate"][1] == 0.0, "내려온 컵(dz < hold_dz)은 면제"
    assert torch.allclose(t["cmd_rate"][2:], torch.full((2,), -cfg.cmd_rate_scale * 10.0))


def test_cmd_rate_fires_on_the_lift_step_itself():
    """래치가 서는 스텝(just_lifted)부터 벌점 — keypoint_progress 와 같은 lifted_f 게이트."""
    cfg = PR.ProgressRewardCfg()
    high = torch.full((N,), 0.30 + cfg.lift_latch_height + 0.01)
    _, t, o = _run(obj_z=high, cmd_rate=torch.full((N,), 2.0))
    assert o["just_lifted"].all()
    assert torch.allclose(t["cmd_rate"], torch.full((N,), -cfg.cmd_rate_scale * 2.0))


def test_cmd_rate_scale_must_be_nonnegative():
    with pytest.raises(ValueError):
        PR.ProgressRewardCfg(cmd_rate_scale=-0.1)
    with pytest.raises(ValueError):
        PR.ProgressRewardCfg(cmd_rate_hold_dz=-0.01)


# =============================================================================
# total: 합·NaN 방어
# =============================================================================
def test_total_is_sum_of_terms():
    total, terms, _ = _run(obj_z=torch.full((N,), 0.32), near_goal=torch.ones(N, dtype=torch.bool),
                           arm_qd=torch.ones(N, ARM), hand_z_min=torch.full((N,), 0.2))
    assert torch.allclose(total, sum(terms.values()))


def test_total_finite_with_nan_inputs():
    ft = torch.full((N, K), 0.15); ft[0, 0] = float("nan")
    kp = torch.full((N,), float("nan"))
    obj_z = torch.full((N,), 0.30); obj_z[1] = float("inf")
    total, _, _ = _run(ft_dist=ft, closest_ft=torch.full((N, K), 0.20),
                       kp_dist=kp, closest_kp=torch.full((N,), 0.3),
                       obj_z=obj_z, arm_qd=torch.full((N, ARM), float("nan")))
    assert torch.isfinite(total).all()
