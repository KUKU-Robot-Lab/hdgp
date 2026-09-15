"""보상 게이트용 에피소드 단계 래치 — 순수 torch (Isaac 불요).

잠그는 것(09.15 사용자 "기본 핸드 자세에서 컵으로 접근 → 접근한 상태에서 인벨롭 파지 → 리프트"):
  · 접근 완료 = 손바닥 중심 ↔ 파지 띠 ≤ 2 cm · 손바닥이 컵 축을 향함(cos ≥ 0.7) · 움직이는 손 관절이 기본 자세 ±0.15 안 — 셋 동시.
  · 인벨롭 완료 = 접근 완료 **뒤에만** · 손바닥 + 엄지 + 닿은 손가락 ≥ 4 가 5 스텝 연속.
  · 둘 다 에피소드 래치(한 번 서면 리셋까지 유지) — 끊기면 연속 카운트만 0.
  · ctx 주석(생성기가 읽는 환경 설명)이 같은 수치를 적는다.

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_fj_t2r/tests -q
"""

from __future__ import annotations

from pathlib import Path

import torch

from openarm.agnostic.tasks.grasp_fj_t2r import grasp_gates as G

_CTX = (Path(__file__).resolve().parent.parent / "t2r" / "context.py").read_text(encoding="utf-8")


def test_palm_facing_is_plus_one_when_the_palm_normal_points_at_the_cup_axis():
    cup = torch.zeros(3, 3)
    axis = torch.tensor([[0.0, 0.0, 1.0]]).expand(3, 3)
    palm = torch.tensor([[0.08, 0.0, 0.03], [0.08, 0.0, 0.0], [0.0, 0.08, 0.0]])
    normal = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    f = G.palm_facing(palm, normal, cup, axis)
    assert torch.allclose(f, torch.tensor([1.0, -1.0, 0.0]), atol=1e-6)


def test_pose_deviation_ignores_locked_joints():
    q = torch.tensor([[0.10, 0.90, 0.30], [0.12, 0.10, 0.50]])
    default = torch.tensor([0.10, 0.10, 0.30])
    movable = torch.tensor([True, False, True])
    dev = G.pose_deviation(q, default, movable)
    assert torch.allclose(dev, torch.tensor([0.0, 0.20]), atol=1e-6)


def _zeros(n):
    return (torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.bool))


def test_approach_needs_gap_facing_and_default_pose_together():
    n = 4
    gap = torch.tensor([0.01, 0.05, 0.01, 0.01])
    facing = torch.tensor([0.9, 0.9, 0.3, 0.9])
    pose_dev = torch.tensor([0.05, 0.05, 0.05, 0.40])
    no_touch = torch.zeros(n, dtype=torch.bool)
    fingers = torch.zeros(n, 5, dtype=torch.bool)
    appr, cnt, env = G.update_gates(*_zeros(n), gap=gap, facing=facing, pose_dev=pose_dev,
                                    palm_touch=no_touch, finger_touch=fingers)
    assert appr.tolist() == [True, False, False, False]
    assert not env.any() and not cnt.any()


def test_envelope_only_after_approach_and_after_consecutive_hold_steps():
    n = 3
    far = torch.full((n,), 0.10)
    face = torch.full((n,), 0.9)
    dev = torch.zeros(n)
    palm = torch.tensor([True, True, True])
    fingers = torch.zeros(n, 5, dtype=torch.bool)
    fingers[:, :4] = True                               # 엄지 + 검지·중지·약지
    appr, cnt, env = _zeros(n)
    # env 0 은 접근 완료, env 1 은 접근 전(손바닥은 이미 닿음), env 2 는 접근 완료지만 엄지가 없다
    appr = torch.tensor([True, False, True])
    no_thumb = fingers.clone()
    no_thumb[2, 0] = False
    no_thumb[2, 4] = True
    for step in range(G.ENVELOPE_HOLD_STEPS):
        appr, cnt, env = G.update_gates(appr, cnt, env, gap=far, facing=face, pose_dev=dev,
                                        palm_touch=palm, finger_touch=no_thumb)
        if step < G.ENVELOPE_HOLD_STEPS - 1:
            assert not env.any(), step
    assert env.tolist() == [True, False, False]
    assert appr.tolist() == [True, False, True], "접근 래치는 간극이 멀어져도 유지된다"


def test_a_break_resets_the_hold_count_but_a_set_latch_stays():
    n = 1
    kw = dict(gap=torch.tensor([0.10]), facing=torch.tensor([0.9]), pose_dev=torch.tensor([0.0]))
    held = torch.ones(n, 5, dtype=torch.bool)
    loose = torch.zeros(n, 5, dtype=torch.bool)
    appr, cnt, env = torch.tensor([True]), torch.zeros(n, dtype=torch.long), torch.tensor([False])
    for _ in range(G.ENVELOPE_HOLD_STEPS - 1):
        appr, cnt, env = G.update_gates(appr, cnt, env, palm_touch=torch.tensor([True]), finger_touch=held, **kw)
    appr, cnt, env = G.update_gates(appr, cnt, env, palm_touch=torch.tensor([False]), finger_touch=held, **kw)
    assert cnt.tolist() == [0] and not env.any()
    for _ in range(G.ENVELOPE_HOLD_STEPS):
        appr, cnt, env = G.update_gates(appr, cnt, env, palm_touch=torch.tensor([True]), finger_touch=held, **kw)
    assert env.tolist() == [True]
    appr, cnt, env = G.update_gates(appr, cnt, env, palm_touch=torch.tensor([False]), finger_touch=loose, **kw)
    assert env.tolist() == [True] and cnt.tolist() == [0]


def test_update_returns_new_tensors_without_mutating_inputs():
    appr, cnt, env = _zeros(2)
    out = G.update_gates(appr, cnt, env, gap=torch.tensor([0.0, 0.0]), facing=torch.tensor([1.0, 1.0]),
                         pose_dev=torch.tensor([0.0, 0.0]), palm_touch=torch.tensor([True, True]),
                         finger_touch=torch.ones(2, 5, dtype=torch.bool))
    assert not appr.any() and not cnt.any() and not env.any()
    assert out[0].all() and (out[1] == 1).all()


def test_context_comments_state_the_same_gate_numbers():
    for tok in ("hand_default_q_norm", "approach_done", "envelope_done",
                "within 2 cm", "about 45 degrees", "within 0.15", "at least 4", "5 consecutive steps", "0.1 N"):
        assert tok in _CTX, tok
    assert G.APPROACH_GAP_M == 0.02 and G.APPROACH_FACING_MIN == 0.7 and G.APPROACH_POSE_TOL == 0.15
    assert G.ENVELOPE_MIN_DIGITS == 4 and G.ENVELOPE_HOLD_STEPS == 5 and G.TOUCH_N == 0.1
