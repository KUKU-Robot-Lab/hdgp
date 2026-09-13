"""fj_reward(B 전용 보상) — 포크 동치 + `goal_bonus` divergence.

`modules/progress_reward.py` 의 포크다(09.08 사용자 확정: A 와 제어 방식이 다르다).
여기서 잠그는 것은 둘이다:
  ① **의도한 항만 갈렸다** — 나머지 8항이 같은 입력에서 A 와 수치가 같다.
  ② `goal_bonus` 는 성공 순간 **1회 전액**이고, `is_success` 는 **필수 인자**다.

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_fj/tests/test_fj_reward.py -q
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from openarm.agnostic.modules.progress_reward import (
    PROGRESS_REWARD_TERMS,
    ProgressRewardCfg,
    compute_progress_reward,
)
from openarm.agnostic.tasks.grasp_fj.fj_reward import (
    FJ_REWARD_TERMS,
    FJRewardCfg,
    compute_fj_reward,
)

N, NT = 6, 4


def _inputs(*, seed: int = 0) -> dict:
    """결정론 더미 배치. lifted/near_goal/is_success 가 서로 다른 env 에 걸리게 만든다."""
    g = torch.Generator().manual_seed(seed)
    settled = torch.full((N,), 0.20)
    dz = torch.tensor([0.0, 0.02, 0.05, 0.15, 0.30, 0.12])          # 3·4·5 번만 래치 위
    return dict(
        obj_z=settled + dz,
        settled_z=settled,
        lifted_prev=torch.tensor([False, False, False, False, True, False]),
        ft_dist=torch.rand(N, NT, generator=g),
        closest_ft=torch.rand(N, NT, generator=g) + 0.1,
        kp_dist=torch.rand(N, generator=g),
        closest_kp=torch.rand(N, generator=g) + 0.1,
        near_goal=torch.tensor([False, True, True, True, True, False]),
        arm_qd=torch.randn(N, 7, generator=g),
        hand_qd=torch.randn(N, 20, generator=g),
        hand_z_min=torch.tensor([0.30, 0.21, 0.19, 0.25, 0.10, 0.22]),
        cmd_rate=torch.rand(N, generator=g) * 3.0,
    )


def _paired_cfgs(one_shot: bool = True) -> tuple[ProgressRewardCfg, FJRewardCfg]:
    """A cfg 에서 B cfg 를 만든다(cfg 배선이 하는 일과 동일). `goal_one_shot` 만 B 고유다."""
    a = ProgressRewardCfg()
    b = FJRewardCfg(goal_one_shot=one_shot,
                    **{f.name: getattr(a, f.name) for f in dataclasses.fields(a)})
    return a, b


def test_field_sets_are_identical():
    """포크가 필드를 조용히 늘리거나 줄이면 cfg 배선(`FJRewardCfg(**...)`)이 깨진다."""
    a = {f.name for f in dataclasses.fields(ProgressRewardCfg)}
    b = {f.name for f in dataclasses.fields(FJRewardCfg)}
    # ★B 고유 필드는 셋뿐이다:
    #   goal_one_shot            지급 방식을 성공 술어에서 유도(09.08 포크 시점)
    #   wrap_scale/wrap_tau_xy/wrap_tau_z 감쌈 보상(09.09 도입 → 09.11 표면 기준 재정의).
    #     `wrap_scale=0.0` 이면 항이 정확히 0 이라 A 와 수치가 같다 — 그 동치는
    #     `test_only_goal_bonus_diverged` 가 계속 고정한다.
    #   ★09.13 palm_scale · grasp_g_min/grasp_q_lo/grasp_q_hi — 손바닥 접근 진행형 · 파지 계수 g(q).
    #     둘 다 기본값이면 꺼짐이라 A 와 수치가 같다(`test_grasp_factor_and_palm_progress_are_off_by_default`).
    assert b - a == {"goal_one_shot", "wrap_scale", "wrap_tau_xy", "wrap_tau_z",
                     "palm_scale", "grasp_g_min", "grasp_q_lo", "grasp_q_hi"}, b - a
    assert a - b == set(), a - b
    # ★항 이름·순서: 공유 9항이 **접두사로** 같고, B 고유 항(wrap · palm_progress)이 끝에 붙는다.
    #   끝에 붙여야 로깅·순서 가드가 기존 항의 자리를 안 바꾼다.
    assert FJ_REWARD_TERMS[:len(PROGRESS_REWARD_TERMS)] == PROGRESS_REWARD_TERMS, \
        "공유 항의 이름·순서가 어긋났다"
    assert FJ_REWARD_TERMS[len(PROGRESS_REWARD_TERMS):] == ("wrap", "palm_progress"), \
        "B 고유 항은 wrap · palm_progress 순서로 맨 끝이어야 한다"


def test_only_goal_bonus_diverged():
    """★포크 동치 — 나머지 8항이 같은 입력에서 A 와 **수치가 같다**.

    이게 깨지면 "B 를 고쳤는데 A 와 왜 다른지 모르는" 상태가 된다. 갈린 것은 의도한 한 항뿐이다.
    """
    a_cfg, b_cfg = _paired_cfgs()
    kw = _inputs()
    is_success = torch.tensor([False, False, False, True, False, False])
    _, a_terms, a_out = compute_progress_reward(cfg=a_cfg, **kw)
    _, b_terms, b_out = compute_fj_reward(cfg=b_cfg, is_success=is_success, **kw)
    for k in PROGRESS_REWARD_TERMS:
        if k == "goal_bonus":
            continue
        assert torch.equal(a_terms[k], b_terms[k]), f"{k} 가 갈렸다"
    for k in a_out:
        assert torch.equal(a_out[k], b_out[k]), f"되먹임 {k} 가 갈렸다"


def test_goal_bonus_is_paid_once_in_full_at_success():
    """SimToolReal env.py:2656-2659 의 forceConsecutive 분기 — near_goal 분할이 아니라 성공 1회 전액."""
    _, cfg = _paired_cfgs()
    kw = _inputs()
    is_success = torch.tensor([False, False, True, False, False, False])
    _, terms, _ = compute_fj_reward(cfg=cfg, is_success=is_success, **kw)
    gb = terms["goal_bonus"]
    assert float(gb[2]) == pytest.approx(cfg.goal_bonus)
    assert float(gb[torch.tensor([0, 1, 3, 4, 5])].abs().sum()) == 0.0, "성공하지 않은 env 는 0"
    # near_goal 인데 성공은 아닌 env(1·3·4)에 한 푼도 안 간다 — A 와 정확히 다른 지점.
    assert float(gb[1]) == 0.0 and float(gb[4]) == 0.0


def test_total_per_goal_matches_the_divided_form():
    """총액 등가 — A 의 10 스텝 분할 합 == B 의 1회 전액. 크기(REWARD_AUDIT Check 1)는 안 바뀐다."""
    a_cfg, b_cfg = _paired_cfgs()
    a_per_step = a_cfg.goal_bonus / a_cfg.success_steps
    assert a_per_step * a_cfg.success_steps == pytest.approx(b_cfg.goal_bonus)


def test_is_success_is_required_not_optional():
    """선택 인자로 두면 안 넘겼을 때 보상 한 항이 **조용히 0** 이 된다 — 시끄럽게 죽어야 한다."""
    _, cfg = _paired_cfgs()
    with pytest.raises(TypeError):
        compute_fj_reward(cfg=cfg, **_inputs())


def test_is_success_shape_and_dtype_are_checked():
    _, cfg = _paired_cfgs()
    kw = _inputs()
    with pytest.raises(ValueError):
        compute_fj_reward(cfg=cfg, is_success=torch.zeros(N + 1, dtype=torch.bool), **kw)
    with pytest.raises(TypeError):
        compute_fj_reward(cfg=cfg, is_success=torch.zeros(N), **kw)


def test_modules_progress_reward_is_untouched_by_the_fork():
    """★A 는 산술이 한 글자도 안 바뀐다 — 공유 모듈에 스위치를 넣지 않았다는 구조적 증거."""
    from pathlib import Path
    src = (Path(compute_progress_reward.__code__.co_filename)).read_text(encoding="utf-8")
    assert "is_success" not in src, "공유 모듈은 포크 인자를 몰라야 한다(이음매가 흡수한다)"
    assert "(cfg.goal_bonus / cfg.success_steps) * near_goal.float()" in src


def test_payout_form_follows_the_success_predicate():
    """★09.08 감사: 형제 트랙 `grasp_fj_rh` 가 이 모듈을 상속하면서 성공 술어는 A 기본값
    (누적)을 쓴다. 지급을 술어와 안 묶으면 "누적 판정 + 1회성 보너스" 가 되어 10회를 채우기
    전까지 near_goal 보상이 **0** 이 된다 — 그 트랙의 학습 신호가 조용히 성겨진다.
    """
    kw = _inputs()
    near = kw["near_goal"]
    succ = torch.tensor([False, False, True, False, False, False])
    _, one = _paired_cfgs(one_shot=True)
    _, div = _paired_cfgs(one_shot=False)
    _, t1, _ = compute_fj_reward(cfg=one, is_success=succ, **kw)
    _, t0, _ = compute_fj_reward(cfg=div, is_success=succ, **kw)
    # 누적(분할)판은 A 와 **완전히 동일**해야 한다 — 형제 트랙이 조용히 안 바뀌는 근거.
    _, ta, _ = compute_progress_reward(cfg=ProgressRewardCfg(), **kw)
    assert torch.equal(t0["goal_bonus"], ta["goal_bonus"]), "누적판이 A 와 갈렸다"
    assert not torch.equal(t1["goal_bonus"], t0["goal_bonus"]), "두 형태가 같으면 시험이 무의미"
    assert float(t1["goal_bonus"][2]) == pytest.approx(one.goal_bonus)
    assert float(t0["goal_bonus"][1]) == pytest.approx(div.goal_bonus / div.success_steps)
    assert bool(near[1]) and not bool(succ[1])   # near_goal 이지만 성공은 아닌 env


def test_cfg_derives_one_shot_from_the_predicate():
    """cfg 배선이 술어에서 유도하는지 — 소스로 고정한다(두 값이 갈리면 형제 트랙이 다친다)."""
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parent.parent / "grasp_fj_env_cfg.py").read_text(encoding="utf-8")
    assert "goal_one_shot=bool(self.goal_force_consecutive)" in src


# ══════════════════════════════════════════════════════════════════════════════
# 09.09 감쌈(wrap) 보상 — reward-audit 이 건 조건을 코드로 잠근다
# ══════════════════════════════════════════════════════════════════════════════
def _wrap_kw(curl, ft):
    """`_inputs()` 위에 curl·ft_dist 만 갈아끼운다."""
    kw = _inputs()
    n = kw["obj_z"].shape[0]
    kw["ft_dist"] = torch.full_like(kw["ft_dist"], ft)
    return kw, torch.full((n,), curl)


def test_wrap_is_off_by_default():
    """기본값이면 wrap 은 정확히 0 — 켜지 않은 런은 현행과 비트 동일해야 한다."""
    kw, frac = _wrap_kw(1.0, 0.01)          # 완전히 감쌌는데도
    _, terms, _ = compute_fj_reward(cfg=FJRewardCfg(), wrap_frac=frac,
                                    is_success=torch.zeros_like(frac, dtype=torch.bool), **kw)
    assert torch.all(terms["wrap"] == 0.0), "기본값인데 wrap 이 0 이 아니다"


def _wrap_step(cfg, frac_v, closest):
    kw, frac = _wrap_kw(frac_v, 0.01)
    _, terms, out = compute_fj_reward(cfg=cfg, wrap_frac=frac, wrap_closest=closest,
                                      is_success=torch.zeros_like(frac, dtype=torch.bool), **kw)
    return float(terms["wrap"][0]), out["wrap_closest"]


def test_wrap_pays_only_new_enclosure_not_holding():
    """★★09.11 — wrap 은 **진행형**: 에피소드 최고 포위도 대비 증가분만 준다.

    매 스텝 `scale × wrap_frac` 을 주던 판(fj_h1)은 리셋 자세의 wrap_frac 0.21 만으로
    0.43/step 이 나와 부트스트랩 구간 총보상의 40% 가 공짜였다. ep69 lifted_frac
    g1 0.618 vs h1 0.0065 — 컵 옆에 붙어 wrap 만 벌고 들지 않았다.
    이 테스트는 그 해를 정확히 막는다: **붙어 있기만 해서는 0**.
    """
    cfg = FJRewardCfg(wrap_scale=50.0)
    n = _inputs()["obj_z"].shape[0]
    cl = torch.full((n,), -1.0)
    r, cl = _wrap_step(cfg, 0.21, cl)
    assert r == 0.0, "첫 스텝(센티널)은 0 이어야 한다 — 리셋 자세의 포위를 사면 안 된다"
    r, cl = _wrap_step(cfg, 0.21, cl)
    assert r == 0.0, "같은 포위를 유지만 하는데 지급됐다 — fj_h1 의 공짜 보상 해"
    r, cl = _wrap_step(cfg, 0.50, cl)
    assert abs(r - 50.0 * 0.29) < 1e-4, f"포위를 늘린 만큼 지급돼야 한다: {r}"
    r, cl = _wrap_step(cfg, 0.40, cl)
    assert r == 0.0, "후퇴에 지급됐다"
    r, cl = _wrap_step(cfg, 0.50, cl)
    assert r == 0.0, "벌렸다 다시 오므려 같은 최고치에 닿았는데 재지급됐다 — 래칫이 없다"


def test_wrap_inputs_are_required_when_enabled():
    """켜 놓고 입력을 안 넘기면 조용히 0 이 되는 대신 **죽어야** 한다."""
    kw, frac = _wrap_kw(0.5, 0.01)
    with pytest.raises(ValueError):
        compute_fj_reward(cfg=FJRewardCfg(wrap_scale=50.0), wrap_frac=frac,
                          is_success=torch.zeros_like(frac, dtype=torch.bool), **kw)


def test_wrap_episode_total_is_below_lift_bonus():
    """★reward-audit Check 1 — 진행형의 **에피소드 총량** 상한은 scale × (1 − 0) 이다.
    그것이 lift_bonus(1회) 보다 작아야 "감싸기만 하고 안 드는" 해가 드는 해를 못 이긴다."""
    import re as _re
    from pathlib import Path as _P
    here = _P(__file__).resolve().parent.parent
    leaf = (here / "grasp_fj_env_cfg.py").read_text(encoding="utf-8")
    base = (here / "fj_kp_cfg.py").read_text(encoding="utf-8")
    scale = float(_re.search(r"rw_wrap_scale: float = ([0-9.]+)", leaf).group(1))
    lift_bonus = float(_re.search(r"rw_lift_bonus: float = ([0-9.]+)", base).group(1))
    assert scale * 1.0 < lift_bonus, f"wrap 총량 상한 {scale} ≥ lift_bonus {lift_bonus}"


# ══════════════════════════════════════════════════════════════════════════════
# 09.13 5손가락 파지 품질 q — 들기·성공 보너스에 곱할 계수의 입력
#   사용자 확정(09.13): "안전한 파지를 위해선 5손가락의 개입이 중요".
#   손가락마다 마디(_3, _4, tip) 표면 커널을 평균하고, 다섯 손가락을 soft-min 으로
#   묶는다(가장 약한 손가락이 지배). 손바닥이 물체를 향하지 않으면 0.
# ══════════════════════════════════════════════════════════════════════════════
_F, _L_PER = 5, 3   # thumb·index·middle·ring·pinky × (_3, _4, tip)


def _gq_inputs(gaps_xy, *, n=2, palm_cos=1.0) -> dict:
    """손가락별 마디 3개의 수평 간극(m) 표 (F, 3) → `grasp_quality` 입력. z 는 띠 안(0)."""
    g = torch.tensor(gaps_xy, dtype=torch.float32)
    e_xy = g.reshape(1, -1).expand(n, -1).clone()             # (n, 15) — 손가락 순서로 펼침
    return dict(e_xy=e_xy, e_z=torch.zeros_like(e_xy),
                finger_sizes=(_L_PER,) * _F,
                palm_cos=torch.full((n,), float(palm_cos)),
                tau_xy=0.02, tau_z=0.03, tau_q=0.1)


def test_grasp_quality_equals_the_common_value_when_all_fingers_agree():
    """soft-min 의 기준점 — 다섯 손가락이 같으면 q 는 그 값 그대로다(임계 해석이 쉬워진다)."""
    from openarm.agnostic.tasks.grasp_fj.fj_reward import grasp_quality
    q, wf = grasp_quality(**_gq_inputs([[0.01] * 3] * _F))
    expect = torch.exp(torch.tensor(-0.01 / 0.02))
    assert wf.shape == (2, _F)
    assert torch.allclose(wf, expect.expand_as(wf), atol=1e-6)
    assert torch.allclose(q, expect.expand_as(q), atol=1e-6)


def test_one_absent_finger_drags_quality_far_below_the_mean():
    """넷이 완벽하게 감싸도 하나가 빠지면 q 는 낮아야 한다 — 산술평균(구 `wrap_frac`)은 여기서 속는다."""
    from openarm.agnostic.tasks.grasp_fj.fj_reward import grasp_quality
    rows = [[0.0] * 3] * 4 + [[0.10] * 3]                     # pinky 만 표면에서 10 cm
    q, wf = grasp_quality(**_gq_inputs(rows))
    assert float(wf.mean()) > 0.79, "산술평균은 0.8 이다"
    assert float(q[0]) < 0.35, f"한 손가락이 빠졌는데 q={float(q[0]):.3f}"


def test_fingertip_only_contact_scores_below_an_envelope():
    """손끝만 대고 중간·끝마디가 뜬 파지(몇 개 손끝 파지) < 마디가 전부 붙은 감쌈."""
    from openarm.agnostic.tasks.grasp_fj.fj_reward import grasp_quality
    q_tip, _ = grasp_quality(**_gq_inputs([[0.05, 0.05, 0.0]] * _F))   # (_3, _4, tip)
    q_env, _ = grasp_quality(**_gq_inputs([[0.0, 0.0, 0.0]] * _F))
    assert float(q_env[0]) == pytest.approx(1.0)
    assert float(q_tip[0]) < 0.45, f"손끝만 댄 파지가 q={float(q_tip[0]):.3f}"


def test_back_of_hand_enclosure_is_not_a_grasp():
    """★관찰 #68 — 위치만 보는 커널은 손등으로 감싸도 만점이다. 손바닥이 물체를 향할 때만 인정한다."""
    from openarm.agnostic.tasks.grasp_fj.fj_reward import grasp_quality
    q, wf = grasp_quality(**_gq_inputs([[0.0] * 3] * _F, palm_cos=-0.5))
    assert torch.all(q == 0.0), "손등 파지에 점수가 나갔다"
    assert torch.all(wf > 0.99), "손가락 기하(진단)는 그대로 보고 q 만 0 이어야 한다"


def test_grasp_quality_is_bounded_between_worst_finger_and_mean():
    """soft-min 성질: min ≤ q ≤ mean, q ∈ [0, 1]. 이게 깨지면 계수 g(q) 의 경계가 무의미하다."""
    from openarm.agnostic.tasks.grasp_fj.fj_reward import grasp_quality
    g = torch.Generator().manual_seed(3)
    n = 64
    q, wf = grasp_quality(e_xy=torch.rand(n, 15, generator=g) * 0.08,
                          e_z=torch.rand(n, 15, generator=g) * 0.05,
                          finger_sizes=(3,) * 5,
                          palm_cos=torch.ones(n), tau_xy=0.02, tau_z=0.03, tau_q=0.1)
    assert torch.all(q >= wf.min(dim=1).values - 1e-6)
    assert torch.all(q <= wf.mean(dim=1) + 1e-6)
    assert torch.all((q >= 0.0) & (q <= 1.0))


def test_grasp_quality_rejects_bad_inputs_loudly():
    """조용한 브로드캐스트·0/0 평균 금지 — 이 값은 보너스 크기를 정한다."""
    from openarm.agnostic.tasks.grasp_fj.fj_reward import grasp_quality
    base = _gq_inputs([[0.0] * 3] * _F)
    with pytest.raises(ValueError):
        grasp_quality(**{**base, "tau_q": 0.0})
    with pytest.raises(ValueError):                             # 마디 0개 손가락 → 평균 0/0
        grasp_quality(**{**base, "finger_sizes": (15, 0, 0, 0, 0)})
    with pytest.raises(ValueError):                             # 마디 수 합이 열 수와 다르다
        grasp_quality(**{**base, "finger_sizes": (3, 3, 3, 3)})
    with pytest.raises(ValueError):
        grasp_quality(**{**base, "e_z": torch.zeros(2, 14)})
    with pytest.raises(ValueError):
        grasp_quality(**{**base, "palm_cos": torch.ones(3)})


# ══════════════════════════════════════════════════════════════════════════════
# 09.13 Phase 1 — 파지 계수 g(q) · 손바닥 접근 진행형 palm_progress
#   g(q) = g_min + (1−g_min)·clip((q−q_lo)/(q_hi−q_lo), 0, 1) 를 **지급 순간**의 보너스에만 곱한다:
#     lift_bonus × g(q) (들어 올리는 순간의 파지 = "들기 전에 파지 형태를 맞춘다")
#     goal_bonus × g(q) (성공 순간의 파지)
#   성공 술어 자체는 그대로다 — 공차 커리큘럼·게이트가 안 흔들린다(fj_h2 사다리 잠김 경로 없음).
#   palm_progress: 손바닥–물체 표면 간극의 에피소드 최단거리 갱신분(들기 전) — 손끝 기준의 대체.
#   두 기능 모두 기본값이면 **꺼짐**이고 현행과 비트 동일해야 한다.
# ══════════════════════════════════════════════════════════════════════════════
def _g_ref(q: float, g_min: float, lo: float, hi: float) -> float:
    return g_min + (1.0 - g_min) * min(max((q - lo) / (hi - lo), 0.0), 1.0)


def test_grasp_factor_and_palm_progress_are_off_by_default():
    """기본 cfg 는 새 입력을 넘겨도 모든 항이 현행과 **비트 동일**하다 — 켜지 않은 런·형제 트랙 보호."""
    kw = _inputs()
    succ = torch.tensor([False, False, False, False, True, False])
    cfg = FJRewardCfg(goal_one_shot=True)
    _, t0, _ = compute_fj_reward(cfg=cfg, is_success=succ, **kw)
    _, t1, _ = compute_fj_reward(cfg=cfg, is_success=succ, grasp_q=torch.zeros(N),
                                 palm_gap=torch.full((N,), 0.05), closest_palm=torch.full((N,), 0.2), **kw)
    for k in FJ_REWARD_TERMS:
        assert torch.equal(t0[k], t1[k]), f"기본값인데 {k} 가 바뀌었다"
    assert torch.all(t1["palm_progress"] == 0.0)


def test_grasp_factor_scales_only_the_bonus_payments():
    """g(q) 는 들기·성공 **지급 스텝**의 보너스에만 걸린다. 나머지 항은 한 비트도 안 바뀐다."""
    kw = _inputs()                       # dz 0.15(3)·0.12(5) 가 이번에 래치 → just_lifted = 3, 5
    succ = torch.tensor([False, False, False, False, True, False])
    q = torch.tensor([0.9, 0.9, 0.9, 0.1, 0.4, 0.9])
    base = FJRewardCfg(goal_one_shot=True)
    cfg = dataclasses.replace(base, grasp_g_min=0.5, grasp_q_lo=0.2, grasp_q_hi=0.6)
    _, t0, _ = compute_fj_reward(cfg=base, is_success=succ, **kw)
    _, t1, _ = compute_fj_reward(cfg=cfg, is_success=succ, grasp_q=q, **kw)
    assert float(t1["lift_bonus"][3]) == pytest.approx(300.0 * _g_ref(0.1, 0.5, 0.2, 0.6))   # 평손 들기 = 절반
    assert float(t1["lift_bonus"][5]) == pytest.approx(300.0 * _g_ref(0.9, 0.5, 0.2, 0.6))   # 감싸 들기 = 전액
    assert float(t1["goal_bonus"][4]) == pytest.approx(1000.0 * _g_ref(0.4, 0.5, 0.2, 0.6))
    for k in FJ_REWARD_TERMS:
        if k not in ("lift_bonus", "goal_bonus"):
            assert torch.equal(t0[k], t1[k]), f"{k} 는 계수와 무관해야 한다"


def test_grasp_factor_is_floor_linear_ceiling():
    """q ≤ q_lo → g_min(평손도 들기·성공 수입은 남는다 — 무행동 함정 방지) · q ≥ q_hi → 1 · 사이는 선형."""
    kw = _inputs()
    succ = torch.ones(N, dtype=torch.bool)
    cfg = FJRewardCfg(goal_one_shot=True, grasp_g_min=0.5, grasp_q_lo=0.2, grasp_q_hi=0.6)
    qs = torch.tensor([0.0, 0.2, 0.3, 0.4, 0.6, 1.0])
    _, t, _ = compute_fj_reward(cfg=cfg, is_success=succ, grasp_q=qs, **kw)
    for i, qv in enumerate(qs.tolist()):
        assert float(t["goal_bonus"][i]) == pytest.approx(1000.0 * _g_ref(qv, 0.5, 0.2, 0.6))


def test_palm_progress_pays_only_new_approach_before_lift():
    """진행형 규약 그대로 — 첫 스텝 0 · 유지 0 · 새로 줄인 만큼 · 후퇴 0 · 들고 나면 0."""
    cfg = FJRewardCfg(palm_scale=50.0)
    kw = _inputs()                       # env 3·4·5 는 lifted
    succ = torch.zeros(N, dtype=torch.bool)

    def step(gap: float, cl: torch.Tensor):
        _, t, o = compute_fj_reward(cfg=cfg, is_success=succ, palm_gap=torch.full((N,), gap),
                                    closest_palm=cl, **kw)
        return t["palm_progress"], o["closest_palm"]

    r, cl = step(0.12, torch.full((N,), -1.0))
    assert torch.all(r == 0.0), "센티널 스텝은 0 — 리셋 자세의 간극을 사면 안 된다"
    r, cl = step(0.12, cl)
    assert torch.all(r == 0.0), "간극 유지에 지급됐다"
    r, cl = step(0.10, cl)
    assert float(r[0]) == pytest.approx(50.0 * 0.02, abs=1e-5)
    assert torch.all(r[3:] == 0.0), "들고 난 env 에 접근 보상이 나갔다"
    r, cl = step(0.11, cl)
    assert torch.all(r == 0.0), "후퇴에 지급됐다"


def test_grasp_factor_and_palm_inputs_are_validated():
    """켜 놓고 입력을 안 넘기면 조용히 1·0 이 되는 대신 죽는다. 경계가 뒤집힌 cfg 도 죽는다."""
    with pytest.raises(ValueError):
        FJRewardCfg(grasp_g_min=0.0)
    with pytest.raises(ValueError):
        FJRewardCfg(grasp_g_min=1.2)
    with pytest.raises(ValueError):
        FJRewardCfg(goal_one_shot=True, grasp_g_min=0.5, grasp_q_lo=0.6, grasp_q_hi=0.6)
    with pytest.raises(ValueError):
        FJRewardCfg(palm_scale=-1.0)
    with pytest.raises(ValueError):                 # g(q) 는 1회 전액 지급과만 짝이다(분할 지급은 설계 밖)
        FJRewardCfg(goal_one_shot=False, grasp_g_min=0.5, grasp_q_lo=0.2, grasp_q_hi=0.6)
    kw = _inputs()
    succ = torch.zeros(N, dtype=torch.bool)
    with pytest.raises(ValueError):
        compute_fj_reward(cfg=FJRewardCfg(goal_one_shot=True, grasp_g_min=0.5, grasp_q_lo=0.2, grasp_q_hi=0.6),
                          is_success=succ, **kw)
    with pytest.raises(ValueError):
        compute_fj_reward(cfg=FJRewardCfg(palm_scale=50.0), is_success=succ, **kw)


def test_flat_grasp_still_beats_doing_nothing_when_first_success_is_late():
    """★reward-audit Check 1(할인형, 관찰 #160) — 평손 파지(g = g_min)의 과제 가치가 무행동 가치보다 커야
    FRESH 초반에 '안 드는' 해로 무너지지 않는다. 무행동 가치 = 들기 전 lift 상수 20×0.05 = 1.0/step → 100.

    느린 초반을 가정한다: 첫 들기 290 · 첫 성공 300 스텝 · 목표 간격 12 스텝 · 5목표.
    g_min 0.5 → 106(1.06배) 로 통과, 0.3 → 63 으로 미달 — 09.13 D2(0.5) 결정의 근거를 잠근다.
    """
    import re as _re
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parent.parent / "grasp_fj_env_cfg.py").read_text(encoding="utf-8")
    leaf = src.split("class GraspFJTesolloRightEnvCfg")[1].split("\nclass ")[0]
    # 아래 산술은 들기·성공 보너스와 lift 상수를 `FJRewardCfg` 기본값으로 읽는다 — leaf 가 덮으면 틀린 수를 검사한다.
    for f in ("rw_lift_bonus", "rw_goal_bonus", "rw_lift_scale", "rw_lift_base"):
        assert f not in leaf, f"leaf 가 {f} 를 덮는다 — 이 테스트의 기본값 전제가 깨졌다"
    g_min = float(_re.search(r"rw_grasp_g_min: float = ([0-9.]+)", leaf).group(1))
    cfg = FJRewardCfg()
    gamma, t_lift, t_first, gap = 0.99, 290, 300, 12
    v_task = g_min * (cfg.lift_bonus * gamma ** t_lift
                      + cfg.goal_bonus * sum(gamma ** (t_first + k * gap) for k in range(5)))
    v_idle = cfg.lift_scale * cfg.lift_base / (1.0 - gamma)
    assert v_task > v_idle, f"평손 과제 가치 {v_task:.1f} ≤ 무행동 {v_idle:.1f} (g_min {g_min})"
