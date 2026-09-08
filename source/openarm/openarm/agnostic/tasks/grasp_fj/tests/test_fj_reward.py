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
    # ★`goal_one_shot` 만 B 고유다 — 지급 방식을 성공 술어에서 유도하기 위한 필드.
    assert b - a == {"goal_one_shot"}, b - a
    assert a - b == set(), a - b
    assert FJ_REWARD_TERMS == PROGRESS_REWARD_TERMS, "항 이름·순서는 포크 시점에 같다"


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
