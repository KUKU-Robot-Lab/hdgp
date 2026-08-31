#!/usr/bin/env python3
"""평가 하네스가 태스크를 가리지 않게 하는 층 (Isaac 불필요, 순수 텐서 연산).

문제
----
`play.py --eval_episodes N` 은 grasp_v1/v2 direct-env 속성을 `try` 없이 직참조한다
(`in_success_region`, `_obj_total_episodes.zero_()`, `binary_contact_buf` …).
업그레이드된 신규 태스크는 그 속성을 **하나도** 노출하지 않는다:

  agnostic/tasks/grasp_sensor       — `fingertip_pos` 를 보상 인자로만 쓰고 버퍼 미노출
  agnostic/tasks/grasp_lift_fabric  — `_object_names`/`in_success_region` 없음
  gripper/left/grasp_sensor         — ManagerBasedRLEnv 라 이런 속성이 원천적으로 없다

결과: 재생(rollout·렌더)은 되는데 정량 평가만 AttributeError 로 즉사했다.

방침 — 옮기지 않고 감싼다
--------------------------
grasp_v2 블록은 **play.py 안에 그대로 둔다.** 150줄짜리 밀도 높은 코드를 옮기면 거동이
바뀌었는지 GPU 로 재검증해야 하는데, 그 검증 없이 옮기는 편이 더 위험하다. 대신 진입
직전에 어댑터를 고른다:

  select(env) == GRASP_V2  → 예전 블록 그대로 (바이트 단위 보존)
  select(env) == COMMON    → 여기 CommonEvalAccumulator 로 공통 지표

공통 지표는 env 에 **아무것도 요구하지 않는다** — 리턴·길이·종료사유·액션 통계뿐이라
어떤 태스크에서도 성립한다. 태스크 고유 지표(성공률·접촉 손가락)는 env 가 그 버퍼를
노출할 때만 나온다. 없는 걸 추정해서 만들어내지 않는다.
"""

from __future__ import annotations

GRASP_V2 = "grasp_v2"
COMMON = "common"

# grasp_v2 블록이 `try` 없이 만지는 속성 전부. 하나라도 없으면 그 블록에 들어가면 안 된다.
GRASP_V2_REQUIRED: tuple[str, ...] = (
    "in_success_region",
    "object_pos",
    "object_init_pos",
    "binary_contact_buf",
    "middle_binary_contact_buf",
    "distal_binary_contact_buf",
    "fingertip_pos",
    "_object_names",
    "object_idx",
    "_total_episodes",
    "_successful_episodes",
    "_obj_total_episodes",
    "_obj_success_episodes",
)

SATURATION_LEVEL = 0.99      # |a| 가 이 이상이면 포화로 센다 (action_audit 과 같은 기준)


def missing_grasp_v2_attrs(env) -> tuple[str, ...]:
    return tuple(a for a in GRASP_V2_REQUIRED if not hasattr(env, a))


def select(env) -> str:
    """이 env 에 어느 평가 경로를 쓸 것인가.

    부분 노출을 grasp_v2 로 오인하면 블록 **안에서** AttributeError 가 난다 — 전부
    갖췄을 때만 그 길로 보낸다.
    """
    return COMMON if missing_grasp_v2_attrs(env) else GRASP_V2


def finger_labels(env, count: int, default: tuple = ()) -> list:
    """손가락 라벨 — 5지 하드코딩 금지.

    `["thumb","index","middle","ring","pinky"]` 를 박아 두면 2지 평행 그리퍼
    (`sens_left`: jaw1/jaw2)에 엄지·검지 이름이 붙어 표를 오독하게 만든다.
    프로필이 이름을 알면 그걸 쓰고, 모르면 **지어내지 않고** 인덱스를 쓴다.
    """
    profile = getattr(env, "profile", None)
    names: tuple = ()
    if profile is not None:
        names = tuple(getattr(profile, "fingers", ()) or ())
        if not names:
            names = tuple(getattr(profile, "finger_sensor_bodies", {}) or {})
    if len(names) >= count:
        return list(names[:count])
    # 구 트랙 env 들은 profile 을 노출하지 않는다 — 호출자가 준 기존 라벨을 그대로 써서
    # 출력이 바뀌지 않게 한다. default 도 없으면 지어내지 않고 인덱스를 쓴다.
    if len(default) >= count:
        return list(default[:count])
    return [f"f{i}" for i in range(count)]


class CommonEvalAccumulator:
    """어떤 태스크에서도 성립하는 지표만 모은다.

    env 에 요구하는 것이 없다 — 리워드·done·액션은 루프가 이미 들고 있는 값이다.
    종료 사유(terminated vs truncated)만 있으면 더 읽지만, 없으면 그냥 비운다.
    """

    def __init__(self, num_envs: int):
        import torch

        self.num_envs = int(num_envs)
        self._returns = torch.zeros(self.num_envs, dtype=torch.float64)
        self._lengths = torch.zeros(self.num_envs, dtype=torch.long)
        self.episode_returns: list = []
        self.episode_lengths: list = []
        self.terminated = 0
        self.truncated = 0
        self.steps = 0
        self.nan_steps = 0
        self._action_elems = 0
        self._saturated_elems = 0
        # ★08.30 신설 — 성공률. env 가 `_success_now`(순간 성공 플래그)를 노출하면
        #   **매 스텝 OR 누적**해 "에피소드 중 한 번이라도 성공"을 센다.
        #   ★★종료 시점에 읽으면 안 된다 — `_reset_idx` 가 `step()` 안에서 그 버퍼를
        #   0 으로 지우고 나서 반환하므로 항상 False 다(08.30 실측: 9조건 전부 0.0000).
        #   env 의 `stage/success` 도 같은 OR 누적 규약이라 의미가 일치한다.
        self.successes = 0
        self.success_episodes = 0
        self._succ_flag = torch.zeros(self.num_envs, dtype=torch.bool)

    # ------------------------------------------------------------------
    def add_step(self, rewards, dones, actions, env=None) -> None:
        import torch

        rew = torch.as_tensor(rewards).detach().float().reshape(-1).cpu()
        done = torch.as_tensor(dones).detach().reshape(-1).cpu().bool()
        self.steps += 1

        self._returns += rew.double()
        self._lengths += 1

        act = torch.as_tensor(actions).detach().float().cpu()
        if not torch.isfinite(act).all():
            self.nan_steps += 1
        finite = act[torch.isfinite(act)]
        self._action_elems += act.numel()
        self._saturated_elems += int((finite.abs() >= SATURATION_LEVEL).sum())

        # 성공은 **매 스텝** OR 누적한다(리셋이 버퍼를 지우기 전에 읽는 유일한 방법).
        _succ = getattr(env, "_success_now", None) if env is not None else None
        _has_succ = _succ is not None
        if _has_succ:
            _s = torch.as_tensor(_succ).detach().reshape(-1).cpu().bool()
            if _s.numel() == self._succ_flag.numel():
                self._succ_flag |= _s

        if not bool(done.any()):
            return

        idx = torch.nonzero(done, as_tuple=False).reshape(-1)
        if _has_succ:
            self.successes += int(self._succ_flag[idx].sum())
            self.success_episodes += int(idx.numel())
            self._succ_flag[idx] = False
        for i in idx.tolist():
            self.episode_returns.append(float(self._returns[i]))
            self.episode_lengths.append(int(self._lengths[i]))
            self._returns[i] = 0.0
            self._lengths[i] = 0

        term = getattr(env, "reset_terminated", None) if env is not None else None
        tout = getattr(env, "reset_time_outs", None) if env is not None else None
        if term is not None and tout is not None:
            term = torch.as_tensor(term).detach().reshape(-1).cpu().bool()
            tout = torch.as_tensor(tout).detach().reshape(-1).cpu().bool()
            self.terminated += int((done & term).sum())
            self.truncated += int((done & tout & ~term).sum())

    # ------------------------------------------------------------------
    @property
    def episodes(self) -> int:
        return len(self.episode_returns)

    def saturated_frac(self) -> float:
        return self._saturated_elems / self._action_elems if self._action_elems else 0.0

    def report(self, *, task: str, missing) -> str:
        import statistics

        lines = ["", "EVALSUMMARY" + "=" * 55,
                 f"[EVAL] task={task}  (공통 지표 — 태스크 고유 어댑터 없음)"]
        if missing:
            shown = ", ".join(missing[:6]) + (" …" if len(missing) > 6 else "")
            lines.append(f"  grasp_v2 어댑터 미적용 사유 — env 미노출 속성: {shown}")
        if not self.episode_returns:
            lines.append(f"  에피소드 0 (스텝 {self.steps}) — 아직 종료된 에피소드가 없다")
        else:
            ret, ln = self.episode_returns, self.episode_lengths
            lines.append(
                f"  에피소드 {len(ret)}  리턴 평균 {statistics.fmean(ret):.4g}"
                + (f" (중앙 {statistics.median(ret):.4g})" if len(ret) > 1 else "")
            )
            lines.append(f"  에피소드 길이 평균 {statistics.fmean(ln):.1f} 스텝")
            if self.success_episodes:
                lines.append(
                    f"  ★성공률 {self.successes / self.success_episodes:.4f} "
                    f"({self.successes}/{self.success_episodes})")
            else:
                lines.append("  성공률 — env 가 _success_now 를 안 낸다")
            if self.terminated or self.truncated:
                total = self.terminated + self.truncated
                lines.append(
                    f"  종료 사유: terminated {self.terminated} "
                    f"({self.terminated / total:.2f}) / truncated {self.truncated} "
                    f"({self.truncated / total:.2f})"
                )
            else:
                lines.append("  종료 사유: env 가 reset_terminated/reset_time_outs 를 안 낸다")
        lines.append(
            f"  액션: 포화(|a|≥{SATURATION_LEVEL}) {self.saturated_frac():.3f}  "
            f"비유한 스텝 {self.nan_steps}/{self.steps}"
        )
        lines.append("EVALSUMMARY" + "=" * 55)
        return "\n".join(lines)
