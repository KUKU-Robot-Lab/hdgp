"""BimanualPair — RobotProfile 두 개(좌/우)를 하나의 양팔 태스크 뷰로 합성한다.

`modules/robot_profiles.py` 는 **팔 단위** 프로필만 안다(tasks → modules 단방향 의존).
양팔 태스크가 필요로 하는 것은:
  · 하나의 articulation 에 대한 **양팔 활성** init/actuator (각 프로필의 유휴측을 버림)
  · source(붓는 팔) / receiver(받는 팔) 역할별 프로필 접근

★09.13 레지스트리 교체. 구판은 `modules/robots.py`(a0~a3 자산)를 읽었는데 그 자산 4종은
  09.05 라인업 교체로 전부 폐기(archived)돼 USD 가 없다 — 이 트랙은 부팅 자체가 안 됐다.
  현행 단일 출처는 `modules/robot_profiles.py` 다(프로필에 `side`/`asset` 필드가 없어
  regex 접두사와 usd_relpath 로 유도한다).

★태스크 코드에 `r_`/`l_` 리터럴 금지 계약을 지키기 위해 합성은 전부
  프로필의 regex(arm_joint_regex/hand_joint_regex)로 한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from openarm.agnostic.modules import robot_profiles as _rp

# usd 디렉터리 → gym id 로봇 슬롯. ★train.py 의 run_naming 정규식이 `open-\w+` 라
#   하이픈이 든 자산 이름(dg5f-m-short)을 그대로 쓸 수 없다 — 여기서 짧은 이름을 준다.
PAIR_SHORT_NAMES: dict[str, str] = {
    "openarm_dg5f-m-short_bi_rl": "short",
}


def side_of(profile: _rp.RobotProfile) -> str:
    """프로필이 제어하는 팔의 접두사('r'|'l') — arm_joint_regex 에서 유도."""
    m = re.match(r"^([rl])_aj_", profile.arm_joint_regex)
    if m is None:
        raise ValueError(
            f"프로필 '{profile.name}' 의 arm_joint_regex 에서 측을 유도할 수 없다: "
            f"{profile.arm_joint_regex!r}")
    return m.group(1)


def _side_keys(profile: _rp.RobotProfile, mapping: dict) -> dict:
    """mapping 에서 이 프로필의 활성 팔+손 관절에 해당하는 항목만 추린다."""
    pat = re.compile(
        f"^({profile.arm_joint_regex}|{profile.hand_joint_regex})$")
    return {k: v for k, v in mapping.items() if pat.match(k)}


def _side_actuators(profile: _rp.RobotProfile) -> dict:
    """actuator 그룹 중 joint_names_expr 이 **전부 이 프로필의 측**으로 시작하는 것만."""
    side = side_of(profile)
    out = {}
    for name, spec in profile.actuator_specs.items():
        exprs = [str(e) for e in spec.get("joint_names_expr", [])]
        if exprs and all(e.startswith(f"{side}_") for e in exprs):
            out[name] = spec
    return out


@dataclass(frozen=True)
class BimanualPair:
    """같은 자산의 반대편 프로필 쌍. source 가 붓고 receiver 가 받는다."""

    short: str
    source: _rp.RobotProfile
    receiver: _rp.RobotProfile

    def __post_init__(self) -> None:
        s, r = self.source, self.receiver
        if s.usd_relpath != r.usd_relpath:
            raise ValueError(
                f"양팔 쌍은 같은 자산이어야 한다: {s.usd_relpath} vs {r.usd_relpath}")
        if side_of(s) == side_of(r):
            raise ValueError(f"양팔 쌍은 반대편이어야 한다: 둘 다 '{side_of(s)}'")
        for p in (s, r):
            if p.fabric_class is None:
                raise ValueError(
                    f"프로필 '{p.name}' 은 Fabrics 자산이 없다(fabric_class=None)")

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.short

    @property
    def usd_relpath(self) -> str:
        return self.source.usd_relpath

    @property
    def init_joint_pos(self) -> dict:
        """양팔 **활성** init: source 홈 + receiver 홈 + head.

        ★프로필의 init_joint_pos 에는 반대팔 유휴 자세가 섞여 있다 — 각 프로필에서
          **자기 쪽**(활성) 항목만 추리고, head 등 나머지는 source 에서 가져온다.
        """
        rest = {k: v for k, v in self.source.init_joint_pos.items()
                if not _side_keys(self.source, {k: v})
                and not _side_keys(self.receiver, {k: v})}
        return {
            **rest,
            **_side_keys(self.source, self.source.init_joint_pos),
            **_side_keys(self.receiver, self.receiver.init_joint_pos),
        }

    @property
    def actuator_specs(self) -> dict:
        """양팔 **활성 게인** actuator: 각 프로필에서 자기 측 그룹만. head 는 source 것."""
        out: dict = {}
        for role, p in (("src", self.source), ("rcv", self.receiver)):
            for gname, spec in _side_actuators(p).items():
                out[f"{role}_{gname}"] = spec
        head = {k: v for k, v in self.source.actuator_specs.items()
                if k not in _side_actuators(self.source)
                and k not in _side_actuators(self.receiver)
                and not any(str(e).startswith(("r_", "l_"))
                            for e in v.get("joint_names_expr", []))}
        out.update(head)
        return out


# =============================================================================
# 자산별 쌍 자동 유도 — 같은 USD 의 r/l 프로필이 둘 다 Fabrics 를 가지면 등록.
# =============================================================================
PAIRS: dict[str, BimanualPair] = {}
SKIPPED: dict[str, str] = {}   # 조용히 빠뜨리지 않는다(grasp config 규약과 동일)


def _build_pairs() -> None:
    by_asset: dict[str, dict[str, _rp.RobotProfile]] = {}
    for p in _rp.PROFILES.values():
        by_asset.setdefault(p.usd_relpath, {})[side_of(p)] = p
    for usd, sides in sorted(by_asset.items()):
        key = usd.split("/")[1] if "/" in usd else usd
        short = PAIR_SHORT_NAMES.get(key, key)
        if "r" not in sides or "l" not in sides:
            SKIPPED[short] = f"프로필이 한쪽뿐: {sorted(sides)}"
            continue
        try:
            # source=우(붓기), receiver=좌(받기) — 실기 배치 규약.
            PAIRS[short] = BimanualPair(short=short, source=sides["r"], receiver=sides["l"])
        except ValueError as e:
            SKIPPED[short] = str(e)


_build_pairs()
DEFAULT_PAIR = "short"


def get_pair(name: str) -> BimanualPair:
    if name not in PAIRS:
        reason = SKIPPED.get(name, "미등록 자산")
        raise KeyError(
            f"양팔 쌍 '{name}' 없음 ({reason}). 가능: {sorted(PAIRS)}")
    return PAIRS[name]
