"""BimanualPair — RobotProfile 두 개(좌/우)를 하나의 양팔 태스크 뷰로 합성한다.

modules/ 는 한쪽 팔 단위 프로필만 안다(tasks → modules 단방향 의존).
양팔 태스크가 필요로 하는 것은:
  · 하나의 articulation 에 대한 **양팔 활성** init/actuator (프로필의 tuck/idle 을 버림)
  · source(붓는 팔) / receiver(받는 팔) 역할별 프로필 접근

★태스크 코드에 `r_`/`l_` 리터럴 금지 계약을 지키기 위해 합성은 전부
  프로필의 regex(arm_joint_regex/hand_joint_regex)로 한다.

두 번째 bimanual 태스크가 생기면 modules/ 로 승격한다 — 지금 올리면 학습 중인
grasp 트랙과 공유 표면만 늘어난다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from openarm.agnostic.modules import robots as _rb


def _side_keys(profile: _rb.RobotProfile, mapping: dict) -> dict:
    """mapping 에서 이 프로필의 활성 팔+손 관절에 해당하는 항목만 추린다."""
    pat = re.compile(
        f"^({profile.arm_joint_regex}|{profile.hand_joint_regex})$")
    return {k: v for k, v in mapping.items() if pat.match(k)}


@dataclass(frozen=True)
class BimanualPair:
    """같은 자산의 반대편 프로필 쌍. source 가 붓고 receiver 가 받는다."""

    source: _rb.RobotProfile
    receiver: _rb.RobotProfile

    def __post_init__(self) -> None:
        s, r = self.source, self.receiver
        if s.asset is not r.asset:
            raise ValueError(
                f"양팔 쌍은 같은 자산이어야 한다: {s.asset.name} vs {r.asset.name}")
        if s.side == r.side:
            raise ValueError(f"양팔 쌍은 반대편이어야 한다: 둘 다 '{s.side}'")
        for p in (s, r):
            if p.fabric_class is None:
                raise ValueError(
                    f"프로필 '{p.name}' 은 Fabrics 자산이 없다(fabric_class=None)")

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.source.asset.short

    @property
    def asset(self) -> _rb.RobotAsset:
        return self.source.asset

    @property
    def init_joint_pos(self) -> dict:
        """양팔 **활성** init: source 홈 + receiver 홈 + head.

        ★프로필의 init_joint_pos 에는 반대팔 tuck 이 섞여 있다 — 각 프로필에서
          **자기 쪽**(활성) 항목만 추리고, head 는 어느 쪽에서 가져와도 같다.
        """
        head = {k: v for k, v in self.source.init_joint_pos.items()
                if not _side_keys(self.source, {k: v})
                and not _side_keys(self.receiver, {k: v})}
        return {
            **head,
            **_side_keys(self.source, self.source.init_joint_pos),
            **_side_keys(self.receiver, self.receiver.init_joint_pos),
        }

    @property
    def actuator_specs(self) -> dict:
        """양팔 **활성 게인** actuator: 각 프로필의 active_* 그룹을 역할명으로 rename.

        idle 그룹은 버린다 — 양팔 다 제어 대상이므로 idle 게인(존재한다면)이
        섞이면 안 된다. head 는 한 번만.
        """
        out: dict = {}
        for role, p in (("src", self.source), ("rcv", self.receiver)):
            for gname, spec in p.actuator_specs.items():
                if gname.startswith("active_"):
                    out[f"{role}_{gname[len('active_'):]}"] = spec
                elif gname == "head" and "head" not in out:
                    out["head"] = spec
        return out


# =============================================================================
# 자산별 쌍 자동 유도 — 같은 asset 의 r/l 프로필이 둘 다 Fabrics 를 가지면 등록.
# =============================================================================
PAIRS: dict[str, BimanualPair] = {}
SKIPPED: dict[str, str] = {}   # 조용히 빠뜨리지 않는다(grasp config 규약과 동일)


def _build_pairs() -> None:
    by_asset: dict[str, dict[str, _rb.RobotProfile]] = {}
    for p in _rb.PROFILES.values():
        by_asset.setdefault(p.asset.short, {})[p.side] = p
    for short, sides in sorted(by_asset.items()):
        if "r" not in sides or "l" not in sides:
            SKIPPED[short] = f"프로필이 한쪽뿐: {sorted(sides)}"
            continue
        try:
            # source=우(붓기), receiver=좌(받기) — 실기 배치 규약.
            PAIRS[short] = BimanualPair(source=sides["r"], receiver=sides["l"])
        except ValueError as e:
            SKIPPED[short] = str(e)


_build_pairs()
DEFAULT_PAIR = "bis"


def get_pair(name: str) -> BimanualPair:
    if name not in PAIRS:
        reason = SKIPPED.get(name, "미등록 자산")
        raise KeyError(
            f"양팔 쌍 '{name}' 없음 ({reason}). 가능: {sorted(PAIRS)}")
    return PAIRS[name]
