"""grasp_lift_fabric cfg — `grasp_s2r` 전면 상속 + **손 제어만** 이 트랙 고유.

사용자 지시(08.27): "현재 핸드 제어 부분 빼고, grasp-s2r 세팅으로 변경"
(범위 = 전체 동기: obs·스폰·종료까지).

★왜 상속인가: 두 트랙은 이제 **손 제어 하나만 다르다**(자매 = 3채널 시너지 21D /
  우리 = 13자유 관절 직접 지시 19D). 복제하면 드리프트를 못 막는다 — 08.26 사용자
  확정 "복제는 드리프트를 못 막는다, import 가 막는다". 보상·obs·스폰·goal·종료·
  마커는 전부 자매 정의를 그대로 쓴다.

★자매 파일(`agnostic/tasks/grasp_s2r/*`)은 **다른 세션 소유**다 — 읽기·상속만 하고
  절대 수정하지 않는다.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ..grasp_s2r.grasp_s2r_env_cfg import GraspS2REnvCfg
from ..grasp_s2r.robot_profiles import PROFILES


def side_of(profile_name: str) -> str:
    """프로필 이름 → 관절 이름 규약의 좌우 접두사 ('tesollo_right' → 'r')."""
    return profile_name.split("_")[-1][0]


def resolve_frozen(profile_name: str, override: tuple[str, ...]) -> tuple[str, ...]:
    """`{side}` 치환 후 **그 프로필에 실재하는** 고정 관절 이름만 반환.

    부분 매칭은 오타이므로 fail-loud. 0 개 매칭은 "이 오버라이드가 겨냥하지 않는
    프로필"(예: 2 지 그리퍼)이라 빈 튜플을 준다 — env 가 부팅 로그에 찍는다.
    """
    p = PROFILES[profile_name]
    names = tuple(n.replace("{side}", side_of(profile_name)) for n in override)
    have = [n for n in names if n in p.hand_joint_names]
    if have and len(have) != len(names):
        missing = [n for n in names if n not in p.hand_joint_names]
        raise RuntimeError(
            f"[{profile_name}] 고정 관절 오버라이드 일부만 해석됨 — 없는 이름 "
            f"{missing}. 오타이거나 프로필 관절 이름 규약이 바뀌었다.")
    return tuple(have)


@configclass
class GraspLiftFabricEnvCfg(GraspS2REnvCfg):
    # ================================================================
    # 손 제어 — 이 트랙 고유 (나머지는 전부 GraspS2REnvCfg 상속)
    # ================================================================
    # ★외전(`_1`)은 정책이 제어하지 않고 홈에 고정한다 — 손가락 교차를 자유도
    #   수준에서 없앤다. 소지는 굴곡축이 없어 통째로 고정(pinky_1/2), 엄지 `_2` 는
    #   가동폭 0° 라 액션을 줘도 안 움직인다.
    frozen_hand_joints_override: tuple[str, ...] = (
        "{side}_hj_thumb_1", "{side}_hj_thumb_2",
        "{side}_hj_index_1", "{side}_hj_middle_1", "{side}_hj_ring_1",
        "{side}_hj_pinky_1", "{side}_hj_pinky_2",
    )
    # ★pinky_1 은 **펴진 채** 시작한다. 60° 고정의 원 근거는 "pinky_1 이 굴곡 자유도를
    #   pinky_2 로 재분배한다"였는데 pinky_2 도 고정이라 그 굴곡축을 얼려버렸다 —
    #   근거가 소멸하고 소지가 영구히 벌어진 채 가짜 접촉만 만들었다(08.27 실측).
    hand_home_override: tuple[tuple[str, float], ...] = (("{side}_hj_pinky_1", 0.0),)
    # 감쌈 분모에서 제외할 손가락(굴곡축 부재).
    hand_unusable_fingers: tuple[str, ...] = ("pinky",)
    # 관절 한계 여유 [rad] — a=+1 이 가는 굴곡 한계에서 뺀다.
    hand_limit_margin: float = 0.0

    # ---- 시너지 전용 기구는 OFF, **보상 게이트는 유지** -------------------------
    # 우리 손은 **절대 관절 목표**다: a=−1 이 홈(펴짐), a=+1 이 굴곡 한계.
    #   · contact_freeze — 닿은 마디의 **누산 delta** 를 멈추는 장치. 우리 매핑엔
    #     누산기가 없다(매 스텝 절대 각도) → 표현 불가.
    #   · couple_four    — 4 지를 채널별 평균으로 묶는 장치. 채널이 없다 → 표현 불가.
    synergy_contact_freeze: bool = False
    couple_four_fingers: bool = False
    # ★★close_gate 는 **켠다**(08.27 사용자 원칙: "로봇 특수성을 제외하고 reward
    #   design 은 유지"). 임계가 부팅 FK 로 실측되는 `r_cage` 하나라 로봇 비의존이고,
    #   자매 `6632002` 부터 `grasp` 항이 이 게이트를 직접 곱한다 — 끄면 우리 보상이
    #   자매와 갈린다.
    # ★단 **손 액션에는 걸지 않는다**(핸드 제어는 이 트랙 고유). 자매는 게이트를
    #   보상과 시너지 delta 양쪽에 걸지만, 우리는 보상 쪽만 받는다.
    close_gate_enabled: bool = True

    def __post_init__(self):
        super().__post_init__()
        # 부모가 정한 시너지 액션 차원을 **자유 관절 수**로 갈아끼운다.
        # ★obs/state 공식을 복제하지 않는다 — 둘 다 action_space 를 한 번씩만
        #   포함하므로 차분만 더하면 부모 공식이 그대로 유지된다.
        n_free = (len(PROFILES[self.profile_name].hand_joint_names)
                  - len(resolve_frozen(self.profile_name,
                                       self.frozen_hand_joints_override)))
        delta = (6 + n_free) - self.action_space
        self.action_space += delta
        self.observation_space += delta
        self.state_space += delta


def resolve_cfg(cfg: GraspLiftFabricEnvCfg) -> GraspLiftFabricEnvCfg:
    """hydra CLI 오버라이드 뒤 파생값을 다시 맞춘다 — **멱등**.

    ★hydra 는 cfg **필드만** 덮어쓰고 `__post_init__` 을 다시 돌리지 않는다. 프로필을
      바꾸면 로봇 자산·차원이 옛 프로필 기준으로 남아 "조용히 틀린 조합"으로 돌아간다.
      probe 들이 이 이름을 쓴다(구 설계에서 이어진 공개 API).
    """
    cfg.__post_init__()
    return cfg


@configclass
class GraspLiftFabricTesolloRightEnvCfg(GraspLiftFabricEnvCfg):
    profile_name: str = "tesollo_right"


@configclass
class GraspLiftFabricGripperLeftEnvCfg(GraspLiftFabricEnvCfg):
    profile_name: str = "gripper_left"
