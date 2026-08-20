"""로봇 레지스트리 — 로봇 종속 정보가 모이는 **유일한 곳**.

설계 목표: 잘 설계된 보상·환경만으로 `hdgp/assets/robot/` 의 어떤 로봇을 소환해도
태스크가 성공해야 한다. 태스크 코드는 이 파일의 필드만 참조하고 조인트/바디 **이름을
하드코딩하지 않는다**.

2계층으로 나눈다:
    RobotAsset   — USD 파일 단위 (자산 4종)
    RobotProfile — **팔 단위** (자산 × 좌/우)

새 로봇 추가 = 여기에 항목 1개 추가가 전부여야 한다(합격 조건).

의도적으로 isaaclab 을 import 하지 않는다(순수 데이터). ArticulationCfg 조립은 env_cfg
쪽에서 하고, 계약 테스트는 Isaac 앱 없이 URDF 원본과 직접 대조한다
(`tests/test_robot_registry.py`).

★검증 상태에 대한 정직한 표기
    선언 ≠ 검증이다. `tesollo` 계열 외 프로필은 URDF 이름 계약만 통과했을 뿐
    물리/IK probe 를 돌린 적이 없다. `RobotProfile.probe_verified` 로 구분한다.
"""

from __future__ import annotations

import dataclasses as _dc
from dataclasses import dataclass, field


# =============================================================================
# 자산
# =============================================================================
@dataclass(frozen=True)
class RobotAsset:
    """USD 파일 하나 = 자산 하나."""

    name: str                  # "openarm_tesollo_bi_s_rl" (= USD 디렉터리/파일 이름)
    tag: str                   # run_naming.ASSET_TAGS 와 같은 어휘 ("a2")
    short: str                 # gym id / 로그 경로용 짧은 이름 ("bis")
    note: str = ""

    @property
    def usd_relpath(self) -> str:
        """`hdgp/assets/` 기준 상대 경로."""
        return f"robot/{self.name}/{self.name}.usd"

    @property
    def urdf_relpath(self) -> str:
        """`rl_ws/urdf/generated/rl/` 기준 — 계약 테스트가 이름을 대조하는 원본."""
        return f"{self.name}.urdf"


TESOLLO_BI_S = RobotAsset(
    name="openarm_tesollo_bi_s_rl", tag="a2", short="bis",
    note="좌우 DG-5F-S 20 DOF. grasp_v1/v2 가 쓰는 현행 자산.",
)
TESOLLO_BI = RobotAsset(
    name="openarm_tesollo_bi_rl", tag="a3", short="bi",
    note="좌우 DG-5F 20 DOF (구 기구학). ★run_naming.ASSET_TAGS 에 원래 없어 추가했다.",
)
TESOLLO_SENSOR = RobotAsset(
    name="openarm_tesollo_sensor_rl", tag="a1", short="sens",
    note="우 DG-5F 20 DOF + 좌 2-DOF 평행 그리퍼(prismatic) + D435i 헤드.",
)
RH56F1_BI = RobotAsset(
    name="openarm_bi_rh56f1_rl", tag="a0", short="rh56",
    note="좌우 Inspire RH56F1. 엄지 4 + 나머지 2×4 = 12 관절.",
)

ASSETS: dict[str, RobotAsset] = {
    a.name: a for a in (TESOLLO_BI_S, TESOLLO_BI, TESOLLO_SENSOR, RH56F1_BI)
}


# =============================================================================
# 프로필
# =============================================================================
@dataclass(frozen=True)
class RobotProfile:
    """자산 하나의 한쪽 팔 — 태스크가 참조하는 전부."""

    name: str
    asset: RobotAsset
    side: str                              # "r" | "l"

    # ---- 제어 차원 (env 부팅 시 regex 해석 결과와 대조해 fail-loud) --------------
    num_arm_joints: int
    num_hand_joints: int
    arm_joint_regex: str
    hand_joint_regex: str

    # ---- Fabrics (팔 제어기) ----------------------------------------------------
    # fabric_class 가 None 이면 이 프로필로 Fabrics 태스크를 띄울 수 없다.
    # ★조용히 diffIK 로 폴백하지 않는다 — env 가 RuntimeError 로 멈춘다.
    fabric_class: str | None
    fabric_robot_dir: str | None           # FABRICS models/robots/urdf/<dir>/<dir>.urdf
    palm_body: str                         # Fabrics 가 추종하는 EE body
    # ★★fabric 의 관절 **순서**(자산 조인트 이름으로 표기). articulation 순서와 다르다.
    #   articulation 은 depth-major (index_1, middle_1, pinky_1, ring_1, thumb_1, index_2, …)
    #   fabric URDF 는 **finger-major** (thumb_1..4, index_1..4, …) 다 — 실측 확인.
    #   이 순서로 재조립하지 않으면 fabric 이 엉뚱한 손 자세로 충돌구 FK 를 계산해
    #   존재하지 않는 자기충돌을 피하려고 팔을 밀어낸다.
    fabric_joint_order: tuple

    # ---- 접촉 -------------------------------------------------------------------
    # 손가락 이름 → body 튜플. body 마다 ContactSensor 를 **개별** 생성한다
    # (다중 body 단일 센서는 force_matrix_w 가 0 을 반환한다 — grasp_sensor 실측 함정).
    #   tip  : 핀치 접촉(센서팁)
    #   wrap : 감쌈 접촉(중간·원위 마디) — envelope_frac 의 분자
    # ★_tip / _4 / _3 은 서로 다른 링크다. 인벨롭에선 센서팁이 잘 안 닿으므로
    #   감쌈 판정을 팁으로 하면 안 된다(07.29 재발방지).
    # ★wrap 이 비어 있는 프로필(2지 그리퍼)은 envelope_frac := grip_frac 으로 정의한다.
    #   구조적으로 감쌀 마디가 없는 손을 보상에서 깎지 않기 위해서다.
    finger_tip_bodies: dict
    finger_wrap_bodies: dict
    # 대향 그룹: A(엄지/조1) AND B(나머지/조2) 동시 접촉 = 파지 성립.
    contact_group_a: tuple
    contact_group_b: tuple

    # ---- 관측용 손끝 body (approach 의 tip_side_dist) ----------------------------
    fingertip_bodies: tuple
    # ★손가락이 **서로 교차**할 수 있게 하는 외전(abduction) 관절. 정책 제어에서 빼고
    #   init 값에 고정한다. self-collision 을 끈 채 두면(성능상 그렇게 한다) 손가락이
    #   서로를 통과해 `envelope_frac` 이 물리적으로 불가능한 감쌈을 세게 되는데,
    #   교차의 자유도 자체를 없애면 그 문제가 사라진다.
    #   실측(probe_penetration): self-coll OFF 에서 다른 손가락 링크 최소거리 평균 7.7mm
    #   (마디 반경 ~10mm) → 100% env 겹침.
    #   선례: grasp_v2 도 abduction 을 고정했다가 ADR 로 열었다.
    frozen_hand_joints: tuple

    # ---- 초기 상태 / 액추에이터 --------------------------------------------------
    init_joint_pos: dict
    # 그룹명 → ImplicitActuatorCfg kwargs. **전 DOF 커버 필수** —
    # 커버리지 누락 관절은 조용히 무구동 자유회전한다(adf0b24 교훈).
    actuator_specs: dict

    # ---- palm 워크스페이스 박스 (env-local xyz) -----------------------------------
    # ★★이건 **로봇 종속 정보**다(팔의 도달범위). 태스크 cfg 에 상수로 두면 자산이
    #   바뀔 때 조용히 틀린다 — 실제로 그랬다. grasp_v1(sensor_rl) 박스를 bi_s 에
    #   그대로 물려받았는데 palm 링크가 54.8mm 달라, 박스의 **62% 가 도달 불가**
    #   (오차 >10mm, 중앙값 136mm)였다. 정책 액션이 포화(|a|≈0.77)라 주로 박스
    #   가장자리 = 가장 못 닿는 곳을 명령했고, 액션을 바꿔도 결과가 안 바뀌어
    #   **겨냥을 배울 수 없었다**(approach 0.30 고원, 접촉 0).
    #   → probe_workspace_reach.py 로 재고 채운다. 게이트: 오차<10mm 가 90% 이상.
    palm_box_min: tuple
    palm_box_max: tuple

    # ---- 씬 배치 ----------------------------------------------------------------
    object_spawn_center: tuple             # env-local (x, y)
    # ★작업면 **상면 높이**. 물체 스폰 z 가 아니다 —
    #   스폰 z = surface_z + 물체 원점 오프셋(자산 속성) + 패딩 으로 env 가 계산한다.
    #   둘을 합쳐 하나로 두면 물체가 바뀔 때마다 조용히 틀린 높이가 된다
    #   (실측: cup_big 원점은 바닥+77.3mm, shaker 는 +92.1mm).
    #   env.usd 의 top_plate 상면 = 0.200 (점군 실측).
    surface_z: float = 0.200

    # ---- 에이전트 오버라이드 (선택) ---------------------------------------------
    agent_cfg_name: str | None = None      # None = modules.agents 기본값

    # ---- 검증 상태 --------------------------------------------------------------
    probe_verified: bool = False           # 물리/IK probe 를 통과했는가
    # palm 박스를 probe_workspace_reach 로 실측했는가(게이트: 오차<10mm 가 90% 이상).
    # False 면 다른 로봇 값을 물려받은 것이라 신뢰 금지 — 이번 사고의 원인이 정확히 그것이다.
    palm_box_verified: bool = False
    notes: tuple = ()

    # ------------------------------------------------------------------
    @property
    def fingers(self) -> tuple:
        return tuple(self.finger_tip_bodies.keys())

    @property
    def has_wrap_sensors(self) -> bool:
        return any(len(v) > 0 for v in self.finger_wrap_bodies.values())

    def sensor_bodies(self, finger: str) -> tuple:
        """해당 손가락의 전체 센서 body (tip + wrap)."""
        return tuple(self.finger_tip_bodies[finger]) + tuple(
            self.finger_wrap_bodies.get(finger, ())
        )

    @property
    def all_sensor_bodies(self) -> tuple:
        out: list[str] = []
        for f in self.fingers:
            out.extend(self.sensor_bodies(f))
        return tuple(out)

    @property
    def task_short(self) -> str:
        """gym id 의 로봇 슬롯 — 로그가 `log/rl_games/open-<short>/<side>/...` 로 갈린다."""
        return self.asset.short


# =============================================================================
# 액추에이터 게인 — 근거
#   팔  400/80 + friction(0.213/0.493/0.151)  ← real2sim 07.29 우팔 캘리브
#   손  k5/kd2 + effort 1.5 N·m               ← 08.16 S1~S4 스윕
#       (구 400/60 은 토크 포화 레짐: 요구 143 N·m = effort limit 의 19배라
#        목표를 더 밀어도 힘이 안 오른다 = retighten/squeeze 실패의 공통 원인)
# =============================================================================
_ARM_GAINS = dict(stiffness=400.0, damping=80.0)
# ★URDF/USD 실측 effort limit [N·m] — 부위별로 다르다.
#   USD 에 이미 들어 있어(maxForce 40/40/27/27/7/7/7) 지정하지 않아도 적용되지만,
#   **명시해 두면 자산이 바뀌었을 때 조용히 달라지지 않는다.**
_ARM_EFFORT = {"proximal": 40.0, "elbow": 27.0, "wrist": 7.0}
_HAND_GAINS = dict(stiffness=5.0, damping=2.0, effort_limit_sim=1.5)
_FRICTION = {"proximal": 0.213, "elbow": 0.493, "wrist": 0.151}


def _arm_actuators(prefix: str, side: str) -> dict:
    """한쪽 팔의 부위별 actuator 3그룹 (friction 이 부위마다 다르다)."""
    return {
        f"{prefix}_arm_proximal": dict(joint_names_expr=[f"{side}_aj_[1-3]"],
                                       friction=_FRICTION["proximal"],
                                       effort_limit_sim=_ARM_EFFORT["proximal"], **_ARM_GAINS),
        f"{prefix}_arm_elbow":    dict(joint_names_expr=[f"{side}_aj_4"],
                                       friction=_FRICTION["elbow"],
                                       effort_limit_sim=_ARM_EFFORT["elbow"], **_ARM_GAINS),
        f"{prefix}_arm_wrist":    dict(joint_names_expr=[f"{side}_aj_[5-7]"],
                                       friction=_FRICTION["wrist"],
                                       effort_limit_sim=_ARM_EFFORT["wrist"], **_ARM_GAINS),
    }


_HEAD_ACTUATOR = {"head": dict(joint_names_expr=["head_j_(pan|tilt)"], **_ARM_GAINS)}

_TESOLLO_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _tesollo_hand_actuator(prefix: str, side: str) -> dict:
    return {f"{prefix}_hand": dict(
        joint_names_expr=[f"{side}_hj_[a-z]+_[1-4]"], **_HAND_GAINS)}


def _tesollo_hand_rest(side: str) -> dict:
    """엄지 대향 + 나머지 폄.

    ★★thumb_2(대향 관절)의 **부호가 좌우 반대**다(URDF 실측):
          r_hj_thumb_2 = [-2.670, 0.000]   l_hj_thumb_2 = [+0.000, +2.670]
       우측 값을 좌측에 그대로 쓰면 관절한계 밖이라 Articulation 검증에서 막힌다
       (실제로 막혔다). 이 저장소에서 반복된 "엄지 부호" 버그와 같은 부류다.
       thumb_3 은 좌우 대칭([-1.571, 1.571])이지만 대향 자세의 방향이 뒤집히므로
       같이 미러한다.
    """
    sg = 1.0 if side == "r" else -1.0
    q = {f"{side}_hj_{f}_{j}": 0.0 for f in _TESOLLO_FINGERS for j in (1, 2, 3, 4)}
    q[f"{side}_hj_thumb_2"] = sg * -1.57
    q[f"{side}_hj_thumb_3"] = sg * -0.5
    return q


# 팔 홈 관절값 — grasp_v1 의 런타임 고정 홈(palm 0.28,-0.38,0.42 / ez90·ey0·ex90)을
# sensor_rl 에서 IK 역산한 값(probe_solve_v1_home 08.20, 오차 2.2mm/0.6°).
# ★★bi_s 에서 그대로 쓰면 palm 자세가 다르다. 우팔 관절 기구학은 sensor_rl 과 **완전
#   동일**하지만 palm 링크 오프셋이 0.0698 → 0.015 (54.8mm) 로 바뀌었다(URDF 실측).
#   즉 손목은 같은 곳에 가지만 palm 은 54.8mm 어긋난다. probe 2 에서 재역산할 것.
_ARM_HOME_R = (0.0380, 0.4012, 0.6015, 0.9643, 0.0294, 0.7060, 0.4213)

# 좌우 미러 규약: aj_4(엘보)만 부호 유지, 나머지 반전.
# grasp_lift 실측값에서 추정했고, **URDF 관절한계로 독립 검증**했다:
#   aj_1/2/3/5/6/7 은 좌우 한계가 [-hi, -lo] 관계(미러), aj_4 만 [0, 2.443] 로 동일.
_MIRROR_SIGN = (-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0)

# 유휴 팔이 접고 있는 고정 자세(작업공간 밖). ★이 수치는 **좌팔** 자세다 —
# 우팔에 그대로 쓰면 관절한계 밖이다(URDF 실측 r_aj_2 ∈ [-0.175, +3.316] 인데 -0.671).
# grasp_lift 가 두 프로필에 같은 수치를 썼던 것이 그래서 우연히만 맞았다.
_ARM_TUCK_L = (-0.0431, -0.6706, -0.0961, 0.7342, -0.3750, -0.5678, -0.6709)

_HEAD_HOME = {"head_j_pan": 0.0, "head_j_tilt": 0.0}


def _mirror(q: tuple) -> tuple:
    return tuple(sg * v for sg, v in zip(_MIRROR_SIGN, q))


def _arm_q(side: str, values: tuple) -> dict:
    return {f"{side}_aj_{i}": float(v) for i, v in enumerate(values, start=1)}


def _arm_home(side: str) -> dict:
    """활성 팔 홈. 좌팔은 우팔 홈의 미러(★probe 2 에서 IK 재역산 대상)."""
    return _arm_q(side, _ARM_HOME_R if side == "r" else _mirror(_ARM_HOME_R))


def _arm_tuck(side: str) -> dict:
    """유휴 자세. 좌팔 기준값이라 우팔은 미러한다."""
    return _arm_q(side, _ARM_TUCK_L if side == "l" else _mirror(_ARM_TUCK_L))


_HOME_NOTE = (
    "팔 홈은 sensor_rl 기준 IK 역산값이다. bi_s 는 palm 오프셋이 54.8mm 다르므로 "
    "probe 2(홈 자세)에서 재역산해야 한다 — 추정 금지."
)
_MIRROR_NOTE = "좌팔 홈은 우팔 홈의 부호 미러(aj_4 제외)라 IK 검증을 거치지 않았다."

# grasp_v1 원좌표(우측). palm-pose 홈 기준 x0.22~0.38 × y-0.30~-0.10 전 격자 quiet 실측.
_SPAWN_R = (0.30, -0.20)
_SPAWN_L = (0.30, 0.20)

# probe_workspace_reach 실측(bi_s 우팔, 5^3 격자·180스텝) — 게이트 PASS:
#   오차<30mm 90.4% · 중앙값 8.4mm · p90 29.4mm · **컵 15cm 이내 8.7mm** · 홈 포함 ✓
#
# 대조 (같은 probe):
#   구 grasp_v1 승계 박스 x[0.20,0.65] y[-0.55,0.22] z[0.20,0.65]
#     → 오차<10mm 12.0% · 중앙값 136mm · 컵 주변 62mm   ★62% 가 도달 불가
#   중간 후보 x[0.16,0.44] y[-0.46,-0.06] z[0.30,0.56]
#     → <30mm 70.4% · 중앙값 8.6mm · 컵 주변 22mm       가장자리(x.44/z.30/y-.46)가 나쁨
#
# 왜 중요한가: 박스가 도달범위보다 크면 정책 액션이 포화(|a|≈0.77)했을 때 주로
# 못 닿는 곳을 명령하게 되고, 액션을 바꿔도 결과가 안 바뀌어 **겨냥을 배울 수 없다**
# (fab_test1~3 실측: approach 0.30 고원 · 명령목표가 컵에서 41cm · 접촉 0).
# ★z 상한 0.56 → 0.64. 0.56 이면 홈(0.418)에서 올릴 수 있는 최대가 0.142m 인데
#   과제가 요구하는 상승량은 0.15m 이다 — 정책이 a[2] 를 **최대치까지 밀어야** 겨우
#   닿고, action_rate 페널티까지 있어 그 극단을 탐색하기 어렵다.
#   (도달성 대가는 거의 없다: <30mm 90.4% → 87.2%, z 0.64 자체는 오차 8mm.)
#   리프트 여유 0.222m 확보.
_BOX_R = ((0.16, -0.42, 0.34), (0.40, -0.08, 0.64))
_BOX_L = ((0.16, 0.08, 0.34), (0.40, 0.42, 0.64))     # y 미러


# =============================================================================
# Tesollo DG-5F / DG-5F-S 양팔 자산 (bi_s, bi) — 좌우 20 DOF, 이름 규약 동일
# =============================================================================
def _tesollo_profile(
    *, asset: RobotAsset, side: str, fabric_dir: str, fabric_class: str,
    spawn_center: tuple, box: tuple, box_verified: bool = False, notes: tuple = (),
) -> RobotProfile:
    other = "l" if side == "r" else "r"
    return RobotProfile(
        name=f"{asset.short}_{'right' if side == 'r' else 'left'}",
        asset=asset,
        side=side,
        num_arm_joints=7,
        num_hand_joints=20,
        arm_joint_regex=f"{side}_aj_[1-7]",
        hand_joint_regex=f"{side}_hj_(thumb|index|middle|ring|pinky)_[1-4]",
        fabric_class=fabric_class,
        fabric_robot_dir=fabric_dir,
        palm_body=f"{side}_hl_palm",
        fabric_joint_order=(
            tuple(f"{side}_aj_{i}" for i in range(1, 8))
            + tuple(f"{side}_hj_{f}_{j}" for f in _TESOLLO_FINGERS for j in (1, 2, 3, 4))
        ),
        # 원위마디(_4)·중간마디(_3)가 감쌈, 센서팁(_tip)이 핀치.
        finger_tip_bodies={f: (f"{side}_hl_{f}_tip",) for f in _TESOLLO_FINGERS},
        finger_wrap_bodies={
            f: (f"{side}_hl_{f}_3", f"{side}_hl_{f}_4") for f in _TESOLLO_FINGERS
        },
        contact_group_a=("thumb",),
        contact_group_b=("index", "middle", "ring", "pinky"),
        fingertip_bodies=tuple(f"{side}_hl_{f}_tip" for f in _TESOLLO_FINGERS),
        # URDF 한계로 판별: index/middle/ring 의 _1 은 작고 비대칭(외전),
        # _2 는 큰 단방향(MCP 굴곡). ★pinky 만 _1=굴곡 / _2=외전 으로 뒤바뀐다.
        # thumb_1=외전·thumb_2=대향 — 대향은 rest(-1.57) 에 고정해 미리 마주보게 둔다.
        # ★grasp_v2 방식: **_1 관절 전체 고정** + 엄지 대향(_2) + pinky 외전(_2).
        #   grasp_v2 는 thumb_1/thumb_2/index_1/pinky_1/pinky_2 를 고정했다가 ADR 로 열었다.
        #   여기서는 _1 을 전부 고정해 손가락이 벌어지는 자유도를 없앤다.
        #   → 남는 자유도 = 굴곡(_2,_3,_4)뿐이라 손가락이 평행 평면에서만 움직인다.
        frozen_hand_joints=(
            f"{side}_hj_thumb_1", f"{side}_hj_thumb_2",
            f"{side}_hj_index_1", f"{side}_hj_middle_1",
            f"{side}_hj_ring_1", f"{side}_hj_pinky_1", f"{side}_hj_pinky_2",
        ),
        init_joint_pos={
            **_arm_home(side), **_tesollo_hand_rest(side),
            **_arm_tuck(other), **_tesollo_hand_rest(other),
            **_HEAD_HOME,
        },
        actuator_specs={
            **_arm_actuators("active", side), **_tesollo_hand_actuator("active", side),
            **_arm_actuators("idle", other), **_tesollo_hand_actuator("idle", other),
            **_HEAD_ACTUATOR,
        },
        palm_box_min=box[0],
        palm_box_max=box[1],
        palm_box_verified=box_verified,
        object_spawn_center=spawn_center,
        notes=notes,
    )


BIS_RIGHT = _tesollo_profile(
    asset=TESOLLO_BI_S, side="r",
    fabric_dir="openarm_tesollo_bi_s", fabric_class="OpenArmTeoslloPoseFabric",
    spawn_center=_SPAWN_R, box=_BOX_R, box_verified=True,
    notes=(_HOME_NOTE, "fabric URDF FK 오차 0.000mm 검증됨(grasp_v1, 08.17)."),
)
BIS_LEFT = _tesollo_profile(
    asset=TESOLLO_BI_S, side="l",
    fabric_dir="openarm_tesollo_bi_s_left", fabric_class="OpenArmTeoslloLeftPoseFabric",
    spawn_center=_SPAWN_L, box=_BOX_L,
    notes=(_HOME_NOTE, _MIRROR_NOTE,
           "좌측 fabric URDF 는 FK 교차검증 기록이 없다 — probe 4 대상."),
)
BI_RIGHT = _tesollo_profile(
    asset=TESOLLO_BI, side="r",
    fabric_dir="openarm_tesollo", fabric_class="OpenArmTeoslloPoseFabric",
    spawn_center=_SPAWN_R, box=_BOX_R,
    notes=(_HOME_NOTE, "구 DG-5F 기구학 — 마디 길이·회전축이 DG-5FS 와 다르다."),
)
BI_LEFT = _tesollo_profile(
    asset=TESOLLO_BI, side="l",
    fabric_dir="openarm_tesollo_left", fabric_class="OpenArmTeoslloLeftPoseFabric",
    spawn_center=_SPAWN_L, box=_BOX_L,
    notes=(_HOME_NOTE, _MIRROR_NOTE, "구 DG-5F 기구학."),
)


# =============================================================================
# sensor_rl — 우 DG-5F 20 DOF + 좌 2-DOF 평행 그리퍼(prismatic) + D435i 헤드
#   ★agnosticism 의 진짜 시험대다: 손 자유도 20 → 1, 감쌈 마디 없음, 대향 그룹이
#     엄지/4지가 아니라 조1/조2. 태스크 코드 수정 0 으로 돌아야 한다.
# =============================================================================
_GRIPPER_JAWS = ("jaw1", "jaw2")

# 헬퍼는 유휴측을 20관절 손으로 가정하므로, sensor_rl(좌측 2-DOF 그리퍼)은
# 유휴측 init/actuator 만 갈아끼운다.
SENS_RIGHT = _dc.replace(
    _tesollo_profile(
        asset=TESOLLO_SENSOR, side="r",
        fabric_dir="openarm_tesollo_sensor", fabric_class="OpenArmTeoslloPoseFabric",
        spawn_center=_SPAWN_R, box=_BOX_R,
        notes=(_HOME_NOTE,
               "좌측이 20관절 손이 아니라 2-DOF 그리퍼(prismatic)라 유휴측 규약이 다르다."),
    ),
    init_joint_pos={
        **_arm_home("r"), **_tesollo_hand_rest("r"),
        **_arm_tuck("l"),
        "l_hj_gripper_1": 0.044, "l_hj_gripper_2": 0.044,
        **_HEAD_HOME,
    },
    actuator_specs={
        **_arm_actuators("active", "r"), **_tesollo_hand_actuator("active", "r"),
        **_arm_actuators("idle", "l"),
        "idle_gripper": dict(joint_names_expr=["l_hj_gripper_[1-2]"], **_ARM_GAINS),
        **_HEAD_ACTUATOR,
    },
)

SENS_LEFT_GRIPPER = RobotProfile(
    name="sens_left",
    asset=TESOLLO_SENSOR,
    side="l",
    num_arm_joints=7,
    num_hand_joints=1,                      # l_hj_gripper_2 는 USD PhysX mimic(gearing=-1)
    arm_joint_regex="l_aj_[1-7]",
    hand_joint_regex="l_hj_gripper_1",
    fabric_class="OpenArmGripperLeftPoseFabric",
    fabric_robot_dir="openarm_tesollo_sensor_left_gripper",
    # ★이 fabric URDF 는 손 부분이 **2지 그리퍼가 아니라 DG-5F 손**이다(실측 rj_dg_*).
    #   arm IK 에는 쓸 수 있으나 손 충돌구는 존재하지 않는 손가락을 가리킨다 —
    #   probe 없이 신뢰 금지. 자산에 대응 관절이 없어 순서를 비워 둔다(팔만).
    fabric_joint_order=tuple(f"l_aj_{i}" for i in range(1, 8)),
    palm_body="l_hl_gripper_base",
    # 그리퍼에는 감쌀 마디가 없다 → wrap 비움. envelope_frac := grip_frac 로 정의된다.
    finger_tip_bodies={
        "jaw1": ("l_hl_gripper_left_finger",),
        "jaw2": ("l_hl_gripper_right_finger",),
    },
    finger_wrap_bodies={"jaw1": (), "jaw2": ()},
    contact_group_a=("jaw1",),
    contact_group_b=("jaw2",),
    fingertip_bodies=("l_hl_gripper_left_finger", "l_hl_gripper_right_finger"),
    frozen_hand_joints=(),      # 2지 그리퍼는 1-DOF, 교차 불가

    init_joint_pos={
        **_arm_q("l", _mirror(_ARM_HOME_R)),
        "l_hj_gripper_1": 0.044, "l_hj_gripper_2": 0.044,
        **_arm_tuck("r"), **_tesollo_hand_rest("r"),
        **_HEAD_HOME,
    },
    actuator_specs={
        **_arm_actuators("active", "l"),
        "active_gripper": dict(joint_names_expr=["l_hj_gripper_[1-2]"], **_ARM_GAINS),
        **_arm_actuators("idle", "r"), **_tesollo_hand_actuator("idle", "r"),
        **_HEAD_ACTUATOR,
    },
    palm_box_min=_BOX_L[0],
    palm_box_max=_BOX_L[1],
    object_spawn_center=_SPAWN_L,
    notes=(_HOME_NOTE, _MIRROR_NOTE,
           "2지 그리퍼는 jaw 수평 자세가 강제된다 — Fabrics 로 손목 ±45° 를 못 맞춰 "
           "gripper 트랙에서 ABORTED 된 이력이 있다(28°). 이 프로필은 그 재현 실험용."),
)


# =============================================================================
# rh56f1 — Inspire RH56F1 양손. 엄지 4 + (검·중·약·소)×2 = 12 관절
#   링크: _1 / _2 (+엄지 _3 _4) / _sensor(팁 F/T) / _tip
# =============================================================================
_RH_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _rh56_hand_rest(side: str) -> dict:
    q = {f"{side}_hj_thumb_{j}": 0.0 for j in (1, 2, 3, 4)}
    q.update({f"{side}_hj_{f}_{j}": 0.0
              for f in ("index", "middle", "ring", "pinky") for j in (1, 2)})
    return q


def _rh56_hand_actuator(prefix: str, side: str) -> dict:
    return {f"{prefix}_hand": dict(
        joint_names_expr=[f"{side}_hj_[a-z]+_[1-4]"], **_HAND_GAINS)}


def _rh56_profile(*, side: str, spawn_center: tuple) -> RobotProfile:
    other = "l" if side == "r" else "r"
    return RobotProfile(
        name=f"rh56_{'right' if side == 'r' else 'left'}",
        asset=RH56F1_BI,
        side=side,
        num_arm_joints=7,
        num_hand_joints=12,
        arm_joint_regex=f"{side}_aj_[1-7]",
        hand_joint_regex=f"{side}_hj_(thumb_[1-4]|(index|middle|ring|pinky)_[1-2])",
        # ★좌측 fabric URDF 가 **없다**(models/robots/urdf 실측). 클래스만 있고 자산이
        #   없으므로 좌팔 프로필은 fabric_class=None 으로 두어 env 가 fail-loud 하게 한다.
        #   조용히 우측 URDF 로 IK 를 풀면 좌팔이 우팔 기구학을 쓴다.
        fabric_class="OpenArmRh56f1PoseFabric" if side == "r" else None,
        fabric_robot_dir="openarm_rh56f1" if side == "r" else None,
        fabric_joint_order=(
            (
                tuple(f"{side}_aj_{i}" for i in range(1, 8))
                + tuple(f"{side}_hj_thumb_{j}" for j in (1, 2, 3, 4))
                + tuple(f"{side}_hj_{f}_{j}"
                        for f in ("index", "middle", "ring", "pinky") for j in (1, 2))
            )
            if side == "r" else ()
        ),
        palm_body=f"{side}_hl_palm_sensor",
        finger_tip_bodies={f: (f"{side}_hl_{f}_sensor", f"{side}_hl_{f}_tip")
                           for f in _RH_FINGERS},
        # 감쌈은 원위 마디(_2, 엄지는 _4)로 판단.
        finger_wrap_bodies={
            "thumb": (f"{side}_hl_thumb_3", f"{side}_hl_thumb_4"),
            **{f: (f"{side}_hl_{f}_2",) for f in ("index", "middle", "ring", "pinky")},
        },
        contact_group_a=("thumb",),
        contact_group_b=("index", "middle", "ring", "pinky"),
        fingertip_bodies=tuple(f"{side}_hl_{f}_tip" for f in _RH_FINGERS),
        # RH56F1: 엄지 _1(외전)만 고정. 나머지는 굴곡 2관절뿐이라 교차 자유도가 없다.
        frozen_hand_joints=(f"{side}_hj_thumb_1",),
        init_joint_pos={
            **_arm_home(side), **_rh56_hand_rest(side),
            **_arm_tuck(other), **_rh56_hand_rest(other),
            **_HEAD_HOME,
        },
        actuator_specs={
            **_arm_actuators("active", side), **_rh56_hand_actuator("active", side),
            **_arm_actuators("idle", other), **_rh56_hand_actuator("idle", other),
            **_HEAD_ACTUATOR,
        },
        palm_box_min=(_BOX_R if side == "r" else _BOX_L)[0],
        palm_box_max=(_BOX_R if side == "r" else _BOX_L)[1],
        object_spawn_center=spawn_center,
        notes=(_HOME_NOTE,
               "손 게인 5/2·effort 1.5 는 Tesollo 실측값을 그대로 쓴 것이라 "
               "RH56F1 에는 근거가 없다 — probe 5 전까지 신뢰 금지.")
        + (() if side == "r" else
           ("좌측 fabric URDF(openarm_rh56f1_left)가 없어 Fabrics 태스크로 못 띄운다. "
            "generate_left_fabric_urdf.py 로 생성해야 사용 가능.",)),
    )


RH56_RIGHT = _rh56_profile(side="r", spawn_center=_SPAWN_R)
RH56_LEFT = _rh56_profile(side="l", spawn_center=_SPAWN_L)


# =============================================================================
PROFILES: dict[str, RobotProfile] = {
    p.name: p
    for p in (
        BIS_RIGHT, BIS_LEFT,
        BI_RIGHT, BI_LEFT,
        SENS_RIGHT, SENS_LEFT_GRIPPER,
        RH56_RIGHT, RH56_LEFT,
    )
}

# 착수 기본값. 나머지는 선언만 돼 있고 probe 를 통과하지 않았다(§probe_verified).
DEFAULT_PROFILE = "bis_right"


def get(name: str) -> RobotProfile:
    if name not in PROFILES:
        raise KeyError(
            f"알 수 없는 로봇 프로필 '{name}'. 가능: {sorted(PROFILES)}"
        )
    return PROFILES[name]
