"""물체 뱅크 — 어떤 물체를 몇 종 스폰할지 한 곳에서 정한다.

Phase A 는 컵 1종으로 시작하고, 나중에 grasp_v2 식 다물체 일반화로 갈 때
`object_bank` 문자열 하나만 바꾸면 되게 한다.

★모듈이 강제하는 함정 2개 (둘 다 재발 이력이 있다)
  ① 뱅크 크기 > 1 이면 `scene.replicate_physics` 를 **반드시** False 로 둬야 한다.
     MultiAsset(env 별 다른 물체)은 physics 복제가 불가능하다.
  ② RigidObject 생성은 **`clone_environments` 이후**여야 한다. 그 전에는 env_0 만
     존재해 MultiAssetSpawner 가 assets_cfg[0] 하나만 스폰하고, 전 env 가 같은
     물체를 받는다(배정 어긋남 → warm/판정 붕괴). pour_sensor 포함 3회 재발.
     `assert_spawned_after_clone()` 로 호출 순서를 검사한다.

★자산 선택 근거 (08.16 실측)
  visdex 의 cup_big/shaker 는 USD 에 `physics:approximation="sdf"` 를 적어놓고도
  apiSchemas 에 PhysxSDFMeshCollisionAPI 가 없어 PhysX 가 **convexHull 로 폴백**한다
  = 속이 찬 원통. 그래서 파지 자세가 pour(진짜 SDF 컵)로 전달되지 않았다.
  authoring 을 맞춘 사본이 `assets/cup/*_rl.usd` 이고, 컵 계열 뱅크는 그쪽을 쓴다.
  (visdex 원본은 grasp_v2 가 sorted-glob 으로 obs 차원을 파생시키므로 불가침이다.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# openarm 패키지 루트에서 assets 까지
_MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
_HDGP_ROOT = os.path.normpath(os.path.join(_MODULES_DIR, "..", "..", "..", "..", ".."))
ASSETS_DIR = os.path.join(_HDGP_ROOT, "assets")
CUP_ROOT = os.path.join(ASSETS_DIR, "cup")
VISDEX_ROOT = os.path.join(ASSETS_DIR, "multi_obj", "visdex_objects", "USD")

# 전 물체 공통 기본 질량 [kg] — pour_v1 실컵과 동일.
# ADR mass DR 의 곱셈 기준이라 여기를 바꾸면 실효 질량 범위 전체가 이동한다.
# ★USD 기본질량이 제각각이면(shaker 0.263 / 원기둥 0.100) 같은 scale DR 을 걸어도
#   물체마다 절대 질량이 2.6배 벌어져 무거운 물체만 미검증 외삽 영역으로 튄다.
BASE_OBJECT_MASS = 0.134

# bead 가 컵 바닥에 가라앉았을 때도 "안"으로 세기 위한 바닥 위 여유 [m].
# ★bead 반경(≈0.010)보다 작아야 한다 — 크면 가라앉은 bead 가 판정에서 빠지고,
#   pour_sensor 의 spill 식이 그것을 "손실"로 세어 성공을 실패로 뒤집는다.
#   구 상수(-0.070 = cup_big 바닥 -0.0773 + 0.0073)의 여유를 그대로 승계한다.
BEAD_FLOOR_MARGIN = 0.0073

# grasp_v2 에서 구조적으로 못 잡는다고 판정된 물체(반경이 작아 force closure 불가).
VISDEX_EXCLUDED = (
    "small_5_cyl", "small_8_cyl", "small_12_cyl",
    "small_5_cuboid", "small_8_cuboid", "small_12_cuboid",
)


@dataclass(frozen=True)
class ObjectSpec:
    """물체 하나. scale 로 같은 USD 를 여러 크기로 쓸 수 있다."""

    id: str
    usd_path: str
    scale: tuple = (1.0, 1.0, 1.0)
    mass: float = BASE_OBJECT_MASS
    # ★USD 원점이 물체 **바닥에서 얼마나 위**인가 [m] (scale 적용 전).
    #   이걸 모르면 "작업면에 놓인 상태"의 z 를 계산할 수 없다.
    #   실측(pxr bbox): cup_big 0.0773 · shaker_closed 0.0921.
    #   None = 미측정 → 그 뱅크를 쓰면 env 가 fail-loud 한다(조용히 틀린 높이 금지).
    base_origin_offset_z: float | None = None
    # ★접촉 필터가 가리켜야 할 **RigidBodyAPI prim 이름**.
    #   루트 Xform 을 가리키면 PhysX 가 "GPU contact filter … not supported" 를 내고
    #   force_matrix_w 가 **항상 0** 이 된다(fab_test1~4 를 그렇게 날렸다).
    #   실측: cup/shaker/visdex 표본 전부 "baseLink". 새 자산은 확인 후 채울 것.
    rigid_body_name: str = "baseLink"
    # ★파지 표면 기하 [m, scale 적용 전] — tip_bridge 의 목표면(측면 원통 띠).
    #   None = 미측정 → 그 뱅크를 쓰면 env 가 fail-loud(조용히 틀린 목표면 금지).
    #   실측 출처: cup_big 반경 0.062(접촉 실측 중앙 59~65mm) · shaker 0.044(메시).
    base_grasp_radius: float | None = None
    base_grasp_halfheight: float | None = None
    # ★림(입구) z [m, scale 적용 전, **원점 기준**] — 붓기 지점·받는 입구의 진실원천.
    #   pour 는 이 값을 `cup_pose + R·[0,0,rim_z]` 로 써서 붓는 점을 만든다. 다물체에서
    #   상수 하나(구 `SOURCE_CUP_POUR_POINT_POS_B=0.100`)를 쓰면 −17~+30 mm 어긋난다:
    #     cup_big s085 0.0853 · s100 0.1003 · s130 0.1304 · shaker 0.0829 (09.01 pxr 실측).
    #   None = 미측정 → 소비 시 fail-loud(조용히 허공에 붓는 것 금지).
    base_rim_z: float | None = None
    # ★내벽/외벽 반경 [m, scale 적용 전] — bead 의 "컵 안" 판정과 배출구 위치의 진실원천.
    #   내벽은 **림 높이 단면의 안쪽 반경**을 쓴다(아래로 갈수록 좁아지므로 가장 느슨한
    #   상한이고, 벽이 물리적으로 그보다 큰 반경을 막아준다 → 안에 있는 bead 를 놓치지
    #   않으면서 밖의 bead 를 넣지도 않는다).
    #   09.01 pxr 실측(림 상단 5mm 밴드): cup_big 내 0.0409 / 외 0.0467,
    #                                     shaker_closed 내 0.0432 / 외 0.0440.
    #   None = 미측정 → 소비 시 fail-loud.
    base_inner_radius: float | None = None
    base_outer_radius: float | None = None

    @property
    def grasp_radius_m(self) -> float:
        if self.base_grasp_radius is None:
            raise RuntimeError(
                f"물체 '{self.id}' 의 base_grasp_radius 미측정 — tip_bridge 목표면을 "
                "만들 수 없다. bbox/접촉 실측으로 채워라.")
        return self.base_grasp_radius * float(self.scale[0])

    @property
    def grasp_halfheight_m(self) -> float:
        if self.base_grasp_halfheight is None:
            raise RuntimeError(f"물체 '{self.id}' 의 base_grasp_halfheight 미측정")
        return self.base_grasp_halfheight * float(self.scale[2])

    @property
    def rim_z(self) -> float:
        """스케일 적용 림 z. 원점→입구 높이."""
        if self.base_rim_z is None:
            raise RuntimeError(
                f"물체 '{self.id}' 의 base_rim_z 미측정 — 붓기 지점을 만들 수 없다. "
                "pxr bbox 로 실측해 채워라.")
        return self.base_rim_z * float(self.scale[2])

    @property
    def inner_radius_m(self) -> float:
        """스케일 적용 내벽 반경 — bead 의 "컵 안" xy 판정 상한."""
        if self.base_inner_radius is None:
            raise RuntimeError(
                f"물체 '{self.id}' 의 base_inner_radius 미측정 — bead 내부 판정을 "
                "만들 수 없다. pxr 로 림 단면 반경을 실측해 채워라.")
        return self.base_inner_radius * float(self.scale[0])

    @property
    def outer_radius_m(self) -> float:
        """스케일 적용 외벽 반경 — 기울였을 때 실제 배출구(최하단 림 점) 계산용."""
        if self.base_outer_radius is None:
            raise RuntimeError(
                f"물체 '{self.id}' 의 base_outer_radius 미측정 — 배출구 위치를 "
                "만들 수 없다. pxr 로 림 단면 반경을 실측해 채워라.")
        return self.base_outer_radius * float(self.scale[0])

    @property
    def inside_z_min(self) -> float:
        """바닥에 가라앉은 bead 를 "안"으로 세는 하한 [m, 원점 기준].

        바닥(`-origin_offset_z`)보다 `BEAD_FLOOR_MARGIN` 만큼 위. 이 여유는 bead
        반경(≈0.010)보다 **작아야** 가라앉은 bead(중심 = 바닥+반경)가 하한 위에 남는다.
        컵을 키워도 bead 는 안 커지므로 관계는 스케일과 무관하게 유지된다.
        """
        return -self.origin_offset_z + BEAD_FLOOR_MARGIN

    @property
    def origin_offset_z(self) -> float:
        if self.base_origin_offset_z is None:
            raise RuntimeError(
                f"물체 '{self.id}' 의 base_origin_offset_z 가 미측정이다. "
                "작업면 위 안착 높이를 계산할 수 없다 — USD bbox 로 측정해 채워라."
            )
        return self.base_origin_offset_z * float(self.scale[2])


@dataclass(frozen=True)
class ObjectBank:
    name: str
    specs: tuple
    note: str = ""

    def __len__(self) -> int:
        return len(self.specs)

    @property
    def ids(self) -> tuple:
        return tuple(s.id for s in self.specs)

    @property
    def onehot_dim(self) -> int:
        """`enable_object_onehot` 을 켰을 때 obs 에 더해지는 차원."""
        return len(self.specs)

    @property
    def rigid_body_name(self) -> str:
        """뱅크 전체가 같은 이름이어야 접촉 필터를 하나로 쓸 수 있다."""
        names = {s.rigid_body_name for s in self.specs}
        if len(names) != 1:
            raise RuntimeError(
                f"물체 뱅크 '{self.name}' 의 rigid body 이름이 섞여 있다: {sorted(names)}. "
                "접촉 필터를 하나로 지정할 수 없다."
            )
        return names.pop()

    @property
    def needs_multi_asset(self) -> bool:
        return len(self.specs) > 1

    @property
    def requires_replicate_physics_off(self) -> bool:
        return self.needs_multi_asset

    def missing_files(self) -> tuple:
        return tuple(s.usd_path for s in self.specs if not os.path.isfile(s.usd_path))

    def assign_indices(self, num_envs: int):
        """env_id % N 결정론적 배정 — MultiAssetSpawner(random_choice=False)와 같은 규칙.

        torch 를 import 하지 않기 위해 리스트로 돌려준다(호출부가 텐서화).
        """
        n = len(self.specs)
        return [i % n for i in range(num_envs)]


# =============================================================================
# 뱅크 정의
# =============================================================================
_CUP_BIG = os.path.join(CUP_ROOT, "cup_big_rl.usd")
_SHAKER = os.path.join(CUP_ROOT, "shaker_closed_rl.usd")


# 실측(pxr bbox, scale=1): cup_big 바닥 -0.0773 / 상단 +0.1003
_CUP_BIG_ORIGIN_OFFSET = 0.0773
# 09.01 pxr 실측(scale=1): cup_big 상단 +0.1003 / shaker_closed 상단 +0.0829.
_CUP_BIG_RIM_Z = 0.1003
_SHAKER_RIM_Z = 0.0829
_SHAKER_ORIGIN_OFFSET = 0.0921      # 메모리 기록 "shaker 원점은 바닥+92mm" 와 일치
# 09.01 pxr 실측(림 상단 5mm 밴드의 반경 분포, scale=1):
#   cup_big        내 0.0409 ~ 외 0.0467 (벽 두께 5.8mm, 아래로 갈수록 좁아져 몸통 최소 0.0277)
#   shaker_closed  내 0.0432 ~ 외 0.0440 (얇은 벽 0.8mm, 몸통 최소 0.0332)
_CUP_BIG_INNER_R = 0.0409
_CUP_BIG_OUTER_R = 0.0467
_SHAKER_INNER_R = 0.0432
_SHAKER_OUTER_R = 0.0440


def _cup(scale: float) -> ObjectSpec:
    # ★round 필수 — int(1.15 * 100) 은 부동소수 때문에 114 가 된다(id 가 조용히 어긋남).
    return ObjectSpec(id=f"cup_big_s{round(scale * 100):03d}",
                      usd_path=_CUP_BIG, scale=(scale, scale, scale),
                      base_origin_offset_z=_CUP_BIG_ORIGIN_OFFSET,
                      base_rim_z=_CUP_BIG_RIM_Z,
                      base_inner_radius=_CUP_BIG_INNER_R,
                      base_outer_radius=_CUP_BIG_OUTER_R,
                      base_grasp_radius=0.062, base_grasp_halfheight=0.05)


def _shaker(scale: float) -> ObjectSpec:
    """shaker 를 크기별로. ★`_cup` 과 같은 round 규약(부동소수로 id 가 어긋난다)."""
    return ObjectSpec(id=f"shaker_s{round(scale * 100):03d}",
                      usd_path=_SHAKER, scale=(scale, scale, scale),
                      base_origin_offset_z=_SHAKER_ORIGIN_OFFSET,
                      base_rim_z=_SHAKER_RIM_Z,
                      base_inner_radius=_SHAKER_INNER_R,
                      base_outer_radius=_SHAKER_OUTER_R,
                      base_grasp_radius=0.044, base_grasp_halfheight=0.05)


SINGLE_CUP = ObjectBank(
    name="single_cup",
    specs=(_cup(1.00),),
    note="Phase A 착수용. MultiAsset 불필요 → replicate_physics 를 켠 채로 돌 수 있다.",
)

CUP_FAMILY = ObjectBank(
    name="cup_family",
    specs=(
        _cup(0.85), _cup(1.00), _cup(1.15), _cup(1.30),
        ObjectSpec(id="shaker_closed", usd_path=_SHAKER,
                   base_origin_offset_z=_SHAKER_ORIGIN_OFFSET,
                   base_rim_z=_SHAKER_RIM_Z,
                   base_inner_radius=_SHAKER_INNER_R,
                   base_outer_radius=_SHAKER_OUTER_R,
                   base_grasp_radius=0.044, base_grasp_halfheight=0.05),
        _cup(0.90), _cup(1.05), _cup(1.20),
    ),
    note=("grasp_v1 의 8종. 순서가 env_id % 8 배정과 onehot 인덱스를 동시에 정하므로 "
          "바꾸면 기존 체크포인트와 어긋난다."),
)


SHAKER_FAMILY = ObjectBank(
    name="shaker_family",
    specs=(
        _shaker(0.80), _shaker(0.85), _shaker(0.90), _shaker(0.95),
        _shaker(1.00), _shaker(1.03), _shaker(1.07), _shaker(1.10),
    ),
    note=("★저자유도·소형 손(RH56F1) 전용 8종. `cup_family` 를 그대로 쓸 수 없어서 만든다 "
          "— cup_big r62mm 를 0.85~1.30 으로 쓰면 지름 105~161mm 인데, RH56F1 검지는 "
          "MCP→팁이 URDF 실측 72.5mm(_1 32.9 + _2 39.6)라 감쌈이 구조적으로 불가능하다. "
          "shaker(r44mm)를 0.80~1.10 으로 쓰면 지름 70~97mm 로 손 크기에 들어온다"
          "(사용자 지시 09.02). "
          "★순서는 **오름차순**이다 — env_id % 8 배정과 종별 진단 인덱스를 동시에 정하므로 "
          "크기 순서와 종 인덱스가 일치해 로그를 그대로 읽을 수 있다. 바꾸면 기존 "
          "체크포인트와 어긋난다."),
)


# ---------------------------------------------------------------------------
# ★09.10 신규 자산 — `assets/simulation_setting/shaker` (사용자 지시)
# ---------------------------------------------------------------------------
# 기존 `assets/cup/shaker_closed_rl.usd` 와 **다른 물체**다. 원점 규약부터 다르므로
# 값을 물려받지 말고 아래 실측을 쓸 것(09.10 usda 메시 점 직접 측정, scale=1):
#   · 높이 0.1750 m · 원점 = **바운딩박스 중심**(usda doc "Origin recentred to the
#     bounding-box centre") → 바닥에서 원점까지 0.0875 (구 자산은 0.0921).
#   · 원뿔대다(원통 아님). 외경 바닥 58.0mm → 림 88.0mm 로 선형 확대.
#     벽 두께 5mm(바닥 안쪽 48.0 · 림 안쪽 78.0). 무게중심 z=+0.00138 ≈ 원점.
#   · 충돌 근사 `convexDecomposition`(구 자산의 SDF 아님) — VRAM 이 덜 든다.
#   · 루트 Xform 이름이 `ShakerFDM5mm` 이고 여기에 RigidBodyAPI 가 붙어 있다.
#     `baseLink` 프림이 **없으므로** rigid_body_name 을 반드시 덮어야 한다.
_SHAKER_V1 = os.path.join(ASSETS_DIR, "simulation_setting", "shaker", "usd", "shaker.usda")
_SHAKER_V1_ORIGIN_OFFSET = 0.0875
_SHAKER_V1_RIM_Z = 0.0875            # 닫힌 몸통이라 림 = 상단 = 원점 + 절반높이
_SHAKER_V1_INNER_R = 0.0390          # 림 안쪽(실측 39.0mm)
_SHAKER_V1_OUTER_R = 0.0440          # 림 바깥(실측 44.0mm) = 물체 최대 반경
_SHAKER_V1_PRIM = "ShakerFDM5mm"


def _shaker_v1(scale: float) -> ObjectSpec:
    """신규 셰이커를 크기별로. ★`_cup` 과 같은 round 규약(부동소수로 id 가 어긋난다)."""
    return ObjectSpec(id=f"shakerv1_s{round(scale * 100):03d}",
                      usd_path=_SHAKER_V1, scale=(scale, scale, scale),
                      base_origin_offset_z=_SHAKER_V1_ORIGIN_OFFSET,
                      base_rim_z=_SHAKER_V1_RIM_Z,
                      base_inner_radius=_SHAKER_V1_INNER_R,
                      base_outer_radius=_SHAKER_V1_OUTER_R,
                      rigid_body_name=_SHAKER_V1_PRIM,
                      # 파지 반경은 **파지 높이의 단면**이다. 원뿔대라 높이마다 다르고,
                      # grasp_kp 는 원점(=무게중심) 높이에서 잡는다 → 외경 73.0mm.
                      base_grasp_radius=0.0365, base_grasp_halfheight=0.05)


SHAKER_SWEEP = ObjectBank(
    name="shaker_sweep",
    specs=tuple(_shaker_v1(round(0.80 + 0.05 * i, 2)) for i in range(9)),
    note=("★09.10 사용자 지시: 신규 셰이커를 scale 0.80~1.20, 0.05 단위 = **9종**. "
          "파지 높이(원점=무게중심) 단면 외경 73.0mm 기준으로 58.4~87.6mm 대역이다. "
          "DG5F-M 은 cup_big s100(외경 124mm)을 kp_a1 에서 lifted 0.86 으로 들었으므로 "
          "이 대역은 파지 창 안쪽이다. "
          "★9 는 2 의 거듭제곱이 아니다 — assign_indices 는 i % n 이라 무관하지만 "
          "num_envs 가 9 의 배수가 아니면 종별 env 수가 1개씩 어긋난다(무해). "
          "★순서는 오름차순(종 인덱스 = 크기 순서) — 기존 뱅크와 같은 규약."),
)


SHAKER_SMALL = ObjectBank(
    name="shaker_small",
    specs=(
        _shaker(0.55), _shaker(0.58), _shaker(0.61), _shaker(0.64),
        _shaker(0.67), _shaker(0.70), _shaker(0.73), _shaker(0.75),
    ),
    note=("★★09.03 RH56F1 재설계 — `shaker_family`(0.80~1.10 = 지름 70~97mm)조차 "
          "이 손에 **물리적으로 안 들어간다**. 실측 근거 3가지:\n"
          "  · 엄지-4지 법선 간극이 **83.7mm 가 최대**다(엄지 외전 스윕: 1.57 에서 최대, "
          "    2.09 로 더 벌리면 73.9mm 로 오히려 줄어든다 — 1.57 이 이미 최적).\n"
          "  · 컵을 케이지 중심에 고정하고 재면 s090(79.2mm)부터 **열린 손에서 이미 접촉**\n"
          "    (s085 74.8mm 만 접촉 0). 링크 두께를 빼면 실사용 한계는 ~70mm.\n"
          "  · 그 결과 세 학습 런(rh_b1/c1/d1)에서 `wrap_frac` 이 0.002 에 붙어 있었고,\n"
          "    엄지가 옆으로 못 돌아 컵 rim 안으로 들어가는 접근이 관찰됐다(사용자).\n"
          "지름 48.4~66.0mm — 실제 병·작은 컵 크기대라 s2r 관점에서도 유효한 물체군이다.\n"
          "★순서는 오름차순(종 인덱스 = 크기 순서)."),
)


SHAKER_ONE = ObjectBank(
    name="shaker_one",
    specs=(_shaker(0.65),),
    note=("★★09.07 RH56F1 Track B 전용 — **단일** 셰이커(지름 57.2mm · 높이 114mm).\n"
          "왜 단일인가 둘:\n"
          "  · 다물체(MultiAsset)는 `replicate_physics=False` 를 강제해 env 마다 씬을 통째로\n"
          "    파싱한다. 첫 과제 성립을 보는 단계에서는 비용만 크다.\n"
          "  · 크기 8종은 파지 창(열림 105.5 / 폐쇄 46.6mm)에 다 들어오지만, 종횡비 1.99 라\n"
          "    접촉하면 잘 넘어진다 — 크기 변동까지 얹으면 원인이 섞인다.\n"
          "0.65 를 고른 이유: shaker_small 대역(0.55~0.75)의 중앙이고 지름 57mm 가 폐쇄 종점\n"
          "46.6mm 보다 크고(접촉 성립) 열림 105.5mm 보다 작다(들어옴).\n"
          "★사용자 지시(09.08): '셰이커로 하고'. 컵으로 바꾸지 않는다."),
)


CUP_SMALL = ObjectBank(
    name="cup_small",
    specs=(
        _cup(0.50), _cup(0.52), _cup(0.54), _cup(0.56),
        _cup(0.58), _cup(0.60), _cup(0.62), _cup(0.64),
    ),
    note=("★★09.03 RH56F1 전용. 지름 62.0~79.4mm(파지반경 62mm × scale × 2).\n"
          "★대역은 **검증점 0.58 주변으로만** 잡았다 — 스케일 스윕은 아직 안 했다.\n"
          "cup_family 주석의 \"cup_big 은 감쌈 구조적 불가\" 판정은 **0.85~1.30**\n"
          "(105~161mm)만 보고 내린 것이라 이 대역과 무관하다.\n"
          "근거 — 스크립트 롤아웃(컵 자유·팔이 fabric 으로 실제 접근·폐쇄 후 리프트)에서\n"
          "cup_big 0.58 이 2회 재현으로 들렸다(36점 중 11/8 성립, 7점 교집합).\n"
          "파지 자세는 `thumb:MDT` + 4지 팁 — **엄지로 무는** 파지다.\n"
          "★shaker 를 버린 이유 둘: FP++ 가 텍스처 없는 메시를 놓치고(인식),\n"
          "  파지 창이 좁고 비단조다(셰이커는 테이퍼·림이 있어 스케일을 바꾸면\n"
          "  손가락 평면에 오는 단면이 바뀐다 — 지름이 간섭 기하의 대리값이 못 된다).\n"
          "★순서는 오름차순(종 인덱스 = 크기 순서)."),
)


def _visdex_bank() -> ObjectBank:
    """visdex 디렉터리를 sorted-glob 한다(grasp_v2 규약).

    ★디렉터리에 자산이 하나 추가되면 뱅크 크기가 조용히 변하고, onehot 을 켠 상태면
      obs 차원이 바뀌어 체크포인트 resume 이 깨진다(grasp_v2 에서 실제로 발생 — 148→149).
      그래서 이 뱅크를 쓰는 런은 `expected_size` 로 크기를 고정해 대조해야 한다.
    """
    if not os.path.isdir(VISDEX_ROOT):
        return ObjectBank(name="visdex", specs=(), note="visdex 자산 디렉터리 없음")
    names = sorted(
        n for n in os.listdir(VISDEX_ROOT)
        if os.path.isfile(os.path.join(VISDEX_ROOT, n, f"{n}.usd"))
        and n not in VISDEX_EXCLUDED
    )
    # ★원점 오프셋 미측정(base_origin_offset_z=None). Phase C 에서 이 뱅크를 쓰려면
    #   149종의 USD bbox 를 한 번 계산해 캐시해야 한다 — 안 하면 env 가 fail-loud 한다.
    specs = tuple(
        ObjectSpec(id=n, usd_path=os.path.join(VISDEX_ROOT, n, f"{n}.usd"))
        for n in names
    )
    return ObjectBank(
        name="visdex", specs=specs,
        note=("grasp_v2 물체군. 디렉터리 glob 이라 자산 추가 시 크기가 조용히 변한다 — "
              "onehot 을 켜면 반드시 expected_size 로 고정할 것."),
    )


VISDEX = _visdex_bank()

BANKS: dict[str, ObjectBank] = {
    b.name: b for b in (SINGLE_CUP, CUP_FAMILY, CUP_SMALL,
                        SHAKER_FAMILY, SHAKER_SMALL, SHAKER_ONE,
                        SHAKER_SWEEP, VISDEX)
}
DEFAULT_BANK = "single_cup"


def get(name: str, *, expected_size: int | None = None) -> ObjectBank:
    if name not in BANKS:
        raise KeyError(f"알 수 없는 물체 뱅크 '{name}'. 가능: {sorted(BANKS)}")
    bank = BANKS[name]
    if expected_size is not None and len(bank) != expected_size:
        raise RuntimeError(
            f"물체 뱅크 '{name}' 크기가 {len(bank)} 인데 {expected_size} 를 기대했다. "
            "자산 디렉터리가 바뀌었을 수 있다 — onehot 을 켠 체크포인트와 어긋난다."
        )
    missing = bank.missing_files()
    if missing:
        raise FileNotFoundError(f"물체 뱅크 '{name}' 자산 없음: {missing}")
    return bank


# =============================================================================
# 스폰 순서 강제
# =============================================================================
def spec_by_id(spec_id: str) -> ObjectSpec:
    """전 뱅크에서 스펙 하나를 id 로 찾는다.

    ★받는 컵처럼 **뱅크 밖에서 단독으로 쓰는 자산**도 기하(림·바닥·반경)의 진실원천을
      뱅크와 공유하기 위한 조회구다. 자산 경로만 바꾸고 상수를 안 고치는 드리프트가
      실제로 났다(받는 컵 cup_big→shaker 교체 시 입구 z 가 17.4mm 어긋남, 09.01).
    """
    for bank in BANKS.values():
        for sp in bank.specs:
            if sp.id == spec_id:
                return sp
    raise KeyError(
        f"물체 스펙 '{spec_id}' 을 어느 뱅크에서도 못 찾았다. "
        f"보유: {sorted({sp.id for b in BANKS.values() for sp in b.specs})}")


def assert_spawned_after_clone(bank: ObjectBank, cloned: bool) -> None:
    """MultiAsset 물체를 clone 이전에 만들면 전 env 가 assets_cfg[0] 하나만 받는다.

    3회 재발한 함정이라 조용히 넘기지 않고 여기서 멈춘다.
    """
    if bank.needs_multi_asset and not cloned:
        raise RuntimeError(
            f"물체 뱅크 '{bank.name}'({len(bank)}종)는 clone_environments **이후**에 "
            "스폰해야 한다. clone 이전에는 env_0 만 존재해 MultiAssetSpawner 가 "
            "assets_cfg[0] 하나만 스폰하고, env 별 배정이 통째로 어긋난다."
        )
