# simulation setting — Isaac USD 자산

Fusion 클라우드 `rl_ws / simulation setting` 문서에서 생성한 시뮬레이션 자산입니다.

```
defermable/simulation_setting/
├── env_v1/
│   ├── usd/env_v1.usda            (1.21 MB)  강체 1개 (고정 구조물)
│   └── meshes/visual/env_v1.obj   (0.46 MB)
├── head_v1/
│   ├── usd/head_v1.usda           (4.40 MB)  4링크 아티큘레이션
│   └── meshes/visual/head_v1.obj  (1.66 MB)
└── README.md
```

공통: **미터, Z-up**(`metersPerUnit = 1`, `upAxis = "Z"`), Fusion 원점/좌표계 그대로.
색은 Fusion의 **면(face) 단위** 외관을 읽어 머티리얼별 Mesh로 분리했습니다.
**모든 식별자는 ASCII** — 네 개 파일 전부 비ASCII 문자 0줄로 검증했습니다.

---

## env_v1 — 광학 테이블 + 프레임 (고정)

솔리드 9개를 **강체 하나**로 묶었습니다. `Visuals`(렌더 전용) + `Collision`(병합 메시 1개, invisible).

- `physics:kinematicEnabled = 1` → 중력에 떨어지지 않는 고정 강체
- 충돌 근사 `"none"`(삼각 메시) — kinematic 액터라 PhysX가 허용하며, convex hull처럼 기둥 사이가 메워지지 않습니다
- 충돌 삼각형 9,344개 / 질량 17.0 kg(kinematic이라 무시됨)

| 머티리얼 | RGB | metallic | roughness | 부품 |
|---|---|---|---|---|
| `Metal_999999` | (153,153,153) | 1.0 | 0.40 | 바닥 브레드보드 |
| `Plastic_050505` | (5,5,5) | 0.0 | 0.40 | 플랫폼 + 받침 3개 |
| `PaintedMetal_000000` | (0,0,0) | 1.0 | 0.08 | 상판 |
| `Generic_B2B2B2` | (178,178,178) | 0.0 | 0.80 | 기둥 3개 |

---

## head_v1 — 팬-틸트 헤드 + RealSense D435i (2 DOF)

### 링크 구성

지정해주신 축 링크(**solid_7 = pan**, **solid_25 = tilt**)의 원통 면에서 축을 직접 읽었고,
나머지 바디는 무게중심 위치로 소속을 판별했습니다(팬 링크 부품은 전부 무게중심 x = 22.5 mm =
팬 축 위, 틸트 회전부는 전부 z = 66.0 mm = 틸트 축 위).

| 링크 | 바디 | 질량 | 내용 |
|---|---|---|---|
| `base_link` | 15 | 63.9 g | 베이스 플레이트, 브래킷, 팬 서보 본체 + 케이스 나사 |
| `pan_link` | 16 | 24.3 g | 팬 혼(solid_7)·허브·하부 아이들러, 회전 플랫폼, **틸트 서보 본체 전체** |
| `tilt_link` | 7 | 23.4 g | 틸트 혼(solid_25)·아이들러, U 브래킷, 카메라 마운트 플레이트 |
| `camera_link` | 2 | 44.7 g | **RealSense D435i** (solid_39, solid_39_1) |

헤드 전체 **156.2 g**. 각 링크에 **완전한 질량 특성**을 기록했습니다 — `physics:mass`,
`physics:centerOfMass`, `physics:diagonalInertia`, `physics:principalAxes`. 아래 재질 배분을
반영한 값이라 PhysX가 형상에서 유도하는 균일 밀도 값과 다릅니다.

| 링크 | diagonalInertia (kg·m²) | principalAxes (w,x,y,z) |
|---|---|---|
| `base_link` | (2.354e−5, 4.011e−5, 4.227e−5) | (0.98741, 0.00239, −0.15816, 0.00056) |
| `pan_link` | (5.592e−6, 5.311e−6, 1.777e−6) | (0.999999, 0.00043, 0.00003, −0.00134) |
| `tilt_link` | (1.072e−5, 4.492e−6, 7.770e−6) | (0.97596, 0.00266, −0.21794, 0.00025) |
| `camera_link` | (2.887e−5, 4.122e−6, 2.913e−5) | (0.999984, −0.00001, 0.00524, −0.00192) |

Fusion의 고정밀 주관성 모멘트를 kg·cm² → kg·m²로 변환하고, 주축 행렬을 쿼터니언으로 바꿨습니다.
검증 세 가지를 통과했습니다: ① 평행축 정리
(`trace(원점 기준) − trace(무게중심 기준) = 2m|r|²`)가 네 링크 모두 소수점 6자리까지 일치 —
주관성 모멘트가 무게중심 기준이고 단위가 kg·cm²임을 확인. ② 주축 행렬식이 모두 +1(오른손 좌표계).
③ 쿼터니언 → 행렬 역변환 오차 ~1e−16(기계 정밀도).

교차 검증으로 `camera_link`를 90×25×25 mm / 44.7 g 직육면체로 근사하면 장축 4.66e−6,
횡축 3.25e−5 kg·m²가 나오는데, 실제 계산값 4.12e−6 / 2.89e−5와 잘 맞습니다.

### 재질과 질량 배분

| 재질 | 밀도 | 대상 | 근거 |
|---|---|---|---|
| `Servo_XC330_M288T` | 1.4498 g/cm³ | 서보 바디 30개 | XC330-M288-T **23 g × 2 = 46 g**에 맞춘 역산 밀도 |
| `RealSense_D435i` | 0.9990 g/cm³ | solid_39, solid_39_1 | **44.7 g**에 맞춘 역산 밀도 |
| `ABS_Plastic` | 1.06 g/cm³ | 나머지 구조물 8개 | Fusion 라이브러리 ABS 실제 밀도 |

ABS로 잡은 8개는 베이스 플레이트(solid_1), 브래킷 2개(solid_2/3), 팬 서보 측판 2개(solid_19/20),
회전 플랫폼(solid_21), 틸트 U 브래킷(solid_37), 카메라 마운트 플레이트(solid_38)입니다.

서보 23 g은 **고정자와 회전자로 부피 비례 배분**됩니다. 서보 하나가 두 링크에 걸쳐 있기 때문입니다
— 예를 들어 팬 서보는 본체 21.20 g이 `base_link`에, 혼·허브·아이들러 1.80 g이 `pan_link`에 들어갑니다.
`pan_link`의 서보 몫 23.00 g은 팬 서보 회전자 1.80 g + 틸트 서보 고정자 21.20 g입니다.

### 조인트

| 조인트 | 부모 → 자식 | 타입 | 축 | 회전 중심 (m) | 제한 |
|---|---|---|---|---|---|
| `root_joint` | world → base_link | fixed | — | — | — |
| `pan_joint` | base_link → pan_link | revolute | 월드 **Z** | (0.0225, 0.000034, 0.035603) | ±90° |
| `tilt_joint` | pan_link → tilt_link | revolute | 월드 **Y** | (0.0225, −0.011546, 0.066003) | ±90° |
| `camera_joint` | tilt_link → camera_link | **fixed** | — | — | — |

- 루트에 `PhysicsArticulationRootAPI`, `physxArticulation:enabledSelfCollisions = 0`
- 리볼브 조인트에 `PhysicsDriveAPI:angular`(stiffness 100 / damping 10 / maxForce 2)
- **드라이브 게인과 ±90° 제한은 임시값입니다.** 실제 서보 가동범위를 알려주시면 반영하겠습니다

### 카메라 링크

카메라를 `tilt_link`에서 떼어 **고정 조인트로 연결된 별도 링크**로 만들었습니다. 물리적으로는
틸트 브래킷에 볼트로 붙어 있으니 fixed가 맞고, 링크가 따로 있으면 나중에 센서 프림·좌표 변환·
카메라 파라미터를 이 링크에 바로 붙일 수 있습니다.

#### solid_39는 더 쪼갤 수 없습니다 (그럴 필요도 없습니다)

`solid_39`는 **lump 1개 / shell 5개**입니다. 즉 내부에 공동 4개가 있는 **하나로 연결된 단일
솔리드**라, 지오메트리를 잘라내지 않는 한 별도 바디로 분리되지 않습니다. 그리고 D435i는
통짜 강체 모듈이므로 물리 링크를 더 쪼개도 의미가 없고, 아티큘레이션에 바디·조인트만 늘어납니다.

대신 실제로 필요한 것 — **광학 부품별 개별 프레임** — 을 `camera_link` 안에 넣었습니다.
Isaac의 카메라 센서는 프림 경로에 붙으므로, 프레임 하나가 곧 독립 카메라 하나입니다.

| 프레임 | translate (m) | 정체 |
|---|---|---|
| `depth_frame` | (0.0340, +0.01755, 0.10233) | 뎁스 원점 = 좌측 IR 이미저 (RealSense 관례) |
| `left_ir_frame` | (0.0340, +0.01755, 0.10233) | 좌측 IR 이미저 |
| `right_ir_frame` | (0.0340, −0.03245, 0.10233) | 우측 IR 이미저 |
| `color_frame` | (0.0340, −0.01145, 0.10233) | RGB 컬러 카메라 |
| `ir_projector_frame` | (0.03486, +0.03255, 0.10232) | IR 프로젝터 (소구경) |

방향은 5개 모두 동일한 ROS 광학 규약 `orient = (0.5, −0.5, 0.5, −0.5)` (quatf w,x,y,z):
**+Z 전방 = 월드 +X**, +X 우측 = 월드 −Y, +Y 하방 = 월드 −Z. 즉 **카메라는 +X를 봅니다.**

**측정 근거** (추정이 아니라 면 지오메트리에서 뽑았습니다) — 전면에 개구 4개가 x ≈ 34 mm,
z = 102.3 mm 평면에 일렬로 있습니다:

- `mirror` 외관 구면 2개 → y = −32.45 / +17.55 mm, 간격 **정확히 50 mm** = D435i 스테레오 베이스라인
- `green high gloss` 구면 + 동일 규격 경통(r = 4.1 mm) → y = −11.45 mm, 두 이미저 사이 = RGB
  (초록빛은 IR 차단 코팅)
- y = +32.55 mm 의 소구경(r = 0.5 mm, x = 34.9 mm로 더 돌출) = IR 프로젝터

> 좌/우 판정은 광학 규약 기준입니다(전방 +X, 상방 +Z ⇒ 우측 = 월드 −Y).
> IR 프로젝터로 본 소구경은 주변광 센서일 가능성도 있습니다 — 벤더 CAD에 라벨이 없어
> 크기·위치로 추정했습니다.

### 부속품 정리

- **서피스 바디 12개(`Srf1`~`Srf12`)는 Fusion 문서에서 삭제했습니다.** 부피가 없어 충돌 메시를
  오염시키고 메시 생성 시 오해를 만들던 것들입니다.
- 나사·탭·허브 등 **부피 0.15 cm³ 미만 20개 바디를 충돌 메시에서 제외**했습니다. 전부 조립
  내부에 묻혀 접촉에 기여하지 않으면서 convex decomposition에서 파편만 만들던 것들입니다.
  **비주얼에는 그대로 남아 있어 외형은 동일합니다.**

| 링크 | 충돌 바디 | 제외 | 충돌 삼각형 | 비주얼 삼각형 |
|---|---|---|---|---|
| base_link | 8 / 15 | 7 | 10,260 | 11,716 |
| pan_link | 6 / 16 | 10 | 5,336 | 7,376 |
| tilt_link | 4 / 7 | 3 | 6,420 | 7,004 |
| camera_link | 2 / 2 | 0 | 4,086 | 4,086 |

### 색상 9종

| 머티리얼 | RGB | roughness | 비고 |
|---|---|---|---|
| `PaintedMetal_000000` | (0,0,0) | 0.08 | 브래킷·서보·베이스 (metallic 1.0) |
| `MatteAluminum` | (255,255,255) | 0.95 | 카메라 본체 |
| `Generic_CBD2EF` | (203,210,239) | 0.84 | |
| `BlackSoftTouchPlastic` | (18,18,18) | 0.84 | |
| `PolishedZinc` | (204,206,204) | 0.84 | |
| `DarkGreySatinFinishPlastic` | (76,76,76) | 0.90 | |
| `GreenHighGlossPlastic` | (73,169,84) | 0.80 | LED |
| `TranslucentPlastic` | (131,131,131) | 0.84 | opacity 0.10 |
| `Mirror` | (172,172,172) | 0.84 | IR 이미저 렌즈 |

> `matte aluminum` / `mirror` / `polished zinc` 는 **이름만 금속**입니다. Fusion 안에서는
> generic 셰이더에 `generic_is_metal = False`로 저장돼 있어 metallic 0으로 뽑았고,
> 이게 Fusion 화면에 보이는 색과 같습니다.

---

## Fusion에서 회전 확인하기

`head_v1` 문서 브라우저에 `base_link` / `pan_link` / `tilt_link` / `camera_link` 컴포넌트와
`pan_joint` / `tilt_joint`(리볼브) / `camera_joint`(고정)가 있습니다.

- **드래그**: 카메라나 플랫폼을 직접 잡아끌면 해당 축으로 돕니다
- **정확한 각도**: 브라우저의 조인트 우클릭 → 조인트 편집 → 각도 입력
- `base_link`는 고정(grounded), 두 리볼브 조인트 모두 **±90° 제한**, 정지값 0°

검증 완료: 팬 40° / 틸트 −30°에서 카메라·마운트·U브래킷이 한 덩어리로 따라 움직였고,
변환 행렬이 정확히 Z축 40° 회전(cos40 = 0.766)을 보였습니다.

---

## 불러오기

```python
from isaacsim.core.utils.stage import add_reference_to_stage

add_reference_to_stage("simulation_setting/env_v1/usd/env_v1.usda",   "/World/EnvV1")
add_reference_to_stage("simulation_setting/head_v1/usd/head_v1.usda", "/World/HeadV1")
```

Isaac Lab:

```python
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg

ENV_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/EnvV1",
    spawn=sim_utils.UsdFileCfg(
        usd_path="simulation_setting/env_v1/usd/env_v1.usda",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    ),
)

HEAD_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/HeadV1",
    spawn=sim_utils.UsdFileCfg(usd_path="simulation_setting/head_v1/usd/head_v1.usda"),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"pan_joint": 0.0, "tilt_joint": 0.0},
    ),
)
```

카메라 센서는 원하는 광학 프레임에 각각 붙입니다:

```python
from isaaclab.sensors import CameraCfg

RGB_CFG = CameraCfg(                       # 컬러
    prim_path="{ENV_REGEX_NS}/HeadV1/camera_link/color_frame/rgb",
    width=848, height=480,                 # D435i
    data_types=["rgb"],
)
DEPTH_CFG = CameraCfg(                     # 뎁스 (= 좌측 IR 이미저 원점)
    prim_path="{ENV_REGEX_NS}/HeadV1/camera_link/depth_frame/depth",
    width=848, height=480,
    data_types=["distance_to_image_plane"],
)
```

`left_ir_frame` / `right_ir_frame`에 각각 붙이면 스테레오 IR 쌍(베이스라인 50 mm)도 재현됩니다.

---

## 알아둘 점

1. **head_v1 질량 특성은 실제 부품 사양 기반이며 관성 텐서까지 완비돼 있습니다.** PhysX가
   형상에서 유도하지 않고 명시값을 씁니다. 부품 질량이 바뀌면 재질 밀도만 고쳐 재생성하면 됩니다.
2. **env_v1 질량 17 kg은 명목값입니다**(generic 밀도 1.0 g/cm³). kinematic 강체라 물리적으로는
   무시되므로 그대로 두었습니다.
2. Fusion 문서 안에는 **외관 2개와 재질 1개가 아직 한글 이름**입니다. 라이브러리에 연결된
   읽기 전용 항목이라 API로 이름이 바뀌지 않습니다. 다만 내보내기에서 전부 ASCII 슬러그로
   치환되므로 **USD/OBJ에는 한글이 전혀 없습니다**(검증 완료).
3. **`defermable/env/` 폴더(이전 생성분)는 폐기 대상입니다.** env_v1은 그 `env` 문서보다
   기둥·상판이 5 mm 높습니다(상판 z = 19.5~20.5 cm).
4. 재생성은 Fusion MCP 스크립트로 합니다(로컬 pxr 불필요). 테셀레이션 공차 env_v1 = 0.05 cm,
   head_v1 = 0.03 cm.
5. 검증 완료: 인덱스 범위·삼각형 수·법선 개수·괄호 균형·ASCII 이상 없음
   (env_v1 메시 5개, head_v1 메시 15개).
