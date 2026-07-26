# Cocktail Shaker — Isaac Sim / Isaac Lab 에셋 가이드

로봇 손(rh10 / LEAP)으로 **컵에 bead(유체 근사)를 붓고(pouring) → 흔드는(shaking)** RL 태스크용 에셋입니다.
Fusion 360 원본: `Cocktail_Shaker_Cobbler_700ml` (3피스 코블러, 총 높이 238 mm, 최대 외경 92 mm).

## 파일 구성 (표준 패키지 레이아웃)
```
isaac/
├── README_isaac_KR.md
├── load_in_isaac.py                 # Isaac 로딩 + bead 스폰 예제
├── meshes/
│   ├── visual/                      # 렌더용 OBJ (스무스 노멀 포함)
│   │   ├── shaker_body.obj  shaker_lid.obj  shaker_cap.obj
│   │   ├── shaker_full.obj          # 3파트 병합
│   │   ├── shaker_materials.mtl     # 금속 재질(stainless 기본 / aluminum)
│   │   └── shaker_full.stl          # 병합 STL(호환용)
│   └── collision/                   # 충돌용 STL
│       ├── shaker_body.stl  shaker_lid.stl  shaker_cap.stl   # 속 빈 셸 → SDF
│       ├── shaker_body_thick3mm.stl # 컵 벽3mm 대체본 (tunneling 저감)
│       └── shaker_solid_envelope.stl# 외형 채운 솔리드 — ※ bead 컨테이너엔 쓰지 말 것
├── urdf/                            # 파트별 독립 + 조립본
│   ├── shaker_body.urdf             # 몸체(열린 컵)  ← POURING
│   ├── shaker_lid.urdf              # 뚜껑(독립 강체)
│   ├── shaker_cap.urdf              # 캡(독립 강체)
│   └── shaker_assembled.urdf        # 닫힘 1강체(단일 링크) ← SHAKING 대안
└── usd/                             # ★ physics 완비 USD + 설정 분리 보관
    ├── shaker_body.usda             # 몸체 독립 강체 (원점=무게중심 COM)
    ├── shaker_lid.usda              # 뚜껑 독립 강체 (원점=무게중심 COM)
    ├── shaker_cap.usda              # 캡 독립 강체 (원점=무게중심 COM)
    ├── shaker_assembled.usda        # 닫힘 = 위 3파트 참조 + fixed joint 결합
    └── config/
        ├── physics.yaml             # 이식용 physics·조립·끼워맞춤 설정(원천값)
        └── shaker_cfg.py            # Isaac Lab RigidObjectCfg (파트별 + 조립)
```
> **분리형 구조**: 뚜껑·몸체·캡이 각각 독립 강체 USD. 각 파트는 자체 로컬 원점을 가짐.
> `shaker_assembled.usda` 는 이들을 조립 변환으로 **참조**하고 fixed joint 로 묶은 "닫힘" 상태.
> 관례: **비주얼=OBJ, 충돌=STL, URDF**, 그리고 **USD는 자체 폴더+config 로 분리**.
> 단위: 메시(OBJ/STL)는 **mm** → URDF `scale="0.001"` 로 m 변환. USD는 이미 **m 단위**로 저작됨.

## 재질 (금속 질감)
셰이커는 **스테인리스(기본)** 로 마감, **알루미늄** 대체 포함.
- **USD**: `Looks` 스코프에 `StainlessSteel` / `Aluminum` (UsdPreviewSurface, metallic=1).
  기본 바인딩은 스테인리스. 알루미늄으로 바꾸려면 각 Mesh 의
  `rel material:binding` 을 `.../Looks/Aluminum` 로 변경.
- **OBJ**: `meshes/visual/shaker_materials.mtl` 의 `stainless`(기본) / `aluminum`.
  OBJ 의 `usemtl stainless` → `usemtl aluminum` 로 변경하면 알루미늄.
- **Fusion 문서**: 3개 바디에 `Stainless Steel - Satin` appearance 적용됨(보기용).
- 값: stainless diffuse≈(0.56,0.57,0.58) rough0.32 / aluminum≈(0.91,0.91,0.92) rough0.28.

## physics 정보는 어디에 저장되는가 (중요)
| 파일 | 저장된 physics | 비고 |
|------|----------------|------|
| Fusion 문서 / STL | 없음 (STL은 형상만, 단위조차 없음) | 재료·충돌·질량 미포함 |
| URDF | 질량·관성만 | SDF 충돌은 주석일 뿐 |
| **`.usda`** | **전부** — RigidBody, Mass/COM/관성, SDF 메시충돌, 접촉 offset, SDF해상도 | **← 이걸 쓰면 됨** |

즉 **컵의 physics는 `shaker_body.usda` / `shaker_assembled.usda` 에 완전히 내장**돼 있습니다.
Isaac에서 이 USD를 참조하면 별도 물성 세팅 없이 바로 시뮬 가능합니다.
포함 내용(검증됨):
- `PhysicsRigidBodyAPI` + `PhysicsMassAPI`: mass, centerOfMass, diagonalInertia, principalAxes
- 각 메시에 `PhysicsCollisionAPI`+`MeshCollisionAPI`+`PhysxCollisionAPI`+`PhysxSDFMeshCollisionAPI`
- `physics:approximation = "sdf"`, `contactOffset=0.002`, `restOffset=0.0005`, `sdfResolution=256`
- 단위: `metersPerUnit=1`(미터), `upAxis=Z`. 컵 z 0~0.175 m, 조립 0~0.238 m.

### 값 바꾸는 법
- 질량/재료: `.usda` 의 `physics:mass` 와 `physics:diagonalInertia` 수정
  (관성은 질량에 비례: `I = mass × [xx=4.728e-3, zz=1.335e-3] m²`, 조립체 기준).
- tunneling 심하면 `sdfResolution` 상향(256→512) 또는 `restOffset` 소폭 증가.

### Isaac 로딩 (핵심)
```python
from omni.isaac.core.utils.stage import add_reference_to_stage
add_reference_to_stage(usd_path="usd/shaker_body.usda", prim_path="/World/ShakerBody")
# 물성·충돌 이미 내장 → 그대로 RigidObject로 사용. 전체 예제는 load_in_isaac.py 참고.
```
> URDF 경로를 선호하면 `.urdf` 를 Isaac URDF Importer 로 변환 후, 컵/뚜껑 메시의
> Collision Approximation 을 수동으로 **SDF** 로 바꿔야 합니다. USD 를 직접 쓰면 이 수동 단계가 불필요합니다.

## ⚠️ 가장 중요한 원칙 — 충돌체 선택
| 용도 | 충돌 방식 | 이유 |
|------|-----------|------|
| **컵(컨테이너)** | **SDF 메시** (`shaker_body.stl`) | 내부 공동+얇은 벽 보존, bead가 들어가고 벽과 충돌 |
| 뚜껑/스트레이너 | SDF 메시 | Ø4 홀 straining 거동 유지 |
| 컵을 단순 강체로만 취급(내부 무시) | convex decomposition (`collision/shaker_solid_envelope.stl`) | 접촉 안정 최고, 단 내부 사용 불가 |

**절대 금지:** 컵을 convex hull / convex decomposition 로 충돌 처리 → 공동이 메워져 bead가 못 들어감.

## 치수 / 물성 (설계값)
- 컵 내부 용량 ≈ **706 mL** (사양 700 mL 일치)
- 컵 입구 내경 ≈ 86 mm, 바닥 내경 ≈ 56 mm, 내부 높이 ≈ 174 mm
- 스트레이너 홀: **Ø4 mm × 13개** (중앙 1 + PCD 34 위 12개)
- 질량(속 빈 셸 기준): 스테인리스304 **0.50 kg** / 알루미늄 0.17 kg / PLA 0.079 kg / PETG 0.081 kg
- 무게중심(조립·월드): z = **139.5 mm** (x=y=0, 축대칭)
- 관성(스테인리스 0.50 kg, COM 기준): Ixx=Iyy=2.36e-3, Izz=6.68e-4 kg·m²
  - 다른 질량으로 스케일: `I = mass × [Ixx/m=4.728e-3, Izz/m=1.335e-3] (m²)`

## bead(유체 근사) 셋업 권장값
- **모델**: rigid-sphere bead 다수 (PhysX particle system 대신 강체 구 근사).
- **bead 반경**: 4~6 mm 권장.
  - Ø4 스트레이너 홀보다 **크게**(r>2 mm) 하면 흔들 때 새지 않음(=straining 됨).
  - "따라내기(strain-out)"를 시뮬하려면 bead를 홀보다 작게 → 별도 태스크로 분리.
- **개수**: 컵의 1/3 채움 기준 200~500개(r=5 mm 구 ≈ 0.52 mL). 많을수록 사실적이나 느려짐.
- bead 총 질량으로 액체량 근사 (예: 물 233 mL ≈ 0.233 kg 를 bead 질량 합으로 배분).

## tunneling(벽 관통) 대책 — 얇은 벽(0.8mm) + 격한 shaking
우선순위대로:
1. **컵 충돌을 SDF** 로. (PhysX SDF는 얇은/오목 벽에 강함)
2. 그래도 새면 컵 충돌 메시를 **`meshes/collision/shaker_body_thick3mm.stl`** 로 교체(벽 3mm).
3. 시뮬 스텝/솔버:
   - `sim.dt` 작게 (예 1/240 s 이하), `decimation` 로 policy step 조정.
   - PhysX solver iterations 상향: position ≈ 16, velocity ≈ 4.
   - bead `contact_offset` ≈ 0.002 m, `rest_offset` ≈ 0.0005 m 부터 튜닝.
   - 필요 시 bead에 CCD 활성 / substep 증가.
4. bead 반경을 벽 두께보다 훨씬 크게(≥4 mm) 유지.

## 분리형 조립 워크플로우 (뚜껑 ↔ 몸체 분리)
목표 시퀀스: **① 열린 몸체에 음료 붓기 → ② 뚜껑 닫기 → ③ 흔들어 섞기**.
세 파트는 각각 **독립 강체**이므로 이 순서가 그대로 구현됩니다.

- **① POUR**: `shaker_body.usda`(열린 컵, 넓은 입구 Ø86) 에 음료를 붓는다. 뚜껑/캡은 스테이징 위치에 별도로 스폰.
- **② CLOSE**: 로봇이 뚜껑을 집어 몸체 위에 안착. 목표 상대 포즈(몸체 프레임 기준):
  - 뚜껑 = `(0, 0, 0.09126) m`  (몸체 COM→뚜껑 COM; 스커트가 상단 림에 18mm 물림, Ø88.7/Ø88)
  - 캡  = `(0, 0, 0.12815) m`  (= 뚜껑 프레임 기준 `(0,0,0.03689)`)
  - ※ 모든 원점이 COM이라 mate 포즈 = (파트 COM − 기준 COM)
- **③ SHAKE**: 뚜껑이 안착된 순간 **fixed joint(또는 grasp-weld)를 런타임 생성**해 하나로 묶고 흔든다.
  - 미리 묶인 닫힘 상태가 필요하면 `shaker_assembled.usda` 사용(3파트 참조 + fixed joint 내장).
  - 흔든 뒤 분리하려면 joint 를 제거하거나, 처음부터 파트 USD 를 개별 스폰.

**끼워맞춤(mate) 치수** — `usd/config/physics.yaml` 의 `mate` 참조:
- 뚜껑↔몸체: 몸체 상단 외경 Ø88.0 / 뚜껑 스커트 내경 Ø88.7 (편측 0.35mm), 물림 18mm
- 캡↔뚜껑넥: 넥 외경 Ø57.0 / 캡 내경 Ø57.6, 물림 12mm

> 참고: bead(유체) 는 사용하지 않는 구성 기준입니다. 아래 bead/ tunneling 섹션은 추후 유체를 넣을 때만 참고하세요.

## Isaac Lab 임포트 순서(요약)
1. URDF Importer 로 `.urdf` → USD 변환 (Fix Base = false, Merge Fixed Joints 취향).
2. 변환된 USD 에서 컵/뚜껑 메시의 **Collision Approximation = SDF Mesh** 로 설정.
3. `RigidObjectCfg` 로 shaker 등록, bead 는 `spawn` 으로 구 다수 생성(또는 particle).
4. 물성(mass/inertia) 위 값으로 지정하거나 auto-compute.
5. 소규모(bead 50개)로 tunneling 먼저 검증 후 개수/스텝 스케일업.

## 원본 편집이 필요하면
Fusion 문서 `Cocktail_Shaker_Cobbler_700ml` 에서 형상 수정 후 동일 파이프라인으로 재-export.
벽 두께·홀 크기·용량 변경은 원본 revolve 프로파일/스트레이너 스케치에서 조정.
