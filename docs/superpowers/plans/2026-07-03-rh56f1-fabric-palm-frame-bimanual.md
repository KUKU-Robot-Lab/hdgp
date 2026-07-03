# rh56f1 fabric palm 프레임 정합 + 양팔 통합 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** rh56f1 fabric IK가 실제 `r_hl_palm_sensor`(및 `l_hl_palm_sensor`)를 정확히 제어하도록 정합하고, 양팔 통합 fabric(26 DOF, 인프라만) 인프라를 구축한다.

**Architecture:** fabric URDF의 palm IK 프레임을 Tesollo 가상프레임(palm_link)에서 실제 palm_sensor 링크 + 그 로컬 축점 6개로 교체한다. env는 offset 변환을 제거하고 palm orientation 규약을 palm_sensor 기준으로 재보정한다. Phase 1은 오른쪽만(13 DOF 유지) 정합해 오른손을 즉시 working 상태로 만들고, Phase 2에서 왼팔+왼손을 대칭 추가해 26 DOF 통합 fabric으로 확장한다.

**Tech Stack:** Python, PyTorch, fabrics_sim(커스텀 Fabric IK), Isaac Lab, URDF(xml.etree), numpy FK.

## Global Constraints

- 소스 URDF: `hdgp/assets/robot/openarm_bi_rh56f1_rl/openarm_bi_rh56f1_rl.urdf` (USD 정렬, `r_hl_*`/`l_hl_*` 네이밍). 구본 `hdgp/assets/openarm_bi_rh56f1/openarm_bi_rh56f1.urdf` 사용 금지.
- 팔 체인 기하는 기존 Tesollo 공유본과 동일(팔 동일·검증됨). 오른팔 fabric joint 이름은 기존 `openarm_right_joint1~7` 유지.
- palm_sensor 실기하(참고 URDF): `r_hl_palm_sensor` = parent `r_hl_palm_2`, origin `(0.0159401947506102, -0.00135045394126701, 0.0737460952299602)` rpy `(1.5707963267949, 0, 1.5707963267949)`. 왼쪽 `l_hl_palm_sensor` = parent `l_hl_palm_2`, origin `(0.01594, -0.0013505, 0.073746)` rpy `(1.5708, 0, 1.5708)`.
- palm 축점 규약: fabric `convert_transform_to_points`가 원점 + `±x/±y/±z @0.25m`(로컬)로 펼침. 축점 프레임은 palm_sensor 자식으로 `(±0.25,0,0)/(0,±0.25,0)/(0,0,±0.25)` origin.
- 검증 기준: 정적 FK 정합 위치 <2mm·자세 <1°. 좌우 both.
- obs/action 차원 변경 금지(인프라만). Phase 2에서 왼팔 target은 고정 중립 상수.
- tesollo fabric/env 파일은 절대 수정 금지.
- 커밋은 각 태스크 끝에서. 브랜치 `pour`(hdgp repo). fabrics_sim은 `hdgp/source/FABRICS`(hdgp repo에 vendored) — 같은 repo에서 커밋.
- reward/gate/weight 변경 시 reward-audit 통과(이 계획엔 reward 변경 없음. euler 규약 재보정은 IK 기하이지 reward 아님).

---

## 파일 구조 (생성/수정 맵)

- `hdgp/source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_rh56f1/generate_openarm_rh56f1_urdf.py` — 수정(소스 _rl, palm_sensor+축점 graft, 양팔). URDF 생성기.
- `hdgp/source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_rh56f1/openarm_rh56f1.urdf` — 재생성 산출물.
- `hdgp/source/FABRICS/src/fabrics_sim/fabrics/openarm_rh56f1_pose_fabric.py` — 수정(control_point_frames, TIP_FRAMES, default_palm_euler, cspace 26DOF, 좌우 palm attractor).
- `hdgp/source/FABRICS/src/fabrics_sim/fabric_params/openarm_rh56f1_pose_params.yaml` — 수정(collision_sphere_frames palm_link→palm_sensor, 양팔 충돌).
- `hdgp/source/openarm/openarm/rh56f1/right/grasp_v1/grasp_right_env.py` — 수정(offset 제거, euler 재보정, fabric_q 확장).
- `hdgp/source/openarm/openarm/rh56f1/right/pour_v1/pour_right_env.py` — 수정(동일 offset 제거·euler 재보정).
- `hdgp/source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_rh56f1/verify_palm_sensor_fk.py` — 생성(FK 정합 검증 도구, numpy).
- 테스트: `.../grasp_v1/tests/test_phase4_env_static.py`, `.../pour_v1/tests/test_pour_rh56f1_static.py` — offset 참조 갱신.

---

## Phase 1 — 오른쪽 palm_sensor 정합 (13 DOF 유지)

### Task 1.0: FK 정합 검증 도구 작성

**Files:**
- Create: `hdgp/source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_rh56f1/verify_palm_sensor_fk.py`

**Interfaces:**
- Produces: CLI `python3 verify_palm_sensor_fk.py --urdf <fabric_urdf> --ref <ref_urdf> --side right|left|both` → 각 side에서 랜덤 cspace N개에 대해 `fabric URDF FK(*_hl_palm_sensor)` vs `참고 URDF FK(*_hl_palm_sensor)`를 비교, 위치 mm·자세 deg 최대오차 출력 + PASS/FAIL(위치<2mm, 자세<1°).

- [ ] **Step 1: FK 유틸 작성**

numpy 기반 URDF FK. 두 URDF에서 지정 링크의 base(robot root) 기준 4x4 변환을 랜덤 관절각으로 계산해 비교한다. 팔 joint는 두 URDF에서 이름이 다를 수 있으므로(`openarm_right_joint*` vs `r_aj_*`) **양쪽 공통 조상(오른팔 root)부터 palm_sensor까지의 상대 변환**만 비교한다. 링크 체인은 URDF parent/child 그래프에서 자동 추적.

```python
# verify_palm_sensor_fk.py (骨: 실행자는 urdfpy 미설치 가정, 순수 xml+numpy)
import argparse, numpy as np, xml.etree.ElementTree as ET

def rpy_to_R(r,p,y):
    cr,sr,cp,sp,cy,sy=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
    return (np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])@
            np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])@
            np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]]))

def load(urdf):
    root=ET.parse(urdf).getroot()
    joints={}
    for j in root.findall('joint'):
        o=j.find('origin'); ax=j.find('axis')
        joints[j.find('child').get('link')]=dict(
            name=j.get('name'), type=j.get('type'), parent=j.find('parent').get('link'),
            xyz=[float(v) for v in (o.get('xyz','0 0 0').split())] if o is not None else [0,0,0],
            rpy=[float(v) for v in (o.get('rpy','0 0 0').split())] if o is not None else [0,0,0],
            axis=[float(v) for v in ax.get('xyz').split()] if ax is not None else [0,0,1])
    return joints

def chain(joints, link):
    c=[]
    while link in joints:
        c.append(joints[link]); link=joints[link]['parent']
    return list(reversed(c))

def fk(joints, link, qmap):
    T=np.eye(4)
    for j in chain(joints, link):
        M=np.eye(4); M[:3,:3]=rpy_to_R(*j['rpy']); M[:3,3]=j['xyz']
        if j['type']=='revolute':
            q=qmap.get(j['name'],0.0); a=np.array(j['axis'])
            R=rpy_to_R(0,0,0)  # axis-angle
            ax=a/np.linalg.norm(a); K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
            R=np.eye(3)+np.sin(q)*K+(1-np.cos(q))*K@K
            Rj=np.eye(4); Rj[:3,:3]=R; M=M@Rj
        T=T@M
    return T
# main: 두 urdf 공통 상위(오른팔 root) 기준 상대변환 비교, 위치/자세 오차 리포트
```

- [ ] **Step 2: 자기검증 — 참고 URDF를 두 인자에 동일 지정**

Run: `python3 verify_palm_sensor_fk.py --urdf <ref> --ref <ref> --side right`
Expected: 위치오차 0.0mm, 자세오차 0.0° PASS (동일 URDF 자기비교 → 0)

- [ ] **Step 3: Commit**

```bash
git -C hdgp add source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_rh56f1/verify_palm_sensor_fk.py
git -C hdgp commit -m "test: add palm_sensor FK alignment verifier for rh56f1 fabric"
```

---

### Task 1.1: generate 스크립트 — 오른손 palm_sensor + 축점 graft

**Files:**
- Modify: `.../urdf/openarm_rh56f1/generate_openarm_rh56f1_urdf.py`
- Regenerate: `.../urdf/openarm_rh56f1/openarm_rh56f1.urdf`

**Interfaces:**
- Consumes: 참고 URDF `openarm_bi_rh56f1_rl.urdf`.
- Produces: fabric URDF에 `r_hl_base→r_hl_palm_1/2→r_hl_palm_sensor` 실체 체인 + `r_hl_palm_sensor` 자식 축점 `ps_r_x/x_neg/y/y_neg/z/z_neg`. Tesollo `palm_link`/`palm_x`… 프레임 제거. 손가락 joint 이름은 fabric cspace 순서 유지 위해 기존 `rh56f1_right_right_*` 유지하되 origin은 _rl에서 취득(동일값).

- [ ] **Step 1: 소스 경로 교체**

`generate_openarm_rh56f1_urdf.py`의 `SRC_ROBOT_URDF`를 `/home/user/rl_ws/hdgp/assets/robot/openarm_bi_rh56f1_rl/openarm_bi_rh56f1_rl.urdf`로 변경. `_rl` 네이밍(`r_hj_*`/`r_hl_*`)과 기존 fabric joint 이름(`rh56f1_right_right_*_joint`)의 매핑 딕셔너리를 추가한다(손 기하 동일, 이름만 변환). 실행자는 먼저 참고 URDF에서 오른손 drive/mimic joint 이름·origin·limit·axis를 추출해 매핑을 확정한다.

- [ ] **Step 2: Tesollo palm 프레임 제거 + palm_sensor 체인 삽입**

`build()`에서 Tesollo 복사 루프의 palm 관련 링크/조인트(`palm_link`, `palm_x*`, `palm_y*`, `palm_z*`, `palm_center`, `palm_link_sphere*`)를 DROP 목록에 추가. 대신 참고 URDF의 `r_hl_base→r_hl_palm_1→r_hl_palm_2→r_hl_palm_sensor` 체인을 `openarm_right_link7`에 마운트해 삽입(참고 URDF의 `r_al_7→r_hl_base` origin `(0,0.0000003,0.0595695)` rpy`(0,0,-1.5707964)` 사용). 이어서 palm_sensor 자식 축점 6개를 추가:

```python
PALM_AXIS_POINTS_R = {
    "ps_r_x":     "0.25 0 0",   "ps_r_x_neg": "-0.25 0 0",
    "ps_r_y":     "0 0.25 0",   "ps_r_y_neg": "0 -0.25 0",
    "ps_r_z":     "0 0 0.25",   "ps_r_z_neg": "0 0 -0.25",
}
for name, off in PALM_AXIS_POINTS_R.items():
    link = ET.SubElement(out, "link", {"name": name}); _massless_inertial(link)
    j = ET.SubElement(out, "joint", {"name": f"{name}_joint", "type": "fixed"})
    ET.SubElement(j, "origin", {"xyz": off, "rpy": "0 0 0"})
    ET.SubElement(j, "parent", {"link": "r_hl_palm_sensor"})
    ET.SubElement(j, "child", {"link": name})
```

- [ ] **Step 3: 재생성 실행**

Run: `python3 generate_openarm_rh56f1_urdf.py`
Expected: `[OK] wrote .../openarm_rh56f1.urdf`. 이어서 grep으로 `r_hl_palm_sensor`, `ps_r_z` 존재 + `palm_link` 부재 확인.

- [ ] **Step 4: FK 정합 검증 (게이트 1, 오른쪽)**

Run: `python3 verify_palm_sensor_fk.py --urdf openarm_rh56f1.urdf --ref <ref_urdf> --side right`
Expected: PASS (위치<2mm, 자세<1°). FAIL 시 마운트 origin/체인 순서 재점검 — 진행 금지.

- [ ] **Step 5: Commit**

```bash
git -C hdgp add source/FABRICS/src/fabrics_sim/models/robots/urdf/openarm_rh56f1/
git -C hdgp commit -m "feat(fabric): graft real r_hl_palm_sensor + axis points into rh56f1 urdf (right)"
```

---

### Task 1.2: fabric 코드 — 오른쪽 palm_sensor control point + euler 재계산

**Files:**
- Modify: `.../fabrics/openarm_rh56f1_pose_fabric.py`

**Interfaces:**
- Consumes: Task 1.1 URDF 프레임(`r_hl_palm_sensor`, `ps_r_*`, `r_hl_*` tip 링크).
- Produces: `control_point_frames = ["r_hl_palm_sensor","ps_r_x","ps_r_x_neg","ps_r_y","ps_r_y_neg","ps_r_z","ps_r_z_neg"]`; `TIP_FRAMES`를 r_hl 말단 링크 기준 tip 프레임 이름으로; `default_palm_euler`를 palm_sensor 로컬 기준 재계산값으로.

- [ ] **Step 1: control_point_frames / TIP_FRAMES 교체**

`add_palm_points_attractor`의 `control_point_frames`를 palm_sensor 기준 7프레임으로. `TIP_FRAMES`를 재생성 URDF의 실제 tip 프레임 이름으로 맞춤(URDF grep으로 확인).

- [ ] **Step 2: default_palm_euler / default_config 재계산**

기존 default euler `(1.5708,0,1.5708)`는 palm_link 기준이라 90° 어긋난 프레임 값. `default_config`(팔 자연자세+손)로 FK를 돌려 palm_sensor의 world 자세를 얻고, 그 자세를 표현하는 euler_zyx를 산출해 `default_palm_euler`에 대입. 산출은 Task 1.0 FK 유틸 재사용 또는 fabric FK.

Run: `python3 -c "..."` (default_config에서 palm_sensor FK 자세 → euler 출력)
Expected: 산출된 euler 3값. 이 값을 코드에 하드코딩하고 주석에 근거 명시.

- [ ] **Step 3: 정적 로드 스모크**

Run: `python3 hdgp/source/openarm/openarm/rh56f1/right/grasp_v1/tests/smoke_rh56f1_fabric.py` (있으면). 없으면 fabric을 batch=2로 인스턴스화해 `construct_fabric` 통과 + palm taskmap 프레임 해상 확인하는 임시 스모크.
Expected: 예외 없음, palm taskmap이 7프레임 해상.

- [ ] **Step 4: Commit**

```bash
git -C hdgp add source/FABRICS/src/fabrics_sim/fabrics/openarm_rh56f1_pose_fabric.py
git -C hdgp commit -m "feat(fabric): retarget rh56f1 palm attractor to r_hl_palm_sensor (right)"
```

---

### Task 1.3: params yaml — collision frame 정합

**Files:**
- Modify: `.../fabric_params/openarm_rh56f1_pose_params.yaml`

- [ ] **Step 1: palm_link 참조 교체**

`body_repulsion.collision_sphere_frames`와 `collision_link_prefix_pairs`의 `palm_link`/`palm_link_sphere2`를 palm_sensor 근사 충돌구로 교체(URDF에 대응 충돌구 프레임을 palm_sensor 자식으로 추가하거나, 기존 손 충돌구로 대체). radii 배열 길이 일치 유지(assert).

- [ ] **Step 2: 로드 검증**

Run: fabric batch=2 인스턴스화 스모크 재실행.
Expected: collision_sphere_frames 길이 == radii 길이 assert 통과, 예외 없음.

- [ ] **Step 3: Commit**

```bash
git -C hdgp add source/FABRICS/src/fabrics_sim/fabric_params/openarm_rh56f1_pose_params.yaml
git -C hdgp commit -m "fix(fabric): align rh56f1 collision frames to palm_sensor (right)"
```

---

### Task 1.4: env(grasp_v1) — offset 제거 + euler 규약 재보정

**Files:**
- Modify: `.../grasp_v1/grasp_right_env.py`
- Modify: `.../grasp_v1/grasp_right_env_cfg.py` (pregrasp/palm_pose_mins,maxs 재보정 시)
- Modify: `.../grasp_v1/tests/test_phase4_env_static.py`

**Interfaces:**
- Consumes: Task 1.2 fabric(palm_sensor 직접 제어, offset 불필요).
- Produces: `_fabric_palm_pose_from_sensor_target`가 항등(입력=출력) 또는 호출부 제거; `_PALM_SENSOR_OFFSET_IN_FABRIC_PALM` 삭제; pregrasp/mins/maxs/upright euler가 palm_sensor 기준.

- [ ] **Step 1: offset 항등화**

`_fabric_palm_pose_from_sensor_target`를 위치 offset 제거(항등 반환)하거나 호출부(`env.py:685`, reset)에서 직접 palm_sensor pose를 fabric에 전달. `_PALM_SENSOR_OFFSET_IN_FABRIC_PALM` 상수 및 참조 5곳 제거. 실행자는 먼저 `grep -n _PALM_SENSOR_OFFSET_IN_FABRIC_PALM\|_fabric_palm_pose_from_sensor_target grasp_right_env.py`로 전 참조 목록화.

- [ ] **Step 2: pregrasp/mins/maxs/upright euler 재보정**

reset pregrasp(`env.py:670-677`)의 palm_sensor euler(`[:,3]=90°,[:,5]=90°`)와 `palm_pose_mins/maxs`, `_apply_upright_palm_orientation_correction`을 palm_sensor 기준으로 재계산. Task 1.2에서 얻은 palm_sensor 자연 자세 euler를 기준으로 "손바닥(+z)이 컵(아래/전방)을 향하는" 목표 euler를 산출한다. 산출값은 FK로 검증(손바닥 +z world 방향이 컵 방향과 이루는 각 확인).

- [ ] **Step 3: 정적 테스트 갱신 + 실행**

`test_phase4_env_static.py`의 offset 관련 3개 참조(assert)를 새 규약으로 갱신.
Run: `python3 -m pytest .../grasp_v1/tests/test_phase4_env_static.py .../tests/test_approach_cfg_static.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git -C hdgp add source/openarm/openarm/rh56f1/right/grasp_v1/
git -C hdgp commit -m "refactor(grasp_v1): drop palm offset, recalibrate euler to palm_sensor frame"
```

---

### Task 1.5: env(pour_v1) — 동일 재보정

**Files:**
- Modify: `.../pour_v1/pour_right_env.py`
- Modify: `.../pour_v1/tests/test_pour_rh56f1_static.py`

- [ ] **Step 1: offset/euler 재보정 이식**

pour_v1의 `_fabric_palm_pose_from_sensor_target`/offset 참조(1곳)와 pregrasp/upright euler를 Task 1.4와 동일 규약으로 맞춘다. pour 고유의 tilt 목표 자세도 palm_sensor 기준으로 재계산.

- [ ] **Step 2: 정적 테스트**

Run: `python3 -m pytest .../pour_v1/tests/test_pour_rh56f1_static.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git -C hdgp add source/openarm/openarm/rh56f1/right/pour_v1/
git -C hdgp commit -m "refactor(pour_v1): align palm target to r_hl_palm_sensor frame"
```

---

### Task 1.6: Phase 1 통합 검증 게이트 (IK 왕복 + 육안)

**Files:** (검증 전용, 코드 변경 없음. 필요 시 임시 스크립트)

- [ ] **Step 1: IK 왕복 (게이트 2, 오른쪽)**

fabric을 batch로 띄워 N개 target palm_sensor pose(작업공간 내)를 set_features로 주고 integrator를 수렴까지 step → `get_palm_pose`로 실제 palm_sensor pose 읽어 target과 위치/자세 오차 확인.
Expected: 위치<5mm·자세<2° 수렴(IK 특성상 정적 FK보다 느슨). FAIL 시 attractor gain/축점 스케일 점검.

- [ ] **Step 2: 육안 (게이트 3, 오른쪽) — 서버/GPU**

Run(서버): `./play.sh` 또는 grasp_v1 play로 reset 초기 자세 렌더 → 오른손바닥(+z)이 컵을 향하는지 확인.
Expected: 손바닥이 컵을 향함(90° 어긋남 해소). 사용자 육안 승인.

- [ ] **Step 3: Phase 1 태그**

```bash
git -C hdgp tag rh56f1-fabric-palm-right-aligned
```

---

## Phase 2 — 양팔 통합 (26 DOF, 인프라만)

### Task 2.1: generate 스크립트 — 왼팔+왼손 + l_hl_palm_sensor graft

**Files:**
- Modify: `.../urdf/openarm_rh56f1/generate_openarm_rh56f1_urdf.py`
- Regenerate: `openarm_rh56f1.urdf`

**Interfaces:**
- Produces: URDF에 왼팔 arm 체인 + 왼손 drive/mimic 체인 + `l_hl_palm_sensor` + 축점 `ps_l_*`. cspace joint 26개.

- [ ] **Step 1: 왼쪽 대칭 graft**

Task 1.1의 오른손 graft를 왼쪽(`l_al_*`/`l_hl_*`, 참고 URDF origin)으로 미러 추가. 왼팔 arm joint는 Tesollo 공유본에 있으면 재사용, 없으면 참고 URDF `l_al_*` 삽입. 왼손 drive joint 이름은 `rh56f1_left_left_*`(구 fabric 규약) 또는 통일 규약으로 일관되게. 축점 `ps_l_x…ps_l_z_neg`를 `l_hl_palm_sensor` 자식으로.

- [ ] **Step 2: 재생성 + FK 검증 (양쪽)**

Run: `python3 generate_openarm_rh56f1_urdf.py` 후 `python3 verify_palm_sensor_fk.py --urdf openarm_rh56f1.urdf --ref <ref> --side both`
Expected: 좌우 both PASS(위치<2mm·자세<1°).

- [ ] **Step 3: Commit**

```bash
git -C hdgp add source/FABRICS/.../openarm_rh56f1/
git -C hdgp commit -m "feat(fabric): add left arm+hand+l_hl_palm_sensor to rh56f1 urdf (26 DOF)"
```

---

### Task 2.2: fabric 코드 — cspace 26 DOF + 좌우 palm attractor

**Files:**
- Modify: `.../fabrics/openarm_rh56f1_pose_fabric.py`

**Interfaces:**
- Consumes: Task 2.1 URDF(26 joint, `l_hl_palm_sensor`, `ps_l_*`).
- Produces: `NUM_DOF=26`, `default_config`(26), `add_palm_points_attractor` 좌우 2 taskmap(`palm_r`/`palm_l`), `_palm_pose_target_r`/`_l`(각 12D), `set_features` 시그니처가 좌우 palm pose를 받도록 확장(하위호환: 왼쪽 미지정 시 default 유지).

- [ ] **Step 1: BaseFabric set_features 시그니처 확인**

Run: `grep -rn "def set_features" hdgp/source/FABRICS/src/fabrics_sim/` 로 상위 정의 위치 파악 후 palm pose 주입 경로 확인. 오른쪽 fabric의 set_features override 지점을 좌우 palm 지원으로 확장(왼쪽 target 미전달 시 중립 default).

- [ ] **Step 2: cspace/attractor 좌우화**

`NUM_DOF` 26, `default_config` 26값(오른팔/손 기존 + 왼팔 중립 + 왼손 열림). `add_palm_points_attractor`를 side 인자화해 `palm_r`(`r_hl_palm_sensor`+`ps_r_*`), `palm_l`(`l_hl_palm_sensor`+`ps_l_*`) 등록. `convert_transform_to_points`/`get_palm_pose`도 좌우.

- [ ] **Step 3: 로드 스모크 (batch=2)**

Run: fabric 인스턴스화 + construct_fabric + 좌우 palm taskmap 해상 스모크.
Expected: 예외 없음, cspace dim 26, palm_r/palm_l 각 7프레임.

- [ ] **Step 4: Commit**

```bash
git -C hdgp add source/FABRICS/src/fabrics_sim/fabrics/openarm_rh56f1_pose_fabric.py
git -C hdgp commit -m "feat(fabric): bimanual 26 DOF cspace + dual palm attractor"
```

---

### Task 2.3: params yaml — 양팔 충돌

**Files:**
- Modify: `.../fabric_params/openarm_rh56f1_pose_params.yaml`

- [ ] **Step 1: 왼팔 충돌구 + 양팔 충돌쌍**

왼팔/왼손 collision_sphere_frames 추가(오른쪽 대칭), 양팔 간 충돌쌍(`collision_link_prefix_pairs`에 오른팔↔왼팔) 추가. radii 길이 일치 유지.

- [ ] **Step 2: 로드 검증**

Run: fabric batch=2 스모크.
Expected: assert 통과, 예외 없음.

- [ ] **Step 3: Commit**

```bash
git -C hdgp add source/FABRICS/src/fabrics_sim/fabric_params/openarm_rh56f1_pose_params.yaml
git -C hdgp commit -m "feat(fabric): bimanual collision spheres + cross-arm pairs"
```

---

### Task 2.4: env — fabric_q 26 DOF + 왼팔 고정 target 공급

**Files:**
- Modify: `.../grasp_v1/grasp_right_env.py`
- Modify: `.../pour_v1/pour_right_env.py`

**Interfaces:**
- Consumes: Task 2.2 fabric(26 DOF, 좌우 palm).
- Produces: `fabric_q/qd/qdd`가 `num_joints=26`으로 자동 확장(이미 `self.fabric.num_joints` 사용). 왼팔/왼손 cspace 인덱스를 로봇 joint에 반영. 왼팔 palm target은 고정 중립 상수를 좌측 set_features 인자로 공급. action 차원 불변.

- [ ] **Step 1: 왼쪽 인덱싱/공급 추가**

`fabric_q`는 `num_joints` 기반이라 26으로 자동. 왼팔/왼손 cspace 슬라이스(예: `[13:20]`, `[20:26]`)를 `left_arm_dof_indices`/왼손 dof에 `set_joint_position_target`으로 반영(현재 `left_arm_zero_pos` 직접 고정 → fabric 출력 사용). `_apply_action`에서 왼팔 라인(`env.py:1387` 부근) 교체. set_features 호출에 왼쪽 palm 고정 target(중립 상수) 추가.

- [ ] **Step 2: 정적 테스트 (grasp+pour)**

Run: `python3 -m pytest .../grasp_v1/tests/ .../pour_v1/tests/ -q`
Expected: 기존 PASS 유지(차원/인덱스 회귀 없음). 실패 시 cspace 슬라이스/인덱스 정합 점검.

- [ ] **Step 3: Commit**

```bash
git -C hdgp add source/openarm/openarm/rh56f1/right/
git -C hdgp commit -m "feat(env): drive left arm+hand via bimanual fabric (fixed neutral target)"
```

---

### Task 2.5: Phase 2 통합 검증 게이트

- [ ] **Step 1: 양쪽 FK 정합 재확인**

Run: `python3 verify_palm_sensor_fk.py --urdf openarm_rh56f1.urdf --ref <ref> --side both`
Expected: 좌우 both PASS.

- [ ] **Step 2: 오른손 학습 회귀 (서버, 짧게)**

Run(서버): grasp_v1 또는 pour_v1 짧은 학습(수백 epoch) → Phase 1 대비 지표(approach/palm_to_cup) 유지·향상 확인. 통합 fabric 속도 실측(`num_envs=2048` step/s).
Expected: 오른손 태스크가 통합 fabric에서도 정상 동작, 속도 허용범위. 사용자 확인.

- [ ] **Step 3: Phase 2 태그**

```bash
git -C hdgp tag rh56f1-fabric-bimanual-infra
```

---

## Self-Review 결과

- **Spec coverage:** spec §4 (A)URDF→Task1.1/2.1, (B)fabric→Task1.2/2.2, (C)params→Task1.3/2.3, (D)env→Task1.4/1.5/2.4. §5 검증 3게이트→Task1.0/1.4/1.6/2.5. §6 롤백→git 태그/커밋 단위. 모든 spec 요구가 태스크에 매핑됨.
- **측정 의존 값(euler 재보정, default_palm_euler, cspace 슬라이스)**: placeholder가 아니라 "FK 산출 스텝 → 대입"으로 명시. 실행 시 정확한 값 확정.
- **Type/이름 일관성:** `control_point_frames`, `ps_r_*`/`ps_l_*`, `palm_r`/`palm_l`, `_palm_pose_target_r`/`_l`가 Task 간 일관.
- **알려진 리스크:** BaseFabric `set_features` 상위 시그니처 미확인 → Task 2.2 Step 1에서 확인 절차 명시.

---

## 진행 현황 (2026-07-03 갱신)

### 방향 전환 (계획 대비)
원안은 "오른팔 Tesollo 공유본 유지 + 왼팔 `_rl` 추가"였으나, Tesollo 오른팔 base 마운트가
Isaac USD(`_rl`)와 **6.4cm 어긋남**(`(0,-0.0935,0.698)` vs `(0,-0.031,0.698)`)이 확인됨.
→ **fabric 을 `_rl` URDF 단일 소스로 양팔 재생성**하도록 전환. Phase 1 오른팔도 `_rl` 로 재정비됨.
generate 스크립트는 Tesollo 의존을 걷어내고 `_rl` 파싱으로 전면 재작성.

### 완료 (정적/수치 검증)
- [x] Task 1.0 FK 검증기 (자기검증 both 0mm/0°)
- [x] Task 1.1→2.1 통합: **양팔 `_rl` URDF** (cspace 26, palm_sensor+축점 좌우, FK both PASS 0mm/0°)
- [x] Task 1.2 fabric palm attractor → `r_hl_palm_sensor` (control_point; 단 cspace 는 아직 13, 2.2 에서 26)
- [x] Task 1.3 params 충돌 프레임 → palm_sensor
- [x] Task 1.4 grasp_v1 offset 제거 + euler ex+90 (정적 9 pass)
- [x] Task 1.5 pour_v1 동일 (68 pass, 1 pre-existing)
- [x] Task 1.6 **수치 게이트**: 프레임 규약 테스트 3 pass (렌더 대체; palm_sensor +z_world=[0,0,-1], ex+90 손배치 보존)

### 남은 작업 (warp 런타임 서버 검증 필요)
- [ ] Task 2.2 fabric 코드: cspace 13→**26**, `default_config` 26(왼팔 중립+왼손),
      좌우 palm attractor(`palm_r`/`palm_l`), `TIP_FRAMES` `rh56f1_tip_*`→`r_hl_*_tip`,
      `set_features` 좌우 palm. **BaseFabric.set_features 시그니처 파악 선행.**
- [ ] Task 2.3 params: 양팔 충돌구(`r_al_*`/`l_al_*`), `world_body` 참조 정리(현재 Tesollo 유래).
- [ ] Task 2.4 env(grasp_v1/pour_v1): `fabric_q` 26, 왼팔/왼손 fabric 구동(고정 중립 target).
- [ ] Task 2.5 서버 검증: fabric 로드 스모크(num_joints=26), IK 왕복 수치, (play 육안 불가 → 수치).

### 재개 시 주의
- 재생성 URDF tip 프레임 이름이 `r_hl_*_tip`(구 `rh56f1_tip_*` 아님) → fabric `TIP_FRAMES` 갱신 필수(미갱신 시 로드 에러).
- 현재 fabric 코드 `NUM_DOF=13`, `default_config` 13 → URDF(26)와 불일치 상태. Task 2.2 전까지 fabric 로드 불가(warp 없어 로컬 무영향).
- ledger: `hdgp/.superpowers/sdd/progress.md`.
