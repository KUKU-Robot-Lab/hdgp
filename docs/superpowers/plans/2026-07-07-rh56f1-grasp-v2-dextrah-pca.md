# rh56f1 grasp_v2 — 원본 DEXTRAH 직접 이식 (물체 파지 + inspire PCA) 계획

> **진행 상태 (2026-07-07):**
> - ✅ **Phase 1 완료·커밋·푸시** (`8179f49`/`f7e9177`): RH56F1 grasp PCA5 basis 계산. PC1(97.9%)=엄지+4손가락 조율 닫힘=firm envelope 시너지 확인. 산출물 `assets/demograsp_references/rh56f1_grasp_pca5.pt`(force-add), 스크립트 `scripts/pca/compute_rh56f1_grasp_pca.py`. **방향 검증 게이트 통과.**
> - ✅ **Phase 2 정적 완료** (커밋 예정): rh56f1 fabric에 `hand_mode="pca"` 경로 추가 — `RH56F1_HAND_PCA_MATRIX`(5×6, .pt와 일치) 모듈상수, `add_hand_fabric` 분기(pca=(5,26)/direct=(6,26)), `set_features` hand_target 5D/6D. **grasp_v1 불변**(기본 `hand_mode="direct"`, `use_hand_fabric=False`). 정적검증: hand_map (5,26)·오른손블록=basis·팔열0. **핵심 커플링**: fabric taskmap=uncentered(mean 미차감)이라 env action 범위=`pca_action_mins/maxs`(uncentered 투영, .pt에 추가). ⏳ **기능검증(IK 왕복 PCA→qpos) = GPU 대기**(server `verify_fabric_load_ik.py`).
> - ⏭️ **다음 재개점 = Phase 3** (DEXTRAH env 1662줄 → `grasp_v2/` 이식, 최대 덩어리).

**Goal:** grasp_v1(컵 특화, per-finger synergy)로 RH56F1 firm 파지 학습 불가 확정(4개 실험). **원본 DEXTRAH(dextrah_kuka_allegro)의 "일반 물체 파지→goal" 태스크를 OpenArm+RH56F1(inspire PCA 손)로 충실 이식.** 컵/pour 특화(테이퍼·palm-seat·엄지 막힘)를 버리고, **검증된 DEXTRAH 물체 파지가 RH56F1에서 되게** 한 뒤 cup/pour는 나중에 특화.

**방향 전환 (사용자 지시):**
- 태스크 = **컵 파지 아님**. DEXTRAH처럼 **일반 물체(visdex_objects/primitives) 파지 → goal 이동** (object diversity DR).
- reward = DEXTRAH goal-driven (hand_to_object + object_to_goal + lift). 우리 컵 phase/reward 폐기.
- Tesollo grasp_v2 무시. **원본 DEXTRAH 구조·태스크 충실 이식.**

**왜 이게 근본 해결인가:** grasp_v1 실패 = 컵 특화 제약 + per-finger synergy 탐색 난이도. DEXTRAH는 (a)PCA 손 action(자연 파지 manifold), (b)fabric-guided, (c)다양한 물체 파지가 **이미 검증된** 구조. 컵을 빼고 이 검증된 파지를 RH56F1에 올리면 dexterous 파지가 되고, 그 위에 컵/pour 특화.

## 원본 DEXTRAH 구조 (이식 대상)
- repo: `repo/DEXTRAH/dextrah_lab/tasks/dextrah_kuka_allegro/` (env 1662L, cfg, constants, adr).
- **action 11D = 6 palm pose(fabric IK) + 5 hand PCA**.
- **PCA basis는 fabric 안**: `kuka_allegro_pose_fabric.py`의 `pca_matrix`(5×16 하드코딩) → LinearMap "pca_hand" taskmap → 5 PCA를 Allegro 16관절로 매핑.
- fabric-guided: 정책 action(palm pose + PCA) → geometric fabric → arm+hand 관절(충돌회피 포함).
- reward: goal-driven (hand_to_object + object_to_goal + lift). ADR. teacher-student distillation(배포용, 후순위).

## 현 자산 (검증 완료)
- **rh56f1 fabric 존재**(`openarm_rh56f1_pose_fabric.py`) — "pca_hand" taskmap 구조는 있으나 **hand_map=eye(6) identity(직접 제어, PCA 아님)**. → PCA basis로 교체 필요.
- **inspire grasp 시연**: `assets/demograsp_references/grasp_ref_inspire.pkl` — hand_qpos(22,6)+wrist_pos/quat. **6-DOF = RH56F1 6-drive 일치.**
- **inspire PCA5 정규화**: `grasp_ref_inspire_teosollo_pca5.pt` — hand_pca(22,5)+palm_pose(22,6), meta(pinv_subset_6d). reset 레퍼런스 후보.
- fabrics_sim 원본(kuka): `repo/FABRICS/.../kuka_allegro_pose_fabric.py` — PCA fabric 구현 참조.
- grasp_v1의 rh56f1 USD/joint/palm_sensor 이관 경험.

## Global Constraints
- **원본 DEXTRAH 구조 충실**: action 11D(6 palm + 5 PCA), goal reward, ADR. Tesollo synergy/lerp 방식 도입 금지.
- PCA는 **inspire grasp 시연 기반 실제 eigengrasp basis**(identity 금지).
- reward/gate 변경은 reward-audit.
- 검증 3단계: 정적 → fabric IK/PCA 스윕 → play.py 육안.
- 새 태스크 `open-rh56f1_r_grasp_v2` (v1 불변, 독립).

---

## Phase 1: inspire PCA basis 계산 + **firm 파지 존재 검증** [최핵심·최선행]

**목표:** 이 방향의 타당성을 학습 전에 판정 — inspire 22 grasp에 firm/envelope 파지가 있고 PCA5가 그걸 포착하는가.

**Files:** `scripts/tools/compute_inspire_pca.py` (신규)
- `grasp_ref_inspire.pkl`의 hand_qpos(22,6) → **sklearn PCA(n=5)** → basis(5×6) + mean(6) + PCA 좌표 범위(mins/maxs).
- kuka_allegro fabric의 pca_matrix 형식(LinearMap 입력)에 맞춰 저장(`assets/.../rh56f1_pca5_basis.pt`).

**검증 (방향 판정 게이트):**
- 22 grasp를 재구성(5 PCA→6 qpos)해 원본과 오차 확인(설명분산 ≥ ~90%).
- **PCA mins~maxs 스윕 → 각 PCA 샘플의 손 config를 probe(force_close 확장)로 측정: firm 손가락(tip&근위 동시)≥3 되는 PCA 좌표가 존재하는가.**
  - **존재하면** → v2 방향 타당(firm이 action 공간에 내장). Phase 2 진행.
  - **없으면** → inspire 시연 자체에 firm 파지 없음 → 시연 데이터 재수집/다른 소스 필요(방향 재검토). **여기서 조기 판정해 헛수고 방지.**

## Phase 2: rh56f1 fabric에 PCA 손 공간 이식

**Files:** `source/FABRICS/.../openarm_rh56f1_grasp_pca_fabric.py` (신규 또는 기존 fabric 확장), params yaml
- kuka_allegro_pose_fabric의 PCA 구조 이식: `pca_matrix`(identity → Phase1 basis 5×6), arm 7-DOF 앞에 0블록 스택(5×(7+6) 또는 rh56f1 DOF에 맞게), LinearMap "pca_hand" + hand_attractor.
- palm pose taskmap: grasp_v1서 정합한 palm_sensor 프레임·euler 규약 이관.
- mimic(원위)은 fabric이 drive 6 목표를 내면 sim mimic이 추종(하드웨어 결합 유지).

**검증:** fabric IK 왕복(palm pose target→FK <2mm/<1°), PCA target→hand qpos가 Phase1 basis대로 재현.

## Phase 3: DEXTRAH env 이식 (kuka_allegro → openarm_rh56f1)

**Files:** `source/openarm/openarm/rh56f1/right/grasp_v2/` (신규, dextrah_kuka_allegro 구조 이식)
- env.py: DEXTRAH 구조 이식 — action 11D(6 palm + 5 PCA), fabric 통합(위 fabric), obs(state teacher), goal 설정.
  - Kuka 특화 제거, OpenArm(r_aj 7) + RH56F1 매핑.
  - fabric_q/qd/qdd, integrator, capture forward pass(DEXTRAH 방식).
- cfg.py: num_actions=11, fabrics_dt/decimation, ADR, USD/joint(grasp_v1 재사용).
- constants.py: NUM_HAND_PCA=5, HAND_PCA_MINS/MAXS(Phase1), joint 그룹.
- config/__init__.py: `open-rh56f1_r_grasp_v2`(+lstm/play) 등록.

**검증:** 정적 env 생성, 차원 계약(action 11), import/register.

## Phase 4: reward/goal/ADR (DEXTRAH goal-driven 유지)

**Files:** grasp_v2/env.py, grasp_adr.py, cfg.py
- reward: DEXTRAH `compute_rewards`(hand_to_object + object_to_goal + lift) 이식. weight DEXTRAH 기준서 rh56f1 스케일 조정(reward-audit).
- goal: 컵을 목표 위치로 이동+리프트(pour 이관 고려).
- ADR: dextrah_adr 이식(contact/friction/mass/fabric_damping DR).
- reset: inspire PCA5 정규화(palm_pose+hand_pca) 레퍼런스로 pregrasp 초기화(선택).

**검증:** reward 계약 정적, 짧은 GPU 학습서 reward 상승·붕괴 없음.

## Phase 5: GPU 학습 + 물체 파지 검증 (DEXTRAH 기준)

**절차:** 서버 GPU0 `open-rh56f1_r_grasp_v2-lstm` 학습, parse_tfevents 모니터(epoch 3000 기준).
**성공 기준 (DEXTRAH 태스크 = 물체를 잡아 goal로):**
- **object_to_goal 성공률 상승** (물체를 goal 위치로 옮김) — DEXTRAH의 핵심 지표.
- lift_reward·hand_to_object 정상 수렴.
- 다양한 물체(primitives→visdex)에 일반화.
- (참고) firm 파지·held는 컵 특화 지표라 여기선 부차 — 물체를 잡아 옮기면 성공.

**분기:** 물체 파지 성공 → dexterous 파지 확보 → 그 위에 컵/pour 특화(별도 후속). 실패 → PCA basis(Phase1)·fabric(Phase2)·reward(Phase4) 재검토.

---

## 리스크
- **Phase 1이 방향을 가른다**: inspire 22 grasp에 firm 파지 없으면 v2도 실패 → **학습 전 Phase1 스윕으로 조기 판정**(가장 큰 가치).
- inspire hand_qpos 6-DOF가 RH56F1 6-drive 순서/부호와 정합하는지(joint 매핑 검증).
- fabric PCA taskmap의 DOF 정렬(arm 7 + hand 6 = 13, kuka는 7+16=23) 재계산.
- DEXTRAH env가 Kuka/Allegro 특화 가정(관절 수, USD 구조)을 얼마나 담는지 → 이식 공수.
- teacher-student distillation은 배포 단계 → 초기 학습(state 기반)엔 후순위.

## 순서
**Phase 1(inspire PCA + firm 존재 검증 — 학습 전 방향 판정)** → 2(fabric PCA) → 3(env 이식) → 4(reward/ADR) → 5(학습·검증). Phase 1에서 "firm이 PCA 공간에 있는가"를 싸게 먼저 확인하는 게 전체 성패의 관건.

---

## Phase 3 이식 설계 (2026-07-07 확정 · 실행 중)

**사용자 결정:** ① 베이스=grasp_v1 복사 후 개조(rh56f1 fabric/USD/joint 인프라 재사용). ② 물체=primitives 다물체 즉시 도입(MultiAsset + DEXTRAH drop). tesollo grasp_v2의 MultiAsset/drop-settle 자산 참고.

**폴더:** `source/openarm/openarm/rh56f1/right/grasp_v2/` (grasp_v1 전체 복사). gym id `open-rh56f1_r_grasp_v2`(+lstm/play/play-lstm) — config/__init__.py 등록 완료(자동발견 glob).

**핵심 원칙:** grasp_v1의 **인프라 유지**(`__init__`, `_setup_scene`, `_setup_geometric_fabrics`, reset fabric IK, robot/joint indices), **핵심 6메서드는 DEXTRAH 구조로 교체**. 컵특화 phase(approach/grasp/lift/stabilize)·envelope synergy·접촉 reward 전면 제거.

### DEXTRAH → rh56f1 이식 매핑 (원본 dextrah_kuka_allegro_env.py)
| 항목 | DEXTRAH(kuka) | rh56f1 grasp_v2 |
|-----|--------------|-----------------|
| action | 11D=palm6+PCA5, `compute_actions`가 각각 limits로 scale | 동일 11D. palm→palm_pose_targets(palm_pose_mins/maxs), PCA5→hand_pca_targets(pt `pca_action_mins/maxs`) |
| fabric | kuka fabric, `set_features(pca,palm,...)` | `OpenArmRh56f1PoseFabric(use_hand_fabric=True, hand_mode="pca")`. set_features hand_target=(B,5) |
| `_pre_physics_step` | phase 없음: compute_actions→set_features→integrate decimation | 동일. grasp_v1의 phase/synergy/thumb-freeze 전량 제거 |
| `_apply_action` | fabric_q→dof_pos/vel_targets(arm+hand 전체) | fabric_q[:, :13] arm+hand 전체를 로봇에 세팅(mimic은 fabric이 drive6 목표→sim mimic 추종). 왼팔 고정 |
| reward | hand_to_object + object_to_goal + finger_curl_reg + lift | `compute_rewards` 그대로 이식. weight=Phase4 |
| hand_to_object | `hand_pos`(fingertip+palm keypoints)와 object 거리 **MAX** | rh56f1: fingertip5+palm_center → object_pos MAX거리 |
| object_goal | 고정 `[-0.5,0,0.75]` | rh56f1 workspace 맞게 조정(Phase4). 우선 고정값 |
| curled_q | nominal curl(16 hand dof) | rh56f1 6-drive curl 자세(hand_grasp_pose 근처) |
| dones | out_of_reach(xy범위 or z<0.2) \| time_out | 동일. cup_tipping 등 접촉 dones 제거 |
| reset | object xy랜덤+z=0.5 drop+rot랜덤, goal고정 | tesollo grasp_v2 방식: MultiAsset spawn + object_spawn_z drop + settle_steps. 손 pregrasp fabric IK |
| obs | dof/hand_pos/goal/actions/fabric_q/qd/qdd | grasp_v1 96D 골격 유지+개조: last_actions 12→11, **object_goal 3D 추가**, cup→object 개념, 접촉 obs는 net_forces 또는 제거(Phase4 확정). NUM_OBSERVATIONS 재계산 |

### Phase 3 실행 순서 (완결 단위)
1. ✅ 폴더 복사 + config 등록(open-rh56f1_r_grasp_v2 4종)
2. constants.py: NUM_FINGER_ACTION 6→NUM_HAND_PCA 5, NUM_ACTIONS 11, HAND_PCA_MINS/MAXS(pt), obs 차원 재계산
3. env_cfg.py: num_actions=11, fabric use_hand_fabric=True/hand_mode=pca, cup→MultiAsset primitives, replicate_physics=False, object_spawn_z/settle_steps, reward/goal 파라미터, ContactSensor 다물체(net_forces or 제거)
4. env.py: _pre_physics_step/_apply_action(phase제거+PCA), _get_rewards(compute_rewards), _get_observations(goal추가/action11), _reset_idx(drop-settle+goal), _get_dones(out_of_reach). fabric setup use_hand_fabric=True
5. 정적 검증: import/register, action==11 계약, (GPU)env 생성

**Phase 3 검증 목표(계획 원문):** 정적 env 생성, 차원계약(action 11), import/register. reward weight 튜닝·학습은 Phase4~5. 단 다물체+PCA+goal-driven 구조가 정적으로 성립하는 최소 골격까지가 Phase3.
