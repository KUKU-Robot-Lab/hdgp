# 환경 설정 레퍼런스 — hdgp 5g_pour_right_v3

작성일: 2026-05-31  
브랜치: `pour` (hdgp repo)  
목적: 새로운 환경 설치 시 참고할 설정 값 및 수정 사항 정리

---

## 1. 디렉토리 구조

```
~/rl_ws/
├── hdgp/          ← 이 repo (pour branch)
├── IsaacLab/      ← 아래 수정 필요
├── FABRICS/       ← Fabrics sim 패키지
└── datasets/
    ├── grasp_warm_v7_2.hdf5          ← warm state bank (필수)
    └── pour_v1_a{11..20}.hdf5        ← demo 데이터 (선택)
```

---

## 2. IsaacLab 수정 사항

### 2-1. `apps/isaaclab.python.kit`

URDF importer 버전 핀 제거:

```diff
-"isaacsim.asset.importer.urdf" = {version = "2.4.31", exact = true}
+"isaacsim.asset.importer.urdf" = {}
```

> **범위 제한**: 5g_pour_right_v3는 USD를 직접 로드하므로 (`assets/openarm_tesollo_sensor/openarm_tesollo_sensor.usd`)  
> 이 변경은 runtime gripper pose에 직접 영향을 주지 않는다. 저위험 테스트이므로 맞춰두는 것은 무방하나,  
> 렌더링 차이의 root cause가 아닐 수 있다.

### 2-2. `isaaclab.sh` — PYTHONPATH 설정

파일 상단 (export ISAACLAB_PATH 직후)에 추가:

```bash
# hdgp/source/openarm -> openarm package
HDGP_CANDIDATE="${ISAACLAB_PATH}/../hdgp"
if [[ -d "${HDGP_CANDIDATE}/source/openarm" ]]; then
    export PYTHONPATH="${HDGP_CANDIDATE}/source/openarm:${PYTHONPATH}"
fi

# FABRICS/src -> fabrics_sim package
FABRICS_CANDIDATE="${ISAACLAB_PATH}/../FABRICS"
if [[ -d "${FABRICS_CANDIDATE}/src" ]]; then
    export PYTHONPATH="${FABRICS_CANDIDATE}/src:${PYTHONPATH}"
fi

# IsaacLab source tree (offline editable-install fallback)
for module_dir in isaaclab isaaclab_rl isaaclab_tasks isaaclab_assets isaaclab_mimic; do
    if [[ -d "${ISAACLAB_PATH}/source/${module_dir}" ]]; then
        export PYTHONPATH="${ISAACLAB_PATH}/source/${module_dir}:${PYTHONPATH}"
    fi
done

# Isaac Sim warp extension path (warp module import fallback)
WARP_CORE_CANDIDATE="${ISAACLAB_PATH}/_isaac_sim/extscache/omni.warp.core-1.8.2+lx64"
if [[ -d "${WARP_CORE_CANDIDATE}" ]]; then
    export PYTHONPATH="${WARP_CORE_CANDIDATE}:${PYTHONPATH}"
fi

# External Python 3.11 site-packages fallback (환경 이름은 실제 환경에 맞게 수정)
for extra_site in \
    "/home/${USER}/anaconda3/envs/yoon/lib/python3.11/site-packages" \
    "/home/${USER}/anaconda3/envs/isaac_py311/lib/python3.11/site-packages"
do
    if [[ -d "${extra_site}" ]]; then
        export PYTHONPATH="${PYTHONPATH}:${extra_site}"
    fi
done
```

### 2-3. `source/isaaclab/isaaclab/managers/reward_manager.py`

`weight=0` term도 TFEvents에 로깅되도록 수정:

```diff
-            if term_cfg.weight == 0.0:
-                self._step_reward[:, term_idx] = 0.0
+            if term_cfg.weight == 0.0:
+                raw_value = term_cfg.func(self._env, **term_cfg.params)
+                self._episode_sums[name] += raw_value * dt
+                self._step_reward[:, term_idx] = raw_value
                 continue
```

> **범위 제한**: 5g_pour_right_v3는 **DirectRLEnv** 구조이므로 IsaacLab의 RewardManager를 사용하지 않는다.  
> 이 수정은 ManagerBased task (다른 태스크)에는 유용하지만, v3 렌더링 문제와는 무관하다.

---

## 3. 렌더링 차이 — 진단 및 근본 원인

### 실제 구조

```
5g_pour_right_v3
  └─ USD 직접 로드: assets/openarm_tesollo_sensor/openarm_tesollo_sensor.usd
  └─ target cup 위치: compute_left_cup_pose_from_fk(LEFT_ARM_REST_JOINT_POS) → 상수로 고정
                       reset 시 LEFT_TARGET_CUP_POS_ENV_LOCAL + env_origin 에 배치
```

**왼손 gripper가 sim에서 실제로 어디에 있는지 읽지 않는다.**  
FK 상수 위치와 실제 `openarm_left_hand` body pose 사이 오프셋이 있으면 컵이 어긋나 보인다.

### 진단 방법

play.py 실행 후 step 1에서 아래 값을 비교:

```python
# 실제 sim body pose (step 1 이후, stale 아님)
left_hand_pos_w  = env.robot.data.body_pos_w[:, left_hand_body_idx]
left_hand_quat_w = env.robot.data.body_quat_w[:, left_hand_body_idx]

# 현재 배치된 cup world pose
cup_pos_w = env.left_target_cup.data.root_pos_w

# offset
offset = cup_pos_w - left_hand_pos_w   # 0에 가까워야 함 (local_z=0.05 제외)
```

offset이 0이 아니면 → FK 파라미터가 실제 USD와 불일치.

### 근본 수정 방향

현재 방식 대신, reset 후 첫 step에서 실제 body pose 기준으로 cup 재배치:

```python
# _compute_observations 또는 _step_physics에서, step_count == 1인 env에만 적용
if first_step_mask.any():
    left_hand_pos = self.robot.data.body_pos_w[first_step_mask, left_hand_body_idx]
    left_hand_quat = self.robot.data.body_quat_w[first_step_mask, left_hand_body_idx]
    cup_pos = left_hand_pos + quat_apply(left_hand_quat, local_offset)
    self.left_target_cup.write_root_pose_to_sim(cup_pose, env_ids=...)
```

> body_pos_w는 reset 직후 stale이지만, 첫 physics step 후에는 정확하다.

---

## 4. hdgp 파라미터 현황 (test10 기준, 2026-05-31)

### 3-1. 왼팔 preset 관절값 — `pour_right_preset.py`

```python
LEFT_ARM_REST_JOINT_POS = {
    "openarm_left_joint1": -0.315,
    "openarm_left_joint2": -0.079,   # test10: demo a11-a20 평균 (이전: -0.290)
    "openarm_left_joint3":  0.217,   # test10: demo a11-a20 평균 (이전: +0.400)
    "openarm_left_joint4":  0.513,
    "openarm_left_joint5":  0.666,
    "openarm_left_joint6": -0.729,
    "openarm_left_joint7": -0.957,
    "openarm_left_finger_joint1": 0.044,
    "openarm_left_finger_joint2": 0.044,
}
```

이 값에서 FK로 계산된 target cup 위치 (robot-base-local):

```
LEFT_TARGET_CUP_POS_ENV_LOCAL : [0.2904, 0.0290, 0.3238]   (x, y, z 단위: m)
LEFT_TARGET_CUP_QUAT_WXYZ     : [0.5278, 0.0618, -0.0119, 0.8470]
```

### 3-2. 주요 reward weight — `pour_right_env_cfg.py`

| 파라미터 | 값 | 변경 이력 |
|---------|---|----------|
| `weight_tilt` | 40.0 | test5: 8→40 |
| `weight_align` | 6.0 | - |
| `weight_pour_dist` | 12.0 | - |
| `weight_bead_progressive` | 200.0 | - |
| `weight_bead_entry_delta` | 300.0 | - |
| `weight_source_drain` | 20.0 | - |
| `weight_spill` | 40.0 | - |
| `weight_success` | 100.0 | - |
| `weight_j0_ext_rot` | **0.0** | **test9: 3.0→0.0 (IK branch 탈출)** |
| `weight_premature_tilt` | 1.0 | - |
| `weight_action_rate_palm` | 0.02 | - |
| `weight_action_rate_finger` | 0.005 | - |
| `pour_tilt_sharpness` | **4.0** | **test8: 2.0→4.0** |
| `pour_tilt_target_deg` | 120.0 | - |

### 3-3. Null-space 설정 — `pour_right_env.py`

pour phase (stage 3):

```python
_null_cfg[:, 0] = torch.clamp(_null_cfg[:, 0] * 0.70 + 0.37 * 0.30, min=0.0, max=0.46)
# test9: j0 null-space target → +0.37 (demo pour 자세), min=0.0으로 Branch B 유도
_null_cfg[:, 1] = torch.clamp(_null_cfg[:, 1] * 0.95 + 0.39 * 0.05, min=0.00, max=1.05)
_null_cfg[:, 2] = torch.clamp(_null_cfg[:, 2] * 0.95 + (-0.24) * 0.05, min=-0.74, max=0.38)
_null_cfg[:, 6] = torch.clamp(_null_cfg[:, 6] * 0.95 + 0.63 * 0.05, min=0.20, max=1.13)
```

grasp phase (stage 1):
```python
_null_cfg[:, 0] = torch.clamp(_null_cfg[:, 0] * 0.95 + 0.09 * 0.05, min=-0.29, max=0.46)
```

---

## 4. 필수 파일 경로

| 파일 | 경로 | 용도 |
|-----|------|------|
| warm state bank | `~/rl_ws/datasets/grasp_warm_v7_2.hdf5` | pour reset 초기 상태 (필수) |
| grasp checkpoint | `~/rl_ws/hdgp/log/rl_games/pipeline/right/5g_grasp_right_v7_2/test3/nn/5g_grasp_right-v7-2.pth` | warm state rollout용 (disk 모드면 불필요) |
| demo 데이터 | `~/rl_ws/datasets/pour_v1_a{11..20}.hdf5` | demo BC / joint 분석용 |

---

## 5. test 이력 요약

| Test | 핵심 변경 | 결과 |
|------|----------|------|
| test7 | z_window gate 추가 | 안정 수렴, 90° local min |
| test8 | `pour_tilt_sharpness` 2.0→4.0 | XY drift 악화, 학습 붕괴 |
| test9 | `weight_j0_ext_rot` 3.0→0.0, null-space j0→+0.37 | j0 -0.54로 고착, 실패 |
| **test10** | **왼팔 preset joint2/3 demo 값으로 수정** | **진행 중** |

### test10 핵심 가설

왼팔 preset이 demo와 다른 자세였기 때문에 오른팔이 120° tilt를 물리적으로 할 수 없었다.  
demo 값으로 수정 후, 학습 초반 j0 ±탐색에서 Branch B (j0>0, j2<0, j4<0) 발견 시  
120° tilt 가능 → reward spike → Branch B 수렴 기대.

---

## 6. IK Branch 메모

| | warm state bank | demo pour |
|-|----------------|-----------|
| j0 | -0.54 (Branch A) | +0.22 (Branch B) |
| j2 | +0.25 (양수) | -0.34 (음수) |
| j4 | +0.69 (양수) | -1.16 (음수) |

Fabrics는 로컬 IK → episode 중 Branch 전환 불가.  
학습 초반 j0 랜덤 탐색 시 Branch B 진입 → 왼팔 수정으로 120° tilt 가능 → 수렴 기대.

---

*최종 업데이트: 2026-05-31*
