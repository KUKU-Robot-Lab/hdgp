# hdgp/scripts/tools README

`/home/user/rl_ws/hdgp/scripts/tools` 아래 유틸리티 스크립트들의 목적과 기본 사용법을 정리한 문서입니다.

## 개요

이 디렉터리에는 크게 4종류의 도구가 있습니다.

1. IsaacLab/Isaac Sim 환경을 직접 띄워 검증하거나 데이터를 수집하는 도구
2. 학습 로그, TensorBoard, 체크포인트를 후처리하는 분석 도구
3. Phase B/C/D 실험 검증용 리포트 도구
4. 자산/URDF/참조 데이터 변환용 보조 도구

## 실행 방식 요약

### 1) IsaacLab 기반 스크립트

`isaaclab.app.AppLauncher`를 쓰는 스크립트는 보통 IsaacLab 루트에서 아래처럼 실행합니다.

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/tools/<script>.py --headless [추가 옵션]
```

주로 해당되는 파일:

- `confirm_env.py`
- `eval_grasp2g_policy.py`
- `export_scene.py`
- `pipeline_play.py`
- `report_5g_grasp_right_spawn.py`
- `report_phase32_success_metrics.py`
- `report_phaseb3_metrics.py`
- `report_phasec1_contact_metrics.py`
- `report_phasec2_contact_reliability.py`
- `report_phasec4_object_feat.py`
- `save_grasp_terminal_states.py`
- `save_reach_terminal_states.py`
- `save_transfer_terminal_states.py`
- `verify_contact_filter_cpu.py`
- `list_envs.py`

### 2) 일반 Python 스크립트

Isaac Sim 없이 실행 가능한 파일은 보통 아래처럼 실행합니다.

```bash
python3 /home/user/rl_ws/hdgp/scripts/tools/<script>.py [옵션]
```

주로 해당되는 파일:

- `build_run_dataset.py`
- `convert_demograsp_references.py`
- `find_pregrasp_pose.py`
- `freeze_run_analysis.py`
- `generate_object_pc_feature_map.py`
- `patch_swap_lr.py`
- `print_command_slice.py`  (`isaaclab.sh -p` 실행 권장)
- `report_lift_kpi.py`
- `summarize_phase_d_kpi.py`
- `summarize_phase_d_matrix.py`
- `urdf_spec_report.py`

### 3) 셸 래퍼

아래 파일은 여러 실험 케이스를 배치 실행하는 Bash 스크립트입니다.

- `run_phase_d_matrix.sh`
- `run_phase_d_kpi_matrix.sh`

## 파일별 설명

### 환경 확인 / 디버깅

#### `list_envs.py`

- 목적: `openarm`에 등록된 IsaacLab 환경 목록을 표 형태로 출력
- 용도: 사용 가능한 task id 확인
- 예시:

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/tools/list_envs.py --headless
```

#### `confirm_env.py`

- 목적: 환경의 action/observation space, action term 구성, left/right 출력 순서 등을 확인
- 주요 옵션:
- `--task`: 확인할 task id
- `--num_envs`: 생성할 환경 수
- `--steps`: 테스트 스텝 수
- `--split_index`: dual-head 분기 인덱스 확인용
- 예시:

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/tools/confirm_env.py \
  --task Grasp2g_IK-v0 \
  --num_envs 1 \
  --steps 3 \
  --headless
```

#### `print_command_slice.py`

- 목적: 특정 task의 observation layout과 command slice offset 출력
- 용도: 관측 벡터 안에서 command 관련 구간이 어디인지 확인할 때 사용
- 주요 옵션:
- `--task`: Gym task id
- `--group`: observation group 이름, 기본값 `policy`
- `--num_envs`: 생성 환경 수
- `--disable_fabric`: fabric 대신 USD I/O 사용
- 예시:

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/tools/print_command_slice.py \
  --task reach2g-v1 \
  --group policy \
  --num_envs 1 \
  --headless
```

#### `export_scene.py`

- 목적: task의 robot, table, object가 포함된 단일 USD scene export
- 추가 기능: joint state JSON 입력/출력
- 주요 옵션:
- `--task`: export할 task
- `--agent`: Hydra config 해석에 사용할 agent entry point
- `--output`: 출력 USD 경로
- `--joint-input`: 적용할 joint state JSON
- `--joint-output`: 현재 joint state 저장 경로
- 예시:

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/tools/export_scene.py \
  --task GraspIK-v0 \
  --output /tmp/grasp_scene.usd \
  --joint-output /tmp/grasp_joint_state.json \
  --headless
```

### 정책 평가 / 상태 저장

#### `eval_grasp2g_policy.py`

- 목적: grasp2g 정책을 rollout 하며 left/right success metric 평가
- 주요 옵션:
- `--task`: 평가할 task
- `--checkpoint`: 정책 체크포인트 경로
- `--agent`: runner config entry point
- `--num_episodes`: 평가 episode 수
- `--num_envs`: 병렬 환경 수
- `--goal_threshold`: 성공 기준 threshold
- `--seed`: 랜덤 시드
- 예시:

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/tools/eval_grasp2g_policy.py \
  --task Grasp2g_IK-v0 \
  --checkpoint /path/to/model.pt \
  --num_episodes 100 \
  --num_envs 1 \
  --headless
```

#### `save_reach_terminal_states.py`

- 목적: reach 정책의 성공 terminal state를 저장해 다음 curriculum 단계 초기 상태로 사용
- 출력: 기본값 `reach_terminal_states.pt`
- 주요 옵션:
- `--task`, `--checkpoint`, `--num_episodes`, `--num_envs`, `--output`
- `--success_threshold`: 성공 거리 기준
- `--seed`
- 예시:

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/tools/save_reach_terminal_states.py \
  --task Reach-v1 \
  --checkpoint /path/to/reach.pt \
  --num_episodes 1000 \
  --output ../hdgp/data/reach_terminal_states.pt \
  --headless
```

#### `save_grasp_terminal_states.py`

- 목적: grasp 정책의 성공 terminal state를 저장해 transfer curriculum 초기 상태로 사용
- 추가 기준:
- `--lift_height`: 리프트 성공 높이 기준
- `--grasp_dist`: end-effector 와 cup 거리 기준
- 예시:

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/tools/save_grasp_terminal_states.py \
  --task GraspIK-v0 \
  --checkpoint /path/to/grasp.pt \
  --num_episodes 1000 \
  --output ../hdgp/data/grasp_terminal_states.pt \
  --headless
```

#### `save_transfer_terminal_states.py`

- 목적: transfer 정책의 성공 terminal state를 저장해 pour curriculum 초기 상태로 사용
- 추가 기준:
- `--close_threshold`: object2-object 거리 기준
- `--target_offset_z`, `--source_offset_z`
- 예시:

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/tools/save_transfer_terminal_states.py \
  --task TransferIK-v0 \
  --checkpoint /path/to/transfer.pt \
  --num_episodes 1000 \
  --output ../hdgp/data/transfer_terminal_states.pt \
  --headless
```

#### `pipeline_play.py`

- 목적: Reach -> Grasp -> Transfer -> Pour 정책을 순차 실행
- 용도: 파이프라인 전체 연결 상태 확인, stage별 checkpoint smoke test
- 주요 옵션:
- `--reach-task`, `--grasp-task`, `--transfer-task`, `--pour-task`
- `--reach-ckpt`, `--grasp-ckpt`, `--transfer-ckpt`, `--pour-ckpt`
- `--reach-steps`, `--grasp-steps`, `--transfer-steps`, `--pour-steps`
- `--seed`
- 예시:

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/tools/pipeline_play.py \
  --reach-ckpt /path/to/reach.pt \
  --grasp-ckpt /path/to/grasp.pt \
  --transfer-ckpt /path/to/transfer.pt \
  --pour-ckpt /path/to/pour.pt \
  --headless
```

#### `collect_grasp_v1_warm_states.py`

- 목적: `grasp_v1`(rh56f1 / tesollo) 최신 LSTM 체크포인트를 rollout 해
  grasp 성공 시점 상태를 pour warmstart HDF5(`warm_states` 그룹)로 저장
- 산출물: `hdgp/data/grasp_warm_<robot>.hdf5` → 해당 로봇 pour `warm_state_paths` 입력
  - rh56f1 → `rh56f1/right/pour_v1` warm_state_bank
  - tesollo → `tesollo/right/pour_v5`·`pour_v6` warm_state_bank (동일 hdf5 공유)
- 동작: `play.py` 를 서브프로세스로 실행하며 `env.enable_warm_state_export=true`
  등 hydra override 로 export 를 켜고, HDF5 가 원자적으로 생성되면 즉시 종료
  (검증된 `collect_grasp_warm_states.py` 의 프로세스그룹/폴링 로직 재사용)
- 주요 옵션: `--robot {rh56f1,tesollo}`(필수), `--checkpoint`, `--out`,
  `--num_envs`, `--target_count`, `--keep_adr`
- 예시:

```bash
cd /home/user/rl_ws/hdgp
# rh56f1 (기본 ckpt: grasp-v1/lstm_test1 ep2000)
python3 scripts/tools/collect_grasp_v1_warm_states.py --robot rh56f1
# tesollo (기본 ckpt: grasp-v1/lstm_test7 ep12500)
python3 scripts/tools/collect_grasp_v1_warm_states.py --robot tesollo
```

### 학습 로그 / 분석 도구

#### `build_run_dataset.py`

- 목적: IsaacLab RSL-RL run 디렉터리에서 compact analysis dataset 생성
- 출력:
- `scalars.csv`: TensorBoard scalar 전체
- `summary.json`: tag 통계, config 포인터, 기본 집계
- `grasp_debug.csv`: stdout 로그에서 `GRASP_DEBUG` 파싱 시 생성
- 주요 옵션:
- `--log_dir`: run 디렉터리
- `--out_dir`: 출력 디렉터리, 기본값 `<log_dir>/analysis`
- `--last_n`: 최근 N개 window 통계
- `--stdout_log`: 표준출력 로그 경로
- 예시:

```bash
python3 /home/user/rl_ws/hdgp/scripts/tools/build_run_dataset.py \
  --log_dir /home/user/rl_ws/hdgp/log/rsl_rl/grasp2g/test3 \
  --out_dir /home/user/rl_ws/hdgp/log/rsl_rl/grasp2g/test3/analysis
```

#### `report_lift_kpi.py`

- 목적: TensorBoard event 파일에서 lift task KPI 요약
- 주요 옵션:
- `--log_dir`: `summaries/`를 포함한 run 디렉터리
- 예시:

```bash
python3 /home/user/rl_ws/hdgp/scripts/tools/report_lift_kpi.py \
  --log_dir /home/user/rl_ws/hdgp/log/rl_games/pipeline/left/5g_lift_left_v2/test2
```

#### `freeze_run_analysis.py`

- 목적: 이벤트 파일 + checkpoint를 고정 스냅샷(JSON)과 프롬프트로 변환하고, 선택적으로 Gemini 응답까지 저장
- 생성물:
- `snapshot.json`
- `prompt.txt`
- `manifest.json`
- `compare_to_base.json` (비교 모드)
- 주요 옵션:
- `--run`: 분석할 run 디렉터리, 여러 번 지정 가능
- `--base-run`: 비교 기준 run
- `--out-dir`: 출력 디렉터리
- `--tail-n`: tail 통계 window
- `--gemini`: 생성 프롬프트로 gemini CLI 호출
- `--gemini-cmd`: gemini 실행 파일명
- `--gemini-no-browser`: 브라우저 없는 인증
- `--force-refresh`: 캐시 무시 후 재생성
- 예시:

```bash
python3 /home/user/rl_ws/hdgp/scripts/tools/freeze_run_analysis.py \
  --run /home/user/rl_ws/hdgp/log/rl_games/pipeline/left/5g_lift_left_v2/test2 \
  --base-run /home/user/rl_ws/hdgp/log/rl_games/pipeline/left/5g_lift_left_v1/test1 \
  --out-dir /home/user/rl_ws/hdgp/log/analysis_llm
```

관련 상세 설명은 같은 폴더의 `freeze_run_analysis.md` 참고.

### Phase B/C/D 검증 리포트

이 섹션의 스크립트는 대부분 `5g_grasp_right-v2` 검증용이며, 짧은 rollout 뒤 `extras`와 observation shape를 점검해 JSON 리포트를 남기는 용도입니다.

#### `report_5g_grasp_right_spawn.py`

- 목적: `5g_grasp_right-v1` 또는 지정 task의 spawn 시점 palm/cup 위치와 상대 벡터 리포트
- 주요 옵션:
- `--task`
- `--agent`
- `--num_envs`
- `--env_id`: 보고할 env index
- `--output`: JSON 출력 경로
- 예시:

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/tools/report_5g_grasp_right_spawn.py \
  --task 5g_grasp_right-v2 \
  --num_envs 1 \
  --output /tmp/spawn_report.json \
  --headless
```

#### `report_phaseb3_metrics.py`

- 목적: 짧은 rollout으로 Phase B 지표 유효성 검증
- 주요 옵션:
- `--task`, `--agent`, `--num_envs`, `--steps`, `--output`
- 용도: observation dimension, tracked metric, feature bind 상태 확인
- 예시:

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/tools/report_phaseb3_metrics.py \
  --task 5g_grasp_right-v2 \
  --num_envs 8 \
  --steps 32 \
  --output /tmp/phaseb3.json \
  --headless
```

#### `report_phase32_success_metrics.py`

- 목적: Phase 3.2 success/contact metric 점검
- 주요 옵션:
- `--task`, `--agent`, `--num_envs`, `--steps`, `--output`
- 용도: success/contact/penalty KPI 비교용
- 예시:

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/tools/report_phase32_success_metrics.py \
  --task 5g_grasp_right-v2 \
  --num_envs 8 \
  --steps 32 \
  --output /tmp/phase32_metrics.json \
  --headless
```

#### `report_phasec1_contact_metrics.py`

- 목적: contact split metric 검증
- 주요 옵션:
- `--task`, `--agent`, `--num_envs`, `--steps`, `--output`
- 예시:

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/tools/report_phasec1_contact_metrics.py \
  --task 5g_grasp_right-v2 \
  --num_envs 8 \
  --steps 32 \
  --output /tmp/phasec1_contact.json \
  --headless
```

#### `report_phasec2_contact_reliability.py`

- 목적: contact reliability metric 검증
- 주요 옵션:
- `--task`, `--agent`, `--num_envs`, `--steps`, `--output`
- 예시:

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/tools/report_phasec2_contact_reliability.py \
  --task 5g_grasp_right-v2 \
  --num_envs 8 \
  --steps 32 \
  --output /tmp/phasec2_contact.json \
  --headless
```

#### `report_phasec4_object_feat.py`

- 목적: object point-cloud feature observation 차원/바인딩 검증
- 주요 옵션:
- `--task`, `--agent`, `--num_envs`, `--steps`, `--output`
- 용도: observation 차원 일치 여부, feature index/type 갱신 여부 확인
- 예시:

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/tools/report_phasec4_object_feat.py \
  --task 5g_grasp_right-v2 \
  --num_envs 32 \
  --steps 4 \
  --output /tmp/phasec4_object_feat.json \
  --headless
```

#### `verify_contact_filter_cpu.py`

- 목적: CPU 시뮬레이션에서 object/table contact filter 분리 동작 검증
- 시나리오:
- baseline
- object contact 강제
- table contact 강제
- 주요 옵션:
- `--task`, `--agent`, `--num_envs`, `--steps`, `--output`
- `--force_sim_cpu`: CPU 강제
- 예시:

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/tools/verify_contact_filter_cpu.py \
  --task 5g_grasp_right-v2 \
  --num_envs 1 \
  --steps 8 \
  --output /tmp/contact_filter_cpu.json \
  --headless
```

#### `run_phase_d_matrix.sh`

- 목적: Phase D integration matrix 8개 조합을 일괄 실행
- 내부적으로 `report_phaseb3_metrics.py` 호출
- 토글:
- `A = use_reference_replay`
- `B = use_tip_contact_gate`
- `C = use_object_pc_feat`
- 주요 환경변수:
- `ISAACLAB_ROOT`
- `TASK`
- `NUM_ENVS`
- `STEPS`
- `HEADLESS`
- `ENABLE_CUP`
- `ENABLE_PRIMITIVES`
- `OUT_DIR`
- 예시:

```bash
OUT_DIR=/tmp/phaseD_reports \
NUM_ENVS=256 \
STEPS=2048 \
bash /home/user/rl_ws/hdgp/scripts/tools/run_phase_d_matrix.sh
```

#### `summarize_phase_d_matrix.py`

- 목적: `A?B?C?.json` 리포트를 읽어 observation 일치 여부와 핵심 metric을 하나의 summary JSON으로 정리
- 주요 옵션:
- `--input_dir`
- `--output`
- 예시:

```bash
python3 /home/user/rl_ws/hdgp/scripts/tools/summarize_phase_d_matrix.py \
  --input_dir /tmp/phaseD_reports \
  --output /tmp/phaseD_reports/summary.json
```

#### `run_phase_d_kpi_matrix.sh`

- 목적: 대표 Phase D 조합 4개에 대해 KPI 매트릭스 실행
- 내부적으로 `report_phase32_success_metrics.py` 호출
- 주요 환경변수:
- `ISAACLAB_ROOT`
- `TASK`
- `NUM_ENVS`
- `STEPS`
- `HEADLESS`
- `ENABLE_CUP`
- `ENABLE_PRIMITIVES`
- `OUT_DIR`
- 예시:

```bash
OUT_DIR=/tmp/phaseD_kpi_reports \
NUM_ENVS=128 \
STEPS=512 \
bash /home/user/rl_ws/hdgp/scripts/tools/run_phase_d_kpi_matrix.sh
```

#### `summarize_phase_d_kpi.py`

- 목적: KPI matrix JSON을 읽어 case별 지표와 간단한 tuning note를 생성
- 출력 예:
- `ranked_cases`
- `tuning_notes`
- 주요 옵션:
- `--input_dir`
- `--output`
- 예시:

```bash
python3 /home/user/rl_ws/hdgp/scripts/tools/summarize_phase_d_kpi.py \
  --input_dir /tmp/phaseD_kpi_reports \
  --output /tmp/phaseD_kpi_reports/summary.json
```

### 자산 / 데이터 변환 / 패치 도구

#### `convert_demograsp_references.py`

- 목적: DemoGrasp reference `.pkl` 파일을 정규화된 palm(6) + pca(5) `.pt` 형식으로 변환
- 주요 옵션:
- `--src_dir`: 입력 `grasp_ref_*.pkl` 디렉터리
- `--dst_dir`: 출력 디렉터리
- `--fit_to_bounds`: PCA 시퀀스를 설정된 bound 안으로 매핑
- `--fit_margin`: bound 내부 점유 margin
- 예시:

```bash
python3 /home/user/rl_ws/hdgp/scripts/tools/convert_demograsp_references.py \
  --src_dir /home/user/rl_ws/hdgp/assets/demograsp_references \
  --dst_dir /home/user/rl_ws/hdgp/assets/demograsp_references/normalized \
  --fit_to_bounds
```

#### `find_pregrasp_pose.py`

- 목적: `5g_grasp_right`에서 컵 위치 기준 pregrasp `ARM_START_POSE` 7-DOF를 IK로 계산
- 원리:
- 컵 중심이 thumb base 와 middle fingertip 사이 중점에 오도록 palm_link 목표 위치를 정의
- `scipy.optimize.minimize(L-BFGS-B)`로 관절 한계 내 최적 자세 탐색
- 출력:
- 추천 `RIGHT_ARM_START_POSE`
- `grasp_right_env_cfg.py`에 넣을 joint 초기값 형식
- 주요 옵션:
- `--cup X Y Z`: 컵 root 위치
- `--pregrasp-y`: Y축 접근 오프셋
- `--restarts`: 다중 초기값 시도 횟수
- `--quiet`: 간략 출력
- 예시:

```bash
python3 /home/user/rl_ws/hdgp/scripts/tools/find_pregrasp_pose.py \
  --cup 0.40 -0.15 0.215 \
  --pregrasp-y -0.09 \
  --restarts 10
```

#### `generate_object_pc_feature_map.py`

- 목적: object code -> feature vector `.pt` 매핑 파일 생성
- 특징: SHA256 기반 deterministic feature, 기본 차원 64
- 주요 옵션:
- `--assets_dir`
- `--dim`
- `--output`
- 예시:

```bash
python3 /home/user/rl_ws/hdgp/scripts/tools/generate_object_pc_feature_map.py \
  --assets_dir /home/user/rl_ws/hdgp/assets \
  --dim 64 \
  --output /home/user/rl_ws/hdgp/assets/object_pc_features/openarm_right_object_code_feat_dim64.pt
```

#### `urdf_spec_report.py`

- 목적: URDF에서 geometry spec을 읽어 치수/상단 지름 등을 리포트
- 지원: mesh, box, cylinder, sphere
- 주요 옵션:
- `urdf`: 입력 URDF 경로
- `--from`: `visual` 또는 `collision`
- `--top-frac`: 상단 지름 추정에 사용할 상단 높이 비율
- 예시:

```bash
python3 /home/user/rl_ws/hdgp/scripts/tools/urdf_spec_report.py \
  /home/user/Desktop/dd.urdf \
  --from visual \
  --top-frac 0.02
```

#### `patch_swap_lr.py`

- 목적: IsaacLab의 `RslRlVecEnvWrapper` 등에 `swap_lr` 지원 패치 적용
- 사용 시점: left/right action/observation swap 실험을 위한 로컬 패치가 필요할 때
- 주요 옵션:
- `--isaaclab-dir`: IsaacLab 루트
- `--target`: 직접 패치할 `vecenv_wrapper.py` 경로
- `--dry-run`: 실제 쓰기 없이 diff 성격 검토
- `--force`: 이미 일부 패치가 있어도 강제 진행
- 예시:

```bash
python3 /home/user/rl_ws/hdgp/scripts/tools/patch_swap_lr.py \
  --isaaclab-dir /home/user/rl_ws/IsaacLab \
  --dry-run
```

## 참고 파일

#### `freeze_run_analysis.md`

- `freeze_run_analysis.py`의 상세 사용 예시와 배경 설명 문서

## 실무 팁

- 대부분의 IsaacLab 스크립트는 `--headless`를 붙여 실행하는 편이 안전합니다.
- Hydra override가 필요한 스크립트는 일반 옵션 뒤에 `env.xxx=value` 형태를 추가로 넘길 수 있습니다.
- JSON 리포트를 남기는 도구는 `--output /tmp/<name>.json`을 같이 주면 비교와 자동화가 쉬워집니다.
- 배치 스크립트 실행 전 `ISAACLAB_ROOT`와 출력 디렉터리(`OUT_DIR`)를 먼저 확인하는 것이 좋습니다.
