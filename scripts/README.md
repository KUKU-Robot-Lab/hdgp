# scripts/ 지도

역할별로 나눈다. **모든 디렉토리는 `scripts/` 바로 아래 한 단계**로 유지한다.

이 규칙은 취향이 아니다. 여기 있는 스크립트 대부분이 hdgp 루트를 이렇게 잡는다.

```python
_HDGP_ROOT = Path(__file__).resolve().parents[2]   # scripts/<dir>/x.py → hdgp/
```

한 단계라도 더 내리면 이 상수가 조용히 다른 디렉토리를 가리킨다. 새 하위 디렉토리를
만들고 싶으면 깊이 상수를 함께 고쳐야 한다.

| 디렉토리 | 내용 |
|---|---|
| `tools/` | 상시 사용하는 핵심 도구. `parse_tfevents.py`, `record_test_snapshot.py`, `openarm_fk.py`, `list_envs.py`, `confirm_env.py`, `freeze_run_analysis.py` |
| `reinforcement_learning/` | RL 학습·재생 진입점 (`rl_games/train.py`, `rl_games/play.py`) |
| `reinforcement_learning/probes/` | 일회성 확인 스크립트 (07.06 live-policy-fluid 평가의 `p1`~`p4`, `verify_fluid_*`, `replay_pour_fluid.py`) |
| `imitation_learning/` | BC / robomimic 학습 |
| `analysis/` | TFEvents 파싱 및 실험 분석 |
| `warm_states/` | grasp 성공 상태 수집 → pour warm start (`collect_*`, `save_*_terminal_states`, `balance_*`, `migrate_*`) |
| `reports/` | phase별 KPI·지표 리포트 (`report_*`, `summarize_phase_d_*`, `run_phase_d_*.sh`) |
| `assets_tools/` | USD/URDF/메시 생성·변환 (`convert_*`, `generate_*`, `export_scene.py`, `make_cm_cup.py`, `urdf_spec_report.py`, `patch_swap_lr.py`) |
| `pca/` | 손 시너지(eigengrasp) 산출·이식 (`compute_rh56f1_grasp_pca.py`, `retarget_allegro_pca_to_tesollo.py`, `render_synergy_sweep.py`) |
| `probes/` | 일회성 검증·탐침 (`probe_palm_orientation.py`, `find_pregrasp_pose.py`, `verify_contact_filter_cpu.py`, `check_reset_symmetry.py`, `test_finger_joints.py`, `pipeline_play.py`) |
| `datasets/` | 데이터셋 빌드 (`build_run_dataset.py`, `build_pre_pour_bc_dataset.py`) |
| `r2s_autotune/` | Real2Sim actuator autotune (자체 README 참조) |

## 주의

- `probes/test_finger_joints.py`는 이름이 `test_`로 시작하지만 pytest 테스트가 아니라
  Isaac Sim 스크립트다. `scripts/`에서 pytest를 돌리면 수집되어 `isaaclab` import로 실패한다.
  테스트는 각 디렉토리 안에서 돌린다. 예: `cd r2s_autotune && pytest tests/`.
- `probes/eval_grasp2g_policy.py:49`의 `repo_root = parents[1]`은 `scripts/`를 가리켜
  `scripts/source`를 import path에 넣는다. 이 스크립트는 이전부터 깨져 있다.
- `tools/README.MD`는 도구별 사용법을 담는다.
