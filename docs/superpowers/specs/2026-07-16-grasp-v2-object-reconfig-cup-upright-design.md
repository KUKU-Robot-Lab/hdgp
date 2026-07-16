# grasp_v2 물체군 재구성(실패 제거 + cup upright) → teacher 재학습 → vision distillation

- 날짜: 2026-07-16
- 대상: `tesollo/right/grasp_v2`, `tesollo/left/grasp_v2`, `openarm/distillation`
- 상태: 설계 승인됨. **본 spec의 구현 범위는 "코드 준비"까지.** 학습 실행은 사용자 요청 시까지 보류.

---

## 1. 목표 / 배경

vision student(distillation)를 최종적으로 얻는 것이 목적이다. distillation은 DAgger 모방 학습이라
teacher가 못 하는 건 student도 배울 수 없다(`dagger.py:173` teacher 체크포인트 필수). 따라서:

- **실패 물체 제거**: teacher onehot 슬롯을 유지한 채 distillation rollout에서 스폰만 막으면 재학습 없이도 가능.
  단, 이번엔 물체군 자체를 정리해 재학습하므로 물체군에서 제외한다.
- **cup 추가(upright)**: teacher가 cup을 잡을 줄 알아야 student가 배운다. 또한 cup을 넣으면 onehot 차원이
  바뀌어 기존 teacher 체크포인트가 로드 불가. **→ 새 물체군으로 teacher 재학습이 필수(경로 A).**

## 2. 확정된 결정

| 항목 | 결정 |
|---|---|
| 실패 물체 선정 | `episode_success_rate/{object}` TFEvents 로그에서 자동 추출 (클린 성공률 임계 < 0.3, 조정 가능) |
| 실패 물체 제거 범위 | **right·left 합집합** — 좌우 동일 물체군, 미러/onehot 차원 대칭 유지 |
| cup 대상 | `cup_big.usd` 1종, 이름 `"cup"`, **side 접근**(기존 인프라 재사용) |
| cup upright 정의 | **yaw(Z축)만 랜덤, X/Y 틸트 0** |
| 진행 경로 | A — 물체군 확정 → right·left teacher 재학습 → distillation |
| 학습 실행 | **보류.** 본 구현은 코드 변경·테스트까지. 실행은 사용자가 별도로 |

## 3. 현재 구조 (근거)

- 활성 물체군: `assets/visdex_objects/USD` 디렉토리 스캔 153종
  (`grasp_right_env_cfg.py:96` `_VISDEX_NAMES`, `_ACTIVE_OBJECT_NAMES = _VISDEX_NAMES`).
- 물체 배정: `env_id % N` 결정적 (`grasp_right_env.py:326`), onehot 로깅.
- obs/state 차원: `NUM_OBS_BASE + N`, `NUM_CRITIC_OBS_BASE + N` (`grasp_right_env_cfg.py:218-224`).
- spawn cfg: `MultiAssetSpawnerCfg(assets_cfg=[_primitive_usd_cfg(n) …], random_choice=False)`
  (`:129`). `_primitive_usd_cfg`는 `_ACTIVE_OBJECT_ROOT/name/name.usd` **단일 root** 가정 (`:110`).
- cup 인프라(이미 존재하나 활성군에서 빠짐):
  - `SIDE_APPROACH_OBJECT_NAMES = ("cup",)` (`grasp_right_preset.py:229`)
  - `side_object_idx` = active_object_names 중 side_approach 이름의 인덱스 (`grasp_right_env.py:329-335`)
  - `object_bbox.json`에 `'cup'` 키 존재.
- spawn 회전: `_sample_spawn_rotation(n)` — 전 물체 공통 X·Y 틸트(ADR 0→1) (`grasp_right_env.py:621`).
  리셋에서 `spawn_rot = self._sample_spawn_rotation(n)`으로 뽑아 `cup_root_state`에 zero velocity로 기록
  (`:1798, :1819, :1955`).

## 4. 설계 변경 (컴포넌트별)

### 4.1 물체군 재구성 — `grasp_{right,left}_env_cfg.py`

- **제외 상수 도입**: 모듈 수준 `_EXCLUDED_OBJECT_NAMES: frozenset[str] = frozenset()`
  (초기 비어 있음; 로그 추출 후 채운다). active 목록을 차집합으로 계산:
  ```
  _ACTIVE_OBJECT_NAMES = tuple(n for n in _VISDEX_NAMES if n not in _EXCLUDED_OBJECT_NAMES) + ("cup",)
  ```
  left도 동일 상수를 갖고, **합집합 목록을 좌우 동일하게 반영**한다.
- **cup 경로 해석 확장**: `_primitive_usd_cfg(name)`가 단일 root 대신 이름→경로 매핑을 쓴다.
  visdex 이름은 `_VISDEX_ROOT/name/name.usd`, `"cup"`은 `assets/cup/cup_big.usd`로 해석하는
  헬퍼(`_object_usd_path(name)`) 추가. 나머지 rigid/articulation 속성은 기존과 동일.
- onehot/obs 차원은 `len(_ACTIVE_OBJECT_NAMES)` 참조라 자동 반영. **명시 검증 테스트로 고정.**

### 4.2 cup upright 소환 — `grasp_{right,left}_env.py`

- `_sample_spawn_rotation`이 `obj_idx: torch.Tensor`(리셋 env의 물체 인덱스)를 받도록 시그니처 확장.
- cup(=`side_object_idx`에 속하는 env)에 대해 **yaw-only quat**로 덮어쓴다:
  Z축 회전각만 `[-π, π] * rot_f`에서 샘플, X/Y 성분 0. cup이 아닌 env는 기존 X·Y 틸트 유지.
- 호출부(`:1798`, `:1819`)에 `self.object_idx[env_ids]` 전달. `_spawn_rot_for_z`와 `spawn_rot`이
  동일 텐서를 공유하는 기존 계약(`:1837` 주석) 유지.
- (선택·YAGNI 보류) cup spawn z를 settled 높이로 강제하는 보강은 넣지 않는다 — zero velocity + yaw-only면
  수직 원통이 그대로 안착한다. 실측에서 흔들리면 그때 추가.

### 4.3 teacher 재학습 (실행 보류)

- 새 물체군으로 rl_games PPO 재학습(right, left). reset warm-state·ADR 인프라 재사용, 네트워크는
  obs 차원 변경으로 **fresh**. 체크포인트 warm-start는 입력 차원 불일치로 불가(정정된 사실).
- **본 spec 범위 밖의 실행.** 코드가 준비되면 사용자가 `train.sh`로 구동.

### 4.4 distillation 연결 (실행 보류)

- `dagger.py` config의 teacher `ckpt`/`cfg`를 새 teacher 산출물로 지정. `num_teacher_observations`가
  새 N과 일치해야 함(teacher cfg가 자동 반영). student obs는 물체 pose 미포함이라 물체군 변경 무관.
- **실행 보류.**

## 5. 테스트 (구현 범위)

- `test_object_set`: `_ACTIVE_OBJECT_NAMES`에 `"cup"` 포함, 제외 상수 차집합 동작, `observation_space ==
  NUM_OBS_BASE + len(active)` 정합. left/right 동일 목록 확인.
- `test_cup_usd_path`: `_object_usd_path("cup")`가 `assets/cup/cup_big.usd`, visdex 이름은 visdex root.
- `test_upright_spawn_rotation`: cup obj_idx에 대해 `_sample_spawn_rotation` 결과 quat의 x,y 성분 ≈ 0
  (yaw-only), 비-cup은 틸트 존재 가능.
- 기존 `test_approach_branch`: cup이 active set에 들어가며 `side_object_idx` 매핑 확인 (회귀).

## 6. 리스크

- **실패 목록은 현재 run 종료 후에만 확정.** run 진행 중이므로 `_EXCLUDED_OBJECT_NAMES`는 빈 상태로
  코드 준비만 하고, 추출 후 값 주입은 별도 단계(보류).
- **cup 단일 물체 → 배정 1/N로 낮음.** 학습 신호 부족 가능. 일단 균등 배정, 부족 시 cup 가중 배정 고려(YAGNI).
- **left 미러.** 좌우 동일 적용, left 착수 전 right 최신 커밋 동기화(메모리 교훈 `left-grasp-v2-mirror-port`).
- **cup 물리 안착.** yaw-only + zero velocity 가정. 렌더/실측에서 넘어지면 4.2 보강 옵션 적용.

## 7. 범위 밖(명시적 제외)

- teacher 재학습 실행, distillation 실행 — 사용자가 별도로.
- 실패 물체 실제 추출 실행 — run 종료 후.
- cup 가중 배정, cup spawn-z 보강 — 필요 입증 전까지 미도입.
