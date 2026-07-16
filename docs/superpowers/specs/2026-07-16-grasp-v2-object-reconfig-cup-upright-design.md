# grasp_v2 물체군 재구성(실패 제거 + cup upright) → teacher 재학습 → vision distillation

- 날짜: 2026-07-16
- 대상: `tesollo/right/grasp_v2`, `tesollo/left/grasp_v2`, `openarm/distillation`
- 상태: 설계 승인됨. **본 spec은 Phase 2 작업.** Phase 1(현재 teacher distillation 검증) 완료 후 착수.

---

## 0. 실행 순서 (2-phase) — 중요

물체군 재구성은 **현재 학습 중인 teacher와 호환되지 않는다.** distillation env는 teacher와 동일한
물체군을 써야 `num_teacher_observations`가 teacher 체크포인트와 일치한다. 지금 물체군을 바꾸면
현재 teacher를 distillation에 물릴 때 obs 차원 불일치로 깨진다.

- **Phase 1 (선행, 현재 우선순위)**: 물체군 **불변(153 visdex)**. 현재 server teacher 완주 →
  **비전 셋업(카메라 정면-사선 + depth) 적용** → `distill.sh --teacher <last.pth>`로 distillation 실행 →
  구조/파이프라인이 제대로 학습되는지 검증. 비전 셋업은 teacher(상태 기반)와 무관해 물체군을 안 건드린다.
  상세는 §A.
- **Phase 2 (후행)**: Phase 1 검증 통과 후, 아래 물체군 재구성(실패 제거 + cup upright)을 적용 →
  새 teacher 재학습 → 새 distillation. 상세는 §1~§7.

> ⚠️ Phase 1이 끝나기 전에 §4의 물체군 변경을 grasp_v2 cfg에 적용하면 현재 teacher의 distillation이
> 깨진다. server가 git pull로 코드를 받는 구조이므로, 학습 중 물체군 변경 커밋을 push하지 않는다.

---

## A. Phase 1 — distillation 비전 셋업 (카메라 정면-사선 + depth)

> 현재 teacher(153 visdex, 상태 기반)로 vision student를 증류해 **구조/파이프라인이 제대로 학습되는지
> 검증**하기 위한 셋업. teacher와 무관하므로 물체군을 건드리지 않는다. **학습 실행은 teacher 완료 후**
> (현재 미완료) 사용자가 별도 수행 — 본 spec은 코드/설정 준비까지.

### A.1 문제: top-down 접근에서 카메라 가림

현재 카메라(월드 고정 3인칭)는 베이스 근처 높이 0.97m에서 **65° 하향**(over-the-shoulder).
접근이 top-down으로 바뀌며 **위에서 내려오는 손·팔뚝이 접촉 직전 물체를 가린다** → 가장 중요한 순간에
depth에서 물체가 사라진다.

### A.2 카메라 재배치 (정면-사선)

- 월드 고정이라 `CAMERA_POS`/`CAMERA_ROT`(preset, env 로컬 좌표) 2개만 변경.
- 목표: **물체 앞쪽(+x, 물체 x=0.27 앞)에서 로봇 향해, 물체와 ≥0.3m(D435i 최소거리), 높이 ~0.4~0.55m,
  약 30~45° 하향**. 물체 윗면·앞면 + 내려오는 손의 옆모습이 보여 마지막 순간까지 가림 최소.
- DEXTRAH-G도 정면 3인칭 depth 뷰. 완전 수평(0°)은 윗면 depth 단서 손실·완전가림 위험 → 사선 채택.
- **기하 문제라 눈 검증이 정답**: `place_camera.py`로 후보 pose 잡고 `preview_camera.py`로 top-down
  동작 내내 가림 확인 후 확정. left는 y 부호 반전 미러(`grasp_left_preset.py`).

### A.3 student modality: RGB → depth

- 근거: depth가 sim2real 갭이 작다(RGB는 텍스처·조명 DR 의존, depth는 기하 기반). DEXTRAH 방식.
- **이미 준비된 것**: env가 `"img"`=depth(1ch, 유효밴드 마스킹) 제공(`env.py:1185-1200`);
  `DepthAug`+`depth_randomization_cfg`가 `img_aug_type=="depth"`에서 켜짐(`dagger.py:215`).
- **변경 필요(네트워크 코드)**: 활성 인코더가 `MonoEncoder(resnet, use_depth=False 하드코딩)`으로 3ch RGB
  전용(`a2c_mono_transformer.py:454,460`). ResNet은 1ch depth 불가. 방식:
  - (권장) 주석 처리된 **DEXTRAH `CustomCNN` depth 경로 부활**(`num_channel=1`, `:278,304`).
  - 대안: resnet 첫 conv 1ch 교체(pretrained 손실) / depth 3ch 타일링(특성 약화).
- 설정: `img_aug_type="depth"`, `aug_depth=True`(기본), student가 obs `"img"` 소비.
- **점검**: 인코더 하드코딩 `img_height=240`(`:452`)이 카메라 180과 어긋남 — resnet adaptive pool로
  흡수되는지 구현 중 확인.

### A.4 순서

카메라 정면-사선 + depth 인코더를 **둘 다 먼저 적용**해 최종 구성으로 검증(사용자 결정). teacher 완료 후
`distill.sh open-tesol_{r,l}_grasp_v2-distill <label> <teacher.pth>`로 실행.

### A.5 테스트

- `preview_camera.py` 시각 검증(가림 없음) — 실행형, 사용자 GPU.
- depth 인코더 유닛: 1ch 입력 forward 통과, 출력 shape 정합.
- `img_aug_type="depth"` 경로에서 `DepthAug` 적용·RgbAug 미적용 확인.

### A.6 Phase 1 리스크

- 카메라 pose는 렌더로만 확정 가능(GPU 필요) — 코드는 후보값 + preview 절차까지.
- depth 인코더 교체가 student 수렴 특성을 바꿈 — 첫 검증에서 aux(object_pos 회귀) 손실 추세로 확인.
- 좌우 미러 카메라 pose 동기화 필요.

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

### 4.3 teacher 재학습 (Phase 2, 실행 보류)

- 새 물체군으로 rl_games PPO 재학습(right, left). reset warm-state·ADR 인프라·하이퍼파라미터 재사용.
- **재학습 방식은 미정(사용자 보류)**: (a) fresh — 새 obs 차원으로 처음부터, 단순. (b) warm-start —
  입력층 weight 수동 재매핑(겹치는 물체 슬롯 + hidden/LSTM/value 복사, cup 슬롯 랜덤)으로 전이,
  수렴 빠르나 체크포인트 수술 코드 추가 필요. rl_games **자동** 로드는 입력 차원 불일치로 불가.
  Phase 1 검증·run 종료 후 결정한다.
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
