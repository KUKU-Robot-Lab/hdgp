# eval_s2r — grasp_v1 sim2real 평가 하네스

물체를 사용자가 지정한 그리드/단일 위치에 스폰하여 학습된 grasp_v1 정책을 평가하고, 작업공간 성공률 히트맵·CSV·JSON을 산출한다.

**설계 및 아키텍처**: [docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp1-design.md](../../docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp1-design.md)

---

## 사용법

### 1. 배치 히트맵 (무인 그리드 스윕)

```bash
python scripts/eval_s2r/eval_sim2real.py \
  --robot left \
  --checkpoint <path/to/ep_20000.pth> \
  --pose_source state_frozen \
  --grid_x 0.21 0.33 --grid_nx 5 \
  --grid_y -0.16 0.02 --grid_ny 5 \
  --grid_repeats 8 \
  --episodes_per_env 3 \
  --out log/eval_s2r/left_lstm_test12
```

- `--grid_x MIN MAX` / `--grid_y MIN MAX`: 스폰 범위
- `--grid_nx N` / `--grid_ny N`: 격자점 개수 (nx × ny × grid_repeats 개 env 자동 생성)
- `--grid_repeats`: 각 셀에서 독립적으로 반복 실행 횟수 (기본값 8)
- `--episodes_per_env`: 각 env마다 순차 에피소드 개수 (기본값 3)
- `--pose_source {state_frozen,live,camera_frozen}`: 지각 모드 (기본값 state_frozen)
- `--out`: 결과 CSV/JSON/PNG 저장 경로

### 2. 단일 위치 시각 확인

```bash
python scripts/eval_s2r/eval_sim2real.py \
  --robot right \
  --checkpoint <path/to/ep_20000.pth> \
  --object_x 0.27 --object_y -0.10 \
  --render --real-time
```

- `--object_x X --object_y Y`: 스폰 좌표 (단일)
- `--object_z Z`: Z 좌표 지정 (생략 시 물체별 기본값)
- `--render`: GUI 단일-env 시각 재생 활성화
- `--real-time`: 실시간 페이싱 (기본 고속 재생)

### 3. 인터랙티브 상주 세션 (주 사용 방식)

```bash
python scripts/eval_s2r/eval_sim2real.py \
  --robot right --checkpoint <path/to/ep_20000.pth> \
  --interactive
```

GUI 상주 세션을 시작한다. 터미널에서 다음 명령어로 물체 소환·평가·리셋:

| 명령 | 인자 | 설명 |
|------|------|------|
| `spawn` | `X Y [Z]` | 물체를 (X, Y, Z)에 스폰 후 평가. Z 생략 시 기본값 사용. 대기 후 완료 후 결과 출력. |
| `repeat` | 없음 | 마지막 `spawn` 위치에서 재실행 (객체 위치 유지). |
| `obj` | `N` | 평가할 물체 인덱스를 N으로 변경 (다중 물체 에피소드의 경우). |
| `sweep` | 없음 | 그리드 스윕 미지원 안내: "배치 모드를 사용하세요 (--grid_x/y)" |
| `quit` | 없음 | 세션 종료. |

---

## GPU 스모크 체크리스트 (사용자 게이트)

### Phase 1: 기본 모드 검증

1. [ ] **right lstm_test3, 3×3 그리드 × repeats 8, episodes_per_env 2**
   - 히트맵 2장 생성 확인 (전·중심 셀)
   - 중심 셀 `episode_success_buf` ≈ play.py --eval_episodes 값(0.89) ±0.05 범위 확인
   - CSV/JSON 로그 생성 확인

2. [ ] **left lstm_test12 동일 절차**
   - 좌팔 정책 로드 및 평가 성공
   - 히트맵·CSV 생성

3. [ ] **--pose_source live vs state_frozen 중심 셀 비교**
   - live (현행 학습 동등) vs state_frozen (고정 추적) 성공률 차이 기록
   - freeze staleness 1차 수치화

4. [ ] **인터랙티브 (로컬 pc5090, GUI) 기본 흐름**
   - 기동 → `spawn 0.27 -0.10` → 결과 출력 → `spawn` 재실행 2회 → `quit` 정상 종료
   - GUI 마우스 상호작용(뷰포트 회전) 대기 중 가능 확인

5. [ ] **훅 무동작 회귀 (기존 play.py 검증)**
   - 기존 play.py 재생이 이전과 동일 동작 (랜덤 스폰, obs 정상)
   - 평가 세션 종료 후 play.py 실행 무영향 확인

### Phase 2: GPU 심화 검증

6. [ ] **episode_success_buf 프리스텝 스냅샷 타이밍 vs play.py --eval_episodes 교차검증**
   - 동일 체크포인트에서 eval_s2r과 play.py 성공률 ±0.05 범위 일치 확인

7. [ ] **체크포인트 params/env.yaml·agent.yaml 복원 실동작**
   - 로깅된 환경 설정 복원 검증
   - BC-pretrained 체크포인트인 경우 optimizer restore 이슈 주의 (현재 _patch_optimizer_restore 미이식)

8. [ ] **vars(args_cli) JSON 직렬화 가능 여부**
   - AppLauncher 주입 필드가 로그·전달에서 JSON 형식으로 손실 없이 처리되는지 확인

9. [ ] **rl_games get_action(is_deterministic) API 호환**
   - 결정적 action 추출 정상 동작 확인

10. [ ] **num_envs=1 truncation이 dones로 실제 전달 (무한루프 방지)**
    - 단일 env 에피소드 자동 종료 확인 (종료 조건 미전달 = 무한루프)

11. [ ] **--real-time 페이싱 정상**
    - 실시간 렌더링 시 wall-clock과 sim 시간 일치 확인

12. [ ] **GUI 창 닫기 mid-episode → invalid 처리·정상 종료**
    - 에피소드 중 창 닫기 → 에러 없이 graceful 종료

13. [ ] **파이프 stdin EOF → 정상 quit**
    - 인터랙티브 모드에서 입력 종료(EOF) 시 hang 없이 정상 종료

---

## 출력 형식

각 모드의 결과:

- **그리드 모드**: `{out}/heatmap_{prefix}.png` (히트맵 이미지)
                 `{out}/results.csv` (셀별 성공률·통계)
                 `{out}/results.json` (전체 메타데이터)
- **단일 모드**: 터미널 출력 (성공 여부, 상세 지표)
- **인터랙티브**: 각 `spawn` 이후 터미널 출력

---

## 주의사항

- **ADR 비활성화**: 그리드 고정 스폰과 충돌하므로 평가 중 자동 비활성화
- **pose_source**: SP1은 `state_frozen`, `live` 만 지원. `camera_frozen`은 SP2에서 구현.
- **체크포인트 경로**: play.py와 동일한 prefix-glob 지원 (`*_ep_*.pth` 자동 해석)
