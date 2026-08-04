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
| `spawn` | `X Y [Z]` | 물체를 (X, Y, Z)에 스폰 후 평가. Z 생략 시 기본값 사용. 완료 후 결과 출력. |
| `repeat` | 없음 | 마지막 `spawn` 위치에서 재실행 (객체 위치 유지). |
| `obj` | `N` | 물체 인덱스를 N으로 변경 시도. **단, 단일 env 세션에서는 무시되며 물체 0 고정** (다음 spawn 시 경고 메시지). |
| `sweep` | `XMIN XMAX NX YMIN YMAX NY REPEATS` | 문법상 7개 인자 필수이나 인터랙티브 모드에서는 미지원: `[ERR] sweep 은 인터랙티브 num_envs=1 세션에서 미지원` 메시지 출력. 배치 모드(--grid_x/y 등)로 별도 실행하세요. |
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

그리드/단일 모드는 둘 다 `main()`을 거치므로 `--out`을 지정했는지 여부와 무관하게 항상
`results.csv`+`summary.json`을 남긴다(`--out` 미지정 시 `log/eval_s2r/{robot}_{timestamp}/`에 자동 생성).
히트맵 PNG만 그리드 모드 전용.

- **그리드 모드**: `{out}/heatmap_success.png`, `{out}/heatmap_lifted.png` (히트맵 이미지 2장)
                 `{out}/results.csv` (셀별 성공률·통계, `per_obj_success`/`finger_contact_rates`는 JSON 인코딩된 문자열 컬럼)
                 `{out}/summary.json` (전체 메타데이터 + 셀별 집계)
                 터미널에 셀별 요약 표 출력
- **단일 모드**: 위와 동일하게 `{out}/results.csv`+`{out}/summary.json` 기록(셀 1개, 히트맵 없음) + 터미널 출력
- **인터랙티브**: 각 `spawn` 이후 터미널 출력(`fingers=[...]` 포함). `--out` 지정 시 세션 종료 시
                `{out}/interactive_history.json` 에 세션 전체 에피소드 이력을 **JSON**으로 저장
                (다른 모드의 CSV/JSON 이중 출력과 달리 인터랙티브는 JSON 단일 파일만 — 스펙 문서의
                CSV 언급과는 의도적으로 다르다: 세션 이력은 셀 집계가 없는 원시 `EpisodeResult` 리스트라
                CSV 스키마가 맞지 않는다)

---

## 주의사항

- **ADR 비활성화**: 그리드 고정 스폰과 충돌하므로 평가 중 자동 비활성화
- **pose_source**: SP1은 `state_frozen`, `live` 만 지원. `camera_frozen`은 SP2에서 구현.
- **finger_contact_rates**: `results.csv`/`summary.json`에 손가락별(엄지·검지·중지·약지·소지, T/I/M/R/P 순)
  접촉률이 셀당 원소별 평균으로 포함된다(표본 0이면 `null`).
- **체크포인트 경로**: play.py와 동일한 prefix-glob 지원 (`*_ep_*.pth` 자동 해석)
