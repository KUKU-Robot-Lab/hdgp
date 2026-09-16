# IKER 트랙 — 작업·보고 규칙

> 도메인 규칙은 hdgp 루트 `CLAUDE.md` 를 따른다. 이 파일은 **이 트랙에서의 일하는 방식**만 담는다.

## 1. 보고는 끝에 한 번

**작업 중에는 말하지 않는다. 도구만 호출한다.**

금지:
- `필요(비공개 점검)` · `필요한 것 정리` 같은 사전 정리 블록 출력 — **"비공개로 정리하라"는 지시는
  머릿속에서만 하라는 뜻이다.** 제목에 "비공개"를 붙여 출력하는 것은 위반이다. 다음에 무엇을 할지
  따지는 과정은 한 글자도 화면에 내보내지 않는다
- 단계마다 "무엇을 왜 고쳤는지" 상세 서술
- 도구 호출 전후의 예고·중계 (`이제 …를 확인합니다`, `…를 동시에 실행합니다`)
- 이미 말한 상태를 다시 요약하는 중간 보고

허용(짧게 한 줄):
- 승인이 필요한 시점 — 되돌리기 어려운 작업, 외부 영향, 실기 동작
- 막혀서 사용자 판단이 있어야 진행되는 시점
- 오래 걸리는 작업을 시작했다는 사실 한 줄

## 2. 최종 요약 한 번

모든 수정이 끝난 뒤 **한 번만** 보고한다. 담을 것:

1. **무엇이 바뀌었나** — 파일·커밋 해시, 한 줄씩
2. **판정과 근거** — 수치로. 추측과 확인된 사실을 섞지 않는다
3. **실패·미완** — 통과 못 한 것은 출력과 함께 그대로 적는다
4. **남은 결정** — 사용자가 정해야 할 것만

분량은 내용에 맞춘다. 표가 목록보다 짧으면 표를 쓴다.

## 3. 학습 기동

worktree 에서는 `train.sh`(→`isaaclab.sh`)를 쓰지 않는다. `isaaclab.sh` 가 PYTHONPATH 맨 앞에
`IsaacLab/../hdgp/source/openarm` 를 꽂아 `train.py` 의 로컬-패키지 가드가 죽인다.
인터프리터를 직접 부른다 — `RUN_LABEL`/`NOTE` 로 `test_history.md` 스냅샷은 그대로 남는다.

```bash
RUN_LABEL=<라벨> NOTE="<가설>" PYTHONUNBUFFERED=1 PYTHONPATH=source/openarm \
OMNI_KIT_ACCEPT_EULA=YES TERM=xterm \
/home/user/rl_ws/IsaacLab/_isaac_sim/python.sh \
  scripts/reinforcement_learning/rl_games/train.py \
  --task <task_id> --num_envs 4096 --headless env.<키>=<값> \
  > log/<트랙>/<라벨>.log 2>&1
```

학습 종료는 **PID 로만**. `pkill`·`killall` 금지(동시 세션을 죽인다).

## 4. 진행 감시

학습 로그 감시 필터는 **마일스톤과 실패 신호만** 잡는다. `fps step` 처럼 매 epoch 나오는 줄을
넣으면 알림이 수백 번 울린다. 실패 쪽은 넓게(`Traceback|RuntimeError|AssertionError|Killed|out of memory`),
진행 쪽은 좁게(50 epoch 배수 등).

## 5. reward-audit 안 씀

IKER 트랙은 t2r 생성 보상을 쓰므로 `reward-audit` 를 돌리지 않는다(사용자 결정 09.15).
검증은 t2r 검증기 + 부팅 스모크 + 학습 지표로 한다.
