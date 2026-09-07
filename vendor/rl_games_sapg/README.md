# rl_games_sapg — SimToolReal 포크 rl_games (SAPG 포함)

출처: `rl_ws/repo/simtoolreal/rl_games/rl_games` 를 그대로 복사.
용도: **SAPG 를 쓰는 태스크에서만** `PYTHONPATH` 앞에 붙여 설치본(rl_games 1.6.1)을 대신한다.

## 왜 통째로 두는가
09.07 함수 단위 분석에서 SAPG 배선이 키워드 없는 이름으로도 흩어져 있음이 확인됐다:
- `models.py` `extra_info_start_idx` — 블록 스칼라를 **정규화에서 제외**한다.
  이게 없으면 `coef_cond`/`extra_param` 의 `== ids` 비교가 정규화된 값과 원값을 비교해
  전부 거짓이 되고 `argmax` 가 조용히 0(블록 0)을 돌려준다.
- `datasets.py` `update_values_dict` 의 동적 `length` — 증강된 배치가 원래보다 크다.
둘 다 SAPG 키워드가 없어 기계적 분류로는 놓친다. 손으로 골라 옮기면 **조용히 다르게 학습**한다.

## 규약
- 이 트리는 **읽기 전용 사본**이 아니다 — 재생 경로 결함 수정이 들어간다(아래).
- 설치본에 없는 모듈은 없다. 포크에 없는 설치본 모듈은 `spatial_softmax`(CNN)·`smac_v2_env`
  둘뿐이고 우리 경로가 안 쓴다.
- 쓰지 않는 태스크는 영향받지 않는다(PYTHONPATH 를 붙일 때만 활성).
