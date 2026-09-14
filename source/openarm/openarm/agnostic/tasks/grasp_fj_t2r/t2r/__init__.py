"""grasp_fj_t2r 전용 text2reward 포크 — 컨텍스트·로더·검증기·프롬프트.

`modules/t2r`(양팔 붓기)와 코드를 공유하지 않는다(09.10 fj 트랙 공유 금지). 규약은 같다:
생성 코드는 `compute_reward(ctx) -> (reward (N,), {name: (N,)})` 이고 `ctx` 가 유일한 입력이다.
"""
