"""계층 보상 — 자매 `grasp_sensor/rewards_tip_cyl.py` 의 **재수출**(단일 소스).

사용자 확정(08.26): 두 트랙의 리워드는 이름부터 수식까지 동일, 로봇 대상만 다르다.
복제는 드리프트를 못 막는다 — import 가 막는다. 로봇 종속 적응(미러 부호·폐쇄도·
표면 거리·코리더)은 전부 env 조립부에 있다.
"""

from __future__ import annotations

from ..grasp_sensor.rewards_tip_cyl import (  # noqa: F401  (재수출)
    compute_tip_cyl_rewards,
    smoothstep,
)

# 구 이름 호환 — env 와 계약이 이 이름을 쓴다. 본문은 자매 함수 그 자체다.
compute_stage_rewards = compute_tip_cyl_rewards
