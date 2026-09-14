"""grasp_fj_t2r cfg — Track B leaf(`GraspFJTesolloRightShortEnvCfg`: short · thumb_1 용접)에
생성 보상 코드 경로만 더한다.

관측·액션·종료·성공 판정·커리큘럼·자산은 전부 B leaf 그대로다. B 의 보상 계수(`rw_*`, g(q))는
에피소드 상태(lifted 래치·최단거리 추적기)를 만드는 데만 쓰이고 **총보상에는 쓰이지 않는다** —
총보상은 `reward_code_path` 의 `compute_reward(ctx)` 가 낸다.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ..grasp_fj.grasp_fj_env_cfg import GraspFJTesolloRightShortEnvCfg


@configclass
class GraspFJT2RRightShortEnvCfg(GraspFJTesolloRightShortEnvCfg):
    """text2reward 생성 보상판. 보상 코드는 런처가 `env.reward_code_path=<절대경로>` 로 준다."""

    #: 생성 보상 코드(`compute_reward(ctx)`) 절대경로. 비면 **영 보상** — 부팅·무작위 롤아웃 전용.
    #:   env 가 `__init__` 에서 읽는다(런타임) — hydra override 가 `__post_init__` 에 구워지는 함정이 없다.
    reward_code_path: str = ""
