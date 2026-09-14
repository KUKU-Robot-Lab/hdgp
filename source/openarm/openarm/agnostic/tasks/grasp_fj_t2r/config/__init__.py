"""gym 등록 — grasp_fj_t2r(Track B + text2reward 생성 보상).

★gym id 규약: `open-short_r_grasp_fj_t2r-lstm-sapg` → `log/rl_games/open-short/right/grasp-fj-t2r/<label>/`.
  09.14 최종 목표판 `open-short_r_grasp_fj_t2r_reach-lstm-sapg` → `.../grasp-fj-t2r-reach/<label>/`.
★에이전트 yaml 은 Track B 것을 그대로 쓴다(관측·액션·네트워크가 같다).
★`-play` id 를 반드시 같이 등록해야 `play.py` 가 동작한다.
"""

import gymnasium as gym

from ...grasp_fj.config import agents
from ..grasp_fj_t2r_env_cfg import GraspFJT2RReachEnvCfg, GraspFJT2RRightShortEnvCfg

_ENTRY = "openarm.agnostic.tasks.grasp_fj_t2r.grasp_fj_t2r_env:GraspFJT2REnv"


def _play(cls):
    class _Play(cls):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 50
            self.scene.env_spacing = 2.5
            # 왜: 커리큘럼은 체크포인트에 없다 — 평가 tol 은 학습 종료 목표(tol_floor) 고정.
            self.tol_eval = self.tol_floor
    _Play.__name__ = cls.__name__ + "_PLAY"
    return _Play


_CFGS = {
    ("short_r", "grasp_fj_t2r"): GraspFJT2RRightShortEnvCfg,
    ("short_r", "grasp_fj_t2r_reach"): GraspFJT2RReachEnvCfg,
}

SKIPPED: dict[str, str] = {}
REGISTERED: list[str] = []

for (_tag, _task), _cls in _CFGS.items():
    _play_cls = _play(_cls)
    globals()[_cls.__name__] = _cls
    globals()[_play_cls.__name__] = _play_cls
    for _suffix, _cfg_name, _agent in (
        ("-lstm", _cls.__name__, "rl_games_ppo_lstm_cfg.yaml"),
        ("-play-lstm", _play_cls.__name__, "rl_games_ppo_lstm_cfg.yaml"),
        ("-lstm-sapg", _cls.__name__, "rl_games_ppo_lstm_sapg_cfg.yaml"),
        ("-play-lstm-sapg", _play_cls.__name__, "rl_games_ppo_lstm_sapg_cfg.yaml"),
    ):
        gym.register(
            id=f"open-{_tag}_{_task}{_suffix}",
            entry_point=_ENTRY,
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:{_cfg_name}",
                "rl_games_cfg_entry_point": f"{agents.__name__}:{_agent}",
            },
        )
        REGISTERED.append(f"open-{_tag}_{_task}{_suffix}")
