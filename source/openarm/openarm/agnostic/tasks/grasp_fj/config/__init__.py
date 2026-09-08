"""gym 등록 — grasp_fj. 로봇당 id 4개(train/play × mlp/lstm). grasp_kp 등록부와 같은 규약.

로봇 추가 = `robot_profiles.py` 프로필 + 여기 `_CFGS` 한 줄.

★gym id 규약: train.py 의 run_naming 정규식 `^(open-\\w+)_([rbl])_(.+)$` 에 걸려야
  로그가 `log/rl_games/<robot>/<side>/<task>/` 로 분리된다.
★`-play` id 를 반드시 같이 등록해야 `play.py`·warm-state 수집이 동작한다.
★Track B 는 Fabrics 자산이 필요 없다 — A 의 `fabric_class` 게이트를 두지 않는다.
  `SKIPPED` 는 도구(task_matrix) 규약상 이름만 유지한다(건너뛸 사유가 없으므로 항상 비어 있다).
"""

import gymnasium as gym

from . import agents
from ..grasp_fj_env_cfg import GraspFJTesolloRightEnvCfg, GraspFJTesolloRightShortEnvCfg

_ENTRY = "openarm.agnostic.tasks.grasp_fj.grasp_fj_env:GraspFJEnv"


def _play(cls):
    class _Play(cls):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 50
            self.scene.env_spacing = 2.5
            # 왜: 커리큘럼은 체크포인트에 없다 — play 가 0.06 에서 다시 굴리면 성공수가 비교 불가.
            #   평가 tol 은 학습 종료 목표(tol_floor) 고정. 다른 값은 hydra `env.tol_eval=`.
            self.tol_eval = self.tol_floor
    _Play.__name__ = cls.__name__ + "_PLAY"
    return _Play


_CFGS = {
    "sens_r": GraspFJTesolloRightEnvCfg,
    "short_r": GraspFJTesolloRightShortEnvCfg,
}

SKIPPED: dict[str, str] = {}
REGISTERED: list[str] = []

for _tag, _cls in _CFGS.items():
    _play_cls = _play(_cls)
    # config entry point 는 "모듈:속성" 문자열 — 동적 클래스를 모듈 네임스페이스에 노출.
    globals()[_cls.__name__] = _cls
    globals()[_play_cls.__name__] = _play_cls
    for _suffix, _cfg_name, _agent in (
        ("", _cls.__name__, "rl_games_ppo_cfg.yaml"),
        ("-play", _play_cls.__name__, "rl_games_ppo_cfg.yaml"),
        ("-lstm", _cls.__name__, "rl_games_ppo_lstm_cfg.yaml"),
        ("-play-lstm", _play_cls.__name__, "rl_games_ppo_lstm_cfg.yaml"),
        # ★SAPG 변형. **벤더 rl_games**(hdgp/vendor/rl_games_sapg)를 PYTHONPATH 앞에
        #   붙여야 돈다 — 설치본 1.6.1 에는 SAPG 가 없어 expl_* 가 조용히 무시되고
        #   `fixed_sigma: coef_cond` 도 해석 못 해 죽는다. 그래서 id 를 따로 둔다.
        ("-lstm-sapg", _cls.__name__, "rl_games_ppo_lstm_sapg_cfg.yaml"),
        ("-play-lstm-sapg", _play_cls.__name__, "rl_games_ppo_lstm_sapg_cfg.yaml"),
    ):
        gym.register(
            id=f"open-{_tag}_grasp_fj{_suffix}",
            entry_point=_ENTRY,
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:{_cfg_name}",
                "rl_games_cfg_entry_point": f"{agents.__name__}:{_agent}",
            },
        )
        REGISTERED.append(f"open-{_tag}_grasp_fj{_suffix}")
