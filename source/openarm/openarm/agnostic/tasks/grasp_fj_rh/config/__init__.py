"""gym 등록 — grasp_fj_rh(Track B · RH56F1). 로봇당 id 4개(train/play × mlp/lstm).

로봇 추가 = `grasp_s2r/robot_profiles.py` 프로필 + 여기 `_CFGS` 한 줄.

★gym id 규약: train.py 의 run_naming 정규식
  `^(open-[A-Za-z0-9]+)_([rbl])_(.+?)(?:-play|-lstm|-sapg|…)*$` 에 걸려야 로그가
  `log/rl_games/<robot>/<side>/<task>/` 로 분리된다.
  `open-rh_r_grasp_fj_rh-lstm` → `log/rl_games/open-rh/right/grasp-fj-rh/<label>/`.
★`-play` id 를 반드시 같이 등록해야 `play.py`·warm-state 수집이 동작한다.
★**SAPG id 는 두지 않는다** — 이 트랙은 로컬 단일 GPU 2048 env 전용이다(사용자 확정).
  SAPG 는 벤더 rl_games 포크와 블록 ≥2(env ÷ block_size)를 요구해 서버 전용이다.
★Track B 는 Fabrics 자산이 필요 없다 — A 의 `fabric_class` 게이트를 두지 않는다.
"""

import gymnasium as gym

from . import agents
from ..grasp_fj_rh_env_cfg import GraspFJRH56F1RightEnvCfg

_ENTRY = "openarm.agnostic.tasks.grasp_fj_rh.grasp_fj_rh_env:GraspFJRHEnv"


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
    "rh_r": GraspFJRH56F1RightEnvCfg,
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
    ):
        gym.register(
            id=f"open-{_tag}_grasp_fj_rh{_suffix}",
            entry_point=_ENTRY,
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:{_cfg_name}",
                "rl_games_cfg_entry_point": f"{agents.__name__}:{_agent}",
            },
        )
        REGISTERED.append(f"open-{_tag}_grasp_fj_rh{_suffix}")
