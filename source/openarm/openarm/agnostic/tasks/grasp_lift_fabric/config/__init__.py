"""gym 등록 — grasp_lift_fabric. 로봇당 id 4개(train/play × mlp/lstm).

★프로필 소스가 `grasp_s2r.robot_profiles` 다(08.27 전면 동기). 자매와 **같은 로봇
  자산·같은 palm 박스·같은 컵 스폰 중심**을 쓰므로 두 트랙의 차이가 손 제어 하나로
  좁혀진다 — 그게 이 트랙의 목적이다.

★gym id 규약: train.py 의 run_naming 정규식 `^(open-\\w+)_([rbl])_(.+)$` 에 걸려야
  로그가 `log/rl_games/<robot>/<side>/<task>/` 로 분리된다.
      open-sens_r_grasp_lift_fab → log/rl_games/open-sens/right/grasp-lift-fab/
★`-play` id 를 반드시 같이 등록해야 `play.py`·warm-state 수집이 동작한다.
"""

import dataclasses as _dc

import gymnasium as gym

from . import agents
from ..grasp_lift_fabric_env_cfg import (
    GraspLiftFabricGripperLeftEnvCfg,
    GraspLiftFabricTesolloRightEnvCfg,
)
from ...grasp_s2r import robot_profiles as _rp

_ENTRY = (
    "openarm.agnostic.tasks.grasp_lift_fabric.grasp_lift_fabric_env:GraspLiftFabricEnv"
)


def _play(cls):
    class _Play(cls):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 50
            self.scene.env_spacing = 2.5
    _Play.__name__ = cls.__name__ + "_PLAY"
    return _Play


_CFGS = {
    "sens_r": GraspLiftFabricTesolloRightEnvCfg,
    "sens_l": GraspLiftFabricGripperLeftEnvCfg,
}


def _profile_name_of(cls) -> str:
    """cfg 클래스가 어느 프로필로 조립되는지 — **인스턴스를 만들지 않고** 읽는다.

    ★`cls.profile_name` 은 안 된다. @configclass 는 클래스 속성을 dataclass 필드로
      옮기면서 클래스에서 제거하고, 기본값이 `default_factory` 로 가기도 한다.
    """
    f = cls.__dataclass_fields__["profile_name"]
    if f.default is not _dc.MISSING:
        return f.default
    return f.default_factory()


SKIPPED: dict[str, str] = {}
REGISTERED: list[str] = []

for _tag, _cls in _CFGS.items():
    # 이 태스크는 Fabrics 로만 돈다(폴백 금지). 자산이 없는 프로필을 등록하면
    # "존재하지만 띄우면 죽는" id 가 생기므로, 등록하지 않고 사유를 남긴다.
    _profile = _rp.PROFILES[_profile_name_of(_cls)]
    if _profile.fabric_class is None:
        SKIPPED[_tag] = (
            f"프로필 '{_profile.name}': Fabrics 자산 없음(fabric_class=None) — "
            "이 태스크는 Fabrics 전용이라 띄울 수 없다")
        continue

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
            id=f"open-{_tag}_grasp_lift_fab{_suffix}",
            entry_point=_ENTRY,
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:{_cfg_name}",
                "rl_games_cfg_entry_point": f"{agents.__name__}:{_agent}",
            },
        )
        REGISTERED.append(f"open-{_tag}_grasp_lift_fab{_suffix}")
