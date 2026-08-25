"""gym 등록 — robot-agnostic grasp-sensor. 로봇당 id 4개(train/play × mlp/lstm).

로봇 추가 = robot_profiles.py 프로필 + 여기 _register() 한 줄.
"""

import dataclasses as _dc

import gymnasium as gym

from . import agents
from .. import robot_profiles as _rp
from ..grasp_sensor_env_cfg import (
    GraspSensorGripperLeftEnvCfg,
    GraspSensorTesolloRightEnvCfg,
)

_ENTRY = "openarm.agnostic.tasks.grasp_sensor.grasp_sensor_env:GraspSensorEnv"


def _play(cls):
    class _Play(cls):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 50
            self.scene.env_spacing = 2.5
    _Play.__name__ = cls.__name__ + "_PLAY"
    return _Play


# ★gym id 규약: train.py 의 run_naming 정규식 ^(open-\w+)_([rbl])_(.+)$ 에 걸려야
#   로그가 log/rl_games/<robot>/<side>/<task>/ 로 분리된다. 구 id
#   (open-agn_grasp_sensor_tesollo_r)는 두 번째 슬롯이 r/l/b 가 아니라 매칭에 실패해
#   로그가 pipeline/left/... 로 오분류됐다.
_CFGS = {
    "sens_r": GraspSensorTesolloRightEnvCfg,
    "sens_l": GraspSensorGripperLeftEnvCfg,
}

# ★이 태스크는 Fabrics 로만 돈다 — `_setup_fabrics` 가 fabric_class=None 이면
#   RuntimeError 로 멈춘다(폴백 금지). 그런데 등록은 되고 있어서 `open-sens_l_grasp_sensor`
#   4종이 "존재하지만 띄우면 죽는" 상태였다. 형제 태스크(grasp_lift_fabric/config)와 같은
#   SKIPPED 규약으로 **등록하지 않고 사유를 남긴다** — 조용히 빠뜨리지 않기 위해서다.
def _profile_name_of(cls) -> str:
    """cfg 클래스가 어느 프로필로 조립되는지 — **인스턴스를 만들지 않고** 읽는다.

    ★`cls.profile_name` 은 안 된다. @configclass 는 클래스 속성을 dataclass 필드로
      옮기면서 클래스에서 **제거**하고(isaaclab/utils/configclass.py
      `_process_mutable_types`: "things that became a dataclass field were removed
      from class members"), 그 과정에서 기본값도 `default_factory` 쪽으로 간다.
      그래서 `.default` 만 봐도 MISSING 이 나온다 — 둘 다 봐야 한다.
      인스턴스화는 답이 아니다: __post_init__ 이 로봇 자산까지 조립하므로
      "등록할지 말지"를 정하는 이 시점에 돌리면 순환이 된다.
    """
    f = cls.__dataclass_fields__["profile_name"]
    if f.default is not _dc.MISSING:
        return f.default
    return f.default_factory()


SKIPPED: dict[str, str] = {}
REGISTERED: list[str] = []

for _tag, _cls in _CFGS.items():
    # ★`_cls.profile_name` 으로는 못 읽는다 — @configclass 가 클래스 속성을
    #   dataclass 필드로 옮기면서 **클래스에서 제거**한다
    #   (isaaclab/utils/configclass.py `_process_mutable_types`:
    #    "things that became a dataclass field were removed from class members").
    #   인스턴스를 만들면 __post_init__ 이 로봇 자산까지 조립하므로 필드 기본값을 읽는다.
    _profile = _rp.PROFILES[_profile_name_of(_cls)]
    if _profile.fabric_class is None:
        SKIPPED[_tag] = (
            f"프로필 '{_profile.name}': Fabrics 자산 없음(fabric_class=None) — "
            "이 태스크는 Fabrics 전용이라 띄울 수 없다"
        )
        continue

    _play_cls = _play(_cls)
    # config entry point 는 "모듈:속성" 문자열 — 동적 클래스를 모듈 네임스페이스에 노출
    globals()[_cls.__name__] = _cls
    globals()[_play_cls.__name__] = _play_cls
    for _suffix, _cfg_name, _agent in (
        ("", _cls.__name__, "rl_games_ppo_cfg.yaml"),
        ("-play", _play_cls.__name__, "rl_games_ppo_cfg.yaml"),
        # ★-lstm = **논문 구조**(actor LSTM+MLP / critic MLP)
        ("-lstm", _cls.__name__, "rl_games_ppo_lstm_cfg.yaml"),
        ("-play-lstm", _play_cls.__name__, "rl_games_ppo_lstm_cfg.yaml"),
        # ★-lstm-cvrnn = DEXTRAH **배포 설정 원본**(critic 에도 LSTM 2048).
        #   배포본과 논문이 어긋나 둘 다 남긴다 — 태스크 id 접미사로 토글한다.
        ("-lstm-cvrnn", _cls.__name__, "rl_games_ppo_lstm_cvrnn_cfg.yaml"),
        ("-play-lstm-cvrnn", _play_cls.__name__, "rl_games_ppo_lstm_cvrnn_cfg.yaml"),
    ):
        gym.register(
            id=f"open-{_tag}_grasp_sensor{_suffix}",
            entry_point=_ENTRY,
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:{_cfg_name}",
                "rl_games_cfg_entry_point": f"{agents.__name__}:{_agent}",
            },
        )
        REGISTERED.append(f"open-{_tag}_grasp_sensor{_suffix}")
