"""gym 등록 — grasp_kp. 로봇당 id 4개(train/play × mlp/lstm). grasp_s2r 등록부와 같은 규약.

로봇 추가 = `robot_profiles.py` 프로필 + 여기 `_CFGS` 한 줄.

★gym id 규약: train.py 의 run_naming 정규식 `^(open-\\w+)_([rbl])_(.+)$` 에 걸려야
  로그가 `log/rl_games/<robot>/<side>/<task>/` 로 분리된다. 두 번째 슬롯이 r/l/b 가
  아니면 매칭에 실패해 로그가 `pipeline/left/...` 로 오분류된다(선행 트랙 사고).
★`-play` id 를 반드시 같이 등록해야 `play.py`·warm-state 수집이 동작한다.
★Track A 는 Fabrics 로만 돈다 — fabric 자산 없는 프로필은 등록하지 않고 사유를 남긴다.
"""

import dataclasses as _dc

import gymnasium as gym

from . import agents
from .. import robot_profiles as _rp
from ..grasp_kp_env_cfg import GraspKPTesolloRightEnvCfg

_ENTRY = "openarm.agnostic.tasks.grasp_kp.grasp_kp_env:GraspKPEnv"


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
    "sens_r": GraspKPTesolloRightEnvCfg,
}


def _profile_name_of(cls) -> str:
    """cfg 클래스가 어느 프로필로 조립되는지 — **인스턴스를 만들지 않고** 읽는다.

    ★`cls.profile_name` 은 안 된다. @configclass 는 클래스 속성을 dataclass 필드로
      옮기면서 클래스에서 **제거**하고, 기본값이 `default_factory` 쪽으로 가기도 한다.
      인스턴스화는 답이 아니다(`__post_init__` 이 로봇 자산까지 조립한다).
    """
    f = cls.__dataclass_fields__["profile_name"]
    if f.default is not _dc.MISSING:
        return f.default
    return f.default_factory()


SKIPPED: dict[str, str] = {}
REGISTERED: list[str] = []

for _tag, _cls in _CFGS.items():
    # ★fabric 게이트(grasp_s2r 동일): 자산이 없는 프로필을 등록하면 "존재하지만 띄우면 죽는"
    #   id 가 생기므로 등록하지 않고 사유를 남긴다(record-loud).
    _profile = _rp.PROFILES[_profile_name_of(_cls)]
    if _profile.fabric_class is None:
        SKIPPED[_tag] = (
            f"프로필 '{_profile.name}': Fabrics 자산 없음(fabric_class=None) — "
            "Track A 는 Fabrics 전용이라 띄울 수 없다")
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
            id=f"open-{_tag}_grasp_kp{_suffix}",
            entry_point=_ENTRY,
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:{_cfg_name}",
                "rl_games_cfg_entry_point": f"{agents.__name__}:{_agent}",
            },
        )
        REGISTERED.append(f"open-{_tag}_grasp_kp{_suffix}")
