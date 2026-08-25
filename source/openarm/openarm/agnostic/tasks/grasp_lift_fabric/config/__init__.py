"""gym 등록 — 로봇 프로필마다 id 4개(train/play × mlp/lstm).

로봇 추가 = `modules/robots.py` 프로필 1개 추가가 전부다. 여기는 손댈 필요가 없다.

★id 의 로봇 슬롯에 **자산 short** 를 넣는다:
      open-<short>_<side>_grasp_lift_fab
  train.py 가 `open-<robot>_<side>_<task>` → `log/rl_games/<robot>/<side>/<task-ver>/`
  로 푸는 기존 규약을 그대로 타므로, **train.py 수정 없이** 로봇 USD 별로 로그가 갈린다.
      open-bis_r_grasp_lift_fab  →  log/rl_games/open-bis/right/grasp-lift-fab/
"""

import gymnasium as gym

from openarm.agnostic.modules import agents as _ag
from openarm.agnostic.modules import robots as _rb

from . import agents
from ..grasp_lift_fabric_env_cfg import GraspLiftFabricEnvCfg

_ENTRY = (
    "openarm.agnostic.tasks.grasp_lift_fabric.grasp_lift_fabric_env:GraspLiftFabricEnv"
)

# Fabrics 자산이 없는 프로필은 등록하지 않는다. **조용히 빠뜨리지 않고** 사유를 남긴다.
SKIPPED: dict[str, str] = {}
REGISTERED: dict[str, str] = {}


def _cfg_class(profile, play: bool):
    name = f"GraspLiftFabric_{profile.name}{'_PLAY' if play else ''}_Cfg"

    # ★★클래스 속성으로 두면 안 된다. 베이스가 configclass(데이터클래스)라 상속된
    #   __init__ 이 **베이스 필드 기본값**("bis_right")을 인스턴스 속성으로 다시 쓴다 —
    #   서브클래스의 일반 클래스 속성은 조용히 가려진다. 실측: 좌팔 id
    #   `open-bis_l_...` 로 부팅했는데 profile=bis_right 가 로드됐다(08.22).
    #   __post_init__ 에서 **인스턴스 속성**으로 강제한 뒤 베이스 resolve 를 태운다.
    #   (기본 인자 바인딩 `_pn=profile.name` 은 루프 late-binding 함정 방지.)
    _play = play

    def __post_init__(self, _pn=profile.name, _play=_play):
        self.profile_name = _pn
        if _play:
            self.scene.num_envs = 50
        GraspLiftFabricEnvCfg.__post_init__(self)

    cls = type(name, (GraspLiftFabricEnvCfg,), {"__post_init__": __post_init__})
    globals()[name] = cls          # entry point 는 "모듈:속성" 문자열이라 노출 필요
    return cls


for _p in _rb.PROFILES.values():
    if _p.fabric_class is None:
        SKIPPED[_p.name] = (
            "Fabrics 자산 없음(fabric_class=None). "
            + " / ".join(_p.notes[-1:])
        )
        continue

    _train_cls = _cfg_class(_p, play=False)
    _play_cls = _cfg_class(_p, play=True)
    _side = "r" if _p.side == "r" else "l"
    _base = f"open-{_p.asset.short}_{_side}_grasp_lift_fab"
    REGISTERED[_p.name] = _base

    # ★`-paper` = DEXTRAH 논문 E.6 구성(critic 을 MLP 로). 레포 yaml 은 critic 에도
    #   LSTM 2048 을 두는데 논문은 "크리틱은 특권 정보를 다 보므로 시간 의존성이
    #   불필요" 라며 MLP 라고 명시한다. 어느 쪽이 나은지는 실험 문제라 **둘 다 돌릴 수
    #   있게** 태스크를 나눠 둔다(기본 태스크는 건드리지 않는다 — 진행 중인 런 보호).
    _PAPER_CFG = "rl_games_ppo_paper_cfg.yaml"
    for _suffix, _cls, _agent_yaml in (
        ("", _train_cls, _ag.resolve_agent_cfg(_p, use_lstm=False)),
        ("-play", _play_cls, _ag.resolve_agent_cfg(_p, use_lstm=False)),
        ("-lstm", _train_cls, _ag.resolve_agent_cfg(_p, use_lstm=True)),
        ("-play-lstm", _play_cls, _ag.resolve_agent_cfg(_p, use_lstm=True)),
        ("-paper", _train_cls, _PAPER_CFG),
        ("-play-paper", _play_cls, _PAPER_CFG),
    ):
        gym.register(
            id=f"{_base}{_suffix}",
            entry_point=_ENTRY,
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:{_cls.__name__}",
                "rl_games_cfg_entry_point": f"{agents.__name__}:{_agent_yaml}",
            },
        )
