"""에이전트(RL 망) 프리셋 해석 — 기존 규약을 그대로 유지한다.

규약(grasp_v1/v2/grasp_lift 공통):
    `config/agents/` 에 rl_games yaml 을 두고, gym id 의 `-lstm` 접미사로 고른다.
    이 규약을 지켜야 기존 도구(train.sh, play.py, parse_tfevents, record_test_snapshot)가
    수정 없이 동작한다.

이 모듈이 하는 일은 하나뿐이다: 프로필과 lstm 여부로 yaml **파일 이름**을 고른다.
프로필이 `agent_cfg_name` 을 선언했으면 그것이 우선한다 — 손 자유도가 크게 다른
로봇(RH56F1 12 vs Tesollo 20 vs 그리퍼 1)에서 망 크기를 바꾸고 싶을 때
프로필 한 줄로 끝내기 위한 통로다.
"""

from __future__ import annotations

MLP_CFG = "rl_games_ppo_cfg.yaml"
LSTM_CFG = "rl_games_ppo_lstm_cfg.yaml"


def resolve_agent_cfg(profile, *, use_lstm: bool = False) -> str:
    """이 프로필/모드에 쓸 rl_games yaml 파일 이름.

    ★기본은 MLP 다. obs 가 거의 Markov(관절+물체 상대 pose+접촉력+prev_action)이므로
      LSTM 1024 + seq16 의 비용을 정당화할 근거가 아직 없다. 부분관측(occlusion)이
      **실측**된 뒤에 켠다.
    """
    override = getattr(profile, "agent_cfg_name", None)
    if override:
        return override
    return LSTM_CFG if use_lstm else MLP_CFG
