from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def test_v10_3_lstm_agent_matches_v11_stable_recurrent_layout() -> None:
    cfg = (_ROOT / "config" / "agents" / "rl_games_ppo_lstm_cfg.yaml").read_text(
        encoding="utf-8"
    )

    assert "name: 5g_grasp_right-v10-3-target26-lstm" in cfg
    assert "Actor 132D MLP [512, 512] -> LSTM 1024" in cfg
    assert "before_mlp: False" in cfg
    assert "units: [512, 512, 256, 128]" in cfg
    central_value = cfg.split("central_value_config:", 1)[1]
    assert "rnn:" not in central_value
