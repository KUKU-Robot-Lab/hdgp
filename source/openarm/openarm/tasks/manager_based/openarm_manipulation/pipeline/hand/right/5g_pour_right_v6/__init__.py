# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# ── PourLstmBCAgent / PourChunkBCAgent 자동 등록 ──────────────────────────
# rl_games Runner 가 생성될 때마다 알고리즘을 algo_factory / player_factory에
# 자동 등록한다. train.py 수정 없이 yaml 의 algo.name 만으로 동작.
try:
    from rl_games.torch_runner import Runner as _Runner
    from rl_games.algos_torch import players as _rl_players
    from .lstm_bc_agent import PourLstmBCAgent as _Agent
    from .pour_chunk_bc_agent import PourChunkBCAgent as _ChunkAgent

    _orig_runner_init = _Runner.__init__

    def _patched_runner_init(self, *args, **kwargs):
        _orig_runner_init(self, *args, **kwargs)
        self.algo_factory.register_builder(
            "a2c_continuous_lstm_bc",
            lambda **kw: _Agent(**kw),
        )
        self.player_factory.register_builder(
            "a2c_continuous_lstm_bc",
            lambda **kw: _rl_players.PpoPlayerContinuous(**kw),
        )
        self.algo_factory.register_builder(
            "pour_chunk_bc",
            lambda **kw: _ChunkAgent(**kw),
        )
        self.player_factory.register_builder(
            "pour_chunk_bc",
            lambda **kw: _rl_players.PpoPlayerContinuous(**kw),
        )

    _Runner.__init__ = _patched_runner_init
except Exception:
    # rl_games 미설치 환경(unit test 등)에서는 조용히 무시
    pass
