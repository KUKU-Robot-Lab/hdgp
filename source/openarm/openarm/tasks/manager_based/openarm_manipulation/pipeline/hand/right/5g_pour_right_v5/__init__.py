# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ── PourLstmBCAgent 자동 등록 ──────────────────────────────────────────────
# rl_games Runner 가 생성될 때마다 'a2c_continuous_lstm_bc' 알고리즘을 자동으로
# algo_factory 에 등록한다. train.py 수정 없이 yaml 의 algo.name 만으로 동작.
try:
    from rl_games.torch_runner import Runner as _Runner
    from rl_games.algos_torch import players as _rl_players
    from .pour_lstm_bc_agent import PourLstmBCAgent as _PourLstmBCAgent

    _orig_runner_init = _Runner.__init__

    def _patched_runner_init(self, *args, **kwargs):
        _orig_runner_init(self, *args, **kwargs)
        # 학습용 agent 등록 (algo_factory)
        self.algo_factory.register_builder(
            "a2c_continuous_lstm_bc",
            lambda **kw: _PourLstmBCAgent(**kw),
        )
        # 추론/play용 player 등록 (player_factory) — PpoPlayerContinuous가 LSTM 포함한 표준 player
        self.player_factory.register_builder(
            "a2c_continuous_lstm_bc",
            lambda **kw: _rl_players.PpoPlayerContinuous(**kw),
        )

    _Runner.__init__ = _patched_runner_init
except Exception:
    # rl_games 미설치 환경(unit test, import-only 환경)에서는 조용히 무시
    pass
