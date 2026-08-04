import math

import pytest

from scripts.eval_s2r.console import Command, SessionState, parse_command, apply_command
from scripts.eval_s2r.grid import GridSpec


class TestParse:
    def test_spawn_xy(self):
        c = parse_command("spawn 0.27 -0.10")
        assert c.kind == "spawn" and c.x == pytest.approx(0.27) and c.y == pytest.approx(-0.10)
        assert math.isnan(c.z)  # z 생략 → NaN(물체별 기본 높이)

    def test_spawn_xyz(self):
        c = parse_command("spawn 0.27 -0.10 0.30")
        assert c.z == pytest.approx(0.30)

    def test_repeat_obj_quit(self):
        assert parse_command("repeat").kind == "repeat"
        assert parse_command("obj 3").obj == 3
        assert parse_command("quit").kind == "quit"

    def test_sweep(self):
        c = parse_command("sweep 0.21 0.33 3 -0.16 0.02 2 4")
        assert c.kind == "sweep"
        assert c.grid == GridSpec(0.21, 0.33, 3, -0.16, 0.02, 2, 4)

    @pytest.mark.parametrize("line", [
        "", "bogus", "spawn", "spawn 0.1", "spawn a b",
        "obj", "obj 8", "obj -1",              # 물체 0~7
        "sweep 0.1 0.2 3",                     # 인자 부족
    ])
    def test_invalid_raises(self, line):
        with pytest.raises(ValueError):
            parse_command(line)


class TestApply:
    def test_spawn_updates_last(self):
        s0 = SessionState(last_spawn=None, obj_idx=None)
        s1, act = apply_command(s0, parse_command("spawn 0.3 0.0"))
        assert act["action"] == "spawn"
        assert s1.last_spawn[0] == pytest.approx(0.3)
        assert s0.last_spawn is None  # 불변성: 원본 상태 미변경

    def test_repeat_requires_last(self):
        s0 = SessionState(last_spawn=None, obj_idx=None)
        with pytest.raises(ValueError):
            apply_command(s0, parse_command("repeat"))

    def test_repeat_reuses_last(self):
        s0 = SessionState(last_spawn=(0.3, 0.0, float("nan")), obj_idx=None)
        s1, act = apply_command(s0, parse_command("repeat"))
        assert act["action"] == "spawn" and act["x"] == pytest.approx(0.3)

    def test_obj_sets_idx_noop(self):
        s1, act = apply_command(SessionState(None, None), parse_command("obj 5"))
        assert s1.obj_idx == 5 and act["action"] == "noop"

    def test_quit(self):
        _, act = apply_command(SessionState(None, None), parse_command("quit"))
        assert act["action"] == "quit"
