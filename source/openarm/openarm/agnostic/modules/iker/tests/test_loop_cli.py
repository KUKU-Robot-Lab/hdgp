"""scripts/iker/loop.py — init, the user's actions and the generator request (design auto-loop §3, §6; no Isaac, no probe)."""

import importlib.util
import json

import pytest

from openarm.agnostic.modules.iker import loop_state as ls
from openarm.agnostic.modules.iker import prompts
from openarm.agnostic.tasks.iker_shoe import layout

SPEC = importlib.util.spec_from_file_location("iker_loop_cli", layout.HDGP_ROOT / "scripts" / "iker" / "loop.py")
loop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loop)


def _history(state_dir):
    return [json.loads(line) for line in (state_dir / "history.jsonl").read_text().splitlines()]


def test_init_writes_the_policy_overrides_once(tmp_path, capsys):
    state_dir = tmp_path / "mock_loop"
    policy = json.dumps({"stage2_epochs": 10, "labels": {"stage2": "iker_vlm_mock"}})
    assert loop.main(["--state-dir", str(state_dir), "init", "--track", "iker_shoe_c00_mock", "--phase", "vlm_target", "--policy", policy]) == 0
    state = ls.load_state(state_dir / "LOOP_STATE.json")
    assert (state["track"], state["phase"], state["policy"]["stage2_epochs"]) == ("iker_shoe_c00_mock", "vlm_target", 10)
    assert state["policy"]["labels"]["stage2"] == "iker_vlm_mock" and state["runs"] == {}
    assert _history(state_dir)[0]["action"] == "init"
    with pytest.raises(SystemExit, match="exists"):
        loop.main(["--state-dir", str(state_dir), "init"])


def test_resume_and_approve_are_recorded_without_a_probe(tmp_path, capsys):
    state_dir = tmp_path / "mock_loop"
    loop.main(["--state-dir", str(state_dir), "init", "--phase", "completion_review"])
    paused = ls.apply(ls.load_state(state_dir / "LOOP_STATE.json"), ls.Decision("pause", "review", {"needs": "watch"}), "t")
    ls.save_state(state_dir / "LOOP_STATE.json", paused)
    assert loop.main(["--state-dir", str(state_dir), "--no-commit", "act", "resume", "--policy", '{"success_target": 0.6}']) == 0
    resumed = ls.load_state(state_dir / "LOOP_STATE.json")
    assert resumed["status"] == "running" and resumed["policy"]["success_target"] == 0.6
    assert loop.main(["--state-dir", str(state_dir), "--no-commit", "act", "approve"]) == 0
    done = ls.load_state(state_dir / "LOOP_STATE.json")
    assert (done["phase"], done["status"]) == ("done", "done")
    assert [record["action"] for record in _history(state_dir)] == ["init", "resume", "approve"]


def test_vlm_request_gives_the_generator_only_its_files(tmp_path):
    paths = loop.lp.LoopPaths.of(tmp_path, 0, "iker_shoe_c00")
    target = loop.vlm_request(paths, ls.Decision("vlm_generate", "attempt 01", {"kind": "target", "attempt": 1}))
    assert target["response"] == str(paths.stage_dir / "attempt_01" / "response.md")
    assert target["images"] == {prompts.IMAGE_MARKER: str(paths.stage_dir / "snapshot.png")}
    requery = loop.vlm_request(paths, ls.Decision("vlm_generate", "requery", {"kind": "requery"}))
    assert requery["prompt"] == str(paths.observe_dir / "prompt.md")
    assert set(requery["images"]) == {prompts.IMAGE_MARKER, loop.loop_vlm.STAGE_IMAGE_MARKER}
    assert requery["brief"].startswith(f"Read the prompt file {paths.observe_dir / 'prompt.md'}")
