"""scripts/iker/loop.py — init, the user's actions and the generator request (design auto-loop §3, §6; no Isaac, no probe)."""

import importlib.util
import json

import pytest

from openarm.agnostic.modules.iker import loop_state as ls
from openarm.agnostic.modules.iker import prompts, run_files
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


def _mock_loop(tmp_path, phase, runs=None, track="iker_shoe_c00_mock"):
    state_dir = tmp_path / "mock_loop"
    loop.main(["--state-dir", str(state_dir), "init", "--track", track, "--phase", phase])
    paths = loop.lp.LoopPaths.of(tmp_path, 0, track, state_dir)
    state = {**ls.load_state(paths.state_file), "runs": runs or {}}
    ls.save_state(paths.state_file, state)
    return state, paths


def test_resume_clears_a_dead_run_that_reads_as_crashed(tmp_path, monkeypatch, capsys):
    log = tmp_path / "harvest_ep350.log"
    log.write_text("HARVEST seed 0 envs 512\n")
    harvest = {"label": "iker_shoe_c00_mock_harvest", "log": str(log), "started_s": 0.0, "key": "/b/ep350.pth", "epoch": 350}
    state, paths = _mock_loop(tmp_path, "harvest", {"harvest": harvest})
    ls.save_state(paths.state_file, ls.apply(state, ls.Decision("pause", "harvest crashed", {"needs": "fix"}), "t"))
    monkeypatch.setattr(loop.lp, "label_pids", lambda label, proc_root=None: [])
    assert loop.main(["--state-dir", str(paths.state_dir), "--no-commit", "act", "resume"]) == 0
    resumed = ls.load_state(paths.state_file)
    assert resumed["status"] == "running" and "cleared" in resumed["runs"]["harvest"]
    assert _history(paths.state_dir)[-1]["result"] == {"cleared_runs": ["harvest"]}


def test_executors_mark_a_run_stopping_in_the_state_file_before_they_signal_it(tmp_path, monkeypatch, capsys):
    record = {"log": "/l/x.log", "started_s": 0.0}
    runs = {name: {**record, "label": label, "key": label}
            for name, label in (("stage1_a", "iker_grasp_c00_a"), ("stage1_b", "iker_grasp_c00_b"), ("stage2", "iker_vlm_c00_s1"))}
    state, paths = _mock_loop(tmp_path, "harvest", runs)
    seen = {}

    def fake_stop(label, force=False):
        seen[label] = ls.load_state(paths.state_file)["runs"]
        return [42]

    monkeypatch.setattr(loop, "stop", fake_stop)
    monkeypatch.setattr(loop, "launch", lambda label, argv, log, note: {"label": label, "log": str(log)})
    bank = loop.commit_bank(state, paths, ls.Decision("commit_bank", "bank", {"checkpoint": "/b/ep350.pth", "verified": 80}))
    b = loop.launch_b(state, paths, ls.Decision("launch_b", "b", {"key": "/a/ep250.pth", "checkpoint": "/a/ep250.pth"}))
    advance = loop.advance(state, paths, ls.Decision("advance", "target met", {"to": "observe_requery", "epoch": 500, "stop": "stage2"}))
    assert "stopping" in seen["iker_grasp_c00_b"]["stage1_b"] and bank["stopped_runs"] == ["stage1_b"]
    assert "stopping" in seen["iker_grasp_c00_a"]["stage1_a"] and b["stopped_runs"] == ["stage1_a"]
    assert "stopping" in seen["iker_vlm_c00_s1"]["stage2"] and advance["stopped_runs"] == ["stage2"]
    assert loop.advance(state, paths, ls.Decision("advance", "b booted", {"to": "harvest"})) == {}


def test_act_vlm_generate_counts_the_request_and_writes_the_generator_record(tmp_path, monkeypatch, capsys):
    state, paths = _mock_loop(tmp_path, "vlm_target")
    paths.stage_dir.mkdir(parents=True)
    for name in ("prompt.md", "snapshot.png", "keypoints.json"):
        (paths.stage_dir / name).write_text("x")
    monkeypatch.setattr(loop, "gpu_used_mib", lambda: 0)
    capsys.readouterr()
    assert loop.main(["--state-dir", str(paths.state_dir), "--no-commit", "act", "vlm_generate"]) == 0
    out = json.loads(capsys.readouterr().out)
    generator = run_files.read_json(paths.stage_dir / "attempt_00" / "generator.json")
    assert generator["agent_type"] == "iker-vlm-generator" and generator["tools"] == ["Read", "Write"] and generator["request"] == 1
    assert out["result"]["vlm"]["brief"] == generator["brief"] and generator["response"] == str(paths.stage_dir / "attempt_00" / "response.md")
    assert ls.load_state(paths.state_file)["vlm_requests"] == {"target_00": 1}


def test_vlm_request_gives_the_generator_only_its_files(tmp_path):
    paths = loop.lp.LoopPaths.of(tmp_path, 0, "iker_shoe_c00")
    target = loop.vlm_request(paths, ls.Decision("vlm_generate", "attempt 01", {"kind": "target", "attempt": 1}))
    assert target["response"] == str(paths.stage_dir / "attempt_01" / "response.md")
    assert target["images"] == {prompts.IMAGE_MARKER: str(paths.stage_dir / "snapshot.png")}
    requery = loop.vlm_request(paths, ls.Decision("vlm_generate", "requery", {"kind": "requery"}))
    assert requery["prompt"] == str(paths.observe_dir / "prompt.md")
    assert set(requery["images"]) == {prompts.IMAGE_MARKER, loop.loop_vlm.STAGE_IMAGE_MARKER}
    assert requery["brief"].startswith(f"Read the prompt file {paths.observe_dir / 'prompt.md'}")
