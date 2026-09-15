"""scripts/iker/loop.py — init, the user's actions and the generator request (design auto-loop §3, §6; no Isaac, no probe)."""

import importlib.util
import json

import pytest

from openarm.agnostic.modules.iker import loop_state as ls
from openarm.agnostic.modules.iker import loop_t2r
from openarm.agnostic.modules.iker import prompts, run_files
from openarm.agnostic.tasks.iker_shoe import layout

SPEC = importlib.util.spec_from_file_location("iker_loop_cli", layout.HDGP_ROOT / "scripts" / "iker" / "loop.py")
loop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loop)


def test_loop_t2r_file_names_agree_with_the_pipeline():
    """Pre-flight ruling A: loop_t2r's file-name constants (no torch) must not drift from pipeline's (needs torch)."""
    from openarm.agnostic.tasks.iker_shoe.t2r import pipeline

    assert (pipeline.PROMPT, pipeline.RESPONSE, pipeline.CODE, pipeline.VALIDATION, pipeline.FEEDBACK) == (
        loop_t2r.PROMPT, loop_t2r.RESPONSE, loop_t2r.CODE, loop_t2r.VALIDATION, loop_t2r.FEEDBACK,
    )


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
    state, paths = _mock_loop(tmp_path, "stage1_t2r", {"harvest": harvest})
    ls.save_state(paths.state_file, ls.apply(state, ls.Decision("pause", "harvest crashed", {"needs": "fix"}), "t"))
    monkeypatch.setattr(loop.lp, "label_pids", lambda label, proc_root=None: [])
    assert loop.main(["--state-dir", str(paths.state_dir), "--no-commit", "act", "resume"]) == 0
    resumed = ls.load_state(paths.state_file)
    assert resumed["status"] == "running" and "cleared" in resumed["runs"]["harvest"]
    assert _history(paths.state_dir)[-1]["result"] == {"cleared_runs": ["harvest"]}


def test_executors_mark_a_run_stopping_in_the_state_file_before_they_signal_it(tmp_path, monkeypatch, capsys):
    record = {"log": "/l/x.log", "started_s": 0.0}
    label = "iker_grasp_c00_t2r_i00"
    runs = {name: {**record, "label": lbl, "key": lbl}
            for name, lbl in (("stage1_t2r", label), ("stage2", "iker_vlm_c00_s1"))}
    state, paths = _mock_loop(tmp_path, "stage1_t2r", runs)
    seen = {}

    def fake_stop(label, force=False):
        seen[label] = ls.load_state(paths.state_file)["runs"]
        return [42]

    monkeypatch.setattr(loop, "stop", fake_stop)
    bank = loop.commit_bank(state, paths, ls.Decision("commit_bank", "bank", {"checkpoint": "/b/ep350.pth", "verified": 80, "iter": 0}))
    advance = loop.advance(state, paths, ls.Decision("advance", "target met", {"to": "observe_requery", "epoch": 500, "stop": "stage2"}))
    assert "stopping" in seen[label]["stage1_t2r"] and bank["stopped_runs"] == ["stage1_t2r"]
    assert "stopping" in seen["iker_vlm_c00_s1"]["stage2"] and advance["stopped_runs"] == ["stage2"]
    assert loop.advance(state, paths, ls.Decision("advance", "no stop needed", {"to": "vlm_target"})) == {}


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


def test_t2r_request_gives_the_generator_only_the_round_prompt_and_response(tmp_path):
    paths = loop.lp.LoopPaths.of(tmp_path, 0, "iker_shoe_c00_t2r")
    request = loop.t2r_request(paths, ls.Decision("t2r_generate", "iter 02", {"iter": 2}))
    folder = paths.state_dir / "stage1_t2r" / "iter_02"
    assert request["prompt"] == str(folder / "prompt.md") and request["response"] == str(folder / "response.md")
    assert request["brief"] == loop_t2r.generator_brief(folder / "prompt.md", folder / "response.md")


def test_act_t2r_generate_counts_the_request_and_writes_the_generator_record(tmp_path, monkeypatch, capsys):
    state, paths = _mock_loop(tmp_path, "stage1_t2r")
    folder = paths.t2r_iter_dir(0)
    folder.mkdir(parents=True)
    (folder / "prompt.md").write_text("p")
    monkeypatch.setattr(loop, "gpu_used_mib", lambda: 0)
    capsys.readouterr()
    assert loop.main(["--state-dir", str(paths.state_dir), "--no-commit", "act", "t2r_generate"]) == 0
    out = json.loads(capsys.readouterr().out)
    record = run_files.read_json(folder / "generator.json")
    assert record["agent_type"] == "iker-t2r-generator" and record["request"] == 1 and out["result"]["t2r"]["brief"] == record["brief"]
    assert ls.load_state(paths.state_file)["t2r"]["requests"] == 1


def test_a_failed_ingest_moves_the_attempt_aside(tmp_path, monkeypatch):
    state, paths = _mock_loop(tmp_path, "stage1_t2r")
    folder = paths.t2r_iter_dir(0)
    folder.mkdir(parents=True)
    (folder / "response.md").write_text("r")

    def fake_run(argv, **kwargs):
        (folder / "compute_reward.py").write_text("c")
        (folder / "validation.json").write_text(json.dumps({"ok": False, "errors": ["bad field"]}))
        return loop.subprocess.CompletedProcess(argv, 1, "", "")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    result = loop.ingest_reward(state, paths, ls.Decision("ingest_reward", "iter 00", {"iter": 0}))
    assert result == {"passed": False, "errors": ["bad field"], "attempt": 1}
    assert sorted(p.name for p in folder.iterdir()) == ["compute_reward_attempt_1.py", "response_attempt_1.md", "validation_attempt_1.json"]


def test_record_smoke_miss_moves_the_rounds_files_aside(tmp_path):
    state, paths = _mock_loop(tmp_path, "stage1_t2r")
    folder = paths.t2r_iter_dir(0)
    folder.mkdir(parents=True)
    (folder / "response.md").write_text("r")
    (folder / "compute_reward.py").write_text("c")
    (folder / "validation.json").write_text(json.dumps({"ok": True, "reward_sha256": "a" * 64}))
    (folder / "smoke.json").write_text(json.dumps({"passed": False}))
    result = loop.record_smoke_miss(state, paths, ls.Decision("record_smoke_miss", "iter 00", {"iter": 0, "digest": "a" * 64}))
    assert result == {"attempt": 1, "digest": "a" * 64}
    assert sorted(p.name for p in folder.iterdir()) == [
        "compute_reward_attempt_1.py", "response_attempt_1.md", "smoke_attempt_1.json", "validation_attempt_1.json",
    ]


def test_launch_t2r_trains_fresh_with_the_round_reward(tmp_path, monkeypatch):
    state, paths = _mock_loop(tmp_path, "stage1_t2r")
    folder = paths.t2r_iter_dir(0)
    folder.mkdir(parents=True)
    (folder / "compute_reward.py").write_text("c")
    seen = {}
    monkeypatch.setattr(loop, "launch", lambda label, argv, log, note: seen.update(label=label, argv=argv) or {"label": label, "log": str(log)})
    result = loop.launch_t2r(state, paths, ls.Decision("launch_t2r", "iter 00", {"key": "iker_grasp_c00_t2r_i00", "iter": 0}))
    argv = seen["argv"]
    assert seen["label"] == "iker_grasp_c00_t2r_i00" and argv[argv.index("--task") + 1] == "open-sens_l_iker_shoe_grasp_t2r"
    assert argv[argv.index("--max_iterations") + 1] == "500" and argv[argv.index("--num_envs") + 1] == "4096"
    assert f"env.reward_code_path='{folder / 'compute_reward.py'}'" in argv and "--checkpoint" not in argv
    assert folder / "compute_reward.py" in result["commit_paths"]


def test_write_t2r_prompt_renders_the_first_round_prompt(tmp_path):
    state, paths = _mock_loop(tmp_path, "stage1_t2r")
    result = loop.write_t2r_prompt(state, paths, ls.Decision("write_t2r_prompt", "iter 00", {"iter": 0}))
    text = (paths.t2r_iter_dir(0) / "prompt.md").read_text(encoding="utf-8")
    assert result["prompt"] == str(paths.t2r_iter_dir(0) / "prompt.md") and "Grasp the shoe with the left hand" in text


def test_end_round_records_the_run_folder_and_the_reward_digest(tmp_path, monkeypatch):
    label = "iker_grasp_c00_t2r_i00"
    record = {"label": label, "log": "/l/t.log", "started_s": 0.0, "key": label}
    state, paths = _mock_loop(tmp_path, "stage1_t2r", {"stage1_t2r": record})
    run_dir = paths.task_dir("stage1_t2r") / label
    run_dir.mkdir(parents=True)
    folder = paths.t2r_iter_dir(0)
    folder.mkdir(parents=True)
    (folder / "compute_reward.py").write_text("c")
    monkeypatch.setattr(loop.lp, "label_pids", lambda label, proc_root=None: [])
    params = {"iter": 0, "label": label, "end_epoch": 500, "ended": "round_epochs", "success_max": 0.01, "latched_max": 0.02}
    result = loop.end_round(state, paths, ls.Decision("end_round", "done", params))
    assert result["run_dir"] == str(run_dir) and len(result["reward_sha256"]) == 64 and result["stopped_runs"] == []
