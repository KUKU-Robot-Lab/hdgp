"""loop_state — the IKER auto-loop state machine (design 2026-09-14-iker-auto-loop §3, §4, §9; no Isaac)."""

from dataclasses import replace

import pytest

from openarm.agnostic.modules.iker import loop_state as ls

NOW = "2026-09-14T20:00:00"
RUNNING = ls.RunStatus(started=True, alive=True)
ENDED = ls.RunStatus(started=True, finished=True)
DONE_PASS = ls.RunStatus(started=True, finished=True, passed=True, marker="X passed True")
DONE_FAIL = ls.RunStatus(started=True, finished=True, passed=False, marker="X passed False")


def _state(phase="stage1_a", **changes):
    return {**ls.new_state(NOW, phase=phase), **changes}


def _flat(value, upto, start=1):
    return tuple((epoch, value) for epoch in range(start, upto + 1))


def test_new_state_merges_policy_overrides_and_rejects_unknown_keys():
    state = ls.new_state(NOW, policy={"stage2_epochs": 10, "labels": {"stage2": "iker_vlm_mock"}})
    assert state["policy"]["stage2_epochs"] == 10 and state["policy"]["eval_every"] == 250
    assert state["policy"]["labels"] == {"stage1_a": "iker_grasp_c00_a", "stage1_b": "iker_grasp_c00_b", "stage2": "iker_vlm_mock"}
    assert ls.DEFAULT_POLICY["labels"]["stage2"] == "iker_vlm_c00_s1"
    with pytest.raises(ValueError, match="unknown policy keys"):
        ls.new_state(NOW, policy={"gate_epochs": 300})


def test_state_round_trips_and_validation_rejects_inconsistent_states(tmp_path):
    state = _state()
    path = ls.save_state(tmp_path / "loop" / "LOOP_STATE.json", state)
    assert ls.load_state(path) == state
    with pytest.raises(ValueError, match="awaiting"):
        ls.validate_state({**state, "status": "awaiting"})
    with pytest.raises(ValueError, match="phase"):
        ls.validate_state({**state, "phase": "stage3"})
    with pytest.raises(ValueError, match="unknown runs"):
        ls.validate_state({**state, "runs": {"stage1_c": {}}})


def test_bin_mean_takes_the_epochs_after_end_minus_width_through_end():
    points = ((190, 1.0), (191, 2.0), (200, 4.0), (201, 8.0))
    assert ls.bin_mean(points, 200, 10) == 3.0
    assert ls.bin_mean(points, 180, 10) is None
    assert ls.last_epoch(points) == 201 and ls.last_epoch(()) == 0


def test_first_calibration_checkpoint_is_the_earliest_saved_epoch_over_the_threshold():
    latched = _flat(0.05, 100) + _flat(0.12, 150, start=101)
    checkpoints = {50: "a", 100: "b", 150: "c"}
    assert ls.first_calibration_checkpoint(latched, checkpoints, 0.10, 10) == 150
    assert ls.first_calibration_checkpoint(latched, {50: "a", 100: "b"}, 0.10, 10) is None


def test_pick_observe_env_best_eval_and_eval_epochs():
    rows = [{"env": 0, "success": False, "end_dist": 0.01}, {"env": 1, "success": True, "end_dist": 0.04},
            {"env": 2, "success": True, "end_dist": 0.02}, {"env": 3, "success": True, "end_dist": 0.03},
            {"env": 4, "success": True, "end_dist": 0.01}]
    assert ls.pick_observe_env(rows) == 2
    assert ls.pick_observe_env(rows[:1]) is None
    evals = {"250": {"success": 0.4}, "500": {"success": 0.6}, "750": {"success": 0.6}}
    assert ls.best_eval(evals) == (500, {"success": 0.6})
    assert ls.best_eval({}) is None
    assert ls.eval_epochs(ls.DEFAULT_POLICY) == (250, 500, 750)


def test_a_paused_loop_only_waits():
    paused = ls.apply(_state(), ls.Decision("pause", "why", {"needs": "what"}), NOW)
    assert paused["status"] == "awaiting" and paused["awaiting"] == {"reason": "why", "needs": "what", "phase": "stage1_a"}
    assert ls.decide(paused, ls.Probe()).action == "wait"
    with pytest.raises(ValueError, match="not recorded"):
        ls.apply(_state(), ls.Decision("wait", "nothing"), NOW)


def test_stage1_a_waits_for_the_gate_epoch_then_records_the_last_bin():
    state = _state()
    assert ls.decide(state, ls.Probe(latched=_flat(0.05, 199), runs={"stage1_a": RUNNING})).action == "wait"
    passing = ls.decide(state, ls.Probe(latched=_flat(0.0, 190) + _flat(0.03, 205, start=191), runs={"stage1_a": RUNNING}))
    assert passing.action == "record_gate" and passing.params["passed"] and passing.params["latched"] == pytest.approx(0.03)
    failing = ls.decide(state, ls.Probe(latched=_flat(0.019, 200), runs={"stage1_a": RUNNING}))
    assert failing.action == "record_gate" and not failing.params["passed"]
    paused = ls.apply(state, failing, NOW)
    assert paused["status"] == "awaiting" and paused["gate1"]["passed"] is False


def test_stage1_a_moves_to_calibrate_at_the_first_checkpoint_over_ten_percent():
    state = _state(gate1={"epoch": 200, "latched": 0.08, "passed": True})
    checkpoints = {200: "/a/ep200.pth", 250: "/a/ep250.pth"}
    latched = _flat(0.08, 240) + _flat(0.11, 255, start=241)
    decision = ls.decide(state, ls.Probe(latched=latched, checkpoints=checkpoints, runs={"stage1_a": RUNNING}))
    assert decision.action == "advance" and decision.params == {"to": "calibrate", "epoch": 250, "checkpoint": "/a/ep250.pth"}
    moved = ls.apply(state, decision, NOW)
    assert moved["phase"] == "calibrate"
    assert moved["calibration"] == {"checkpoint": "/a/ep250.pth", "epoch": 250, "tried": [], "passed": False}
    below = _flat(0.08, 255)
    assert ls.decide(state, ls.Probe(latched=below, checkpoints=checkpoints, runs={"stage1_a": RUNNING})).action == "wait"
    assert ls.decide(state, ls.Probe(latched=below, checkpoints=checkpoints, runs={"stage1_a": ENDED})).action == "pause"


def test_a_crash_pauses_only_the_phase_that_uses_the_run():
    crashed = ls.RunStatus(started=True, crashed=True, marker="Traceback (most recent call last)")
    decision = ls.decide(_state(gate1={"epoch": 200, "latched": 0.08, "passed": True}), ls.Probe(runs={"stage1_a": crashed}))
    assert decision.action == "pause" and decision.reason.startswith("stage1_a crashed: Traceback")
    assert ls.decide(_state("vlm_target"), ls.Probe(runs={"stage1_a": crashed})).action == "write_prompt"


def test_a_side_run_alive_two_minutes_after_its_result_is_killed_as_hung():
    hung = ls.RunStatus(started=True, alive=True, finished=True, passed=True, idle_s=121.0)
    decision = ls.decide(_state("vlm_target"), ls.Probe(runs={"video": hung}))
    assert decision.action == "kill_stale" and decision.params == {"run": "video"}
    fresh = replace(hung, idle_s=60.0)
    assert ls.decide(_state("vlm_target"), ls.Probe(runs={"video": fresh})).action == "write_prompt"


def test_calibrate_launches_below_the_gpu_limit_and_commits_a_passing_measurement():
    calibration = {"checkpoint": "/a/ep250.pth", "epoch": 250, "tried": [], "passed": False}
    state = _state("calibrate", calibration=calibration)
    probe = ls.Probe(gpu_used_mib=13000, checkpoints={250: "/a/ep250.pth"}, runs={"stage1_a": RUNNING})
    launch = ls.decide(state, probe)
    assert launch.action == "run_calibrate" and launch.params == {"key": "/a/ep250.pth", "checkpoint": "/a/ep250.pth", "epoch": 250}
    assert ls.decide(state, replace(probe, gpu_used_mib=21000)).action == "wait"
    assert ls.decide(state, replace(probe, runs={"stage1_a": RUNNING, "video": RUNNING})).action == "wait"
    launched = ls.apply(state, launch, NOW, {"run": {"label": "iker_shoe_c00_calibrate", "pid": 11}})
    assert launched["runs"]["calibrate"] == {"label": "iker_shoe_c00_calibrate", "pid": 11, "key": "/a/ep250.pth", "epoch": 250}
    runs = {"stage1_a": RUNNING, "calibrate": DONE_PASS}
    doc = {"checkpoint": "/a/ep250.pth", "q_lo": 0.1, "q_hi": 0.3}
    commit = ls.decide(launched, replace(probe, runs=runs, calibration=doc))
    assert commit.action == "commit_calibration"
    assert ls.apply(launched, commit, NOW)["phase"] == "stage1_b"
    other = {**doc, "checkpoint": "/a/ep200.pth"}
    assert ls.decide(launched, replace(probe, runs=runs, calibration=other)).action == "pause"


def test_calibrate_retries_at_the_next_checkpoint_and_pauses_when_phase_a_is_over():
    calibration = {"checkpoint": "/a/ep250.pth", "epoch": 250, "tried": [], "passed": False}
    state = _state("calibrate", calibration=calibration, runs={"calibrate": {"label": "c", "key": "/a/ep250.pth"}})
    later = {250: "/a/ep250.pth", 300: "/a/ep300.pth"}
    retry = ls.decide(state, ls.Probe(checkpoints=later, runs={"calibrate": DONE_FAIL, "stage1_a": RUNNING}))
    assert retry.action == "next_calibration" and retry.params == {"epoch": 300, "checkpoint": "/a/ep300.pth"}
    moved = ls.apply(state, retry, NOW)
    assert moved["calibration"]["tried"] == ["/a/ep250.pth"] and moved["calibration"]["epoch"] == 300
    assert ls.decide(moved, ls.Probe(checkpoints=later, runs={"calibrate": DONE_FAIL, "stage1_a": RUNNING})).action == "run_calibrate"
    only = {250: "/a/ep250.pth"}
    assert ls.decide(state, ls.Probe(checkpoints=only, runs={"calibrate": DONE_FAIL, "stage1_a": RUNNING})).action == "wait"
    assert ls.decide(state, ls.Probe(checkpoints=only, runs={"calibrate": DONE_FAIL, "stage1_a": ENDED})).action == "pause"


def test_stage1_b_stops_a_launches_from_the_calibrated_checkpoint_and_checks_the_boot_line():
    calibration = {"checkpoint": "/a/ep250.pth", "epoch": 250, "tried": [], "passed": True}
    state = _state("stage1_b", calibration=calibration, runs={"stage1_a": {"label": "iker_grasp_c00_a", "key": "a"}})
    launch = ls.decide(state, ls.Probe(runs={"stage1_a": RUNNING}))
    assert launch.action == "launch_b" and launch.params["checkpoint"] == "/a/ep250.pth"
    launched = ls.apply(state, launch, NOW, {"run": {"label": "iker_grasp_c00_b"}, "stopped_runs": ["stage1_a"]})
    assert launched["runs"]["stage1_a"]["stopped"] == NOW and launched["runs"]["stage1_b"]["key"] == "/a/ep250.pth"
    probe = ls.Probe(runs={"stage1_b": RUNNING}, calibration={"checkpoint": "/a/ep250.pth", "q_lo": 0.11, "q_hi": 0.33})
    assert ls.decide(launched, probe).action == "wait"
    good = ls.decide(launched, replace(probe, boot_reward={"g_min": 0.5, "q_lo": 0.11, "q_hi": 0.33}))
    assert good.action == "advance" and good.params == {"to": "harvest"}
    assert ls.decide(launched, replace(probe, boot_reward={"g_min": 1.0, "q_lo": 0.11, "q_hi": 0.33})).action == "pause"


def test_harvest_tries_the_newest_new_checkpoint_and_commits_only_a_learned_bank():
    state = _state("harvest", runs={"stage1_b": {"label": "b", "key": "b"}})
    probe = ls.Probe(checkpoints={300: "/b/ep300.pth", 350: "/b/ep350.pth"}, runs={"stage1_b": RUNNING})
    launch = ls.decide(state, probe)
    assert launch.action == "run_harvest" and launch.params == {"key": "/b/ep350.pth", "checkpoint": "/b/ep350.pth", "epoch": 350}
    launched = ls.apply(state, launch, NOW, {"run": {"label": "iker_shoe_c00_harvest"}})
    assert ls.decide(launched, replace(probe, runs={"stage1_b": RUNNING, "harvest": RUNNING})).action == "wait"
    miss = ls.decide(launched, replace(probe, runs={"stage1_b": RUNNING, "harvest": DONE_FAIL}))
    assert miss.action == "record_harvest_miss"
    missed = ls.apply(launched, miss, NOW)
    assert missed["bank"] == {"tried": ["/b/ep350.pth"], "last_epoch": 350}
    assert ls.decide(missed, replace(probe, runs={"stage1_b": RUNNING, "harvest": DONE_FAIL})).action == "wait"
    newer = replace(probe, checkpoints={**probe.checkpoints, 400: "/b/ep400.pth"}, runs={"stage1_b": RUNNING, "harvest": DONE_FAIL})
    assert ls.decide(missed, newer).params["checkpoint"] == "/b/ep400.pth"
    assert ls.decide(missed, replace(probe, runs={"stage1_b": ENDED, "harvest": DONE_FAIL})).action == "pause"
    learned = {"source": "learned_grasp", "checkpoint": "/b/ep350.pth", "verified": 80}
    passed = replace(probe, runs={"stage1_b": RUNNING, "harvest": DONE_PASS}, bank_meta=learned)
    commit = ls.decide(launched, passed)
    assert commit.action == "commit_bank" and commit.params == {"checkpoint": "/b/ep350.pth", "verified": 80}
    committed = ls.apply(launched, commit, NOW, {"path": "/c/grasp_bank.json", "stopped_runs": ["stage1_b"]})
    assert committed["phase"] == "vlm_target" and committed["bank"]["verified"] == 80
    assert committed["runs"]["stage1_b"]["stopped"] == NOW
    assert ls.decide(launched, replace(passed, bank_meta={"side_sign": 1.0, "seed": 0})).action == "pause"


def test_vlm_target_writes_the_prompt_generates_ingests_and_stops_after_three_failures():
    state = _state("vlm_target")
    assert ls.decide(state, ls.Probe()).action == "write_prompt"
    files = {"stage_prompt": True}
    first = ls.decide(state, ls.Probe(files=files))
    assert first.action == "vlm_generate" and first.params == {"kind": "target", "attempt": 0}
    pending = ls.decide(state, ls.Probe(files=files, attempts=(ls.AttemptStatus(0, None),)))
    assert pending.action == "ingest" and pending.params == {"attempt": 0}
    assert ls.apply(state, pending, NOW, {"passed": False})["attempts"] == {"1": 1}
    two_failed = (ls.AttemptStatus(0, False), ls.AttemptStatus(1, False))
    assert ls.decide(state, ls.Probe(files=files, attempts=two_failed)).params == {"kind": "target", "attempt": 2}
    three_failed = two_failed + (ls.AttemptStatus(2, False),)
    assert ls.decide(state, ls.Probe(files=files, attempts=three_failed)).action == "pause"
    passed = ls.decide(state, ls.Probe(files=files, attempts=(ls.AttemptStatus(0, False), ls.AttemptStatus(1, True))))
    assert passed.action == "commit_interaction" and ls.apply(state, passed, NOW)["phase"] == "stage2_train"


def test_stage2_train_smokes_launches_evaluates_and_advances_on_the_success_target():
    state = _state("stage2_train")
    smoke = ls.decide(state, ls.Probe())
    assert smoke.action == "run_env_smoke"
    smoked = ls.apply(state, smoke, NOW, {"run": {"label": "s"}})
    assert ls.decide(smoked, ls.Probe(runs={"env_smoke": DONE_FAIL})).action == "pause"
    runs = {"env_smoke": DONE_PASS}
    assert ls.decide(smoked, ls.Probe(runs={**runs, "stage1_b": RUNNING})).action == "wait"
    launch = ls.decide(smoked, ls.Probe(runs=runs))
    assert launch.action == "launch_stage2" and launch.params == {"key": "iker_vlm_c00_s1"}
    training = ls.apply(smoked, launch, NOW, {"run": {"label": "iker_vlm_c00_s1"}})
    runs = {**runs, "stage2": RUNNING}
    checkpoints = {250: "/s/ep250.pth"}
    evaluate = ls.decide(training, ls.Probe(runs=runs, checkpoints=checkpoints))
    assert evaluate.action == "run_eval" and evaluate.params == {"key": "ep250", "epoch": 250, "checkpoint": "/s/ep250.pth"}
    evaluating = ls.apply(training, evaluate, NOW, {"run": {"label": "e"}})
    summary = {"success_5cm_end": 0.3, "checkpoint": "/s/ep250.pth", "dropped": 0.04}
    record = ls.decide(evaluating, ls.Probe(runs={**runs, "eval": DONE_PASS}, checkpoints=checkpoints, evals={250: summary}))
    assert record.action == "record_eval"
    recorded = ls.apply(evaluating, record, NOW)
    assert recorded["eval"] == {"250": {"success": 0.3, "checkpoint": "/s/ep250.pth", "dropped": 0.04}}
    assert ls.decide(recorded, ls.Probe(runs=runs, checkpoints=checkpoints, evals={250: summary})).action == "wait"
    good = {**recorded, "eval": {**recorded["eval"], "500": {"success": 0.55, "checkpoint": "/s/ep500.pth", "dropped": 0.0}}}
    advance = ls.decide(good, ls.Probe(runs=runs, checkpoints=checkpoints, evals={250: summary, 500: summary}))
    assert advance.action == "advance" and advance.params == {"to": "observe_requery", "epoch": 500}
    low = {key: {"success": 0.3, "checkpoint": f"/s/ep{key}.pth", "dropped": 0.0} for key in ("250", "500", "750")}
    ended = {**recorded, "eval": low}
    all_checkpoints = {250: "a", 500: "b", 750: "c"}
    probe = ls.Probe(runs={"env_smoke": DONE_PASS, "stage2": ENDED}, checkpoints=all_checkpoints, evals={250: {}, 500: {}, 750: {}})
    assert ls.decide(ended, probe).action == "pause"
    assert ls.decide(ls.new_state(NOW, phase="stage2_train", policy={"stage2_env_smoke": False}), ls.Probe()).action == "launch_stage2"


def test_observe_requery_rolls_out_renders_requeries_and_stores_the_video():
    state = _state("observe_requery", eval={"500": {"success": 0.6, "checkpoint": "/s/ep500.pth", "dropped": 0.0}})
    rollout = ls.decide(state, ls.Probe())
    assert rollout.action == "run_observe_rollout" and rollout.params["checkpoint"] == "/s/ep500.pth"
    rolled = ls.apply(state, rollout, NOW, {"run": {"label": "o"}})
    failures = ({"env": 0, "success": False, "end_dist": 0.2},)
    assert ls.decide(rolled, ls.Probe(runs={"observe_rollout": DONE_PASS}, final_rows=failures)).action == "pause"
    rows = ({"env": 3, "success": True, "end_dist": 0.02},)
    render = ls.decide(rolled, ls.Probe(runs={"observe_rollout": DONE_PASS}, final_rows=rows))
    assert render.action == "run_observe_render" and render.params == {"key": "env3", "env": 3}
    files = {"observe_snapshot": True}
    assert ls.decide(rolled, ls.Probe(final_rows=rows, files=files)).action == "write_requery_prompt"
    files = {**files, "requery_prompt": True}
    assert ls.decide(rolled, ls.Probe(final_rows=rows, files=files)).params == {"kind": "requery"}
    files = {**files, "requery_response": True}
    assert ls.decide(rolled, ls.Probe(final_rows=rows, files=files)).action == "parse_requery"
    new_stage = {"done": False, "detail": "new stage: move shoe_other"}
    assert ls.decide(rolled, ls.Probe(final_rows=rows, files=files, requery=new_stage)).action == "pause"
    done = {"done": True, "detail": "done"}
    video = ls.decide(rolled, ls.Probe(final_rows=rows, files=files, requery=done))
    assert video.action == "run_video"
    recording = ls.apply(rolled, video, NOW, {"run": {"label": "v"}})
    probe = ls.Probe(runs={"video": DONE_PASS}, final_rows=rows, files={**files, "video_raw": True}, requery=done)
    assert ls.decide(recording, probe).action == "store_video"
    review = ls.decide(recording, replace(probe, files={**files, "video_raw": True, "video": True}))
    assert review.action == "advance" and review.params == {"to": "completion_review"}


def test_completion_waits_for_the_user_and_resume_moves_the_gate():
    review = ls.decide(_state("completion_review"), ls.Probe())
    assert review.action == "pause"
    done = ls.apply(ls.apply(_state("completion_review"), review, NOW), ls.Decision("approve", "user approved"), NOW)
    assert (done["phase"], done["status"]) == ("done", "done") and ls.decide(done, ls.Probe()).action == "wait"
    with pytest.raises(ValueError, match="approve"):
        ls.apply(_state(), ls.Decision("approve", "too early"), NOW)
    low = ls.apply(_state(), ls.Decision("record_gate", "low", {"epoch": 200, "latched": 0.01, "passed": False}), NOW)
    resumed = ls.apply(low, ls.Decision("resume", "user", {"policy": {"gate_epoch": 300}}), NOW)
    assert resumed["status"] == "running" and resumed["gate1"] is None and resumed["policy"]["gate_epoch"] == 300
    with pytest.raises(ValueError, match="unknown policy keys"):
        ls.apply(low, ls.Decision("resume", "user", {"policy": {"gate": 1}}), NOW)


def test_over_rack_rising_past_twice_the_latched_rate_is_noted():
    state, latched = _state(), _flat(0.01, 150)
    noted = ls.decide(state, ls.Probe(latched=latched, over_rack=_flat(0.03, 150), runs={"stage1_a": RUNNING}))
    assert noted.action == "wait" and noted.notes[0].startswith("over-rack raised 3.00 % > 2 x latched 1.00 %")
    quiet = ls.decide(state, ls.Probe(latched=latched, over_rack=_flat(0.015, 150), runs={"stage1_a": RUNNING}))
    assert quiet.notes == ()
