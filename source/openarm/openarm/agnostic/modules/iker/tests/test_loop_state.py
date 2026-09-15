"""loop_state — the IKER auto-loop state machine (design 2026-09-14-iker-auto-loop §3, §4, §9; no Isaac)."""

from dataclasses import replace

import pytest

from openarm.agnostic.modules.iker import loop_state as ls

NOW = "2026-09-14T20:00:00"
RUNNING = ls.RunStatus(started=True, alive=True)
ENDED = ls.RunStatus(started=True, finished=True)
DONE_PASS = ls.RunStatus(started=True, finished=True, passed=True, marker="X passed True")
DONE_FAIL = ls.RunStatus(started=True, finished=True, passed=False, marker="X passed False")


def _state(phase="stage1_t2r", **changes):
    return {**ls.new_state(NOW, phase=phase), **changes}


def _flat(value, upto, start=1):
    return tuple((epoch, value) for epoch in range(start, upto + 1))


LABEL = "iker_grasp_c00_t2r_i00"
REWARD_SHA = "a" * 64
FILES_READY = ls.T2rIter(prompt=True, response=True, validation={"ok": True, "reward_sha256": REWARD_SHA})


def _training(state, **run):
    return {**state, "runs": {"stage1_t2r": {"label": LABEL, "log": "/l/t.log", "started_s": 0.0, "key": LABEL, **run}}}


def test_new_state_holds_the_t2r_round_state_and_rejects_unknown_keys():
    state = ls.new_state(NOW, policy={"t2r_round_epochs": 20, "labels": {"stage2": "iker_vlm_mock"}})
    assert state["phase"] == "stage1_t2r" and state["t2r"] == {"iter": 0, "requests": 0, "rounds": []}
    assert state["policy"]["t2r_round_epochs"] == 20 and state["policy"]["t2r_harvest_success"] == 0.05
    assert state["policy"]["labels"] == {"stage1_t2r": "iker_grasp_c00_t2r", "stage2": "iker_vlm_mock"}
    assert ls.t2r_label(state) == LABEL and "gate1" not in state and "calibration" not in state
    for removed in ("gate_epoch", "calibrate_latched", "b_g_min"):
        with pytest.raises(ValueError, match="unknown policy keys"):
            ls.new_state(NOW, policy={removed: 1})
    with pytest.raises(ValueError, match="t2r"):
        ls.validate_state({**state, "t2r": {"iter": 0}})


def test_a_round_writes_the_prompt_asks_the_generator_and_ingests_its_response():
    state = _state()
    assert ls.decide(state, ls.Probe()).action == "write_t2r_prompt"
    decision = ls.decide(state, ls.Probe(t2r=ls.T2rIter(prompt=True)))
    assert (decision.action, decision.params) == ("t2r_generate", {"iter": 0})
    asked = ls.apply(state, decision, NOW)
    assert asked["t2r"]["requests"] == 1
    assert ls.decide(asked, ls.Probe(t2r=ls.T2rIter(prompt=True, response=True))).action == "ingest_reward"
    exhausted = {**asked, "t2r": {**asked["t2r"], "requests": 3}}
    paused = ls.decide(exhausted, ls.Probe(t2r=ls.T2rIter(prompt=True, failed_attempts=3)))
    assert paused.action == "pause" and "3 generator requests" in paused.reason


def test_a_validated_reward_is_smoked_by_its_digest_then_trained():
    state = _state()
    smoke = ls.decide(state, ls.Probe(t2r=FILES_READY))
    assert (smoke.action, smoke.params) == ("run_t2r_smoke", {"key": REWARD_SHA, "iter": 0, "digest": REWARD_SHA})
    assert ls.decide(state, ls.Probe(t2r=FILES_READY, gpu_used_mib=30000)).action == "wait"
    smoked = ls.apply(state, smoke, NOW, {"run": {"label": "iker_shoe_c00_t2r_smoke", "log": "/l/s.log"}})
    assert smoked["runs"]["t2r_smoke"]["key"] == REWARD_SHA
    assert ls.decide(smoked, ls.Probe(t2r=FILES_READY, runs={"t2r_smoke": RUNNING})).action == "wait"
    launch = ls.decide(smoked, ls.Probe(t2r=FILES_READY, runs={"t2r_smoke": DONE_PASS}))
    assert (launch.action, launch.params) == ("launch_t2r", {"key": LABEL, "iter": 0})
    launched = ls.apply(smoked, launch, NOW, {"run": {"label": LABEL, "log": "/l/t.log", "started_s": 1.0}})
    assert launched["runs"]["stage1_t2r"]["key"] == LABEL
    # finding 1: a new reward (a different digest) in the same iteration is smoked again, not read as already launched
    new_digest = "b" * 64
    fresh = ls.T2rIter(prompt=True, response=True, validation={"ok": True, "reward_sha256": new_digest})
    resmoke = ls.decide(smoked, ls.Probe(t2r=fresh))
    assert (resmoke.action, resmoke.params) == ("run_t2r_smoke", {"key": new_digest, "iter": 0, "digest": new_digest})


def test_a_failed_round_smoke_records_a_miss_and_only_pauses_once_requests_are_exhausted():
    state = _state()
    smoked = ls.apply(state, ls.decide(state, ls.Probe(t2r=FILES_READY)), NOW,
                      {"run": {"label": "iker_shoe_c00_t2r_smoke", "log": "/l/s.log"}})
    # finding 1: a failed smoke with requests still available records a miss (not a pause) and the next decision re-asks
    miss = ls.decide(smoked, ls.Probe(t2r=FILES_READY, runs={"t2r_smoke": DONE_FAIL}))
    assert (miss.action, miss.params) == ("record_smoke_miss", {"iter": 0, "digest": REWARD_SHA})
    missed = ls.apply(smoked, miss, NOW)
    # the executor moved response.md/validation.json/smoke.json aside; the next probe sees neither
    again = ls.decide(missed, ls.Probe(t2r=ls.T2rIter(prompt=True, failed_attempts=1), runs={"t2r_smoke": DONE_FAIL}))
    assert again.action == "t2r_generate" and again.params == {"iter": 0}
    # a passing smoke still leads to launch_t2r regardless of past misses
    passed = ls.decide(missed, ls.Probe(t2r=FILES_READY, runs={"t2r_smoke": DONE_PASS}))
    assert passed.action == "launch_t2r"
    # with the requests exhausted, a failed smoke pauses instead, naming the failed smoke
    exhausted = {**missed, "t2r": {**missed["t2r"], "requests": 3}}
    paused = ls.decide(exhausted, ls.Probe(t2r=FILES_READY, runs={"t2r_smoke": DONE_FAIL}))
    assert paused.action == "pause" and "t2r smoke failed" in paused.reason and "3 generator requests" in paused.reason


def test_launch_t2r_waits_for_the_gpu_limit_like_the_side_runs():
    state = _state()
    smoked = ls.apply(state, ls.decide(state, ls.Probe(t2r=FILES_READY)), NOW,
                      {"run": {"label": "iker_shoe_c00_t2r_smoke", "log": "/l/s.log"}})
    busy_gpu = ls.Probe(t2r=FILES_READY, runs={"t2r_smoke": DONE_PASS}, gpu_used_mib=25000)
    waiting = ls.decide(smoked, busy_gpu)
    assert waiting.action == "wait" and "GPU memory 25000 MiB > 20000 MiB" in waiting.reason
    ok = ls.decide(smoked, replace(busy_gpu, gpu_used_mib=1000))
    assert ok.action == "launch_t2r"


def test_training_harvests_the_newest_checkpoint_at_five_percent_success_and_commits_the_bank():
    state = _training(_state())
    success = _flat(0.02, 100) + _flat(0.06, 150, start=101)
    probe = ls.Probe(t2r=FILES_READY, runs={"stage1_t2r": RUNNING}, success=success, checkpoints={100: "c100", 150: "c150"})
    harvest = ls.decide(state, probe)
    assert (harvest.action, harvest.params) == ("run_harvest", {"key": "c150", "checkpoint": "c150", "epoch": 150})
    harvesting = ls.apply(state, harvest, NOW, {"run": {"label": "iker_shoe_c00_harvest", "log": "/l/h.log"}})
    assert ls.decide(harvesting, replace(probe, runs={"stage1_t2r": RUNNING, "harvest": RUNNING})).action == "wait"
    meta = {"source": ls.LEARNED_BANK_SOURCE, "checkpoint": "c150", "verified": 80}
    commit = ls.decide(harvesting, replace(probe, runs={"stage1_t2r": RUNNING, "harvest": DONE_PASS}, bank_meta=meta))
    assert commit.action == "commit_bank" and commit.params == {"checkpoint": "c150", "verified": 80, "epoch": 150, "iter": 0, "label": LABEL}
    banked = ls.apply(harvesting, commit, NOW, {"path": "/b.json", "run_dir": "/r", "reward_sha256": "ab", "stopped_runs": ["stage1_t2r"]})
    assert banked["phase"] == "vlm_target" and banked["t2r"]["rounds"][-1]["ended"] == "handover" and "stopped" in banked["runs"]["stage1_t2r"]
    few = ls.decide(harvesting, replace(probe, runs={"stage1_t2r": RUNNING, "harvest": DONE_PASS}, bank_meta={**meta, "verified": 10}))
    assert few.action == "pause"
    miss = ls.decide(harvesting, replace(probe, runs={"stage1_t2r": RUNNING, "harvest": DONE_FAIL}))
    assert miss.action == "record_harvest_miss"
    missed = ls.apply(harvesting, miss, NOW)
    assert ls.decide(missed, replace(probe, runs={"stage1_t2r": RUNNING, "harvest": DONE_FAIL})).action == "wait"


def test_a_round_ends_at_its_last_epoch_or_early_and_the_loop_pauses_after_the_last_round():
    state = _training(_state())
    ended = ls.decide(state, ls.Probe(t2r=FILES_READY, runs={"stage1_t2r": ENDED}, success=_flat(0.0, 500), latched=_flat(0.01, 500)))
    assert ended.action == "end_round" and ended.params["ended"] == "round_epochs" and ended.params["end_epoch"] == 500
    next_round = ls.apply(state, ended, NOW, {"run_dir": "/r0", "reward_sha256": "ab"})
    assert next_round["t2r"]["iter"] == 1 and next_round["t2r"]["requests"] == 0 and next_round["t2r"]["rounds"][0]["run_dir"] == "/r0"
    assert ls.t2r_label(next_round) == "iker_grasp_c00_t2r_i01" and ls.decide(next_round, ls.Probe()).action == "write_t2r_prompt"
    early = ls.Probe(t2r=FILES_READY, runs={"stage1_t2r": RUNNING}, success=_flat(0.0, 250), latched=_flat(0.001, 250))
    assert ls.decide(state, early).params.get("ended") == "early"
    assert ls.decide(state, replace(early, success=_flat(0.0, 249), latched=_flat(0.001, 249))).action == "wait"
    assert ls.decide(state, replace(early, latched=_flat(0.01, 250))).action == "wait"
    full = {**state, "t2r": {"iter": 6, "requests": 0, "rounds": [{"iter": i} for i in range(6)]}}
    assert ls.decide(full, ls.Probe()).action == "pause"


def test_a_round_waits_for_the_newest_checkpoint_to_settle_before_ending():
    # finding 4: training_status reads an unsettled checkpoint list, so `finished` can go True before the harvest loop
    # (which only sees settled checkpoints) has had a chance to check the newest one — wait instead of ending the round.
    state = _training(_state())
    probe = ls.Probe(t2r=FILES_READY, runs={"stage1_t2r": ENDED}, success=_flat(0.0, 500), latched=_flat(0.01, 500),
                     checkpoint_pending=True)
    waiting = ls.decide(state, probe)
    assert waiting.action == "wait" and "settl" in waiting.reason
    ended = ls.decide(state, replace(probe, checkpoint_pending=False))
    assert ended.action == "end_round"
    early = ls.Probe(t2r=FILES_READY, runs={"stage1_t2r": RUNNING}, success=_flat(0.0, 250), latched=_flat(0.001, 250),
                     checkpoint_pending=True)
    assert ls.decide(state, early).action == "wait"


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
    assert paused["status"] == "awaiting" and paused["awaiting"] == {"reason": "why", "needs": "what", "phase": "stage1_t2r"}
    assert ls.decide(paused, ls.Probe()).action == "wait"
    with pytest.raises(ValueError, match="not recorded"):
        ls.apply(_state(), ls.Decision("wait", "nothing"), NOW)


def test_a_crash_pauses_only_the_phase_that_uses_the_run():
    crashed = ls.RunStatus(started=True, crashed=True, marker="Traceback (most recent call last)")
    decision = ls.decide(_state(), ls.Probe(runs={"t2r_smoke": crashed}))
    assert decision.action == "pause" and decision.reason.startswith("t2r_smoke crashed: Traceback")
    assert ls.decide(_state("vlm_target"), ls.Probe(runs={"t2r_smoke": crashed})).action == "write_prompt"


def test_a_side_run_alive_two_minutes_after_its_result_is_killed_as_hung():
    hung = ls.RunStatus(started=True, alive=True, finished=True, passed=True, idle_s=121.0)
    decision = ls.decide(_state("vlm_target"), ls.Probe(runs={"t2r_smoke": hung}))
    assert decision.action == "kill_stale" and decision.params == {"run": "t2r_smoke"}
    fresh = replace(hung, idle_s=60.0)
    assert ls.decide(_state("vlm_target"), ls.Probe(runs={"t2r_smoke": fresh})).action == "write_prompt"


def test_resume_clears_a_dead_crashed_run_so_the_phase_launches_it_again_once():
    state = _state()
    probe = ls.Probe(t2r=FILES_READY)
    launched = ls.apply(state, ls.decide(state, probe), NOW, {"run": {"label": "iker_shoe_c00_t2r_smoke"}})
    crashed = replace(probe, runs={"t2r_smoke": ls.RunStatus(started=True, crashed=True)})
    pause = ls.decide(launched, crashed)
    assert pause.action == "pause" and pause.reason.startswith("t2r_smoke crashed")
    paused = ls.apply(launched, pause, NOW)
    assert ls.runs_to_clear(paused, crashed.runs) == ("t2r_smoke",)
    resumed = ls.apply(paused, ls.Decision("resume", "user"), "later", {"cleared_runs": ["t2r_smoke"]})
    assert resumed["status"] == "running" and resumed["runs"]["t2r_smoke"]["cleared"] == "later"
    for seen in (crashed, probe):  # the probe omits a cleared run; a stale reading of its old log is ignored too
        relaunch = ls.decide(resumed, seen)
        assert relaunch.action == "run_t2r_smoke" and relaunch.params == {"key": REWARD_SHA, "iter": 0, "digest": REWARD_SHA}
    relaunched = ls.apply(resumed, relaunch, NOW, {"run": {"label": "iker_shoe_c00_t2r_smoke"}})
    assert "cleared" not in relaunched["runs"]["t2r_smoke"]
    assert ls.decide(relaunched, replace(probe, runs={"t2r_smoke": RUNNING})).action == "wait"
    assert ls.runs_to_clear(paused, {"t2r_smoke": replace(crashed.runs["t2r_smoke"], alive=True)}) == ()


def test_a_run_the_loop_stopped_is_never_a_crash_that_pauses_the_phase():
    dead = ls.RunStatus(started=True, crashed=True)
    for mark in ("stopping", "stopped"):
        state = _training(_state(), **{mark: NOW})
        assert ls.decide(state, ls.Probe(runs={"stage1_t2r": dead})).action != "pause"
        assert ls.runs_to_clear(state, {"stage1_t2r": dead}) == ()


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


def test_a_generator_that_never_writes_its_response_pauses_after_three_requests():
    state = _state("vlm_target")
    files = {"stage_prompt": True}
    request = ls.decide(state, ls.Probe(files=files))
    legacy = {key: value for key, value in state.items() if key != "vlm_requests"}  # a state file from before the counter
    once = ls.apply(legacy, request, NOW)
    assert once["vlm_requests"] == {"target_00": 1}
    thrice = ls.apply(ls.apply(once, request, NOW), request, NOW)
    paused = ls.decide(thrice, ls.Probe(files=files))
    assert paused.action == "pause" and "3" in paused.reason and "stage_01/attempt_00/response.md" in paused.reason
    answered = ls.decide(thrice, ls.Probe(files=files, attempts=(ls.AttemptStatus(0, False),)))
    assert answered.action == "vlm_generate" and answered.params == {"kind": "target", "attempt": 1}
    resumed = ls.apply(ls.apply(thrice, paused, NOW), ls.Decision("resume", "user"), NOW)
    assert ls.decide(resumed, ls.Probe(files=files)).action == "vlm_generate"
    observe = _state("observe_requery", eval={"500": {"success": 0.6, "checkpoint": "/s/ep500.pth", "dropped": 0.0}},
                     vlm_requests={"requery": 3})
    rows = ({"env": 3, "success": True, "end_dist": 0.02},)
    requery = ls.decide(observe, ls.Probe(final_rows=rows, files={"observe_snapshot": True, "requery_prompt": True}))
    assert requery.action == "pause" and "stage_01/observe/response.md" in requery.reason


def test_stage2_train_smokes_launches_evaluates_and_advances_on_the_success_target():
    state = _state("stage2_train")
    smoke = ls.decide(state, ls.Probe())
    assert smoke.action == "run_env_smoke"
    smoked = ls.apply(state, smoke, NOW, {"run": {"label": "s"}})
    assert ls.decide(smoked, ls.Probe(runs={"env_smoke": DONE_FAIL})).action == "pause"
    runs = {"env_smoke": DONE_PASS}
    assert ls.decide(smoked, ls.Probe(runs={**runs, "stage1_t2r": RUNNING})).action == "wait"
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
    assert advance.action == "advance" and advance.params == {"to": "observe_requery", "epoch": 500, "stop": "stage2"}
    advanced = ls.apply(good, advance, NOW, {"stopped_runs": ["stage2"]})
    assert advanced["phase"] == "observe_requery" and advanced["runs"]["stage2"]["stopped"] == NOW
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


def test_observe_requery_pauses_on_a_failed_render_even_when_its_files_exist():
    state = _state("observe_requery", eval={"500": {"success": 0.6, "checkpoint": "/s/ep500.pth", "dropped": 0.0}},
                   runs={"observe_render": {"label": "r", "key": "env3"}})
    rows = ({"env": 3, "success": True, "end_dist": 0.02},)
    files = {"observe_snapshot": True}
    failed = ls.RunStatus(started=True, finished=True, passed=False, marker="OBSERVE config 00 passed False",
                          failed_checks=("margin: keypoint 3 is 2.0 px inside the frame",))
    decision = ls.decide(state, ls.Probe(runs={"observe_render": failed}, final_rows=rows, files=files))
    assert decision.action == "pause" and "margin: keypoint 3 is 2.0 px inside the frame" in decision.reason
    rendered = replace(failed, passed=True, failed_checks=())
    assert ls.decide(state, ls.Probe(runs={"observe_render": rendered}, final_rows=rows, files=files)).action == "write_requery_prompt"
    assert ls.decide(state, ls.Probe(runs={"observe_render": RUNNING}, final_rows=rows, files=files)).action == "wait"


def test_completion_waits_for_the_user_and_resume_moves_the_gate():
    review = ls.decide(_state("completion_review"), ls.Probe())
    assert review.action == "pause"
    done = ls.apply(ls.apply(_state("completion_review"), review, NOW), ls.Decision("approve", "user approved"), NOW)
    assert (done["phase"], done["status"]) == ("done", "done") and ls.decide(done, ls.Probe()).action == "wait"
    with pytest.raises(ValueError, match="approve"):
        ls.apply(_state(), ls.Decision("approve", "too early"), NOW)
    paused = ls.apply(_state(), ls.Decision("pause", "why", {"needs": "what"}), NOW)
    busy = {**paused, "t2r": {**paused["t2r"], "requests": 2}}
    resumed = ls.apply(busy, ls.Decision("resume", "user"), NOW)
    assert resumed["status"] == "running" and resumed["t2r"]["requests"] == 0
    with pytest.raises(ValueError, match="unknown policy keys"):
        ls.apply(busy, ls.Decision("resume", "user", {"policy": {"gate": 1}}), NOW)


def test_over_rack_rising_past_twice_the_latched_rate_is_noted():
    state, latched = _training(_state()), _flat(0.01, 150)
    noted = ls.decide(state, ls.Probe(latched=latched, over_rack=_flat(0.03, 150), runs={"stage1_t2r": RUNNING}))
    assert noted.action == "wait" and noted.notes[0].startswith("over-rack raised 3.00 % > 2 x latched 1.00 %")
    quiet = ls.decide(state, ls.Probe(latched=latched, over_rack=_flat(0.015, 150), runs={"stage1_t2r": RUNNING}))
    assert quiet.notes == ()
