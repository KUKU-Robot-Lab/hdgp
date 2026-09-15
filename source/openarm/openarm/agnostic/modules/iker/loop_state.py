"""IKER auto-loop state machine (design 2026-09-14-iker-auto-loop §3, §4, §8). Pure: no Isaac, torch or subprocess.

A tick reads the world into a ``Probe`` (loop_probe.collect), asks ``decide`` for the one next action, lets the CLI
(scripts/iker/loop.py) carry out its side effect, and records the outcome with ``apply``. Every threshold of the loop is
a key of ``DEFAULT_POLICY`` and is stored in LOOP_STATE.json, so a run's rules travel with its state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

from . import loop_t2r, run_files

SCHEMA = run_files.SCHEMA_VERSION
TRACK = "iker_shoe_c00"
LEARNED_BANK_SOURCE = "learned_grasp"
PHASES = ("stage1_t2r", "vlm_target", "stage2_train", "observe_requery", "completion_review", "done")
STATUSES = ("running", "awaiting", "done")
TRAINING_RUNS = ("stage1_t2r", "stage2")
SIDE_RUNS = ("t2r_smoke", "harvest", "env_smoke", "eval", "observe_rollout", "observe_render", "video")
STALE_AFTER_RESULT_S = 120.0  # an Isaac process alive this long after writing its result is hung (§8)

DEFAULT_POLICY: Mapping = {
    "bin_epochs": 10,
    "t2r_round_epochs": 500,  # stage-1 t2r spec §3.2 (user decision 5): one round's fresh training
    "t2r_early_epoch": 250,
    "t2r_early_latched": 0.005,
    "t2r_harvest_success": 0.05,
    "t2r_max_rounds": 6,
    "t2r_max_requests": 3,
    "t2r_num_envs": 4096,
    "t2r_smoke_envs": 64,
    "t2r_smoke_steps": 150,
    "harvest_min": 64,  # user decision 4
    "vlm_max_attempts": 3,  # the first response and two regenerations (base spec §6)
    "stage2_env_smoke": True,
    "stage2_epochs": 750,
    "stage2_num_envs": 4096,
    "eval_every": 250,
    "success_target": 0.5,  # user decision 5
    "side_gpu_limit_mib": 20000,
    "minibatch_size": 0,  # 0 keeps the PPO yaml value
    "grasp_bank_path": "",  # "" = config_XX/grasp_bank.json
    "labels": {"stage1_t2r": "iker_grasp_c00_t2r", "stage2": "iker_vlm_c00_s1"},
}

PHASE_RUNS: Mapping[str, tuple[str, ...]] = {  # runs whose crash stops the phase
    "stage1_t2r": ("stage1_t2r", "t2r_smoke", "harvest"),
    "vlm_target": (),
    "stage2_train": ("env_smoke", "stage2", "eval"),
    "observe_requery": ("observe_rollout", "observe_render", "video"),
    "completion_review": (),
    "done": (),
}
LAUNCH_RUNS: Mapping[str, str] = {
    "run_t2r_smoke": "t2r_smoke", "launch_t2r": "stage1_t2r", "run_harvest": "harvest", "run_env_smoke": "env_smoke",
    "launch_stage2": "stage2", "run_eval": "eval", "run_observe_rollout": "observe_rollout",
    "run_observe_render": "observe_render", "run_video": "video",
}
SESSION_ACTIONS = ("wait",)  # nothing to record
MANUAL_ACTIONS = ("approve", "resume")  # only on the user's word
FILE_ACTIONS = ("write_prompt", "write_requery_prompt", "parse_requery", "store_video", "kill_stale",
                "write_t2r_prompt", "ingest_reward")  # files are the record
VLM_MAX_REQUESTS = 3  # generator requests for one response file before the loop pauses (a module constant: a new policy key
#                       would make validate_state reject existing state files)
LOOP_END_MARKS = ("stopping", "stopped", "cleared")  # a run record carrying one ended by the loop's hand, never by a crash


@dataclass(frozen=True)
class RunStatus:
    started: bool = False  # the state holds a launch record for the run
    alive: bool = False  # an Isaac process carries the run's RUN_LABEL
    finished: bool = False  # training: the max-epoch checkpoint or MAX EPOCHS NUM!; side run: its result marker or file
    passed: bool | None = None  # side run: "passed True/False" of the result marker (True for a marker without it)
    crashed: bool = False  # a Traceback or FAILED marker, or the process ended without a result
    marker: str = ""
    idle_s: float = 0.0  # seconds since the run's log last changed
    failed_checks: tuple[str, ...] = ()  # side run: its blocking "CHECK FAILED" lines (observe_render)


IDLE = RunStatus()


@dataclass(frozen=True)
class AttemptStatus:
    index: int
    gate_passed: bool | None  # None until attempt_NN/gate.json exists


@dataclass(frozen=True)
class T2rIter:
    iter: int = 0
    prompt: bool = False
    response: bool = False
    validation: Mapping | None = None
    failed_attempts: int = 0


@dataclass(frozen=True)
class Probe:
    gpu_used_mib: int = 0
    runs: Mapping[str, RunStatus] = field(default_factory=dict)
    latched: tuple[tuple[int, float], ...] = ()  # Episode/grasp_episode/latched of the phase's stage-1 run, by epoch
    over_rack: tuple[tuple[int, float], ...] = ()  # Episode/grasp/over_rack_raised_frac, by epoch
    checkpoints: Mapping[int, str] = field(default_factory=dict)  # the phase's training run: epoch -> checkpoint path
    success: tuple[tuple[int, float], ...] = ()  # Episode/grasp_episode/success of the round's run, by epoch
    bank_meta: Mapping | None = None  # metadata of the grasp bank stage 2 loads
    attempts: tuple[AttemptStatus, ...] = ()  # stage-1 VLM attempts holding a response, in order
    evals: Mapping[int, Mapping] = field(default_factory=dict)  # epoch -> eval_iker.py summary
    final_rows: tuple[Mapping, ...] | None = None  # noise-free rollout rows (env, success, end_dist); None before the file
    requery: Mapping | None = None  # observe/requery.json
    files: Mapping[str, bool] = field(default_factory=dict)  # see loop_probe.collect
    t2r: T2rIter = field(default_factory=T2rIter)


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    params: Mapping = field(default_factory=dict)
    notes: tuple[str, ...] = ()


# ------------------------------------------------------------------ measures


def bin_mean(points: Sequence[tuple[int, float]], end_epoch: int, width: int) -> float | None:
    """Mean of the values logged at epochs ``end_epoch - width < e <= end_epoch``; None without any."""
    values = [value for epoch, value in points if end_epoch - width < epoch <= end_epoch]
    return sum(values) / len(values) if values else None


def last_epoch(points: Sequence[tuple[int, float]]) -> int:
    return max((epoch for epoch, _ in points), default=0)


def eval_epochs(policy: Mapping) -> tuple[int, ...]:
    return tuple(range(policy["eval_every"], policy["stage2_epochs"] + 1, policy["eval_every"]))


def best_eval(evals: Mapping[str, Mapping]) -> tuple[int, Mapping] | None:
    """(epoch, record) with the highest success; the earlier epoch wins a tie."""
    if not evals:
        return None
    key = max(evals, key=lambda k: (evals[k]["success"], -int(k)))
    return int(key), evals[key]


def pick_observe_env(rows: Sequence[Mapping]) -> int | None:
    """The succeeded env with the (lower) median end distance (§4 observe_requery); None without a success."""
    successes = sorted((row for row in rows if row["success"]), key=lambda row: (row["end_dist"], row["env"]))
    return int(successes[(len(successes) - 1) // 2]["env"]) if successes else None


# --------------------------------------------------------------------- state


def new_state(now: str, *, track: str = TRACK, phase: str = "stage1_t2r", policy: Mapping | None = None) -> dict:
    overrides = copy.deepcopy(dict(policy or {}))
    unknown = sorted(set(overrides) - set(DEFAULT_POLICY))
    if unknown:
        raise ValueError(f"unknown policy keys {unknown}")
    labels = {**DEFAULT_POLICY["labels"], **overrides.pop("labels", {})}
    state = {
        "schema": SCHEMA, "track": track, "phase": phase, "status": "running", "awaiting": None, "stage": 1,
        "policy": {**copy.deepcopy(dict(DEFAULT_POLICY)), **overrides, "labels": labels},
        "runs": {}, "t2r": {"iter": 0, "requests": 0, "rounds": []}, "bank": None, "attempts": {"1": 0}, "eval": {}, "vlm_requests": {},
        "updated": now,
    }
    validate_state(state)
    return state


def validate_state(state: Mapping) -> None:
    if state.get("schema") != SCHEMA:
        raise ValueError(f"loop state schema {state.get('schema')} != {SCHEMA}")
    if state.get("phase") not in PHASES:
        raise ValueError(f"unknown phase {state.get('phase')!r}")
    if state.get("status") not in STATUSES:
        raise ValueError(f"unknown status {state.get('status')!r}")
    if (state["status"] == "awaiting") != (state.get("awaiting") is not None):
        raise ValueError("an awaiting reason must be set exactly while the status is awaiting")
    policy = state.get("policy", {})
    missing = sorted(set(DEFAULT_POLICY) - set(policy)) + sorted(set(TRAINING_RUNS) - set(policy.get("labels", {})))
    if missing:
        raise ValueError(f"policy lacks {missing}")
    t2r = state.get("t2r")
    if not isinstance(t2r, Mapping) or set(t2r) != {"iter", "requests", "rounds"}:
        raise ValueError(f"the t2r round state must hold iter, requests and rounds, got {t2r!r}")
    unknown = sorted(set(state.get("runs", {})) - set(TRAINING_RUNS) - set(SIDE_RUNS))
    if unknown:
        raise ValueError(f"unknown runs {unknown}")


def load_state(path) -> dict:
    state = run_files.read_json(path)
    validate_state(state)
    return state


def save_state(path, state: Mapping):
    validate_state(state)
    return run_files.write_json(path, state)


def live_record(state: Mapping, run: str) -> Mapping | None:
    """The run's launch record; None without one or once ``resume`` cleared it (the run is then absent)."""
    record = state["runs"].get(run)
    return None if record is None or "cleared" in record else record


def ended_by_loop(state: Mapping, run: str) -> bool:
    """The loop stopped (or is stopping) the run, or ``resume`` cleared it: its end is never a crash."""
    return any(mark in state["runs"].get(run, {}) for mark in LOOP_END_MARKS)


def t2r_label(state: Mapping) -> str:
    """The RUN_LABEL of the current t2r round's training (policy label prefix + iteration)."""
    return loop_t2r.run_label(state["policy"]["labels"]["stage1_t2r"], state["t2r"]["iter"])


def runs_to_clear(state: Mapping, runs: Mapping[str, RunStatus]) -> tuple[str, ...]:
    """The records ``resume`` clears: dead runs that read as crashed, so their phase starts that step again."""
    return tuple(name for name in state["runs"]
                 if name in runs and runs[name].crashed and not runs[name].alive and not ended_by_loop(state, name))


def vlm_request_key(params: Mapping) -> str:
    return f"target_{params['attempt']:02d}" if params["kind"] == "target" else "requery"


def vlm_response_file(params: Mapping) -> str:
    """The generator's response file relative to the loop directory (loop_probe.LoopPaths.stage_dir, observe_dir)."""
    return f"stage_01/attempt_{params['attempt']:02d}/response.md" if params["kind"] == "target" else "stage_01/observe/response.md"


# -------------------------------------------------------------------- decide


def decide(state: Mapping, probe: Probe) -> Decision:
    """The one next action of the loop (§3 tick step 2)."""
    if state["status"] != "running":
        reason = (state.get("awaiting") or {}).get("reason")
        return _wait(f"loop is {state['status']}" + (f": {reason}" if reason else ""))
    hung = next((name for name in SIDE_RUNS if _hung(probe.runs.get(name, IDLE))), None)
    if hung is not None:
        idle = probe.runs[hung].idle_s
        return Decision("kill_stale", f"{hung} wrote its result {idle:.0f} s ago and is still alive", {"run": hung})
    phase = state["phase"]
    crashed = next((name for name in PHASE_RUNS[phase] if probe.runs.get(name, IDLE).crashed and not ended_by_loop(state, name)), None)
    if crashed is not None:
        return _pause(f"{crashed} crashed: {probe.runs[crashed].marker or 'ended without a result'}",
                      "read the end of its log; the loop never edits code - fix the cause, then `loop.py act resume`")
    decision = _DECIDERS[phase](state, probe)
    if phase == "stage1_t2r":
        decision = replace(decision, notes=decision.notes + _over_rack_note(probe, state["policy"]["bin_epochs"]))
    return decision


def _wait(reason: str) -> Decision:
    return Decision("wait", reason)


def _pause(reason: str, needs: str) -> Decision:
    return Decision("pause", reason, {"needs": needs})


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f} %"


def _hung(run: RunStatus) -> bool:
    return run.alive and run.finished and run.idle_s > STALE_AFTER_RESULT_S


def _launched(state: Mapping, run: str, key: str) -> bool:
    return (live_record(state, run) or {}).get("key") == key


def _vlm_generate(state: Mapping, reason: str, params: Mapping) -> Decision:
    requests = state.get("vlm_requests", {}).get(vlm_request_key(params), 0)
    if requests >= VLM_MAX_REQUESTS:
        return _pause(f"{requests} generator requests wrote no {vlm_response_file(params)}",
                      "check the iker-vlm-generator agent and its brief; `loop.py act resume` allows new requests")
    return Decision("vlm_generate", reason, params)


def _busy_side_run(probe: Probe) -> str | None:
    return next((name for name in SIDE_RUNS if probe.runs.get(name, IDLE).alive), None)


def _side_launch(state: Mapping, probe: Probe, action: str, params: Mapping) -> Decision:
    """A read-only Isaac run: one at a time, next to training only below the GPU memory limit (§8)."""
    busy = _busy_side_run(probe)
    if busy is not None:
        return _wait(f"{action} waits for the side run {busy}")
    limit = state["policy"]["side_gpu_limit_mib"]
    if probe.gpu_used_mib > limit:
        return _wait(f"{action} waits: GPU memory {probe.gpu_used_mib} MiB > {limit} MiB")
    return Decision(action, f"{action} {params['key']}", params)


def _over_rack_note(probe: Probe, width: int) -> tuple[str, ...]:
    end = last_epoch(probe.latched)
    latched, over = bin_mean(probe.latched, end, width), bin_mean(probe.over_rack, end, width)
    if latched is None or over is None or over <= 2.0 * latched:
        return ()
    return (f"over-rack raised {_pct(over)} > 2 x latched {_pct(latched)} over epochs {end - width + 1}-{end}",)


def _stage1_t2r(state: Mapping, probe: Probe) -> Decision:
    policy, t2r = state["policy"], state["t2r"]
    iteration, label, width = t2r["iter"], t2r_label(state), policy["bin_epochs"]
    if len(t2r["rounds"]) >= policy["t2r_max_rounds"]:
        return _pause(f"{len(t2r['rounds'])} t2r rounds ended without a handover",
                      "read stage1_t2r/iter_*/feedback.md; resume with a higher t2r_max_rounds, or stop")
    if not _launched(state, "stage1_t2r", label):
        return _t2r_prepare(state, probe, iteration, label)
    return _t2r_training(state, probe, iteration, label, width)


def _t2r_prepare(state: Mapping, probe: Probe, iteration: int, label: str) -> Decision:
    policy, files = state["policy"], probe.t2r
    if not files.prompt:
        return Decision("write_t2r_prompt", f"iter {iteration:02d} prompt", {"iter": iteration})
    if files.validation is None:
        if files.response:
            return Decision("ingest_reward", f"iter {iteration:02d} response", {"iter": iteration})
        requests = state["t2r"]["requests"]
        if requests >= policy["t2r_max_requests"]:
            return _pause(f"{requests} generator requests gave no valid reward for iter {iteration:02d} ({files.failed_attempts} failed validations)",
                          "read stage1_t2r/iter_NN/validation_attempt_*.json and generator.json; `loop.py act resume` allows new requests")
        return Decision("t2r_generate", f"iter {iteration:02d} request {requests + 1}", {"iter": iteration})
    if not files.validation.get("ok"):
        return _pause(f"iter {iteration:02d} validation.json is not ok", "a failed ingest moves its files aside; inspect the iteration folder")
    if not _launched(state, "t2r_smoke", label):
        return _side_launch(state, probe, "run_t2r_smoke", {"key": label, "iter": iteration})
    smoke = probe.runs.get("t2r_smoke", IDLE)
    if not smoke.finished:
        return _wait(f"t2r smoke of iter {iteration:02d} running")
    if not smoke.passed:
        return _pause(f"t2r smoke of iter {iteration:02d} failed: {smoke.marker}", "read its T2R SMOKE CHECK FAILED lines and smoke.json")
    busy = _busy_side_run(probe)
    if busy is not None:
        return _wait(f"launch_t2r waits for the side run {busy}")
    return Decision("launch_t2r", f"iter {iteration:02d}: {policy['t2r_round_epochs']} epochs x {policy['t2r_num_envs']} envs",
                    {"key": label, "iter": iteration})


def _t2r_training(state: Mapping, probe: Probe, iteration: int, label: str, width: int) -> Decision:
    policy = state["policy"]
    bank = state["bank"] or {"tried": [], "last_epoch": 0}
    record, harvest = live_record(state, "harvest"), probe.runs.get("harvest", IDLE)
    if record is not None and record["key"] not in bank["tried"]:
        if not harvest.finished:
            return _wait(f"harvesting iter {iteration:02d} ep {record['epoch']}")
        if harvest.passed:
            meta = probe.bank_meta or {}
            if meta.get("source") != LEARNED_BANK_SOURCE or meta.get("checkpoint") != record["key"]:
                return _pause("HARVEST passed but the grasp bank is not the learned bank of that checkpoint", "inspect grasp_bank.json")
            if meta.get("verified", 0) < policy["harvest_min"]:
                return _pause(f"HARVEST passed but the grasp bank holds {meta.get('verified', 0)} verified grasps < harvest_min {policy['harvest_min']}",
                              "inspect grasp_bank.json and the harvest log")
            return Decision("commit_bank", f"{meta['verified']} verified grasps at iter {iteration:02d} ep {record['epoch']}",
                            {"checkpoint": record["key"], "verified": meta["verified"], "epoch": record["epoch"], "iter": iteration, "label": label})
        return Decision("record_harvest_miss", f"iter {iteration:02d} ep {record['epoch']}: {harvest.marker}",
                        {"checkpoint": record["key"], "epoch": record["epoch"]})
    for epoch in sorted(probe.checkpoints, reverse=True):
        value = bin_mean(probe.success, epoch, width)
        if probe.checkpoints[epoch] not in bank["tried"] and value is not None and value >= policy["t2r_harvest_success"]:
            return _side_launch(state, probe, "run_harvest", {"key": probe.checkpoints[epoch], "checkpoint": probe.checkpoints[epoch], "epoch": epoch})
    last = last_epoch(probe.success)
    success, latched = bin_mean(probe.success, last, width), bin_mean(probe.latched, last, width)
    finished = probe.runs.get("stage1_t2r", IDLE).finished
    early = last >= policy["t2r_early_epoch"] and (success or 0.0) == 0.0 and (latched or 0.0) < policy["t2r_early_latched"]
    if finished or early:
        return Decision("end_round", f"iter {iteration:02d} {'finished' if finished else 'ended early'} at epoch {last}, latched {_pct(latched)}", {
            "iter": iteration, "label": label, "end_epoch": last, "ended": "round_epochs" if finished else "early",
            "success_max": max((v for _, v in probe.success), default=0.0), "latched_max": max((v for _, v in probe.latched), default=0.0),
        })
    return _wait(f"t2r iter {iteration:02d} epoch {last}, success {_pct(success)}, latched {_pct(latched)}")


def _vlm_target(state: Mapping, probe: Probe) -> Decision:
    if not probe.files.get("stage_prompt"):
        return Decision("write_prompt", "the stage-1 prompt, snapshot and keypoints are not in the stage directory")
    attempts = probe.attempts
    passed = next((attempt for attempt in attempts if attempt.gate_passed), None)
    if passed is not None:
        return Decision("commit_interaction", f"attempt {passed.index:02d} passed the gate", {"attempt": passed.index})
    if attempts and attempts[-1].gate_passed is None:
        return Decision("ingest", f"attempt {attempts[-1].index:02d} awaits the gate", {"attempt": attempts[-1].index})
    if len(attempts) >= state["policy"]["vlm_max_attempts"]:
        return _pause(f"{len(attempts)} VLM responses failed the gate",
                      "read attempt_*/gate.json; resume with a higher vlm_max_attempts, or stop")
    return _vlm_generate(state, f"attempt {len(attempts):02d}", {"kind": "target", "attempt": len(attempts)})


def _stage2_train(state: Mapping, probe: Probe) -> Decision:
    policy = state["policy"]
    if policy["stage2_env_smoke"]:
        if not _launched(state, "env_smoke", "vlm_target"):
            return _side_launch(state, probe, "run_env_smoke", {"key": "vlm_target"})
        smoke = probe.runs.get("env_smoke", IDLE)
        if not smoke.finished:
            return _wait("stage-2 environment smoke running")
        if not smoke.passed:
            return _pause(f"stage-2 environment smoke failed: {smoke.marker}", "read its SMOKE CHECK FAILED lines")
    if live_record(state, "stage2") is None:
        if any(probe.runs.get(name, IDLE).alive for name in ("stage1_t2r",)):
            return _wait("a stage-1 training is still alive")
        return Decision("launch_stage2", f"{policy['stage2_epochs']} epochs x {policy['stage2_num_envs']} envs", {"key": policy["labels"]["stage2"]})
    new = sorted(epoch for epoch in probe.evals if str(epoch) not in state["eval"])
    if new:
        summary = probe.evals[new[0]]
        return Decision("record_eval", f"ep {new[0]} success {_pct(summary['success_5cm_end'])}", {"epoch": new[0], "summary": dict(summary)})
    best = best_eval(state["eval"])
    if best is not None and best[1]["success"] >= policy["success_target"]:
        return Decision("advance", f"ep {best[0]} success {_pct(best[1]['success'])}; stop stage 2",
                        {"to": "observe_requery", "epoch": best[0], "stop": "stage2"})
    due = [epoch for epoch in eval_epochs(policy) if epoch in probe.checkpoints and str(epoch) not in state["eval"]]
    if due:
        epoch = due[0]
        if _launched(state, "eval", f"ep{epoch}"):
            if not probe.runs.get("eval", IDLE).finished:
                return _wait(f"evaluating ep {epoch}")
            return _pause(f"the evaluation of ep {epoch} printed its result but wrote no summary", "inspect the eval log and output path")
        return _side_launch(state, probe, "run_eval", {"key": f"ep{epoch}", "epoch": epoch, "checkpoint": probe.checkpoints[epoch]})
    if not probe.runs.get("stage2", IDLE).alive and len(state["eval"]) == len(eval_epochs(policy)):
        return _pause(f"stage 2 ended with best success {_pct(best[1]['success']) if best else 'n/a'} < {_pct(policy['success_target'])}",
                      "decide on stage 2: train longer, query a new target, or stop")
    return _wait(f"stage 2 training; evaluated epochs {sorted(int(k) for k in state['eval'])}")


def _observe_requery(state: Mapping, probe: Probe) -> Decision:
    chosen = best_eval(state["eval"])
    if chosen is None:
        return _pause("observe_requery has no stage-2 evaluation to pick a checkpoint from", "inspect LOOP_STATE.json eval")
    epoch, checkpoint = chosen[0], chosen[1]["checkpoint"]
    if probe.final_rows is None:
        if _launched(state, "observe_rollout", checkpoint):
            if not probe.runs.get("observe_rollout", IDLE).finished:
                return _wait("noise-free observation rollout running")
            return _pause("the observation rollout finished without final states", "inspect the observe_rollout log")
        return _side_launch(state, probe, "run_observe_rollout", {"key": checkpoint, "checkpoint": checkpoint, "epoch": epoch})
    env = pick_observe_env(probe.final_rows)
    if env is None:
        return _pause(f"no env succeeded in the noise-free rollout of ep {epoch}", "watch a play video of the checkpoint; decide on stage 2")
    files, rendered, render = probe.files, _launched(state, "observe_render", f"env{env}"), probe.runs.get("observe_render", IDLE)
    if rendered and not render.finished:
        return _wait(f"rendering env {env}")
    if rendered and render.passed is False:  # its files may exist: the render writes them before judging them
        return _pause(f"the observation render of env {env} failed its checks: {'; '.join(render.failed_checks) or render.marker}",
                      "read its OBSERVE CHECK FAILED lines; the loop does not requery a scene that failed its checks")
    if not files.get("observe_snapshot"):
        if rendered:
            return _pause(f"the observation render finished without its files: {render.marker}", "inspect the observe_render log")
        return _side_launch(state, probe, "run_observe_render", {"key": f"env{env}", "env": env})
    if not files.get("requery_prompt"):
        return Decision("write_requery_prompt", "multi-step prompt with the stage-1 code as history")
    if not files.get("requery_response"):
        return _vlm_generate(state, "requery on the executed scene", {"kind": "requery"})
    if probe.requery is None:
        return Decision("parse_requery", "the requery response is not parsed yet")
    if not probe.requery["done"]:
        return _pause(f"the VLM did not report done: {probe.requery['detail']}", "a further stage is outside this loop; decide by hand")
    if not files.get("video"):
        if _launched(state, "video", checkpoint):
            if probe.runs.get("video", IDLE).alive:
                return _wait("recording the play video")
            if files.get("video_raw"):
                return Decision("store_video", "copy the play video to our_source", {"checkpoint": checkpoint, "epoch": epoch})
            return _pause("the play video run ended without a file", "inspect the video log")
        return _side_launch(state, probe, "run_video", {"key": checkpoint, "checkpoint": checkpoint, "epoch": epoch})
    return Decision("advance", "the VLM reported done and the video is stored", {"to": "completion_review"})


def _completion_review(state: Mapping, probe: Probe) -> Decision:
    return _pause("stage 2 met the success target and the VLM reported done",
                  "watch the video named in stage_01/video.txt; approve with `loop.py act approve`")


_DECIDERS = {
    "stage1_t2r": _stage1_t2r, "vlm_target": _vlm_target,
    "stage2_train": _stage2_train, "observe_requery": _observe_requery, "completion_review": _completion_review,
    "done": lambda state, probe: _wait("done"),
}


# --------------------------------------------------------------------- apply


def apply(state: Mapping, decision: Decision, now: str, result: Mapping | None = None) -> dict:
    """The state after ``decision`` was carried out; ``result`` is what the CLI's side effect returned
    (``run``: a launch record, ``stopped_runs``: runs it stopped by PID, ``passed``: an ingest verdict)."""
    action, params, outcome = decision.action, dict(decision.params), dict(result or {})
    known = action in LAUNCH_RUNS or action in _APPLIERS or action in FILE_ACTIONS
    if not known:
        raise ValueError(f"action {action!r} is not recorded by the loop state")
    new = copy.deepcopy(dict(state))
    new["updated"] = now
    if action in LAUNCH_RUNS:
        new["runs"][LAUNCH_RUNS[action]] = {**outcome.get("run", {}), "key": params["key"], "epoch": params.get("epoch")}
    for name in outcome.get("stopped_runs", ()):
        if name in new["runs"]:
            new["runs"][name] = {**new["runs"][name], "stopped": now}
    if action in _APPLIERS:
        _APPLIERS[action](new, decision, params, outcome)
    validate_state(new)
    return new


def _apply_pause(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    new["status"], new["awaiting"] = "awaiting", {"reason": decision.reason, "needs": params["needs"], "phase": new["phase"]}


def _apply_advance(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    new["phase"] = params["to"]


def _apply_record_harvest_miss(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    bank = new["bank"] or {"tried": [], "last_epoch": 0}
    bank["tried"].append(params["checkpoint"])
    bank["last_epoch"] = max(bank["last_epoch"], params["epoch"])
    new["bank"] = bank


def _apply_commit_bank(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    bank = new["bank"] or {"tried": [], "last_epoch": 0}
    new["bank"] = {**bank, "checkpoint": params["checkpoint"], "verified": params["verified"], "path": outcome.get("path")}
    new["t2r"]["rounds"].append({"iter": params["iter"], "label": params["label"], "end_epoch": params["epoch"], "ended": "handover",
                                 "run_dir": outcome.get("run_dir"), "reward_sha256": outcome.get("reward_sha256")})
    new["phase"] = "vlm_target"


def _apply_end_round(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    t2r = new["t2r"]
    t2r["rounds"].append({"iter": params["iter"], "label": params["label"], "end_epoch": params["end_epoch"], "ended": params["ended"],
                          "success_max": params["success_max"], "latched_max": params["latched_max"],
                          "run_dir": outcome.get("run_dir"), "reward_sha256": outcome.get("reward_sha256")})
    t2r["iter"], t2r["requests"] = params["iter"] + 1, 0


def _apply_t2r_generate(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    new["t2r"]["requests"] += 1


def _apply_ingest(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    new["attempts"]["1"] = max(new["attempts"].get("1", 0), params["attempt"] + 1)


def _apply_commit_interaction(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    new["phase"] = "stage2_train"


def _apply_record_eval(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    summary = params["summary"]
    new["eval"][str(params["epoch"])] = {
        "success": summary["success_5cm_end"], "checkpoint": summary["checkpoint"], "dropped": summary.get("dropped"),
    }


def _apply_approve(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    if new["phase"] != "completion_review":
        raise ValueError(f"approve is for completion_review; the loop is at {new['phase']}")
    new["phase"], new["status"], new["awaiting"] = "done", "done", None


def _apply_resume(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    updates = copy.deepcopy(dict(params.get("policy", {})))
    unknown = sorted(set(updates) - set(DEFAULT_POLICY))
    if unknown:
        raise ValueError(f"unknown policy keys {unknown}")
    labels = {**new["policy"]["labels"], **updates.pop("labels", {})}
    new["policy"] = {**new["policy"], **updates, "labels": labels}
    new["status"], new["awaiting"] = "running", None
    for name in outcome.get("cleared_runs", ()):  # dead runs that read as crashed: their phase starts that step again
        if name in new["runs"]:
            new["runs"][name] = {**new["runs"][name], "cleared": new["updated"]}
    new["vlm_requests"] = {}
    new["t2r"] = {**new["t2r"], "requests": 0}


def _apply_vlm_generate(new: dict, decision: Decision, params: Mapping, outcome: Mapping) -> None:
    requests, key = dict(new.get("vlm_requests", {})), vlm_request_key(params)
    new["vlm_requests"] = {**requests, key: requests.get(key, 0) + 1}


_APPLIERS = {
    "pause": _apply_pause, "advance": _apply_advance,
    "record_harvest_miss": _apply_record_harvest_miss, "commit_bank": _apply_commit_bank, "ingest": _apply_ingest,
    "commit_interaction": _apply_commit_interaction, "record_eval": _apply_record_eval, "approve": _apply_approve,
    "resume": _apply_resume, "vlm_generate": _apply_vlm_generate,
    "end_round": _apply_end_round, "t2r_generate": _apply_t2r_generate,
}
