#!/usr/bin/env python3
"""IKER auto-loop CLI (design docs/superpowers/specs/2026-09-14-iker-auto-loop-design.md §3, §8).

System python3; Isaac runs only in child processes started here, identified by RUN_LABEL and stopped by PID.

    python3 scripts/iker/loop.py [--state-dir DIR] init [--track T] [--phase P] [--policy JSON] [--adopt RUN ...]
    python3 scripts/iker/loop.py [--state-dir DIR] status
    python3 scripts/iker/loop.py [--state-dir DIR] [--no-commit] act ACTION [--policy JSON]

``act`` decides again from a fresh probe and refuses an action that is no longer the decision. ``approve`` and
``resume`` are the user's words. scripts/iker/LOOP_PROMPT.md is the tick procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "openarm"))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from parse_tfevents import load_tfevents  # noqa: E402

from openarm.agnostic.modules.iker import loop_probe as lp  # noqa: E402
from openarm.agnostic.modules.iker import loop_state as ls  # noqa: E402
from openarm.agnostic.modules.iker import loop_t2r, loop_vlm, prompts, run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout  # noqa: E402

CONFIG_INDEX = 0
ISAAC_PYTHON = ROOT.parent / "IsaacLab" / "_isaac_sim" / "python.sh"
TRAIN_SCRIPT = "scripts/reinforcement_learning/rl_games/train.py"
PLAY_SCRIPT = "scripts/reinforcement_learning/rl_games/play.py"
T2R_SCRIPT = "scripts/iker/t2r_reward.py"
STAGE1_T2R_TASK, STAGE2_TASK = "open-sens_l_iker_shoe_grasp_t2r", "open-sens_l_iker_shoe"
VIDEO_ENVS, VIDEO_STEPS = 16, 200
VIDEO_DIR = ROOT.parent / "our_source"
STOP_TIMEOUT_S, PID_WAIT_S = 300.0, 120.0
SESSION_URL = "https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4"
COMMIT_ACTIONS = (
    "advance", "launch_t2r", "end_round", "commit_bank", "commit_interaction", "record_eval", "store_video", "pause",
    "approve", "resume",
)
OBSERVE_FILES = ("prompt.md", "response.md", "generator.json", "requery.json", "snapshot.png", "snapshot_raw.png", "keypoints.json",
                 "state.json", "rollout_summary.json")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return lp.parse_gpu_used_mib(out.stdout)


def head_commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


# ------------------------------------------------------------------ processes


def launch(label: str, argv: list[str], log: Path, note: str) -> dict:
    """Start a detached Isaac child and return its launch record once a process with its RUN_LABEL is visible."""
    if lp.label_pids(label):
        raise RuntimeError(f"a process already carries RUN_LABEL={label}")
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TERM": "xterm", "OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONPATH": str(ROOT / "source" / "openarm"),
           "RUN_LABEL": label, "NOTE": note}
    started = time.time()
    with log.open("wb") as out:
        subprocess.Popen([str(ISAAC_PYTHON), *argv], cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out,
                         stderr=subprocess.STDOUT, start_new_session=True)
    while not (pids := lp.label_pids(label)):
        if time.time() > started + PID_WAIT_S:
            raise RuntimeError(f"no Isaac process with RUN_LABEL={label} after {PID_WAIT_S:.0f} s; see {log}")
        time.sleep(2.0)
    return {"label": label, "log": str(log), "pid": pids[0], "started": datetime.fromtimestamp(started).astimezone().isoformat(timespec="seconds"),
            "started_s": started, "commit": head_commit(), "argv": argv}


def stop(label: str, force: bool = False) -> list[int]:
    """Stop the Isaac processes of ``label`` by PID (never by a name pattern); SIGKILL only for a hung run (§8)."""
    pids = lp.label_pids(label)
    for pid in pids:
        os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
    deadline = time.time() + STOP_TIMEOUT_S
    while any(Path(f"/proc/{pid}").exists() for pid in pids):
        if time.time() > deadline:
            raise RuntimeError(f"RUN_LABEL={label} PIDs {pids} are alive {STOP_TIMEOUT_S:.0f} s after the signal")
        time.sleep(2.0)
    return pids


def commit(paths: list[Path], message: str, session_url: str) -> str | None:
    """Commit exactly ``paths`` (no -A); None when nothing changed."""
    rel = sorted({str(Path(p).resolve().relative_to(ROOT)) for p in paths if Path(p).exists()})
    subprocess.run(["git", "-C", str(ROOT), "add", "--", *rel], check=True)
    if subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet", "--", *rel]).returncode == 0:
        return None
    body = f"{message}\n\nCo-Authored-By: Claude Opus 5 <noreply@anthropic.com>\nClaude-Session: {session_url}"
    subprocess.run(["git", "-C", str(ROOT), "commit", "-q", "-m", body, "--", *rel], check=True)
    return head_commit()


# ------------------------------------------------------------------ executors


def side_label(state: dict, run: str) -> str:
    return f"{state['track']}_{run}"


def stop_run(state: dict, paths: lp.LoopPaths, run: str) -> list[str]:
    """Stop a training run by PID after marking its record ``stopping`` in the state file, so a probe meanwhile (or after a
    failed act) never reads the dying run as crashed; returns the ``stopped_runs`` for apply."""
    record = state["runs"].get(run)
    if record is None:
        return []
    marked = {**state, "runs": {**state["runs"], run: {**record, "stopping": now_iso()}}}
    ls.save_state(paths.state_file, marked)
    stop(record["label"])
    return [run]


def stage2_inputs(state: dict, paths: lp.LoopPaths) -> list[str]:
    bank = state["policy"]["grasp_bank_path"]
    return ["--interaction", str(paths.interaction_file), *(["--grasp-bank", bank] if bank else [])]


def stage2_overrides(state: dict, paths: lp.LoopPaths) -> list[str]:
    """Hydra overrides of the stage-2 inputs; quoted, since a path segment may start with '-'."""
    bank = state["policy"]["grasp_bank_path"]
    return [f"env.interaction_path='{paths.interaction_file}'", *([f"env.grasp_bank_path='{bank}'"] if bank else [])]


def _t2r_env() -> dict:
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "source" / "openarm"), str(ROOT / "scripts" / "tools")])}


def _round_outputs(state: dict, paths: lp.LoopPaths, iteration: int) -> dict:
    record = state["runs"].get("stage1_t2r")
    run_dir = lp.find_run_dir(paths.task_dir("stage1_t2r"), record["label"], record.get("started_s", 0.0)) if record else None
    code = paths.t2r_iter_dir(iteration) / loop_t2r.CODE
    return {"run_dir": str(run_dir) if run_dir else None, "reward_sha256": hashlib.sha256(code.read_bytes()).hexdigest() if code.is_file() else None}


def write_t2r_prompt(state, paths, decision):
    iteration = decision.params["iter"]
    target = paths.t2r_iter_dir(iteration)
    if iteration == 0:
        argv = [sys.executable, T2R_SCRIPT, "render", "--iter-dir", str(target)]
    else:
        previous = state["t2r"]["rounds"][-1]
        events = sorted((Path(previous["run_dir"]) / "summaries").glob("events.out.tfevents.*")) if previous.get("run_dir") else []
        if not events:
            raise RuntimeError(f"round iter {previous['iter']:02d} left no TFEvents to reflect on (run_dir {previous.get('run_dir')})")
        argv = [sys.executable, T2R_SCRIPT, "reflect", "--prev-dir", str(paths.t2r_iter_dir(previous["iter"])), "--next-dir", str(target),
                "--events", str(events[-1])]
    done = subprocess.run(argv, cwd=ROOT, env=_t2r_env(), capture_output=True, text=True)
    if done.returncode != 0 or not (target / loop_t2r.PROMPT).is_file():
        raise RuntimeError(f"t2r_reward.py {argv[2]} failed (exit {done.returncode}):\n{done.stdout}{done.stderr}")
    return {"prompt": str(target / loop_t2r.PROMPT)}


def t2r_request(paths: lp.LoopPaths, decision: ls.Decision) -> dict:
    folder = paths.t2r_iter_dir(decision.params["iter"])
    prompt, response = folder / loop_t2r.PROMPT, folder / loop_t2r.RESPONSE
    return {"prompt": str(prompt), "response": str(response), "brief": loop_t2r.generator_brief(prompt, response)}


def t2r_generate(state, paths, decision):
    """Record one generator request: generator.json next to the response; the tick then dispatches the agent with ``t2r.brief``."""
    request = t2r_request(paths, decision)
    Path(request["response"]).parent.mkdir(parents=True, exist_ok=True)
    record = run_files.write_json(Path(request["response"]).with_name(loop_t2r.GENERATOR),
                                  loop_t2r.generator_record(request, state["t2r"]["requests"] + 1, now_iso()))
    return {"t2r": request, "generator": str(record)}


def ingest_reward(state, paths, decision):
    """Validate the response with Isaac python (the cuda dry run needs it); a failed attempt is moved aside (spec §14)."""
    folder = paths.t2r_iter_dir(decision.params["iter"])
    env = {**_t2r_env(), "TERM": "xterm", "OMNI_KIT_ACCEPT_EULA": "YES"}
    done = subprocess.run([str(ISAAC_PYTHON), T2R_SCRIPT, "ingest", "--iter-dir", str(folder)], cwd=ROOT, env=env, capture_output=True, text=True)
    validation = folder / loop_t2r.VALIDATION
    if not validation.is_file():
        raise RuntimeError(f"t2r ingest wrote no {validation} (exit {done.returncode}):\n{done.stdout[-2000:]}{done.stderr[-2000:]}")
    report = json.loads(validation.read_text(encoding="utf-8"))
    if report["ok"]:
        return {"passed": True}
    attempt = len(list(folder.glob("validation_attempt_*.json"))) + 1
    for name, moved in loop_t2r.failed_attempt_names(attempt).items():
        if (folder / name).exists():
            (folder / name).rename(folder / moved)
    return {"passed": False, "errors": report["errors"], "attempt": attempt}


def run_t2r_smoke(state, paths, decision):
    policy, iteration, digest = state["policy"], decision.params["iter"], decision.params["digest"]
    folder = paths.t2r_iter_dir(iteration)
    argv = ["scripts/iker/t2r_smoke.py", "--mode", "round", "--reward-code", str(folder / loop_t2r.CODE), "--out", str(folder / loop_t2r.SMOKE),
            "--num-envs", str(policy["t2r_smoke_envs"]), "--steps", str(policy["t2r_smoke_steps"]), "--config-index", str(CONFIG_INDEX), "--headless"]
    # finding 1: the smoke is keyed on the reward digest, so a regenerated reward in the same iter gets its own log
    log = paths.side_log("t2r_smoke", f"iter{iteration:02d}_{digest[:8]}")
    return {"run": launch(side_label(state, "t2r_smoke"), argv, log, f"IKER loop t2r smoke iter {iteration:02d} reward {digest[:8]}")}


def record_smoke_miss(state, paths, decision):
    """A round smoke that ran and did not pass: move the round's files aside like a failed ingest, so the next decision
    sees no response/validation/smoke and asks the generator again instead of pausing (finding 1, spec §14)."""
    folder = paths.t2r_iter_dir(decision.params["iter"])
    attempt = len(list(folder.glob("validation_attempt_*.json"))) + 1
    for name, moved in loop_t2r.failed_attempt_names(attempt).items():
        if (folder / name).exists():
            (folder / name).rename(folder / moved)
    return {"attempt": attempt, "digest": decision.params["digest"]}


def launch_t2r(state, paths, decision):
    policy, label, iteration = state["policy"], decision.params["key"], decision.params["iter"]
    folder = paths.t2r_iter_dir(iteration)
    argv = [TRAIN_SCRIPT, "--task", STAGE1_T2R_TASK, "--num_envs", str(policy["t2r_num_envs"]), "--max_iterations", str(policy["t2r_round_epochs"]),
            "--headless", f"env.reward_code_path='{folder / loop_t2r.CODE}'"]
    if policy["minibatch_size"]:
        argv.append(f"agent.params.config.minibatch_size={policy['minibatch_size']}")
    run = launch(label, argv, paths.train_log(label), f"IKER 1단계 t2r iter {iteration:02d} 새 학습, 자동 루프")
    files = sorted(path for path in folder.iterdir() if path.is_file())
    if iteration > 0:
        feedback = paths.t2r_iter_dir(iteration - 1) / loop_t2r.FEEDBACK
        files += [feedback] if feedback.is_file() else []
    return {"run": run, "commit_paths": files, "message": f"iker(loop): 1단계 t2r iter {iteration:02d} 보상·검증·스모크 — {label} 기동"}


def run_harvest(state, paths, decision):
    policy, epoch, iteration = state["policy"], decision.params["epoch"], state["t2r"]["iter"]
    argv = ["scripts/iker/harvest_grasp_bank.py", "--checkpoint", decision.params["checkpoint"], "--config-index", str(CONFIG_INDEX),
            "--min-entries", str(policy["harvest_min"]), "--out", str(paths.harvest_bank_file), "--t2r-iter", str(iteration),
            "--reward-code", str(paths.t2r_iter_dir(iteration) / loop_t2r.CODE), "--headless"]
    log = paths.side_log("harvest", f"iter{iteration:02d}_ep{epoch}")
    return {"run": launch(side_label(state, "harvest"), argv, log, f"IKER loop harvest t2r iter {iteration:02d} ep {epoch}")}


def advance(state, paths, decision):
    run = decision.params.get("stop")
    return {"stopped_runs": stop_run(state, paths, run)} if run else {}


def commit_bank(state, paths, decision):
    outputs = _round_outputs(state, paths, decision.params["iter"])
    stopped = stop_run(state, paths, "stage1_t2r")
    return {"path": str(paths.harvest_bank_file), "stopped_runs": stopped, **outputs, "commit_paths": [paths.harvest_bank_file],
            "message": f"iker(loop): 학습 파지 뱅크 {decision.params['verified']} 개 — t2r iter {decision.params['iter']:02d} "
                       f"{Path(decision.params['checkpoint']).name}"}


def end_round(state, paths, decision):
    iteration = decision.params["iter"]
    outputs = _round_outputs(state, paths, iteration)
    record = state["runs"].get("stage1_t2r")
    stopped = stop_run(state, paths, "stage1_t2r") if record is not None and lp.label_pids(record["label"]) else []
    folder = paths.t2r_iter_dir(iteration)
    files = sorted(path for path in folder.iterdir() if path.is_file()) if folder.is_dir() else []
    return {"stopped_runs": stopped, **outputs, "commit_paths": files,
            "message": f"iker(loop): 1단계 t2r iter {iteration:02d} 라운드 끝({decision.params['ended']}) — 성공 최대 {decision.params['success_max']:.3f}"}


def write_prompt(state, paths, decision):
    paths.stage_dir.mkdir(parents=True, exist_ok=True)
    (paths.stage_dir / "prompt.md").write_text(loop_vlm.target_prompt(layout.TASK_INSTRUCTION), encoding="utf-8")
    for name in ("snapshot.png", "keypoints.json"):
        shutil.copyfile(paths.config_dir / name, paths.stage_dir / name)
    return {}


def vlm_generate(state, paths, decision):
    """Record one generator request: generator.json next to the response; the tick then dispatches the agent with ``vlm.brief``."""
    request = vlm_request(paths, decision)
    count = state.get("vlm_requests", {}).get(ls.vlm_request_key(decision.params), 0) + 1
    record = run_files.write_json(Path(request["response"]).with_name("generator.json"), loop_vlm.generator_record(request, count, now_iso()))
    return {"vlm": request, "generator": str(record)}


def ingest(state, paths, decision):
    attempt = decision.params["attempt"]
    done = subprocess.run([sys.executable, "scripts/iker/ingest.py", "--run-dir", str(paths.stage_dir), "--attempt", str(attempt)], cwd=ROOT,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "source" / "openarm")}, capture_output=True, text=True)
    if not (paths.stage_dir / f"attempt_{attempt:02d}" / "gate.json").is_file():
        raise RuntimeError(f"ingest.py wrote no gate verdict (exit {done.returncode}):\n{done.stdout}{done.stderr}")
    return {"passed": done.returncode == 0, "output": done.stdout.strip()}


def commit_interaction(state, paths, decision):
    stage = paths.stage_dir
    files = [stage / "prompt.md", stage / "snapshot.png", stage / "keypoints.json", paths.interaction_file, *sorted(stage.glob("attempt_*/*"))]
    return {"commit_paths": files, "message": f"iker(loop): 1단계 VLM 목표 — attempt {decision.params['attempt']:02d} 게이트 통과"}


def run_env_smoke(state, paths, decision):
    argv = ["scripts/iker/env_smoke.py", "--config-index", str(CONFIG_INDEX), *stage2_inputs(state, paths), "--headless"]
    return {"run": launch(side_label(state, "env_smoke"), argv, paths.side_log("env_smoke", "vlm_target"), "IKER loop stage-2 env smoke")}


def launch_stage2(state, paths, decision):
    policy, label = state["policy"], state["policy"]["labels"]["stage2"]
    argv = [TRAIN_SCRIPT, "--task", STAGE2_TASK, "--num_envs", str(policy["stage2_num_envs"]), "--max_iterations", str(policy["stage2_epochs"]),
            "--headless", "env.interaction_source=vlm", *stage2_overrides(state, paths)]
    if policy["minibatch_size"]:
        argv.append(f"agent.params.config.minibatch_size={policy['minibatch_size']}")
    return {"run": launch(label, argv, paths.train_log(label), "IKER 2단계 VLM 목표·학습 파지 뱅크, 자동 루프")}


def run_eval(state, paths, decision):
    epoch = decision.params["epoch"]
    argv = ["scripts/iker/eval_iker.py", "--checkpoint", decision.params["checkpoint"], "--config-index", str(CONFIG_INDEX), *stage2_inputs(state, paths),
            "--out", str(paths.eval_file(epoch)), "--rows-out", str(paths.eval_rows_file(epoch)), "--headless"]
    return {"run": launch(side_label(state, "eval"), argv, paths.side_log("eval", f"ep{epoch}"), f"IKER loop eval ep {epoch}")}


def record_eval(state, paths, decision):
    epoch, success = decision.params["epoch"], decision.params["summary"]["success_5cm_end"]
    return {"commit_paths": [paths.eval_file(epoch)], "message": f"iker(loop): 2단계 ep {epoch} 평가 — 성공 {success}"}


def run_observe_rollout(state, paths, decision):
    argv = ["scripts/iker/eval_iker.py", "--checkpoint", decision.params["checkpoint"], "--config-index", str(CONFIG_INDEX), "--no-noise",
            *stage2_inputs(state, paths), "--out", str(paths.observe_dir / "rollout_summary.json"), "--final-states", str(paths.final_states_file),
            "--headless"]
    log = paths.side_log("observe_rollout", f"ep{decision.params['epoch']}")
    return {"run": launch(side_label(state, "observe_rollout"), argv, log, "IKER loop noise-free rollout")}


def run_observe_render(state, paths, decision):
    env = decision.params["env"]
    argv = ["scripts/iker/observe.py", "--config-index", str(CONFIG_INDEX), "--states", str(paths.final_states_file), "--env", str(env),
            "--out-dir", str(paths.observe_dir), "--headless"]
    return {"run": launch(side_label(state, "observe_render"), argv, paths.side_log("observe_render", f"env{env}"), "IKER loop observation render")}


def write_requery_prompt(state, paths, decision):
    passed = next(attempt for attempt in lp.attempts(paths.stage_dir) if attempt.gate_passed)
    response = (paths.stage_dir / f"attempt_{passed.index:02d}" / "response.md").read_text(encoding="utf-8")
    paths.observe_dir.mkdir(parents=True, exist_ok=True)
    (paths.observe_dir / "prompt.md").write_text(loop_vlm.requery_prompt(layout.TASK_INSTRUCTION, response), encoding="utf-8")
    return {}


def parse_requery(state, paths, decision):
    keypoints = run_files.vlm_keypoints(run_files.read_json(paths.observe_dir / "keypoints.json"))
    result = loop_vlm.requery_result((paths.observe_dir / "response.md").read_text(encoding="utf-8"), keypoints)
    run_files.write_json(paths.observe_dir / "requery.json", {"schema": run_files.SCHEMA_VERSION, **result})
    return result


def run_video(state, paths, decision):
    argv = [PLAY_SCRIPT, "--task", f"{STAGE2_TASK}-play", "--num_envs", str(VIDEO_ENVS), "--checkpoint", decision.params["checkpoint"],
            "--video", "--video_length", str(VIDEO_STEPS), "--headless", *stage2_overrides(state, paths)]
    return {"run": launch(side_label(state, "video"), argv, paths.side_log("video", f"ep{decision.params['epoch']}"), "IKER loop play video")}


def store_video(state, paths, decision):
    source = lp.stage2_video(state, paths)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    target = VIDEO_DIR / f"iker_shoe_c{CONFIG_INDEX:02d}_s1_ep{decision.params['epoch']:04d}_{datetime.now():%m%d_%H%M}.mp4"
    shutil.copyfile(source, target)
    paths.video_note.write_text(f"{target}\n", encoding="utf-8")
    files = [paths.video_note, *(paths.observe_dir / name for name in OBSERVE_FILES)]
    return {"video": str(target), "commit_paths": files, "message": f"iker(loop): 재질의 done, 2단계 영상 {target.name}"}


def kill_stale(state, paths, decision):
    run = decision.params["run"]
    return {"stopped_runs": [run], "pids": stop(state["runs"][run]["label"], force=True)}


def pause(state, paths, decision):
    return {"commit_paths": sorted(path for path in paths.state_dir.rglob("*") if path.is_file())}


EXECUTORS = {
    "write_t2r_prompt": write_t2r_prompt, "t2r_generate": t2r_generate, "ingest_reward": ingest_reward, "run_t2r_smoke": run_t2r_smoke,
    "record_smoke_miss": record_smoke_miss,
    "launch_t2r": launch_t2r, "end_round": end_round, "run_harvest": run_harvest,
    "advance": advance, "commit_bank": commit_bank, "write_prompt": write_prompt, "vlm_generate": vlm_generate, "ingest": ingest,
    "commit_interaction": commit_interaction,
    "run_env_smoke": run_env_smoke, "launch_stage2": launch_stage2, "run_eval": run_eval, "record_eval": record_eval,
    "run_observe_rollout": run_observe_rollout, "run_observe_render": run_observe_render, "write_requery_prompt": write_requery_prompt,
    "parse_requery": parse_requery, "run_video": run_video, "store_video": store_video, "kill_stale": kill_stale, "pause": pause,
}


# ------------------------------------------------------------------- commands


def vlm_request(paths: lp.LoopPaths, decision: ls.Decision) -> dict:
    """Paths and the verbatim brief for the fresh generator agent (§6)."""
    response = paths.state_dir / ls.vlm_response_file(decision.params)
    if decision.params["kind"] == "target":
        folder, images = paths.stage_dir, {prompts.IMAGE_MARKER: str(paths.stage_dir / "snapshot.png")}
    else:
        folder = paths.observe_dir
        images = {prompts.IMAGE_MARKER: str(paths.observe_dir / "snapshot.png"), loop_vlm.STAGE_IMAGE_MARKER: str(paths.stage_dir / "snapshot.png")}
    prompt = folder / "prompt.md"
    return {"prompt": str(prompt), "response": str(response), "images": images, "brief": loop_vlm.generator_brief(prompt, response, images)}


def summary_line(state: dict, probe: ls.Probe, decision: ls.Decision) -> str:
    parts = [f"phase {state['phase']}", f"GPU {probe.gpu_used_mib} MiB"]
    if state["phase"] == "stage1_t2r":
        parts.insert(1, f"t2r iter {state['t2r']['iter']:02d}")
    if probe.latched:
        end = ls.last_epoch(probe.latched)
        parts.append(f"epoch {end} latched {100.0 * (ls.bin_mean(probe.latched, end, state['policy']['bin_epochs']) or 0.0):.2f} %")
    if probe.success:
        parts.append(f"success {100.0 * (ls.bin_mean(probe.success, ls.last_epoch(probe.success), state['policy']['bin_epochs']) or 0.0):.2f} %")
    if probe.checkpoints:
        parts.append(f"last checkpoint ep {max(probe.checkpoints)}")
    if state["eval"]:
        parts.append("eval " + ", ".join(f"ep{k} {100.0 * v['success']:.1f} %" for k, v in sorted(state["eval"].items(), key=lambda kv: int(kv[0]))))
    return " · ".join(parts) + f" -> {decision.action}: {decision.reason}"


def append_history(paths: lp.LoopPaths, record: dict) -> None:
    paths.history_file.parent.mkdir(parents=True, exist_ok=True)
    with paths.history_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def adopt(state: dict, paths: lp.LoopPaths, run: str) -> dict:
    """The launch record of a training run started before the loop, found by its policy label."""
    if run not in ls.TRAINING_RUNS:
        raise SystemExit(f"only training runs can be adopted, not {run!r}")
    if run == "stage1_t2r":
        raise SystemExit("t2r rounds are launched by the loop, not adopted")
    label = state["policy"]["labels"][run]
    run_dir, log = paths.task_dir(run) / label, paths.train_log(label)
    if not run_dir.is_dir() or not log.is_file():
        raise SystemExit(f"cannot adopt {run}: {run_dir} or {log} is missing")
    started, pids = run_dir.stat().st_mtime, lp.label_pids(label)
    return {"label": label, "log": str(log), "pid": pids[0] if pids else None, "started_s": started,
            "started": datetime.fromtimestamp(started).astimezone().isoformat(timespec="seconds"), "commit": None, "key": label, "adopted": now_iso()}


def state_dir_of(args) -> Path | None:
    return Path(args.state_dir).resolve() if args.state_dir else None


def load(args) -> tuple[dict, lp.LoopPaths]:
    state = ls.load_state(lp.LoopPaths.of(ROOT, CONFIG_INDEX, ls.TRACK, state_dir_of(args)).state_file)
    return state, lp.LoopPaths.of(ROOT, CONFIG_INDEX, state["track"], state_dir_of(args))


def cmd_init(args) -> int:
    paths = lp.LoopPaths.of(ROOT, CONFIG_INDEX, args.track, state_dir_of(args))
    if paths.state_file.exists():
        raise SystemExit(f"{paths.state_file} exists; the loop continues from it")
    state = ls.new_state(now_iso(), track=args.track, phase=args.phase, policy=json.loads(args.policy))
    state["runs"] = {run: adopt(state, paths, run) for run in args.adopt}
    ls.save_state(paths.state_file, state)
    append_history(paths, {"time": state["updated"], "phase": state["phase"], "action": "init", "adopted": list(args.adopt)})
    print(json.dumps({"state": str(paths.state_file), "phase": state["phase"], "runs": state["runs"]}, ensure_ascii=False, indent=1))
    return 0


def cmd_status(args) -> int:
    state, paths = load(args)
    probe = lp.collect(state, paths, now_s=time.time(), gpu_used_mib=gpu_used_mib(), load_events=load_tfevents)
    decision = ls.decide(state, probe)
    out = {"phase": state["phase"], "status": state["status"], "awaiting": state["awaiting"], "action": decision.action,
           "reason": decision.reason, "params": decision.params, "notes": list(decision.notes), "summary": summary_line(state, probe, decision)}
    if decision.action == "vlm_generate":
        out["vlm"] = vlm_request(paths, decision)
    if decision.action == "t2r_generate":
        out["t2r"] = t2r_request(paths, decision)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def cmd_act(args) -> int:
    state, paths = load(args)
    if args.action in ls.MANUAL_ACTIONS:
        params = {"policy": json.loads(args.policy)} if args.action == "resume" else {}
        decision = ls.Decision(args.action, f"the user asked to {args.action}", params)
        result = {}
        if args.action == "resume":  # dead runs that read as crashed are cleared, so their phase starts that step again
            result = {"cleared_runs": list(ls.runs_to_clear(state, lp.run_statuses(state, paths, now_s=time.time())))}
    else:
        probe = lp.collect(state, paths, now_s=time.time(), gpu_used_mib=gpu_used_mib(), load_events=load_tfevents)
        decision = ls.decide(state, probe)
        if decision.action != args.action:
            raise SystemExit(f"the loop now decides {decision.action!r} ({decision.reason}), not {args.action!r}")
        if decision.action in ls.SESSION_ACTIONS:
            raise SystemExit(f"{decision.action!r} is carried out by the tick session (LOOP_PROMPT.md), not by act")
        result = EXECUTORS.get(decision.action, lambda s, p, d: {})(state, paths, decision)
    new_state = ls.apply(state, decision, now_iso(), result)
    ls.save_state(paths.state_file, new_state)
    record = {"time": new_state["updated"], "phase": state["phase"], "action": decision.action, "reason": decision.reason,
              "params": decision.params, "result": {k: v for k, v in result.items() if k not in ("commit_paths", "message")}}
    append_history(paths, record)
    if decision.action in COMMIT_ACTIONS and not args.no_commit:
        message = result.get("message") or f"iker(loop): {decision.action} — {decision.reason}"
        record["commit"] = commit([*result.get("commit_paths", []), paths.state_file, paths.history_file], message, args.session_url)
    print(json.dumps({**record, "phase_after": new_state["phase"], "status_after": new_state["status"]}, ensure_ascii=False, indent=1, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-dir", default="", help="loop directory (default: iker_runs/shoe_place/config_00/loop)")
    parser.add_argument("--no-commit", action="store_true", help="never commit (a mock loop outside the repository)")
    parser.add_argument("--session-url", default=SESSION_URL, help="Claude-Session trailer of the loop's commits")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--track", default=ls.TRACK)
    init.add_argument("--phase", default="stage1_t2r", choices=ls.PHASES)
    init.add_argument("--policy", default="{}", help="JSON overrides of loop_state.DEFAULT_POLICY")
    init.add_argument("--adopt", nargs="*", default=[], help="training runs already running under their policy labels")
    commands.add_parser("status")
    act = commands.add_parser("act")
    act.add_argument("action")
    act.add_argument("--policy", default="{}", help="resume: JSON policy updates")
    args = parser.parse_args(argv)
    return {"init": cmd_init, "status": cmd_status, "act": cmd_act}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
