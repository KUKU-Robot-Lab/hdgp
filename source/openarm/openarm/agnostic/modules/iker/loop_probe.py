"""Read the IKER auto loop's world into a ``loop_state.Probe`` (design 2026-09-14-iker-auto-loop §3, §8).

No Isaac, torch or subprocess: the checkout root, the /proc root, a TFEvents reader and the GPU memory reading are passed
in, so the tests run on temporary trees. A process belongs to a run only through the RUN_LABEL in its environment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping

from . import loop_state as ls
from . import run_files

RL_LOG_PARTS = ("log", "rl_games", "open-sens", "left")
TASK_DIRS = {"stage1_a": "iker-shoe-grasp", "stage1_b": "iker-shoe-grasp", "stage2": "iker-shoe"}
PHASE_TRAINING = {
    "stage1_a": "stage1_a", "calibrate": "stage1_a", "stage1_b": "stage1_b", "harvest": "stage1_b",
    "stage2_train": "stage2", "observe_requery": "stage2",
}
LATCHED_TAG = "Episode/grasp_episode/latched"
OVER_RACK_TAG = "Episode/grasp/over_rack_raised_frac"
CHECKPOINT_RE = re.compile(r"^last_.+_ep_(\d+)_rew_.*\.pth$")
ATTEMPT_RE = re.compile(r"^attempt_(\d{2})$")
EVAL_RE = re.compile(r"^eval_ep(\d+)\.json$")
TRAIN_FINISHED = "MAX EPOCHS NUM!"
TRAIN_CRASH = ("Traceback (most recent call last)", "CUDA out of memory", "Error executing job")
SIDE_MARKERS: Mapping[str, tuple[str, str]] = {  # run -> (result line prefix, crash marker printed by its excepthook)
    "calibrate": ("QUALITY config", "QUALITY FAILED"),
    "harvest": ("HARVEST config", "HARVEST FAILED"),
    "env_smoke": ("SMOKE passed", "SMOKE FAILED"),
    "eval": ("EVAL {", "EVAL FAILED"),
    "observe_rollout": ("EVAL {", "EVAL FAILED"),
    "observe_render": ("OBSERVE config", "OBSERVE FAILED"),
    "video": ("", ""),  # play.py prints no result line: the video file is the result
}
CHECK_FAILED_PREFIX: Mapping[str, str] = {"observe_render": "OBSERVE CHECK FAILED: "}  # blocking checks; REPORTED lines are not
CHECKPOINT_SETTLE_S = 30.0  # a checkpoint file younger than this may still be written: not yet a candidate
PASSED_RE = re.compile(r"\bpassed (True|False)\b")
REWARD_LINE_RE = re.compile(r"\[iker_grasp\] reward Stage1RewardCfg\(([^)]*)\)")
FIELD_RE = re.compile(r"(\w+)=([-+0-9.eE]+)")
ISAAC_PYTHON_MARK = b"kit/python/bin/python3"
LOG_TAIL_BYTES = 4 << 20
VIDEO_NAME = "rl-video-step-0.mp4"


@dataclass(frozen=True)
class LoopPaths:
    root: Path  # the hdgp checkout
    config_dir: Path  # iker_runs/shoe_place/config_XX
    state_dir: Path  # LOOP_STATE.json, history.jsonl, stage_01/
    log_dir: Path  # side-run logs and uncommitted outputs (gitignored)

    @classmethod
    def of(cls, root: Path, config_index: int, track: str, state_dir: Path | None = None) -> "LoopPaths":
        config_dir = root / "iker_runs" / "shoe_place" / f"config_{config_index:02d}"
        return cls(root, config_dir, state_dir if state_dir is not None else config_dir / "loop", root / "log" / "iker_loop" / track)

    @property
    def state_file(self) -> Path:
        return self.state_dir / "LOOP_STATE.json"

    @property
    def history_file(self) -> Path:
        return self.state_dir / "history.jsonl"

    @property
    def stage_dir(self) -> Path:
        return self.state_dir / "stage_01"

    @property
    def observe_dir(self) -> Path:
        return self.stage_dir / "observe"

    @property
    def interaction_file(self) -> Path:
        return self.stage_dir / "interaction_vlm.json"

    @property
    def video_note(self) -> Path:
        return self.stage_dir / "video.txt"

    @property
    def calibration_file(self) -> Path:
        return self.config_dir / "grasp_quality_calibration.json"  # iker_shoe_grasp_env.QUALITY_CALIBRATION_FILE

    @property
    def final_states_file(self) -> Path:
        return self.log_dir / "observe_final_states.json"

    @property
    def harvest_bank_file(self) -> Path:
        return self.config_dir / "grasp_bank.json"

    def bank_file(self, policy: Mapping) -> Path:
        """The bank stage 2 loads: the harvested bank unless the policy names another (the mock loop)."""
        return Path(policy["grasp_bank_path"]) if policy["grasp_bank_path"] else self.harvest_bank_file

    def task_dir(self, run: str) -> Path:
        return self.root.joinpath(*RL_LOG_PARTS, TASK_DIRS[run])

    def train_log(self, label: str) -> Path:
        return self.root.joinpath(*RL_LOG_PARTS, f"train_{label}.log")

    def side_log(self, run: str, tag: str) -> Path:
        return self.log_dir / f"{run}_{tag}.log"

    def eval_file(self, epoch: int) -> Path:
        return self.stage_dir / f"eval_ep{epoch:04d}.json"

    def eval_rows_file(self, epoch: int) -> Path:
        return self.log_dir / f"eval_ep{epoch:04d}_rows.json"


def read_tail(path: Path, limit: int = LOG_TAIL_BYTES) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        handle.seek(max(0, size - limit))
        return handle.read().decode("utf-8", errors="replace")


def list_checkpoints(nn_dir: Path, settled_before_s: float | None = None) -> dict[int, str]:
    """epoch -> absolute path of rl_games' ``last_<name>_ep_<E>_rew_<R>.pth`` files; on a shared epoch the first name in sorted order (the periodic save) wins.
    With ``settled_before_s``, files modified after that time are left out (they may still be written)."""
    if not nn_dir.is_dir():
        return {}
    found = {}
    for path in sorted(nn_dir.iterdir()):
        match = CHECKPOINT_RE.match(path.name)
        if match and (settled_before_s is None or path.stat().st_mtime <= settled_before_s):
            found.setdefault(int(match.group(1)), str(path.resolve()))
    return found


def find_run_dir(task_dir: Path, label: str, started_s: float) -> Path | None:
    """The folder train.py made for ``label`` (``label``, or ``label-rN`` when that existed), modified since the launch."""
    candidates = [task_dir / label, *task_dir.glob(f"{label}-r*")] if task_dir.is_dir() else []
    fresh = [path for path in candidates if path.is_dir() and path.stat().st_mtime >= started_s - 60.0]
    return max(fresh, key=lambda path: path.stat().st_mtime) if fresh else None


def label_pids(label: str, proc_root: Path = Path("/proc")) -> list[int]:
    """Isaac python processes whose environment holds ``RUN_LABEL=<label>`` (the python.sh bash wrapper is not one)."""
    wanted = b"RUN_LABEL=" + label.encode()
    pids = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            environ, cmdline = (entry / "environ").read_bytes(), (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if wanted in environ.split(b"\0") and ISAAC_PYTHON_MARK in cmdline:
            pids.append(int(entry.name))
    return sorted(pids)


def parse_gpu_used_mib(text: str) -> int:
    """Largest ``memory.used`` in ``nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`` output."""
    values = [int(line.strip()) for line in text.splitlines() if line.strip()]
    if not values:
        raise ValueError(f"no GPU memory reading in {text!r}")
    return max(values)


def boot_reward(text: str) -> dict[str, float] | None:
    """g_min, q_lo and q_hi of the last ``[iker_grasp] reward Stage1RewardCfg(...)`` line."""
    lines = REWARD_LINE_RE.findall(text)
    if not lines:
        return None
    fields = dict(FIELD_RE.findall(lines[-1]))
    return {key: float(fields[key]) for key in ("g_min", "q_lo", "q_hi")}


def training_status(text: str, alive: bool, checkpoints: Mapping[int, str], max_epochs: int | None, idle_s: float) -> ls.RunStatus:
    finished = TRAIN_FINISHED in text or (max_epochs is not None and max_epochs in checkpoints)
    crash = next((marker for marker in TRAIN_CRASH if marker in text), "")
    return ls.RunStatus(started=True, alive=alive, finished=finished, crashed=bool(crash) or (not alive and not finished),
                        marker=crash, idle_s=idle_s)


def side_status(run: str, text: str, alive: bool, result_ready: bool, idle_s: float) -> ls.RunStatus:
    prefix, crash_marker = SIDE_MARKERS[run]
    lines = [line for line in text.splitlines() if prefix and line.startswith(prefix)]
    marker = lines[-1] if lines else ""
    finished = bool(marker) or result_ready
    found = PASSED_RE.search(marker)
    passed = (found.group(1) == "True") if found else (True if finished else None)
    crash = bool(crash_marker) and crash_marker in text
    check_prefix = CHECK_FAILED_PREFIX.get(run, "")
    checks = tuple(line[len(check_prefix):] for line in text.splitlines() if check_prefix and line.startswith(check_prefix))
    return ls.RunStatus(started=True, alive=alive, finished=finished, passed=passed, crashed=crash or (not alive and not finished),
                        marker=marker or (crash_marker if crash else ""), idle_s=idle_s, failed_checks=checks)


def attempts(stage_dir: Path) -> tuple[ls.AttemptStatus, ...]:
    """VLM attempts that hold a response.md, with their gate verdict once gate.json exists."""
    if not stage_dir.is_dir():
        return ()
    found = []
    for path in sorted(stage_dir.iterdir()):
        match = ATTEMPT_RE.match(path.name)
        if match and (path / "response.md").is_file():
            gate = _optional_json(path / "gate.json")
            found.append(ls.AttemptStatus(int(match.group(1)), None if gate is None else bool(gate["passed"])))
    return tuple(found)


def evals(stage_dir: Path) -> dict[int, dict]:
    if not stage_dir.is_dir():
        return {}
    return {int(match.group(1)): run_files.read_json(path)["summary"]
            for path in sorted(stage_dir.iterdir()) if (match := EVAL_RE.match(path.name))}


def final_rows(path: Path) -> tuple[dict, ...] | None:
    doc = _optional_json(path)
    if doc is None:
        return None
    return tuple({"env": int(row["env"]), "success": bool(row["success"]), "end_dist": float(row["end_dist"])} for row in doc["rows"])


def stage2_video(state: Mapping, paths: LoopPaths) -> Path | None:
    """The play video file of the stage-2 run folder (it may not exist yet)."""
    record = state["runs"].get("stage2", {})
    run_dir = find_run_dir(paths.task_dir("stage2"), state["policy"]["labels"]["stage2"], record.get("started_s", 0.0))
    return run_dir / "videos" / "play" / VIDEO_NAME if run_dir is not None else None


def collect(state: Mapping, paths: LoopPaths, *, now_s: float, gpu_used_mib: int,
            load_events: Callable[[str], Mapping[str, list]], proc_root: Path = Path("/proc")) -> ls.Probe:
    policy = state["policy"]
    runs = run_statuses(state, paths, now_s=now_s, proc_root=proc_root)
    latched, over_rack, checkpoints, boot = (), (), {}, None
    training = PHASE_TRAINING.get(state["phase"])
    if training is not None:
        record = state["runs"].get(training, {})  # a cleared record still locates the run folder of its checkpoints
        run_dir = find_run_dir(paths.task_dir(training), policy["labels"][training], record.get("started_s", 0.0))
        if run_dir is not None:
            checkpoints = list_checkpoints(run_dir / "nn", settled_before_s=now_s - CHECKPOINT_SETTLE_S)
            if training != "stage2":
                series = _events(run_dir, load_events)
                latched, over_rack = _series(series, LATCHED_TAG), _series(series, OVER_RACK_TAG)
        live = ls.live_record(state, training)
        if training == "stage1_b" and live:
            boot = boot_reward(read_tail(Path(live["log"])))
    return ls.Probe(
        gpu_used_mib=gpu_used_mib, runs=runs, latched=latched, over_rack=over_rack, checkpoints=checkpoints, boot_reward=boot,
        calibration=_optional_json(paths.calibration_file), bank_meta=(_optional_json(paths.harvest_bank_file) or {}).get("metadata"),
        attempts=attempts(paths.stage_dir), evals=evals(paths.stage_dir), final_rows=final_rows(paths.final_states_file),
        requery=_optional_json(paths.observe_dir / "requery.json"), files=_files(state, paths),
    )


def run_statuses(state: Mapping, paths: LoopPaths, *, now_s: float, proc_root: Path = Path("/proc")) -> dict[str, ls.RunStatus]:
    """The status of every launch record ``resume`` has not cleared (a cleared run is absent)."""
    return {name: _run_status(name, record, state, paths, now_s, proc_root)
            for name, record in state["runs"].items() if ls.live_record(state, name) is not None}


def _optional_json(path: Path) -> dict | None:
    return run_files.read_json(path) if path.is_file() else None


def _series(series: Mapping[str, list], tag: str) -> tuple[tuple[int, float], ...]:
    return tuple((int(step), float(value)) for step, value in series.get(tag, ()))


def _events(run_dir: Path, load_events: Callable[[str], Mapping[str, list]]) -> Mapping[str, list]:
    files = sorted((run_dir / "summaries").glob("events.out.tfevents.*"))
    return load_events(str(files[-1])) if files else {}


def _run_status(name: str, record: Mapping, state: Mapping, paths: LoopPaths, now_s: float, proc_root: Path) -> ls.RunStatus:
    log = Path(record["log"])
    alive = bool(label_pids(record["label"], proc_root))
    idle_s = now_s - log.stat().st_mtime if log.is_file() else 0.0
    text = read_tail(log)
    if name in ls.TRAINING_RUNS:
        run_dir = find_run_dir(paths.task_dir(name), record["label"], record.get("started_s", 0.0))
        checkpoints = list_checkpoints(run_dir / "nn") if run_dir is not None else {}
        max_epochs = {"stage1_a": None, "stage1_b": state["policy"]["b_max_epochs"], "stage2": state["policy"]["stage2_epochs"]}[name]
        status = training_status(text, alive, checkpoints, max_epochs, idle_s)  # every checkpoint: a young final save still finishes
    else:
        status = side_status(name, text, alive, name == "video" and _video_ready(state, paths), idle_s)
    return replace(status, crashed=False) if ls.ended_by_loop(state, name) else status  # stopped by the loop, not crashed


def _video_ready(state: Mapping, paths: LoopPaths) -> bool:
    record, video = ls.live_record(state, "video"), stage2_video(state, paths)
    if record is None or video is None or not video.is_file():
        return False
    stat = video.stat()
    return stat.st_size > 0 and stat.st_mtime >= record.get("started_s", 0.0)


def _files(state: Mapping, paths: LoopPaths) -> dict[str, bool]:
    stage, observe = paths.stage_dir, paths.observe_dir
    return {
        "stage_prompt": all((stage / name).is_file() for name in ("prompt.md", "snapshot.png", "keypoints.json")),
        "observe_snapshot": all((observe / name).is_file() for name in ("snapshot.png", "keypoints.json", "state.json")),
        "requery_prompt": (observe / "prompt.md").is_file(),
        "requery_response": (observe / "response.md").is_file(),
        "video_raw": _video_ready(state, paths),
        "video": paths.video_note.is_file(),
    }
