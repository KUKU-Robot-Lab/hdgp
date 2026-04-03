from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Patterns for progress lines to display on terminal
_RSL_RL_ITER = re.compile(r"Learning iteration \d+/\d+")
_RSL_RL_TIME = re.compile(r"Iteration time:\s*\S+")
_RSL_RL_ELAPSED = re.compile(r"Time elapsed:\s*\S+")
_RSL_RL_ETA = re.compile(r"ETA:\s*\S+")
_SKRL_PROGRESS = re.compile(r"\d+%\|.*?it/s\]")


@dataclass
class TrainResult:
    returncode: int
    log_dir: str | None
    stdout_path: str
    stderr_path: str


def _task_prefix(task: str) -> str:
    return task.split("-")[0]


def _resolve_rl_games_resume_dir(log_root: Path, resume_from: str, task: str) -> Path | None:
    resume_from_path = Path(resume_from)
    if resume_from_path.is_absolute() and resume_from_path.is_dir():
        return resume_from_path

    matches = [p for p in log_root.rglob(resume_from) if p.is_dir()]
    if not matches:
        return None

    task_dir_name = task.replace("-", "_")
    prefix = _task_prefix(task)
    exact_task = [p for p in matches if task_dir_name in str(p)]
    if exact_task:
        exact_task.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return exact_task[0]

    same_prefix = [p for p in matches if prefix in str(p)]
    if same_prefix:
        same_prefix.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return same_prefix[0]

    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _resolve_rl_games_checkpoint(
    log_root: Path, task: str, resume_from: str | None, resume_checkpoint: str | None
) -> Path | None:
    if not resume_checkpoint:
        return None

    cp = Path(resume_checkpoint)
    if cp.is_absolute() and cp.is_file():
        return cp

    if resume_from:
        resume_dir = _resolve_rl_games_resume_dir(log_root, resume_from, task)
        if resume_dir is not None:
            nn_candidate = resume_dir / "nn" / resume_checkpoint
            if nn_candidate.is_file():
                return nn_candidate
            direct_candidate = resume_dir / resume_checkpoint
            if direct_candidate.is_file():
                return direct_candidate

    # Fallback: search by checkpoint filename under log root and prefer task-related path.
    candidates = [p for p in log_root.rglob(resume_checkpoint) if p.is_file()]
    if not candidates:
        return None
    task_dir_name = task.replace("-", "_")
    prefix = _task_prefix(task)
    candidates.sort(
        key=lambda p: (
            task_dir_name in str(p),
            prefix in str(p),
            p.stat().st_mtime,
        ),
        reverse=True,
    )
    return candidates[0]


def _find_latest_log_dir(log_root: Path, task: str, agent: str) -> str | None:
    is_rl_games = str(agent).startswith("rl_games_")
    if is_rl_games:
        task_dir_name = task.replace("-", "_")
        candidates: list[Path] = []
        for entry in log_root.rglob("test*"):
            if not entry.is_dir() or not entry.name.startswith("test"):
                continue
            if not (entry / "params" / "env.yaml").is_file():
                continue
            # Prefer task-matching paths, but keep fallback to any test* under log_root.
            path_str = str(entry)
            if task_dir_name in path_str or _task_prefix(task) in path_str:
                candidates.append(entry)
        if not candidates:
            for entry in log_root.rglob("test*"):
                if entry.is_dir() and entry.name.startswith("test") and (entry / "params" / "env.yaml").is_file():
                    candidates.append(entry)
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0])

    task_root = log_root / _task_prefix(task)
    if not task_root.is_dir():
        return None
    candidates = []
    for entry in task_root.iterdir():
        if entry.is_dir() and entry.name.startswith("test"):
            candidates.append(entry)
    if not candidates:
        return None

    def _score(p: Path) -> tuple[int, float]:
        suffix = p.name[4:]
        num = int(suffix) if suffix.isdigit() else -1
        return (num, p.stat().st_mtime)

    candidates.sort(key=_score, reverse=True)
    return str(candidates[0])


def run_train(
    isaaclab_root: str,
    train_script: str,
    task: str,
    agent: str,
    base_args: Iterable[str],
    hydra_overrides: Iterable[str],
    num_envs: int | None,
    seed: int | None,
    max_iterations: int | None,
    resume_from: str | None,
    resume_checkpoint: str | None,
    extra_env: dict | None,
    stream_logs: bool,
    log_root: str,
    output_dir: str,
) -> TrainResult:
    isaaclab_root = str(Path(isaaclab_root).resolve())
    train_script = str(Path(train_script).resolve())
    log_root_path = Path(log_root).resolve()
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    is_skrl = str(agent).startswith("skrl_")
    is_rl_games = str(agent).startswith("rl_games_")

    cmd = ["./isaaclab.sh", "-p", train_script, "--task", task, "--agent", agent]
    if num_envs is not None:
        cmd += ["--num_envs", str(num_envs)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if max_iterations is not None:
        cmd += ["--max_iterations", str(max_iterations)]

    # Resume handling: rsl_rl vs skrl have different CLI conventions
    if is_skrl:
        # skrl: --checkpoint <full_path_to_checkpoint>
        if resume_from and resume_checkpoint:
            task_prefix = _task_prefix(task)
            ckpt_path = log_root_path / task_prefix / resume_from / "checkpoints" / resume_checkpoint
            cmd += ["--checkpoint", str(ckpt_path)]
        elif resume_checkpoint and Path(resume_checkpoint).is_absolute():
            cmd += ["--checkpoint", str(resume_checkpoint)]
    elif is_rl_games:
        # rl_games: --checkpoint <full_path_to_checkpoint_or_file>
        ckpt_path = _resolve_rl_games_checkpoint(
            log_root=log_root_path,
            task=task,
            resume_from=resume_from,
            resume_checkpoint=resume_checkpoint,
        )
        if ckpt_path is not None:
            cmd += ["--checkpoint", str(ckpt_path)]
    else:
        # rsl_rl: --resume --load_run <run_name> --checkpoint <filename>
        if resume_from:
            cmd += ["--resume", "--load_run", str(resume_from)]
        if resume_checkpoint:
            cmd += ["--checkpoint", str(resume_checkpoint)]

    cmd += list(base_args)
    cmd += list(hydra_overrides)

    stdout_path = output_dir_path / "train.stdout.txt"
    stderr_path = output_dir_path / "train.stderr.txt"

    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
        cmd = ["env"] + [f"{k}={v}" for k, v in extra_env.items()] + cmd

    # Accumulator for rsl_rl multi-line progress
    rsl_parts = {"iter": "", "time": "", "elapsed": "", "eta": ""}

    def _flush_rsl_progress():
        """Print accumulated rsl_rl progress as one line."""
        if rsl_parts["iter"]:
            line = rsl_parts["iter"]
            if rsl_parts["time"]:
                line += ", " + rsl_parts["time"]
            if rsl_parts["elapsed"]:
                line += ", " + rsl_parts["elapsed"]
            if rsl_parts["eta"]:
                line += ", " + rsl_parts["eta"]
            print(f"\r[train][out] {line}          ", end="", file=sys.stdout)
            sys.stdout.flush()
            rsl_parts["iter"] = ""
            rsl_parts["time"] = ""
            rsl_parts["elapsed"] = ""
            rsl_parts["eta"] = ""

    def _tee_stream(stream, fh, prefix: str):
        buf = b""
        while True:
            chunk = stream.read(1)
            if not chunk:
                if buf:
                    text = buf.decode(errors="replace")
                    fh.write(text)
                    fh.flush()
                _flush_rsl_progress()
                break
            buf += chunk
            # split on \n or \r (tqdm uses \r without \n)
            if chunk in (b"\n", b"\r"):
                text = buf.decode(errors="replace")
                buf = b""
                fh.write(text)
                fh.flush()
                if stream_logs:
                    stripped = text.strip()
                    if not stripped:
                        continue
                    if is_rl_games:
                        print(f"{prefix}{stripped}", file=sys.stdout)
                        sys.stdout.flush()
                        continue
                    # skrl tqdm progress
                    if _SKRL_PROGRESS.search(stripped):
                        print(f"\r{prefix}{stripped}          ", end="", file=sys.stdout)
                        sys.stdout.flush()
                        continue
                    # rsl_rl: collect parts then print as one line
                    m = _RSL_RL_ITER.search(stripped)
                    if m:
                        _flush_rsl_progress()
                        rsl_parts["iter"] = m.group()
                        continue
                    m = _RSL_RL_TIME.search(stripped)
                    if m:
                        rsl_parts["time"] = m.group()
                        continue
                    m = _RSL_RL_ELAPSED.search(stripped)
                    if m:
                        rsl_parts["elapsed"] = m.group()
                        continue
                    m = _RSL_RL_ETA.search(stripped)
                    if m:
                        rsl_parts["eta"] = m.group()
                        _flush_rsl_progress()
                        continue

    with open(stdout_path, "w", encoding="utf-8") as stdout_fh, open(
        stderr_path, "w", encoding="utf-8"
    ) as stderr_fh:
        if stream_logs:
            proc = subprocess.Popen(
                cmd, cwd=isaaclab_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
            )
            threads = [
                threading.Thread(target=_tee_stream, args=(proc.stdout, stdout_fh, "[train][out] ")),
                threading.Thread(target=_tee_stream, args=(proc.stderr, stderr_fh, "[train][err] ")),
            ]
            for t in threads:
                t.daemon = True
                t.start()
            returncode = proc.wait()
            for t in threads:
                t.join(timeout=1)
        else:
            returncode = subprocess.run(
                cmd, cwd=isaaclab_root, stdout=stdout_fh, stderr=stderr_fh, check=False, env=env
            ).returncode

    log_dir = _find_latest_log_dir(log_root_path, task, agent)

    meta = {
        "cmd": cmd,
        "returncode": returncode,
        "log_dir": log_dir,
        "hydra_overrides": list(hydra_overrides),
        "resume_from": resume_from,
        "resume_checkpoint": resume_checkpoint,
    }
    (output_dir_path / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return TrainResult(returncode, log_dir, str(stdout_path), str(stderr_path))
