from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import subprocess
import threading

from analyzer import analyze, rule_based_issues, load_learned_rules, merge_rules
from llm import call_ollama_messages
from metrics import summarize_train_metrics, write_metrics_json
from report import write_report_md
from runner import run_train


@dataclass
class EvalResult:
    returncode: int
    metrics_path: str | None
    stdout_path: str
    stderr_path: str


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_path(value: str, base_dir: Path) -> str:
    if not value:
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _task_prefix(task: str) -> str:
    return task.split("-")[0]


def _find_checkpoint(log_dir: Path) -> Path | None:
    """Find best or latest checkpoint. Supports both rsl_rl and skrl layouts."""
    # rsl_rl: model_best.pt / model_*.pt in log_dir root
    best = log_dir / "model_best.pt"
    if best.is_file():
        return best
    # skrl: checkpoints/best_agent.pt / checkpoints/agent_*.pt
    skrl_best = log_dir / "checkpoints" / "best_agent.pt"
    if skrl_best.is_file():
        return skrl_best

    # Search both layouts
    models = []
    # rsl_rl pattern
    for entry in log_dir.iterdir():
        if entry.name.startswith("model_") and entry.name.endswith(".pt"):
            models.append(entry)
    # skrl pattern
    ckpt_dir = log_dir / "checkpoints"
    if ckpt_dir.is_dir():
        for entry in ckpt_dir.iterdir():
            if entry.name.startswith("agent_") and entry.name.endswith(".pt"):
                models.append(entry)

    if not models:
        # rl_games: nn/*.pth
        nn_dir = log_dir / "nn"
        if nn_dir.is_dir():
            pths = [p for p in nn_dir.glob("*.pth") if p.is_file()]
            if pths:
                # Prefer non-"last_" checkpoint if available, otherwise newest file.
                preferred = [p for p in pths if not p.name.startswith("last_")]
                target = preferred if preferred else pths
                target.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return target[0]
        return None

    def _step(p: Path) -> int:
        name = p.stem  # e.g. "model_600" or "agent_80000"
        try:
            # extract last numeric part: model_600 -> 600, agent_80000 -> 80000
            parts = name.split("_")
            return int(parts[-1])
        except Exception:
            return -1

    models.sort(key=_step, reverse=True)
    return models[0]


def _next_test_run_id(log_root: str, task: str, agent: str = "") -> str:
    if str(agent).startswith("rl_games_"):
        task_dir_name = task.replace("-", "_")
        root = Path(log_root)
        nums = []
        for entry in root.rglob("test*"):
            if not entry.is_dir() or not entry.name.startswith("test"):
                continue
            if task_dir_name not in str(entry) and _task_prefix(task) not in str(entry):
                continue
            suffix = entry.name[4:]
            if suffix.isdigit():
                nums.append(int(suffix))
        next_num = (max(nums) + 1) if nums else 1
        return f"test{next_num}"

    task_root = Path(log_root) / _task_prefix(task)
    if not task_root.is_dir():
        return "test1"
    nums = []
    for entry in task_root.iterdir():
        if entry.is_dir() and entry.name.startswith("test"):
            suffix = entry.name[4:]
            if suffix.isdigit():
                nums.append(int(suffix))
    next_num = (max(nums) + 1) if nums else 1
    return f"test{next_num}"


def _resolve_resume_log_dir(
    log_root: str, task: str, resume_from: str, agent: str, resume_checkpoint: str | None = None
) -> Path | None:
    p = Path(resume_from)
    if p.is_absolute() and p.is_dir():
        return p

    root = Path(log_root)
    if str(agent).startswith("rl_games_"):
        matches = [m for m in root.rglob(resume_from) if m.is_dir()]
        if matches:
            task_dir_name = task.replace("-", "_")
            filtered = [m for m in matches if task_dir_name in str(m) or _task_prefix(task) in str(m)]
            use = filtered if filtered else matches
            use.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return use[0]
        if resume_checkpoint:
            ckpt_matches = [p for p in root.rglob(resume_checkpoint) if p.is_file()]
            if ckpt_matches:
                task_dir_name = task.replace("-", "_")
                ckpt_matches.sort(
                    key=lambda p: (
                        task_dir_name in str(p),
                        _task_prefix(task) in str(p),
                        p.stat().st_mtime,
                    ),
                    reverse=True,
                )
                return ckpt_matches[0].parent.parent if ckpt_matches[0].parent.name == "nn" else ckpt_matches[0].parent
        return None

    candidate = root / _task_prefix(task) / resume_from
    if candidate.is_dir():
        return candidate
    return None


def _mean_pair(a: float, b: float) -> float:
    return (a + b) / 2.0


def _load_yaml(path: Path) -> dict | None:
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _key_exists(cfg: object, key_path: str) -> bool:
    cur = cfg
    for part in key_path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False
    return True


def _normalize_hydra_overrides(overrides: list[str], env_yaml_path: Path | None) -> list[str]:
    if not overrides:
        return overrides
    if env_yaml_path is None or not env_yaml_path.is_file():
        return overrides
    env_cfg = _load_yaml(env_yaml_path)
    if not isinstance(env_cfg, dict):
        return overrides

    normalized: list[str] = []
    for item in overrides:
        if "=" not in item:
            normalized.append(item)
            continue
        if item.lstrip().startswith("+"):
            normalized.append(item)
            continue
        key = item.split("=", 1)[0].strip()
        if not key.startswith("env."):
            normalized.append(item)
            continue
        key_path = key[len("env.") :]
        if _key_exists(env_cfg, key_path):
            normalized.append(item)
        else:
            normalized.append("+" + item)
    return normalized


def _extract_reward_keys(env_yaml_path: Path) -> list[str]:
    if not env_yaml_path.is_file():
        return []
    keys: list[str] = []
    in_rewards = False
    for raw in env_yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not in_rewards:
            if line == "rewards:":
                in_rewards = True
            continue
        if not line or (not line.startswith("  ")):
            break
        if line.startswith("    "):
            continue
        if line.endswith(":"):
            name = line.strip()[:-1]
            if name:
                keys.append(name)
    return keys


def _extract_task_ids_from_config_init(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"""gym\.register\(\s*id\s*=\s*["']([^"']+)["']""")
    return sorted(set(pattern.findall(text)))


def _discover_pipeline_hand_tasks(skillblender_root: str) -> set[str]:
    root = (
        Path(skillblender_root)
        / "source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand"
    )
    if not root.is_dir():
        return set()
    task_ids: set[str] = set()
    for cfg_init in root.glob("**/config/__init__.py"):
        try:
            task_ids.update(_extract_task_ids_from_config_init(cfg_init))
        except Exception:
            continue
    return task_ids


def _parse_llm_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if block:
        try:
            return json.loads(block.group(1))
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return {"analysis": text.strip(), "overrides": []}


def _filter_overrides(overrides: list[str], allowed_overrides: list[str]) -> list[str]:
    if not allowed_overrides:
        return overrides
    filtered: list[str] = []
    for item in overrides:
        if "=" not in item:
            continue
        key = item.split("=", 1)[0].strip()
        if any(key.startswith(prefix) for prefix in allowed_overrides):
            filtered.append(item)
    return filtered


def _normalize_override_types(overrides: list[str]) -> list[str]:
    normalized: list[str] = []
    for item in overrides:
        if "=" not in item:
            normalized.append(item)
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key.endswith(".weight"):
            try:
                num = float(value)
            except Exception:
                normalized.append(item)
                continue
            if "." not in value and "e" not in value.lower():
                normalized.append(f"{key}={num:.1f}")
            else:
                normalized.append(f"{key}={num}")
        else:
            normalized.append(item)
    return normalized


def _run_freeze_analysis(
    *,
    script_path: str,
    run_dir: Path,
    out_dir: str,
    report_verbosity: str,
    use_gemini: bool,
    gemini_cmd: str,
    gemini_no_browser: bool,
    stream_logs: bool,
    isaaclab_root: str,
) -> tuple[Path | None, str]:
    cmd = [
        "python3",
        str(Path(script_path).resolve()),
        "--run",
        str(run_dir),
        "--out-dir",
        str(Path(out_dir).resolve()),
        "--report-verbosity",
        report_verbosity,
    ]
    if use_gemini:
        cmd.append("--gemini")
        if gemini_cmd:
            cmd += ["--gemini-cmd", gemini_cmd]
        if gemini_no_browser:
            cmd.append("--gemini-no-browser")

    proc = subprocess.run(
        cmd,
        cwd=str(Path(isaaclab_root).resolve()),
        capture_output=not stream_logs,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        return None, stderr or f"freeze_run_analysis failed (code={proc.returncode})"

    answer_path = Path(out_dir).resolve() / run_dir.name / "gemini_answer.md"
    if not answer_path.is_file():
        return None, f"gemini answer not found: {answer_path}"
    return answer_path, ""


def _ollama_overrides_from_gemini_answer(
    *,
    answer_text: str,
    allowed_overrides: list[str],
    model: str,
    temperature: float,
    api_base: str,
) -> tuple[list[str], str]:
    system_prompt = (
        "You convert RL analysis markdown into strict JSON.\n"
        "Return JSON with keys: analysis, overrides.\n"
        "overrides must be a list of 'key=value' strings only."
    )
    user_prompt = (
        "Read the markdown analysis below and extract practical next-run hydra overrides.\n"
        "Only include overrides whose key starts with one of these allowed prefixes:\n"
        f"{json.dumps(allowed_overrides)}\n\n"
        "Do not include explanations outside JSON.\n\n"
        "Markdown analysis:\n"
        f"{answer_text}"
    )
    response = call_ollama_messages(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=temperature,
        api_base=api_base,
    )
    parsed = _parse_llm_json(response)
    raw = parsed.get("overrides", [])
    overrides = [str(x) for x in raw] if isinstance(raw, list) else []
    overrides = _filter_overrides(overrides, allowed_overrides)
    overrides = _normalize_override_types(overrides)
    summary = str(parsed.get("analysis", "")).strip()
    return overrides, summary


def decide_success(eval_metrics: dict, criteria: dict) -> tuple[bool, dict]:
    lift = _mean_pair(eval_metrics.get("lift_success_left", 0.0), eval_metrics.get("lift_success_right", 0.0))
    hold = _mean_pair(eval_metrics.get("hold_success_left", 0.0), eval_metrics.get("hold_success_right", 0.0))
    goal = _mean_pair(eval_metrics.get("goal_success_left", 0.0), eval_metrics.get("goal_success_right", 0.0))
    goal_track = _mean_pair(
        eval_metrics.get("goal_track_success_left", 0.0), eval_metrics.get("goal_track_success_right", 0.0)
    )
    goal_dist_mean = _mean_pair(
        eval_metrics.get("goal_dist_min_left_mean", 0.0), eval_metrics.get("goal_dist_min_right_mean", 0.0)
    )

    success_rate = _mean_pair(
        eval_metrics.get("success_rate_left", 0.0), eval_metrics.get("success_rate_right", 0.0)
    )

    ok = (
        lift >= criteria.get("lift_success_min", 1.0)
        and hold >= criteria.get("hold_success_min", 1.0)
        and goal >= criteria.get("goal_success_min", 1.0)
        and goal_track >= criteria.get("goal_track_success_min", 1.0)
        and goal_dist_mean <= criteria.get("goal_dist_mean_max", float("inf"))
        and success_rate >= criteria.get("success_rate_min", 0.0)
    )

    return ok, {
        "lift": lift,
        "hold": hold,
        "goal": goal,
        "goal_track": goal_track,
        "goal_dist_mean": goal_dist_mean,
        "success_rate": success_rate,
    }


def run_eval(
    isaaclab_root: str,
    eval_script: str,
    task: str,
    agent: str,
    checkpoint: str,
    num_episodes: int,
    num_envs: int,
    goal_threshold: float,
    lift_threshold: float,
    goal_dist_threshold: float,
    seed: int | None,
    metrics_path: str,
    output_dir: str,
    extra_env: dict | None,
    stream_logs: bool,
    base_args: list[str],
) -> EvalResult:
    isaaclab_root = str(Path(isaaclab_root).resolve())
    eval_script = str(Path(eval_script).resolve())
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "./isaaclab.sh",
        "-p",
        eval_script,
        "--task",
        task,
        "--agent",
        agent,
        "--checkpoint",
        checkpoint,
        "--num_episodes",
        str(num_episodes),
        "--num_envs",
        str(num_envs),
        "--goal_threshold",
        str(goal_threshold),
        "--lift_threshold",
        str(lift_threshold),
        "--goal_dist_threshold",
        str(goal_dist_threshold),
        "--output",
        metrics_path,
    ] + list(base_args)
    if seed is not None:
        cmd += ["--seed", str(seed)]

    stdout_path = output_dir_path / "eval.stdout.txt"
    stderr_path = output_dir_path / "eval.stderr.txt"

    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
        cmd = ["env"] + [f"{k}={v}" for k, v in extra_env.items()] + cmd

    def _tee_stream(stream, fh, prefix: str):
        for line in iter(stream.readline, b""):
            text = line.decode(errors="replace")
            fh.write(text)
            fh.flush()
            if stream_logs:
                print(prefix + text, end="", file=sys.stdout)
                sys.stdout.flush()

    with open(stdout_path, "w", encoding="utf-8") as stdout_fh, open(
        stderr_path, "w", encoding="utf-8"
    ) as stderr_fh:
        if stream_logs:
            proc = subprocess.Popen(
                cmd, cwd=isaaclab_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
            )
            threads = [
                threading.Thread(target=_tee_stream, args=(proc.stdout, stdout_fh, "[eval][out] ")),
                threading.Thread(target=_tee_stream, args=(proc.stderr, stderr_fh, "[eval][err] ")),
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

    return EvalResult(returncode, metrics_path, str(stdout_path), str(stderr_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment OS orchestrator (MVP).")
    parser.add_argument("--config", required=True, help="Path to experiment.json")
    parser.add_argument("--gpu", "--GPU", dest="gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES override")
    parser.add_argument("--num_envs", type=int, default=None, help="Override train --num_envs")
    parser.add_argument("--headless", action="store_true", help="Force headless mode")
    parser.add_argument("--gui", action="store_true", help="Disable headless mode")
    parser.add_argument("--task", type=str, default=None, help="Override task")
    parser.add_argument("--agent", type=str, default=None, help="Override agent")
    parser.add_argument("--max_iterations", "--max_iteration", dest="max_iterations", type=int, default=None, help="Override max_iterations")
    parser.add_argument("--resume_from", type=str, default=None, help="Override resume load_run")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Override resume checkpoint")
    parser.add_argument("--swap_lr", action="store_true", default=False, help="Enable left/right swap augmentation")
    parser.add_argument("--swap_lr_prob", type=float, default=0.5, help="Swap probability per episode (default: 0.5)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_dir = Path(args.config).resolve().parent
    project = cfg["project"]
    project["isaaclab_root"] = resolve_path(project["isaaclab_root"], config_dir)
    project["skillblender_root"] = resolve_path(project["skillblender_root"], config_dir)
    project["log_root"] = resolve_path(project["log_root"], config_dir)
    train = cfg["train"]
    train["train_script"] = resolve_path(train["train_script"], config_dir)
    eval_cfg = cfg["eval"]
    if args.task is not None:
        train["task"] = args.task
    if args.agent is not None:
        train["agent"] = args.agent
    is_skrl = str(train.get("agent", "")).startswith("skrl_")
    is_rl_games = str(train.get("agent", "")).startswith("rl_games_")
    if is_skrl:
        train["train_script"] = resolve_path("../../scripts/reinforcement_learning/skrl/train.py", config_dir)
        project["log_root"] = resolve_path("../../log/skrl", config_dir)
    elif is_rl_games:
        train["train_script"] = resolve_path("../../scripts/reinforcement_learning/rl_games/train.py", config_dir)
        project["log_root"] = resolve_path("../../log/rl_games", config_dir)
    if args.max_iterations is not None:
        train["max_iterations"] = int(args.max_iterations)
    if args.resume_from is not None:
        train["resume_from"] = args.resume_from
    if args.resume_checkpoint is not None:
        train["resume_checkpoint"] = args.resume_checkpoint
    eval_cfg["eval_script"] = resolve_path(eval_cfg["eval_script"], config_dir)
    criteria = cfg.get("success_criteria", {})
    analysis_thresholds = cfg.get("analysis_thresholds", {})
    analysis_rules_path = resolve_path(cfg.get("analysis_rules_path", ""), config_dir)
    analysis_rules = {}
    env_vars = cfg.get("env", {})
    if args.gpu is not None:
        env_vars = dict(env_vars)
        env_vars["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if analysis_rules_path:
        try:
            analysis_rules = json.loads(Path(analysis_rules_path).read_text(encoding="utf-8"))
        except Exception:
            analysis_rules = {}

    # Load and merge learned_rules.json
    learned_rules = load_learned_rules(config_dir)
    if learned_rules.get("issue_to_overrides"):
        analysis_rules = merge_rules(analysis_rules, learned_rules)
        print(f"[orchestrator] Merged {len(learned_rules['issue_to_overrides'])} learned rules")
    llm_cfg = cfg.get("llm", {})
    override_policy = cfg.get("override_policy", {})
    policy = cfg.get("run_policy", {})
    llm_refine = cfg.get("llm_refine_loop", {})

    hand_tasks = _discover_pipeline_hand_tasks(project["skillblender_root"])
    if hand_tasks:
        print(f"[orchestrator] discovered pipeline/hand task ids: {len(hand_tasks)}")
    if train.get("task") in hand_tasks:
        print(f"[orchestrator] task '{train['task']}' is available in pipeline/hand registry.")

    max_runs = int(policy.get("max_runs", 1))
    stop_on_success = bool(policy.get("stop_on_success", True))
    train_chunk_iterations = policy.get("train_chunk_iterations", None)
    max_total_iterations = policy.get("max_total_iterations", None)
    stop_on_collapse = bool(policy.get("stop_on_collapse", True))
    stream_logs = bool(policy.get("stream_logs", True))
    llm_refine_enabled = bool(llm_refine.get("enabled", False))
    freeze_script = resolve_path(
        str(
            llm_refine.get(
                "freeze_analysis_script",
                Path(project["skillblender_root"]) / "scripts" / "tools" / "freeze_run_analysis.py",
            )
        ),
        config_dir,
    )
    freeze_out_dir = resolve_path(
        str(llm_refine.get("out_dir", Path(project["skillblender_root"]) / "log" / "analysis_llm")),
        config_dir,
    )
    freeze_report_verbosity = str(llm_refine.get("report_verbosity", "long"))
    freeze_use_gemini = bool(llm_refine.get("gemini", True))
    freeze_gemini_cmd = str(llm_refine.get("gemini_cmd", "gemini"))
    freeze_gemini_no_browser = bool(llm_refine.get("gemini_no_browser", False))

    seed = train.get("seed", None)
    if args.num_envs is not None:
        train = dict(train)
        train["num_envs"] = int(args.num_envs)

    pending_overrides: list[str] = list(train.get("hydra_overrides", []))

    base_args = list(train.get("base_args", []))
    if args.headless:
        if "--headless" not in base_args:
            base_args.append("--headless")
    if args.gui:
        base_args = [arg for arg in base_args if arg != "--headless"]
    if args.swap_lr:
        if "--swap_lr" not in base_args:
            base_args.append("--swap_lr")
        # swap_lr_prob 값 설정 (기존 값 있으면 교체)
        if "--swap_lr_prob" in base_args:
            idx = base_args.index("--swap_lr_prob")
            base_args[idx + 1] = str(args.swap_lr_prob)
        else:
            base_args += ["--swap_lr_prob", str(args.swap_lr_prob)]
    train["base_args"] = base_args

    # ── Pre-analysis: resume 시 기존 로그를 먼저 분석하여 override 적용 ──
    initial_resume_from = train.get("resume_from", None)
    if initial_resume_from:
        resume_log_dir = _resolve_resume_log_dir(
            project["log_root"],
            train["task"],
            initial_resume_from,
            train["agent"],
            train.get("resume_checkpoint", None),
        )
        if resume_log_dir is None:
            resume_log_dir = Path("__not_found__")
        if resume_log_dir.is_dir():
            print(f"[orchestrator] Pre-analysis: analyzing {resume_log_dir} before first run...")
            pre_metrics = summarize_train_metrics(str(resume_log_dir))
            pre_payload = {"task": train["task"], "train": pre_metrics, "eval": {}}

            env_yaml = resume_log_dir / "params" / "env.yaml"
            reward_names = _extract_reward_keys(env_yaml)
            reward_override_keys = []
            for name in reward_names:
                if name.endswith('_diag'):  # diagnostic 텀은 수정 대상에서 제외
                    continue
                reward_override_keys.append(f"env.rewards.{name}.weight")
                reward_override_keys.append(f"env.rewards.{name}.params.")

            pre_result = analyze(
                payload=pre_payload,
                thresholds=analysis_thresholds,
                llm_cfg=llm_cfg,
                allowed_overrides=reward_override_keys or override_policy.get("allowed_overrides", []),
                rules=analysis_rules,
                config_dir=config_dir,
            )

            if pre_result.applied_overrides:
                pending_overrides = list(pre_result.applied_overrides)
                print(f"[orchestrator] Pre-analysis issues: {pre_result.issues}")
                print(f"[orchestrator] Pre-analysis overrides ({len(pending_overrides)}):")
                for ov in pending_overrides:
                    print(f"  {ov}")
                # Save pre-analysis log
                pre_dir = Path(__file__).resolve().parent / "runs"
                pre_dir.mkdir(parents=True, exist_ok=True)
                pre_log_path = pre_dir / f"pre_analysis_{initial_resume_from}.json"
                pre_log = {
                    "resume_from": initial_resume_from,
                    "log_dir": str(resume_log_dir),
                    "issues": pre_result.issues,
                    "observations": pre_result.observations,
                    "llm_summary": pre_result.llm_summary,
                    "applied_overrides": pre_result.applied_overrides,
                    "override_source": pre_payload.get("override_source", ""),
                }
                pre_log_path.write_text(json.dumps(pre_log, indent=2), encoding="utf-8")
                print(f"[orchestrator] Pre-analysis saved to {pre_log_path}")
            else:
                print("[orchestrator] Pre-analysis: no overrides suggested, starting with original config.")
        else:
            print(f"[orchestrator] Pre-analysis: log dir {resume_log_dir} not found, skipping.")

    for run_idx in range(max_runs):
        if is_skrl:
            agent_prefix = "skrl"
        elif is_rl_games:
            agent_prefix = "rlg"
        else:
            agent_prefix = "rsl"
        runs_root = Path(__file__).resolve().parent / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        # Match IsaacLab log run name: testN
        log_test_id = _next_test_run_id(project["log_root"], train["task"], train["agent"])
        run_id = f"{agent_prefix}_{log_test_id}"
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        train_result = None
        last_log_dir = None
        total_iterations = 0
        chunk = int(train_chunk_iterations) if train_chunk_iterations else None
        max_total = int(max_total_iterations) if max_total_iterations else None

        early_stop_reason = None
        while True:
            resume_from = Path(last_log_dir).name if last_log_dir else train.get("resume_from", None)
            if last_log_dir:
                resume_checkpoint = "best_agent.pt" if is_skrl else "model_best.pt"
            else:
                resume_checkpoint = train.get("resume_checkpoint", None)
            target_iterations = train.get("max_iterations", None)
            if chunk is not None:
                target_iterations = chunk

            env_yaml_path = None
            if last_log_dir:
                env_yaml_path = Path(last_log_dir) / "params" / "env.yaml"
            elif resume_from:
                resume_log_dir = _resolve_resume_log_dir(
                    project["log_root"],
                    train["task"],
                    resume_from,
                    train["agent"],
                    resume_checkpoint,
                )
                if resume_log_dir is not None:
                    env_yaml_path = resume_log_dir / "params" / "env.yaml"

            hydra_overrides = _normalize_hydra_overrides(pending_overrides, env_yaml_path)
            if hydra_overrides:
                print(f"[orchestrator] hydra_overrides ({len(hydra_overrides)}):")
                for ov in hydra_overrides:
                    print(f"  {ov}")
            else:
                print("[orchestrator] hydra_overrides: (none)")

            train_result = run_train(
                isaaclab_root=project["isaaclab_root"],
                train_script=train["train_script"],
                task=train["task"],
                agent=train["agent"],
                base_args=train.get("base_args", []),
                hydra_overrides=hydra_overrides,
                num_envs=train.get("num_envs", None),
                seed=seed,
                max_iterations=target_iterations,
                resume_from=resume_from,
                resume_checkpoint=resume_checkpoint,
                extra_env=env_vars,
                stream_logs=stream_logs,
                log_root=project["log_root"],
                output_dir=str(run_dir),
            )

            last_log_dir = train_result.log_dir
            total_iterations += target_iterations if target_iterations is not None else 0

            if train_result.returncode != 0:
                break
            if chunk is None:
                break
            if max_total is not None and total_iterations >= max_total:
                break

            if last_log_dir:
                train_metrics = summarize_train_metrics(str(last_log_dir))
                probe_payload = {"task": train["task"], "train": train_metrics, "eval": {}}
                issues, _ = rule_based_issues(probe_payload, analysis_thresholds)
                if stop_on_collapse and ("training_collapse" in issues or "entropy_collapse" in issues):
                    early_stop_reason = "training_collapse" if "training_collapse" in issues else "entropy_collapse"
                    break

        report = {
            "run_id": run_id,
            "task": train["task"],
            "agent": train["agent"],
            "train": {
                "returncode": train_result.returncode,
                "log_dir": train_result.log_dir,
                "stdout": train_result.stdout_path,
                "stderr": train_result.stderr_path,
                "seed": seed,
                "hydra_overrides": list(hydra_overrides),
                "total_iterations": total_iterations,
                "chunk_iterations": chunk,
                "early_stop_reason": early_stop_reason,
            },
            "eval": {},
            "decision": {},
        }

        if train_result.returncode != 0 or train_result.log_dir is None:
            report["decision"] = {"status": "train_failed"}
            (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"[orchestrator] run {run_idx+1}/{max_runs} train failed (returncode={train_result.returncode}), continuing to next run...")
            if seed is not None:
                seed += 1
            continue

        log_dir = Path(train_result.log_dir)
        checkpoint = _find_checkpoint(log_dir)
        if checkpoint is None:
            report["decision"] = {"status": "no_checkpoint"}
            (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"[orchestrator] run {run_idx+1}/{max_runs} no checkpoint found, continuing to next run...")
            if seed is not None:
                seed += 1
            continue

        metrics_path = str(log_dir / "metrics.json")
        # rl_games runs were previously hard-skipped here; allow eval unless the config explicitly disables it.
        skip_eval = bool(eval_cfg.get("skip_eval", False)) or is_skrl
        if skip_eval:
            eval_result = EvalResult(0, metrics_path, "", "")
            eval_metrics = {}
        else:
            eval_result = run_eval(
                isaaclab_root=project["isaaclab_root"],
                eval_script=eval_cfg["eval_script"],
                task=train["task"],
                agent=train["agent"],
                checkpoint=str(checkpoint),
                num_episodes=int(eval_cfg.get("num_episodes", 100)),
                num_envs=int(eval_cfg.get("num_envs", 1)),
                goal_threshold=float(eval_cfg.get("goal_threshold", 0.8)),
                lift_threshold=float(eval_cfg.get("lift_threshold", 0.1)),
                goal_dist_threshold=float(eval_cfg.get("goal_dist_threshold", 0.05)),
                seed=eval_cfg.get("seed", None),
                metrics_path=metrics_path,
                output_dir=str(run_dir),
                extra_env=env_vars,
                stream_logs=stream_logs,
                base_args=list(eval_cfg.get("base_args", ["--headless"])),
            )

        train_metrics = summarize_train_metrics(str(log_dir))
        if eval_result.returncode == 0 and os.path.isfile(metrics_path):
            eval_metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))

        payload = {
            "run_id": run_id,
            "task": train["task"],
            "agent": train["agent"],
            "log_dir": str(log_dir),
            "checkpoint": str(checkpoint),
            "train": train_metrics,
            "eval": eval_metrics,
        }
        write_metrics_json(metrics_path, payload)

        if skip_eval:
            train_only_payload = {"task": train["task"], "train": train_metrics, "eval": {}}
            train_issues, _ = rule_based_issues(train_only_payload, analysis_thresholds)
            success = len(train_issues) == 0
            aggregated = {"train_only_issues": train_issues}
        else:
            success, aggregated = decide_success(eval_metrics, criteria)
        analysis_result = None
        applied_overrides = []
        if not success:
            env_yaml = Path(log_dir) / "params" / "env.yaml"
            reward_names = _extract_reward_keys(env_yaml)
            reward_override_keys = []
            for name in reward_names:
                if name.endswith('_diag'):  # diagnostic 텀은 수정 대상에서 제외
                    continue
                reward_override_keys.append(f"env.rewards.{name}.weight")
                reward_override_keys.append(f"env.rewards.{name}.params.")
            allowed_override_keys = reward_override_keys or override_policy.get("allowed_overrides", [])

            llm_refine_summary = ""
            llm_refine_answer_path = ""
            llm_refine_error = ""
            llm_refine_overrides: list[str] = []

            if llm_refine_enabled and str(train.get("agent", "")).startswith("rl_games_"):
                answer_path, refine_err = _run_freeze_analysis(
                    script_path=freeze_script,
                    run_dir=log_dir,
                    out_dir=freeze_out_dir,
                    report_verbosity=freeze_report_verbosity,
                    use_gemini=freeze_use_gemini,
                    gemini_cmd=freeze_gemini_cmd,
                    gemini_no_browser=freeze_gemini_no_browser,
                    stream_logs=stream_logs,
                    isaaclab_root=project["isaaclab_root"],
                )
                if answer_path is None:
                    llm_refine_error = refine_err
                    print(f"[orchestrator] llm_refine_loop skipped: {refine_err}")
                else:
                    llm_refine_answer_path = str(answer_path)
                    try:
                        gemini_answer = answer_path.read_text(encoding="utf-8")
                        llm_refine_overrides, llm_refine_summary = _ollama_overrides_from_gemini_answer(
                            answer_text=gemini_answer,
                            allowed_overrides=allowed_override_keys,
                            model=str(llm_cfg.get("model", "qwen2.5:14b")),
                            temperature=float(llm_cfg.get("temperature", 0.3)),
                            api_base=str(llm_cfg.get("api_base", "http://localhost:11434")),
                        )
                    except Exception as exc:
                        llm_refine_error = str(exc)
                        llm_refine_overrides = []
                        print(f"[orchestrator] llm_refine_loop parse failed: {exc}")

            if llm_refine_overrides:
                applied_overrides = llm_refine_overrides
                pending_overrides = list(applied_overrides)
                payload["override_source"] = "gemini+ollama"
                payload["analysis_response_raw"] = llm_refine_summary
                payload["analysis_prompt"] = "freeze_run_analysis prompt + gemini_answer.md -> ollama override extraction"
                analysis_result = None
            else:
                analysis_result = analyze(
                    payload=payload,
                    thresholds=analysis_thresholds,
                    llm_cfg=llm_cfg,
                    allowed_overrides=allowed_override_keys,
                    rules=analysis_rules,
                    config_dir=config_dir,
                )
                applied_overrides = analysis_result.applied_overrides
                pending_overrides = list(applied_overrides)

        report["eval"] = {
            "returncode": eval_result.returncode,
            "metrics_path": metrics_path,
            "stdout": eval_result.stdout_path,
            "stderr": eval_result.stderr_path,
            "skipped": skip_eval,
        }
        report["decision"] = {
            "status": "success" if success else "failed",
            "aggregated": aggregated,
            "criteria": criteria,
        }
        if analysis_result is not None:
            report["analysis"] = {
                "issues": analysis_result.issues,
                "observations": analysis_result.observations,
                "llm_summary": analysis_result.llm_summary,
                "llm_overrides": analysis_result.llm_overrides,
                "applied_overrides": applied_overrides,
                "override_source": payload.get("override_source", ""),
                "llm_rule_overlap": payload.get("llm_rule_overlap", []),
                "llm_override_count": payload.get("llm_override_count", 0),
                "rule_override_count": payload.get("rule_override_count", 0),
            }
            (run_dir / "analysis_prompt.txt").write_text(
                payload.get("analysis_prompt", ""), encoding="utf-8"
            )
            (run_dir / "analysis_response.txt").write_text(
                payload.get("analysis_response_raw", ""), encoding="utf-8"
            )
            (run_dir / "overrides.json").write_text(
                json.dumps({"overrides": applied_overrides}, indent=2), encoding="utf-8"
            )
        elif applied_overrides:
            report["analysis"] = {
                "issues": aggregated.get("train_only_issues", []),
                "observations": [],
                "llm_summary": payload.get("analysis_response_raw", ""),
                "llm_overrides": applied_overrides,
                "applied_overrides": applied_overrides,
                "override_source": payload.get("override_source", ""),
            }
            (run_dir / "analysis_prompt.txt").write_text(
                payload.get("analysis_prompt", ""), encoding="utf-8"
            )
            (run_dir / "analysis_response.txt").write_text(
                payload.get("analysis_response_raw", ""), encoding="utf-8"
            )
            (run_dir / "overrides.json").write_text(
                json.dumps({"overrides": applied_overrides}, indent=2), encoding="utf-8"
            )

        report_path = run_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_report_md(
            report_path=str(report_path),
            metrics_path=str(metrics_path),
            out_path=str(run_dir / "report.md"),
        )

        # ── Append to cumulative progress log ──
        progress_name = "progress_skrl.md" if is_skrl else "progress_rsl.md"
        progress_name = f"progress_{agent_prefix}_{log_test_id}.md"
        _append_progress_log(run_dir.parent / progress_name, report, payload)

        if success and stop_on_success:
            break

        if seed is not None:
            seed += 1

    return 0


def _append_progress_log(progress_path: Path, report: dict, payload: dict) -> None:
    """Append a run summary to the cumulative progress.md file."""
    run_id = report.get("run_id", "unknown")
    decision = report.get("decision", {})
    analysis = report.get("analysis", {})
    train_info = report.get("train", {})
    scalars = payload.get("train", {}).get("scalars", {})

    mean_reward = scalars.get("mean_reward", {})
    mr_val = mean_reward.get("mean_last_100") if isinstance(mean_reward, dict) else None

    lines = []
    # Header on first write
    if not progress_path.exists():
        lines.append("# Experiment Progress Log")
        lines.append("")
        lines.append("자동 학습 루프 진행 기록. 각 run의 상태, LLM 분석, override 변경 사항 추적.")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append(f"## Run {run_id}")
    lines.append(f"- **status**: {decision.get('status', 'unknown')}")
    lines.append(f"- **iterations**: {train_info.get('total_iterations', 'n/a')}")
    lines.append(f"- **mean_reward**: {mr_val:.3f}" if mr_val is not None else "- **mean_reward**: n/a")
    if train_info.get("early_stop_reason"):
        lines.append(f"- **early_stop**: {train_info['early_stop_reason']}")
    lines.append("")

    # Key physical metrics snapshot
    diag_snapshot = [
        ("eef_dist L/R", "reward_left_eef_dist_diag", "reward_right_eef_dist_diag"),
        ("dist_delta L/R", "reward_left_eef_dist_delta_diag", "reward_right_eef_dist_delta_diag"),
        ("hand_closure L/R", "reward_left_hand_closure_diag", "reward_right_hand_closure_diag"),
        ("phase L/R", "reward_left_grasp2g_phase", "reward_right_grasp2g_phase"),
        ("obj_height L/R", "reward_left_object_height_diag", "reward_right_object_height_diag"),
    ]
    has_any = any(
        isinstance(scalars.get(lk), dict) and scalars.get(lk, {}).get("mean_last_100") is not None
        for _, lk, _ in diag_snapshot
    )
    if has_any:
        lines.append("### Key Metrics")
        lines.append("| Metric | Left | Right |")
        lines.append("|--------|------|-------|")
        for label, lk, rk in diag_snapshot:
            lv = scalars.get(lk, {})
            rv = scalars.get(rk, {})
            l_str = f"{lv.get('mean_last_100', 0):.4f}" if isinstance(lv, dict) and lv.get("mean_last_100") is not None else "n/a"
            r_str = f"{rv.get('mean_last_100', 0):.4f}" if isinstance(rv, dict) and rv.get("mean_last_100") is not None else "n/a"
            lines.append(f"| {label} | {l_str} | {r_str} |")
        lines.append("")

    # Issues + LLM reasoning
    if analysis.get("issues"):
        lines.append(f"### Issues: {', '.join(analysis['issues'])}")
        lines.append("")

    if analysis.get("llm_summary"):
        lines.append("### LLM Reasoning")
        lines.append("```")
        # Truncate very long summaries
        summary = analysis["llm_summary"]
        if len(summary) > 2000:
            summary = summary[:2000] + "\n... (truncated)"
        lines.append(summary)
        lines.append("```")
        lines.append("")

    if analysis.get("applied_overrides"):
        lines.append("### Applied Overrides")
        for item in analysis["applied_overrides"]:
            lines.append(f"- `{item}`")
        source = analysis.get("override_source", "")
        if source:
            lines.append(f"- source: **{source}**")
        lines.append("")

    lines.append("---")
    lines.append("")

    with open(progress_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
