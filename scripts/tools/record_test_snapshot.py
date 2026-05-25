#!/usr/bin/env python3
"""학습 시작 전 현재 코드 상태를 test_history.md에 스냅샷으로 기록.

Usage:
    python record_test_snapshot.py --task 5g_pour_right_v5 --test test6
    python record_test_snapshot.py --task 5g_pour_right_v3 --test test7

실행 위치: /home/user/rl_ws/hdgp (git 루트)
출력: <LOG_ROOT>/<task>/test_history.md 에 새 항목 append
"""

import argparse
import datetime
import re
import subprocess
from pathlib import Path

GIT_ROOT = Path("/home/user/rl_ws/hdgp")
LOG_ROOT = GIT_ROOT / "log/rl_games/pipeline/right"
SRC_ROOT = GIT_ROOT / "source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/right"

# 추출할 핵심 하이퍼파라미터 패턴 (env_cfg.py 기준)
def _param_pattern(name: str, is_bool: bool = False) -> str:
    """Python dataclass 및 일반 대입 형식 모두 처리."""
    val = r"(True|False)" if is_bool else r"([0-9._]+)"
    return rf"{name}\s*(?::\s*\S+\s*)?=\s*{val}"


KEY_PARAM_PATTERNS = [
    _param_pattern("weight_tilt"),
    _param_pattern("weight_align"),
    _param_pattern("weight_demo_arm_pose"),
    _param_pattern("weight_demo_palm_pose"),
    _param_pattern("weight_spill"),
    _param_pattern("weight_cup_collision"),
    _param_pattern("weight_grasp_loss"),
    _param_pattern("pour_tilt_target_deg"),
    _param_pattern("pour_reward_warmup_steps"),
    _param_pattern("cup_collision_margin"),
    _param_pattern("enable_spill_adr", is_bool=True),
    _param_pattern("success_target_fill_ratio"),
    _param_pattern("success_spill_max"),
]


def run(cmd: list[str], cwd: Path = GIT_ROOT) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return result.stdout.strip()


def get_git_head() -> tuple[str, str]:
    """(short_hash, message)"""
    short = run(["git", "rev-parse", "--short", "HEAD"])
    msg = run(["git", "log", "-1", "--format=%s"])
    return short, msg


def get_git_diff_summary(task_dir: str) -> str:
    """현재 HEAD 기준 uncommitted changes (staged+unstaged) 요약."""
    diff = run(["git", "diff", "HEAD", "--", task_dir])
    if not diff:
        diff = run(["git", "diff", "--cached", "--", task_dir])
    if not diff:
        return "(uncommitted 변경 없음)"

    lines = diff.splitlines()
    summary_lines = []
    current_file = None
    add_count = del_count = 0
    changed_files: dict[str, dict] = {}

    for line in lines:
        if line.startswith("diff --git"):
            if current_file and (add_count or del_count):
                changed_files[current_file] = {"add": add_count, "del": del_count, "hunks": []}
            m = re.search(r"b/(.+)$", line)
            current_file = m.group(1).split("/")[-1] if m else "unknown"
            add_count = del_count = 0
        elif line.startswith("+") and not line.startswith("+++"):
            add_count += 1
        elif line.startswith("-") and not line.startswith("---"):
            del_count += 1

    if current_file:
        changed_files[current_file] = {"add": add_count, "del": del_count}

    if not changed_files:
        return "(uncommitted 변경 없음)"

    for fname, counts in changed_files.items():
        summary_lines.append(f"  - {fname}: +{counts['add']} -{counts['del']}")

    return "\n".join(summary_lines)


def get_git_diff_vs_prev_test(task_dir: str, prev_test_commit: str | None) -> str:
    """이전 테스트 커밋 대비 현재 HEAD diff (핵심 변경만)."""
    if not prev_test_commit:
        return "(이전 기록 없음)"
    diff = run(["git", "diff", f"{prev_test_commit}..HEAD", "--", task_dir])
    if not diff:
        return "(이전 커밋과 동일)"

    # 변경된 파일별 통계
    stat = run(["git", "diff", "--stat", f"{prev_test_commit}..HEAD", "--", task_dir])
    return stat if stat else "(diff 있음)"


def extract_key_params(cfg_file: Path) -> dict[str, str]:
    """env_cfg.py에서 핵심 하이퍼파라미터 추출."""
    if not cfg_file.exists():
        return {}
    text = cfg_file.read_text(encoding="utf-8")
    params = {}
    for pattern in KEY_PARAM_PATTERNS:
        m = re.search(pattern, text)
        if m:
            key_m = re.match(r"\w+", pattern)
            if key_m:
                params[key_m.group(0)] = m.group(1)
    return params


def get_previous_test_info(history_file: Path) -> tuple[str | None, dict[str, str]]:
    """test_history.md에서 마지막 test의 (commit_hash, params) 추출."""
    if not history_file.exists():
        return None, {}

    text = history_file.read_text(encoding="utf-8")
    # 마지막 commit 해시
    commit_m = re.findall(r"\*\*Commit\*\*: `([0-9a-f]+)`", text)
    last_commit = commit_m[-1] if commit_m else None

    # 마지막 params 테이블
    params: dict[str, str] = {}
    param_section = re.findall(r"\| (\w+) \| [^|]+ \| ([^|]+) \|", text)
    for key, val in param_section[-20:]:
        params[key.strip()] = val.strip()

    return last_commit, params


def build_param_table(current: dict[str, str], previous: dict[str, str]) -> str:
    """현재/이전 파라미터 비교 테이블."""
    all_keys = list(dict.fromkeys(list(previous.keys()) + list(current.keys())))
    rows = ["| 파라미터 | 이전 | 현재 | 변경 |",
            "|---------|------|------|------|"]
    for key in all_keys:
        prev_val = previous.get(key, "-")
        curr_val = current.get(key, "-")
        changed = "✓" if prev_val != curr_val and prev_val != "-" else ""
        rows.append(f"| {key} | {prev_val} | {curr_val} | {changed} |")
    return "\n".join(rows)


def append_to_history(history_file: Path, entry: str) -> None:
    history_file.parent.mkdir(parents=True, exist_ok=True)
    if history_file.exists():
        existing = history_file.read_text(encoding="utf-8")
        if not existing.endswith("\n"):
            existing += "\n"
        history_file.write_text(existing + "\n---\n\n" + entry, encoding="utf-8")
    else:
        header = f"# {history_file.parent.name} Test History\n\n"
        history_file.write_text(header + entry, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="Task name, e.g. 5g_pour_right_v5")
    ap.add_argument("--test", required=True, help="Test name, e.g. test6")
    ap.add_argument("--note", default="", help="Optional note about this test")
    args = ap.parse_args()

    task_dir = f"source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/right/{args.task}"
    log_dir = LOG_ROOT / args.task
    history_file = log_dir / "test_history.md"
    cfg_file = SRC_ROOT / args.task / "pour_right_env_cfg.py"
    if not cfg_file.exists():
        cfg_file = SRC_ROOT / args.task / "grasp_right_env_cfg.py"

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    short_hash, commit_msg = get_git_head()
    prev_commit, prev_params = get_previous_test_info(history_file)

    current_params = extract_key_params(cfg_file)
    diff_vs_prev = get_git_diff_vs_prev_test(task_dir, prev_commit)
    uncommitted = get_git_diff_summary(task_dir)
    param_table = build_param_table(current_params, prev_params)

    entry = f"""## {args.test}

- **Date**: {now}
- **Commit**: `{short_hash}` — {commit_msg}
- **vs 이전**: `{prev_commit or "없음"}`

### Uncommitted 변경 (학습 시작 시점)
{uncommitted}

### 커밋 기반 코드 변경 (vs 이전 test 커밋)
```
{diff_vs_prev}
```

### 핵심 하이퍼파라미터
{param_table}
"""

    if args.note:
        entry += f"\n### Note\n{args.note}\n"

    append_to_history(history_file, entry)
    print(f"✓ test_history.md 업데이트: {history_file}")
    print(f"  테스트: {args.test} | 커밋: {short_hash} | 날짜: {now}")


if __name__ == "__main__":
    main()
