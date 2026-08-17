#!/usr/bin/env python3
"""학습 시작 전 현재 코드 상태를 test_history.md에 스냅샷으로 기록.

Usage:
    python record_test_snapshot.py --task open-tesol_r_pour_v3 --test test8
    python record_test_snapshot.py --task open-tesol_r_grasp_v11 --test test3

실행 위치: /home/user/rl_ws/hdgp (git 루트)
출력: log/rl_games/<robot>/<side>/<task-ver>/test_history.md 에 새 항목 append
"""

import argparse
import datetime
import re
import subprocess
from pathlib import Path

GIT_ROOT = Path(__file__).resolve().parents[2]  # hdgp/ (scripts/tools/ 기준)
SRC_BASE = GIT_ROOT / "source/openarm/openarm"
LOG_BASE  = GIT_ROOT / "log/rl_games"

# gym ID → (robot_dir, side_dir, task_dir) 파싱
_ROBOT_MAP = {"tesol": "tesollo", "rh56f1": "rh56f1", "gripper": "gripper"}
_SIDE_MAP  = {"r": "right", "l": "left", "b": "both"}
_SUFFIX_RE = re.compile(r"-(lstm|play|bc|il2?|diffusion|nobias).*$")

def parse_task_id(task_id: str) -> tuple[str, str, str, str]:
    """gym task ID → (robot_dir, side_dir, task_dir, log_subpath).

    open-tesol_r_pour_v3       → (tesollo, right, pour_v3,  open-tesol/right/pour-v3)
    open-tesol_r_pour_v3-lstm  → (tesollo, right, pour_v3,  open-tesol/right/pour-v3)
    open-rh56f1_r_grasp_v1    → (rh56f1,  right, grasp_v1, open-rh56f1/right/grasp-v1)
    """
    name = task_id.removeprefix("open-")
    name = _SUFFIX_RE.sub("", name)          # 접미사 제거
    parts = name.split("_")                  # ['tesol', 'r', 'pour', 'v3']
    robot_short = parts[0]
    side_short  = parts[1]
    task        = "_".join(parts[2:])        # pour_v3

    robot_dir = _ROBOT_MAP.get(robot_short, robot_short)
    side_dir  = _SIDE_MAP.get(side_short, side_short)
    log_sub   = f"open-{robot_short}/{side_dir}/{task.replace('_', '-', 1)}"
    return robot_dir, side_dir, task, log_sub


# 추출할 핵심 하이퍼파라미터 패턴 (env_cfg.py 기준)
def _param_pattern(name: str, is_bool: bool = False) -> str:
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
    short = run(["git", "rev-parse", "--short", "HEAD"])
    msg   = run(["git", "log", "-1", "--format=%s"])
    return short, msg


def get_git_diff_summary(task_dir: str) -> str:
    diff = run(["git", "diff", "HEAD", "--", task_dir])
    if not diff:
        diff = run(["git", "diff", "--cached", "--", task_dir])
    if not diff:
        return "(uncommitted 변경 없음)"

    lines = diff.splitlines()
    current_file = None
    add_count = del_count = 0
    changed_files: dict[str, dict] = {}

    for line in lines:
        if line.startswith("diff --git"):
            if current_file and (add_count or del_count):
                changed_files[current_file] = {"add": add_count, "del": del_count}
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

    return "\n".join(f"  - {f}: +{c['add']} -{c['del']}" for f, c in changed_files.items())


def get_git_diff_vs_prev_test(task_dir: str, prev_commit: str | None) -> str:
    if not prev_commit:
        return "(이전 기록 없음)"
    stat = run(["git", "diff", "--stat", f"{prev_commit}..HEAD", "--", task_dir])
    return stat if stat else "(이전 커밋과 동일)"


def extract_key_params(src_dir: Path) -> dict[str, str]:
    for name in ("pour_right_env_cfg.py", "grasp_right_env_cfg.py", "env_cfg.py"):
        cfg_file = src_dir / name
        if cfg_file.exists():
            text = cfg_file.read_text(encoding="utf-8")
            params = {}
            for pattern in KEY_PARAM_PATTERNS:
                m = re.search(pattern, text)
                if m:
                    key_m = re.match(r"\w+", pattern)
                    if key_m:
                        params[key_m.group(0)] = m.group(1)
            return params
    return {}


def get_previous_test_info(history_file: Path) -> tuple[str | None, dict[str, str]]:
    if not history_file.exists():
        return None, {}

    text = history_file.read_text(encoding="utf-8")
    commits = re.findall(r"\*\*Commit\*\*: `([0-9a-f]+)`", text)
    last_commit = commits[-1] if commits else None

    params: dict[str, str] = {}
    for key, val in re.findall(r"\| (\w+) \| [^|]+ \| ([^|]+) \|", text)[-20:]:
        params[key.strip()] = val.strip()

    return last_commit, params


def build_param_table(current: dict[str, str], previous: dict[str, str]) -> str:
    all_keys = list(dict.fromkeys(list(previous.keys()) + list(current.keys())))
    rows = ["| 파라미터 | 이전 | 현재 | 변경 |", "|---------|------|------|------|"]
    for key in all_keys:
        prev_val = previous.get(key, "-")
        curr_val = current.get(key, "-")
        changed = "✓" if prev_val != curr_val and prev_val != "-" else ""
        rows.append(f"| {key} | {prev_val} | {curr_val} | {changed} |")
    return "\n".join(rows)


# /rl-diag가 학습 후 분석을 덧붙이는 앵커. 이 토큰을 바꾸면 rl-diag 커맨드도 같이 바꿀 것.
ANALYSIS_ANCHOR = "<!-- rl-diag-analysis -->"


def write_run_snapshot(task_id: str, run_dir: Path, test_label: str = "", note: str = "") -> Path:
    """run 폴더 안에 self-contained 실험 기록(test_history.md)을 생성(per-run).

    상단 = 코드 스냅샷(학습 시작 시점), 하단 = rl-diag 분석 앵커(학습 후 /rl-diag가 채움).
    전체 cfg(seed·num_envs·모든 weight)는 같은 폴더의 params/env.yaml·agent.yaml이
    authoritative → 여기선 중복하지 않고 포인터만 둔다. 폴더 통째 이동 시 가설+코드+결과가 함께 따라감.
    """
    robot_dir, side_dir, task_dir_name, _ = parse_task_id(task_id)
    src_dir  = SRC_BASE / robot_dir / side_dir / task_dir_name
    task_rel = str(src_dir.relative_to(GIT_ROOT))

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    short_hash, commit_msg = get_git_head()
    uncommitted    = get_git_diff_summary(task_rel)
    label = test_label or run_dir.name
    robot_asset = read_robot_asset(run_dir)

    entry = f"""# {run_dir.name} — 실험 기록

## 코드 스냅샷 (학습 시작 시점)
- **라벨**: {label}
- **Date**: {now}
- **Task**: {task_id}
- **Commit**: `{short_hash}` — {commit_msg}
- **로봇 자산**: `{robot_asset}`
- **전체 설정**: `params/env.yaml` (seed·num_envs·모든 weight) · `params/agent.yaml`

### Uncommitted 변경
{uncommitted}

### Note (가설)
{note or "(없음)"}

## 분석 (rl-diag) — 학습 후 `/rl-diag {run_dir}` 가 채움
{ANALYSIS_ANCHOR}
"""

    out = run_dir / "test_history.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(entry, encoding="utf-8")
    return out


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


def read_robot_asset(run_dir: Path) -> str:
    """params/env.yaml 에서 로봇 USD 를 뽑아 자산 신원을 기록한다.

    ★2026-08-17: 로봇 자산이 DG-5F→DG-5FS 로 바뀌었는데 런 기록 어디에도 자산이 없어
      "이 수치가 어떤 로봇에서 나왔는가"를 사후에 확인할 수 없었다. env.yaml 안에는
      있었지만 사람이 열어보지 않는다 — 헤더로 끌어올린다.
    """
    env_yaml = run_dir / "params" / "env.yaml"
    if not env_yaml.is_file():
        return "(env.yaml 없음 — 스냅샷이 dump 보다 먼저 실행됐다)"
    try:
        text = env_yaml.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(env.yaml 읽기 실패: {exc})"
    # usd_path 중 robot/ 을 포함하는 첫 항목 (컵·테이블 usd 와 구분)
    for line in text.splitlines():
        if "usd_path" in line and "/robot/" in line:
            return line.split("usd_path", 1)[1].lstrip(": ").strip()
    return "(robot usd_path 미발견)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="Gym task ID, e.g. open-tesol_r_pour_v3")
    ap.add_argument("--test", required=True, help="Test name, e.g. test8")
    ap.add_argument("--note", default="", help="Optional note about this test")
    ap.add_argument(
        "--run-dir",
        default=None,
        help="설정 시 해당 run 폴더 안에 self-contained test_history.md 생성(per-run 모드). "
        "미설정 시 task 레벨 누적 파일에 append(레거시).",
    )
    args = ap.parse_args()

    if args.run_dir:
        out = write_run_snapshot(args.task, Path(args.run_dir), args.test, args.note)
        print(f"✓ run 스냅샷 기록: {out}")
        return

    robot_dir, side_dir, task_dir_name, log_sub = parse_task_id(args.task)

    src_dir      = SRC_BASE / robot_dir / side_dir / task_dir_name
    log_dir      = LOG_BASE / log_sub
    history_file = log_dir / "test_history.md"
    task_rel     = str(src_dir.relative_to(GIT_ROOT))

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    short_hash, commit_msg = get_git_head()
    prev_commit, prev_params = get_previous_test_info(history_file)

    current_params = extract_key_params(src_dir)
    diff_vs_prev   = get_git_diff_vs_prev_test(task_rel, prev_commit)
    uncommitted    = get_git_diff_summary(task_rel)
    param_table    = build_param_table(current_params, prev_params)

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
