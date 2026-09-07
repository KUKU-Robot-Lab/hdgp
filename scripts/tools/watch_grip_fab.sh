#!/usr/bin/env bash
# 좌팔 fab 런 모니터링 — 지령/실제/추종오차를 보상과 나란히 본다.
# 사용: scripts/tools/watch_grip_fab.sh <RUN_LABEL> [host]
set -u
RUN="${1:?RUN_LABEL 필요}"; HOST="${2:-vision-3090}"
ROOT=$([ "$HOST" = "vision-3090" ] && echo /home/usr || echo /home/user)
SCRATCH="${SCRATCH_DIR:-/tmp/claude-1000/-home-user-rl-ws/2cf50ea6-1198-4eb8-a7a8-77e79965e055/scratchpad}"
D="$SCRATCH/$RUN"; mkdir -p "$D"

echo "=== $RUN @ $HOST ==="
ssh "$HOST" "tail -1 ~/$RUN.log 2>/dev/null; echo -n 'traceback/CUDA 오류: '; grep -c 'Traceback\|CUDA error' ~/$RUN.log 2>/dev/null; \
  echo -n 'PhysX 경고: '; grep -cE 'buffer overflow|Contacts have been dropped|PxGpuDynamicsMemoryConfig' ~/$RUN.log 2>/dev/null; \
  grep 'saving next best' ~/$RUN.log 2>/dev/null | tail -1"
rsync -a "$HOST:$ROOT/rl_ws/hdgp/log/rl_games/open-grip/left/grasp-sensor-fab/$RUN/summaries/" "$D/" 2>/dev/null
f=$(ls "$D"/events.out.tfevents.* 2>/dev/null | head -1)
[ -z "$f" ] && { echo "tfevents 없음"; exit 1; }
python3 "$(dirname "$0")/parse_tfevents.py" "$f" --out "$D.json" >/dev/null 2>&1 || { echo "파싱 실패"; exit 1; }
RUN="$RUN" SCRATCH="$SCRATCH" python3 - <<'PY'
import json, os, pathlib
d = json.loads(pathlib.Path(f"{os.environ['SCRATCH']}/{os.environ['RUN']}.json").read_text())
def s(k):
    for kk in d:
        if kk.endswith(k): return d[kk]
    # ★HDGP_TB_GROUPS=1 이면 태그가 `Rewards/…`·`task/…` 로 바뀐다 —
    #   옛 런(Episode/Episode_Reward/…)도 계속 읽히게 **잎 이름**으로도 찾는다.
    leaf = k.rsplit('/', 1)[-1]
    for kk in d:
        if kk.rsplit('/', 1)[-1] == leaf: return d[kk]
    return []
last = max((st for st, _ in s("rewards/iter")), default=0)
w = max(50, (last // 4) or 50)
bands = [(0, w), (w, 2*w), (2*w, 3*w), (3*w, last + 1)]
bands = [b for b in bands if b[0] <= last]
def band(k, lo, hi):
    v = [x for st, x in s(k) if lo <= st < hi]
    return sum(v)/len(v) if v else None
# ★TB 의 Episode_Reward 는 (Σ raw·dt)/episode_length_s 다. 에피소드가 만기 전에 끝나면
#   그 비율만큼 작게 찍힌다 — 위치를 m 로 읽으려면 되돌려야 한다.
#   분모는 추정하지 않고 **측정한다** — `diag_duty`(raw=1)의 로깅값이 곧 그 비율이다.
def scale(lo, hi):
    duty = band("Episode_Reward/diag_duty", lo, hi)
    return (1.0 / duty) if duty else 1.0
def show(title, keys, fmt="{:.4f}", norm=False):
    print(f"\n── {title}")
    print(f"{'':<26}" + "".join(f"{f'ep{a}-{b}':>12}" for a, b in bands))
    for k, label in keys:
        row = [band(k, a, b) for a, b in bands]
        if norm:
            row = [None if r is None else r * scale(a, b) for r, (a, b) in zip(row, bands)]
        if all(r is None for r in row):
            print(f"{label:<26}{'(없음)':>12}"); continue
        print(f"{label:<26}" + "".join(
            f"{(fmt.format(r) if r is not None else '-'):>12}" for r in row))
print(f"최근 epoch {last}")
show("지령 vs 실제 (m)", [
    ("Episode_Reward/diag_cmd_x", "지령 x"), ("Episode_Reward/diag_jaw_x", "  실제 x"),
    ("Episode_Reward/diag_cmd_y", "지령 y"), ("Episode_Reward/diag_jaw_y", "  실제 y"),
    ("Episode_Reward/diag_cmd_z", "지령 z"), ("Episode_Reward/diag_jaw_z", "  실제 z"),
    ("Episode_Reward/diag_cmd_jaw_gap", "추종오차 |cmd-jaw|"),
    ("Episode_Reward/diag_cmd_step", "지령 이동/스텝"),
    ("Episode_Reward/diag_jaw_cup_dist", "턱-컵 거리"),
], norm=True)
show("보상", [
    ("Episode_Reward/reaching_object", "reaching"),
    ("Episode_Reward/cup_between_jaws", "between_jaws"),
    ("Episode_Reward/grip_closure_when_enclosed", "closure"),
    ("Episode_Reward/lifting_object", "lifting"),
    ("Episode_Reward/object_goal_tracking", "goal_track"),
    ("Episode_Reward/settled_at_goal", "settled"),
    ("Episode_Reward/dwell_at_goal", "dwell"),
    ("Episode_Reward/grasp_pose", "grasp_pose"),
    ("rewards/iter", "총보상"),
])
show("종료·탐색·ADR", [
    ("Episode_Termination/object_dropping", "drop"),
    ("Episode_Termination/object_out_of_workspace", "workspace 이탈"),
    ("episode_lengths/iter", "ep 길이"),
    ("losses/entropy", "entropy"),
    ("Episode/Curriculum/adr", "ADR 레벨"),
])
PY
