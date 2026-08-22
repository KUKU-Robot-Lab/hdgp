"""self-collision 을 켜도 되는지 **움직이는 중**에 확인하는 게이트.

`audit_self_collision.py` 는 **한 자세**의 raw/hull 거리만 본다. 학습 중 팔은 홈 ±0.5 rad
(scale × 액션 범위)를 훑으므로, 감사를 통과해도 중간 자세에서 유령접촉이 날 수 있다.
그리고 USD 는 raw 도 hull 도 아닌 **convexDecomposition** 으로 나가므로 감사의 raw 여유가
몇 mm 면 근사오차가 그걸 먹는다(좌팔 ABORTED 홈에서 raw 3.2 mm → 실측 5.4 kN 유령접촉).

여기서는 랜덤 액션으로 워크스페이스를 훑으며 **좌팔 링크별 접촉력**을 재고, 컵·테이블 접촉과
구분하기 위해 물체를 멀리 치운 모드를 함께 제공한다.

판정: `--no_object` 에서 좌팔 링크 접촉력이 **전부 0** 이면 켜도 된다.
      0 이 아니면 그 쌍을 자산의 `self_collision_filtered_pairs` 에 추가할 것.

사용:
  python scripts/probes/probe_selfcollision_gate.py --steps 400 --num_envs 64 --no_object
  python scripts/probes/probe_selfcollision_gate.py --self_collision off   # A/B 비교용
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-grip_l_grasp_sensor")
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--no_object", action="store_true",
                    help="컵을 멀리 치워 **자기충돌만** 남긴다.")
parser.add_argument("--self_collision", choices=["on", "off"], default=None,
                    help="cfg 값을 덮어쓴다(A/B 비교용). 미지정이면 cfg 그대로.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P  # noqa: E402

LEFT_BODIES = [f"l_al_{i}" for i in range(1, 8)] + [
    "l_hl_gripper_base", *P.GRIPPER_FINGER_BODIES
]
# 자기충돌 상대가 될 수 있는 전체 바디(반대 팔 포함).
ALL_BODIES = LEFT_BODIES + [f"r_al_{i}" for i in range(1, 8)] + ["body_link"]

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
if args.self_collision is not None:
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = (
        args.self_collision == "on"
    )
# ContactSensor 는 대상 바디에 contact reporter API 가 있어야 한다. 학습 cfg 는 이걸 켜지
# 않으므로(불필요한 비용) 프로브에서만 켠다.
cfg.scene.robot.spawn.activate_contact_sensors = True
sc_on = cfg.scene.robot.spawn.articulation_props.enabled_self_collisions
grav_on = not cfg.scene.robot.spawn.rigid_props.disable_gravity

# ★센서는 gym.make **전에** cfg.scene 에 붙여야 한다. 시뮬이 돌기 시작한 뒤 만들면
#   `'ContactSensor' object has no attribute '_ALL_INDICES'` 로 죽는다(실측).
# ★`net_forces_w` 는 **모든** 접촉을 합산하므로 테이블·컵 충돌이 섞인다(실측: 랜덤 액션이
#   팔을 테이블에 박아 핑거가 self_collision OFF 에서도 2 kN 을 받는다). 로봇 자신을
#   `filter_prim_paths_expr` 로 지정해 `force_matrix_w` 에서 **로봇-로봇 접촉만** 뽑는다.
#   ⚠ 필터 대상은 RigidBodyAPI 가 붙은 바디 prim 이어야 한다 — 루트 Xform 을 가리키면
#     force_matrix_w 가 항상 0 이 되어 "자기충돌 없음"으로 조용히 오판한다(저장소 재발 이력).
_ROBOT_BODIES = ["{ENV_REGEX_NS}/Robot/" + b for b in ALL_BODIES]
for b in LEFT_BODIES:
    setattr(cfg.scene, f"probe_contact_{b}", ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + b,
        filter_prim_paths_expr=[p for p in _ROBOT_BODIES if not p.endswith("/" + b)],
        history_length=1, track_air_time=False))

env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()

n_act = env.action_manager.total_action_dim
obj = env.scene["object"]
peak = {b: 0.0 for b in LEFT_BODIES}
t0 = time.time()
for _ in range(args.steps):
    if args.no_object:
        s = obj.data.root_state_w.clone()
        s[:, 0] = 5.0
        s[:, 1] = 5.0
        s[:, 7:] = 0.0
        obj.write_root_state_to_sim(s)
    # 랜덤 액션 — 정책이 실제로 탐색하는 범위(clip 후 ±1 부근)를 훑는다.
    env.step(torch.empty(env.num_envs, n_act, device=env.device).uniform_(-1.0, 1.0))
    for b in LEFT_BODIES:
        # force_matrix_w: (N, 1, n_filter, 3) — 로봇-로봇 접촉만.
        fm = env.scene.sensors[f"probe_contact_{b}"].data.force_matrix_w
        peak[b] = max(peak[b], float(fm.norm(dim=-1).max()))
fps = args.steps * env.num_envs / (time.time() - t0)

print(f"\n=== self-collision 게이트 · {args.task} ===")
print(f"  self_collision={'ON' if sc_on else 'OFF'} · gravity={'ON' if grav_on else 'OFF'} "
      f"· env {env.num_envs} · {args.steps} 스텝 · 물체 {'제거' if args.no_object else '있음'}")
print(f"  처리량 {fps:,.0f} env-step/s")
print("\n  좌팔 링크별 최대 **자기충돌** 접촉력 (force_matrix_w, 로봇-로봇만)")
worst = 0.0
for b in LEFT_BODIES:
    mark = "  ← 접촉" if peak[b] > 1.0 else ""
    worst = max(worst, peak[b])
    print(f"    {b:30s} {peak[b]:10.2f} N{mark}")

if args.no_object:
    # ★두 핑거가 서로 맞닿는 것은 **정상**이다. 그리퍼가 허공에서 완전히 닫히면 두 조가
    #   만나는데, 둘은 비인접(둘 다 gripper_base 의 자식)이라 PhysX 가 접촉을 만든다.
    #   실측 3.72 N — effort_limit 333 N 의 1% 라 파지에 영향이 없고, 컵(58~88 mm)은
    #   개구(84.5 mm) 안에서 훨씬 먼저 멈춘다. 오히려 self-collision OFF 일 때의
    #   손가락 관통보다 정확하다.
    arm_worst = max(peak[b] for b in LEFT_BODIES if b not in P.GRIPPER_FINGER_BODIES)
    fing_worst = max(peak[b] for b in P.GRIPPER_FINGER_BODIES)
    print(f"\n  팔 링크 최대 {arm_worst:.2f} N · 핑거끼리 최대 {fing_worst:.2f} N")
    if arm_worst > 1.0:
        print("  판정: FAIL — 팔 링크에 유령접촉이 있다. 그 쌍을 자산의 "
              "`self_collision_filtered_pairs` 에 추가할 것.")
    elif fing_worst > 50.0:
        print("  판정: WARN — 핑거끼리 접촉이 과하다. 폐쇄 지령 하한을 확인할 것.")
    else:
        print("  판정: PASS — 팔 링크 유령접촉 없음. self-collision 켜도 된다.")
else:
    print("\n  ※ 물체가 있으면 컵 접촉이 섞인다. 판정은 --no_object 로 할 것.")

env.close()
app.close()
