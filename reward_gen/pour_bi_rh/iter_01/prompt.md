You are an expert in robotics, reinforcement learning and code generation.

We control a bimanual robot: two 7-DOF OpenArm arms, each carrying an Inspire RH56F1 five-finger \
hand, standing at a table. The RH56F1 is an UNDER-ACTUATED hand: only 6 joints per hand are driven \
(thumb abduction, thumb flexion, and one flexion drive for each of the index, middle, ring and pinky \
fingers); the remaining finger joints follow the driven ones through fixed mechanical couplings, so a \
finger always curls as a whole. The RIGHT arm is the "source" arm (src_*), the LEFT arm is the \
"receiver" arm (rcv_*). Two identical cups stand on the table: the source cup (in front of the right \
hand, filled with 20 small beads) and the receiver cup (in front of the left hand, empty). \
Positions are in metres in a frame whose origin is at the robot base; +z is up, +x is forward (away \
from the robot), +y is to the robot's left. The table top is at z = table_z.

The action space is a normalized `Box(-1, 1, (24,), float32)`:
  actions[0:6]   = source palm 6-DoF target offset (xyz + yaw/pitch/roll) — the arm is moved \
by a geometric-fabrics controller toward this target
  actions[6:12]  = source hand closure commands, one per driven joint \
(thumb abduction, thumb flexion, index, middle, ring, pinky; 0 = open, 1 = closed)
  actions[12:18] = receiver palm 6-DoF target offset
  actions[18:24] = receiver hand closure commands
The hand controller stops a finger automatically once one of its links touches its own cup \
(contact freeze); opening is always allowed. Fingers can only close when the palm is near its cup. \
The "cups" are slim shaker bodies about 6 cm in diameter and 11 cm tall, and the hand is small: a \
FINGERTIP (precision) grasp — thumb tip on one side, the tips of some fingers on the other — is the \
expected way to hold the cup; a full wrap-around power grasp is NOT required and usually not \
possible. The cups are light, so a finger brushing the cup easily knocks it over: approach slowly \
and touch it only with the fingertips.

Now I want you to help me write a reward function for reinforcement learning.

Typically, the reward function of a manipulation task consists of these parts (some are optional — include them only if really necessary):
1. the distance between the robot's hand and the target object (one term per arm)
2. the difference between the object's current state and its goal state
3. regularization of the robot's action
4. [optional] extra constraints on the target objects implied by the task (e.g. keep the receiver cup upright, do not spill)
5. [optional] extra constraints on the robot implied by the task

The reward function receives a single argument `ctx`, an instance of this class (all positions env-local, metres; angles rad; forces N):

```python
@dataclass(frozen=True)
class RewardContext:
    # ---- 상수 (에피소드 불변, python float) ---------------------------------------
    table_z: float            # 테이블 상면 높이 [m]
    cup_radius: float         # 컵 안쪽 반경 [m]
    cup_mouth_z: float        # 컵 원점(무게중심 부근) → 림(개구) 높이 [m]
    cup_bottom_z: float       # 컵 원점 → 바닥 높이 [m] (음수)
    num_beads: int            # 소스 컵에 든 비드 수

    # ---- 소스 팔 (비드가 든 컵을 잡아 붓는 팔) ------------------------------------
    src_palm_pos: torch.Tensor          # (N,3) 손바닥 위치
    src_palm_axes: torch.Tensor         # (N,6) 손바닥 회전행렬의 열 0(손바닥 법선)·열 1
    src_tips_pos: torch.Tensor          # (N,F,3) 손끝 위치 (F=손가락 수, 0번=엄지)
    src_hand_closure: torch.Tensor      # (N,) 실측 평균 손 폐쇄도 [0,1] (0=펴짐, 1=완전 파지)
    src_finger_force: torch.Tensor      # (N,F) 손가락별 소스 컵 접촉력 [N]
    src_palm_force: torch.Tensor        # (N,) 손바닥의 소스 컵 접촉력 [N]
    src_grasped: torch.Tensor           # (N,) bool 엄지 AND 다른 손가락이 동시에 소스 컵에 접촉
    src_arm_qd: torch.Tensor            # (N,A) 팔 관절 속도 [rad/s]
    # ---- 리시버 팔 (빈 컵을 잡아 받는 팔) -------------------------------------------
    rcv_palm_pos: torch.Tensor          # (N,3)
    rcv_palm_axes: torch.Tensor         # (N,6)
    rcv_tips_pos: torch.Tensor          # (N,F,3)
    rcv_hand_closure: torch.Tensor      # (N,)
    rcv_finger_force: torch.Tensor      # (N,F) 리시버 컵 접촉력
    rcv_palm_force: torch.Tensor        # (N,)
    rcv_grasped: torch.Tensor           # (N,) bool
    rcv_arm_qd: torch.Tensor            # (N,A)

    # ---- 소스 컵 (비드가 든 컵) ---------------------------------------------------
    src_cup_pos: torch.Tensor           # (N,3) 컵 원점 위치
    src_cup_quat: torch.Tensor          # (N,4) 자세 (w,x,y,z)
    src_cup_up: torch.Tensor            # (N,3) 컵의 위 방향 단위벡터 (직립이면 (0,0,1))
    src_cup_tilt: torch.Tensor          # (N,) 직립 대비 기울기 [rad] (0=직립, π/2=수평)
    src_cup_mouth_pos: torch.Tensor     # (N,3) 컵 개구(림) 중심 위치
    src_cup_lin_vel: torch.Tensor       # (N,3) [m/s]
    src_cup_ang_vel: torch.Tensor       # (N,3) [rad/s]
    src_cup_spawn_pos: torch.Tensor     # (N,3) 에피소드 시작 시 컵 위치(테이블 위)
    # ---- 리시버 컵 (빈 컵) --------------------------------------------------------
    rcv_cup_pos: torch.Tensor           # (N,3)
    rcv_cup_quat: torch.Tensor          # (N,4)
    rcv_cup_up: torch.Tensor            # (N,3)
    rcv_cup_tilt: torch.Tensor          # (N,)
    rcv_cup_mouth_pos: torch.Tensor     # (N,3)
    rcv_cup_lin_vel: torch.Tensor       # (N,3)
    rcv_cup_ang_vel: torch.Tensor       # (N,3)
    rcv_cup_spawn_pos: torch.Tensor     # (N,3)

    # ---- 비드 ----------------------------------------------------------------------
    bead_in_source_frac: torch.Tensor   # (N,) 소스 컵 안에 있는 비드 비율 [0,1]
    bead_in_target_frac: torch.Tensor   # (N,) 리시버 컵 안에 있고 소스 컵 밖인 비드 비율 [0,1] (소스 컵째 끼워 넣은 비드는 0)
    bead_spill_frac: torch.Tensor       # (N,) 어느 컵에도 없이 바닥/테이블로 떨어진 비율 [0,1]
    bead_centroid: torch.Tensor         # (N,3) 비드 무게중심 위치
    d_in_target: torch.Tensor           # (N,) 이번 스텝 bead_in_target_frac 증분 (Δ, 음수 가능)
    d_spill: torch.Tensor               # (N,) 이번 스텝 bead_spill_frac 증분

    # ---- 충돌 (실기 안전 — 09.14) ---------------------------------------------------------
    cup_cup_force: torch.Tensor         # (N,) 두 컵이 서로 부딪히는 접촉력 [N] (0 = 안 닿음)
    src_hand_foreign_force: torch.Tensor  # (N,) 소스 손이 **자기 컵 외**(상대 손·상대 컵·테이블)에 닿는 힘 [N]
    rcv_hand_foreign_force: torch.Tensor  # (N,) 리시버 손이 자기 컵 외에 닿는 힘 [N]

    # ---- 과제 판정 / 시간 ------------------------------------------------------------
    cups_nested: torch.Tensor           # (N,) bool 두 컵 원점 거리 < 9 cm — 소스 컵이 리시버 컵에 끼워져 있음(붓기가 아니라 성공 무효)
    success: torch.Tensor               # (N,) bool 성공 조건 충족 (env 가 판정, 보상이 바꿀 수 없음; cups_nested 면 항상 False)
    episode_progress: torch.Tensor      # (N,) 에피소드 진행도 [0,1]

    # ---- 액션 ------------------------------------------------------------------------
    actions: torch.Tensor               # (N,Dact) 이번 스텝 액션 [-1,1]
    prev_actions: torch.Tensor          # (N,Dact) 직전 스텝 액션
```

Additional knowledge:
1. Everything is BATCHED: every tensor field has a leading dimension N (number of parallel environments). The function must return a reward tensor of shape (N,) — never a Python float — and a dict of named component tensors, each of shape (N,). Use only `torch` and `math`; do not import anything else.
2. Write staged rewards with tensor masks, not `if`/`else` on tensors: `stage2 = torch.where(ctx.src_grasped, r_lift, torch.zeros_like(r_lift))`.
3. Prefer bounded, smooth shaping: `torch.exp(-k * dist)` or `1 - torch.tanh(k * dist)` instead of raw negative distances, so that no single term dominates.
4. `ctx.src_grasped` / `ctx.rcv_grasped` are True when the thumb AND another finger both press on that hand's own cup — this is the grasp-established signal. `ctx.*_hand_closure` is the measured closure (0 = open, 1 = fully closed).
5. Cup tilt: `ctx.src_cup_tilt` is the angle between the cup's up axis and world +z (0 = upright, π/2 = horizontal). Beads leave the source cup once it is tilted past roughly 110° (1.9 rad) with its mouth over the receiver cup's mouth. `ctx.src_cup_mouth_pos` and `ctx.rcv_cup_mouth_pos` are the rim centres.
6. Bead bookkeeping: `bead_in_target_frac` rises as beads land in the receiver cup; `bead_spill_frac` counts beads lost outside both cups (permanent). `d_in_target` and `d_spill` are this step's increments — reward INCREMENTS of beads transferred rather than the level, otherwise the policy is paid for standing still with a filled cup.
7. `ctx.success` is computed by the environment (enough beads in the receiver cup, little spill, cups close together, and the cups NOT nested). You may add a bonus on it but you cannot redefine it. Beads only count as "in the receiver" once they have LEFT the source cup — pushing the source cup into the receiver cup (`ctx.cups_nested`) transfers nothing and is never a success; the beads must fall out of the tilted source cup through the air.
8. Height above the table: `ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]` is how far the source cup has been lifted (0 while it rests on the table).
9. Do not keep any state between calls (no globals, no attributes); the function must be pure.
10. Each component you return is logged separately during training and shown back to you after training, so name them meaningfully (e.g. "approach_src", "grasp_rcv", "pour_delta").
11. This policy will be deployed on the real robot, so collisions are a safety hazard: `ctx.cup_cup_force` is the contact force between the two cups, and `ctx.src_hand_foreign_force` / `ctx.rcv_hand_foreign_force` are the forces each hand exerts on anything other than its own cup (the other hand, the other cup, the table). During a correct pour the cups do not touch each other and the hands touch only their own cup; penalise these forces (bounded, e.g. `torch.tanh(f / 5.0)`) so the policy learns to keep clearance. Cup poses seen by the policy are delayed and noisy as on the real perception system.

I want it to fulfil the following task: Using both arms, grasp each cup (a fingertip grasp is fine, no wrap-around needed): the right hand grasps the source cup (which contains the beads) and the left hand grasps the empty receiver cup. Lift both cups off the table, bring the mouth of the source cup over the mouth of the receiver cup, and tilt the source cup so that the beads pour into the receiver cup. Keep the receiver cup upright, do not drop either cup, and spill as few beads as possible. The task is finished when at least half of the beads are inside the receiver cup with little spill.
1. Please think step by step and tell me what this task means and which stages the robot must go through.
2. Then write a function with exactly this signature:
```python
def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
    return reward, {"component_name": component_tensor, ...}
```
   where `reward` has shape (N,) and every dict value has shape (N,).
3. Take care of tensor shapes and types; never call methods that are not listed in the class definition above.
4. Put the final code in ONE ```python fenced block that starts with `import torch` and `import math` and contains only the function (plus small helper functions if needed). Add short comments explaining weights, e.g. `# weight 0.5 works together with the pour term 5.0 — pouring must dominate once grasped`.

The previous reward function was:
```python
import torch
import math


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Euclidean distance along the last dim -> (...,)."""
    return torch.norm(a - b, dim=-1)


def _near(d: torch.Tensor, k: float) -> torch.Tensor:
    """Bounded closeness in (0, 1]: 1 at d = 0, ~0.37 at d = 1/k."""
    return torch.exp(-k * d)


def _lift_progress(cup_pos: torch.Tensor, spawn_pos: torch.Tensor, full_h: float) -> torch.Tensor:
    """Height above the spawn height, clamped to [0, 1] at full_h metres."""
    h = cup_pos[:, 2] - spawn_pos[:, 2]
    return torch.clamp(h / full_h, 0.0, 1.0)


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = torch.zeros_like(ctx.src_cup_tilt)          # (N,) float template, never touches ctx
    src_g = ctx.src_grasped.to(zero.dtype)              # (N,) 1.0 when thumb + finger on source cup
    rcv_g = ctx.rcv_grasped.to(zero.dtype)              # (N,)
    both_g = src_g * rcv_g

    # ------------------------------------------------------------------ stage 0: reach
    # palm -> own cup centre. k = 8: 1.0 at contact, 0.45 at 10 cm, 0.2 at 20 cm.
    d_src_palm = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv_palm = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * _near(d_src_palm, 8.0)         # weight 1.0 each arm
    approach_rcv = 1.0 * _near(d_rcv_palm, 8.0)

    # ------------------------------------------------------------------ stage 1: fingertip grasp
    # mean fingertip distance to the cup centre; only paid once the palm is near (< ~15 cm),
    # so fingers are not rewarded for curling in mid-air far from the cup. weight 0.5 (helper).
    tips_src_d = _dist(ctx.src_tips_pos, ctx.src_cup_pos[:, None, :]).mean(dim=-1)
    tips_rcv_d = _dist(ctx.rcv_tips_pos, ctx.rcv_cup_pos[:, None, :]).mean(dim=-1)
    palm_near_src = (d_src_palm < 0.15).to(zero.dtype)
    palm_near_rcv = (d_rcv_palm < 0.15).to(zero.dtype)
    tips_src = 0.5 * palm_near_src * _near(tips_src_d, 12.0)
    tips_rcv = 0.5 * palm_near_rcv * _near(tips_rcv_d, 12.0)
    # grasp-established bonus: weight 1.0 each (constant while held -> pays for keeping the grasp)
    grasp_src = 1.0 * src_g
    grasp_rcv = 1.0 * rcv_g

    # ------------------------------------------------------------------ stage 2: lift
    # full credit at 8 cm above spawn; gated by grasp so a knocked-up cup earns nothing.
    lift_p_src = _lift_progress(ctx.src_cup_pos, ctx.src_cup_spawn_pos, 0.08)
    lift_p_rcv = _lift_progress(ctx.rcv_cup_pos, ctx.rcv_cup_spawn_pos, 0.08)
    lift_src = 1.5 * src_g * lift_p_src                  # weight 1.5 each: lifting must beat approach
    lift_rcv = 1.5 * rcv_g * lift_p_rcv
    # "lifted" gate: both held and both at least 3 cm up
    lifted = both_g * (lift_p_src > 0.375).to(zero.dtype) * (lift_p_rcv > 0.375).to(zero.dtype)

    # ------------------------------------------------------------------ stage 3: align rims
    mouth_dxy = _dist(ctx.src_cup_mouth_pos[:, :2], ctx.rcv_cup_mouth_pos[:, :2])
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    # vertical window 4..12 cm above the receiver rim (zero error inside, linear outside);
    # too low -> nesting / rim contact, too high -> beads scatter.
    dz_err = torch.clamp(0.04 - mouth_dz, min=0.0) + torch.clamp(mouth_dz - 0.12, min=0.0)
    align_q = _near(mouth_dxy, 10.0) * _near(dz_err, 15.0)   # (N,) in (0,1]
    align = 2.0 * lifted * align_q                        # weight 2.0: > lift so carrying pays
    # alignment gate for tilting: rims within ~4 cm horizontally and inside the height window
    aligned = lifted * (mouth_dxy < 0.04).to(zero.dtype) * (dz_err < 0.01).to(zero.dtype)

    # ------------------------------------------------------------------ stage 4: tilt and pour
    # tilt progress towards ~2.0 rad (beads leave past ~1.9 rad); only paid when aligned.
    tilt_p = torch.clamp(ctx.src_cup_tilt / 2.0, 0.0, 1.0)
    tilt = 2.0 * aligned * tilt_p                         # weight 2.0 works with pour 30: tilt is only the doorway
    # pay the INCREMENT of beads transferred (episode sum <= 30) -> pouring dominates everything
    pour_delta = 30.0 * ctx.d_in_target
    spill_delta = -20.0 * ctx.d_spill                     # spilled beads are permanent: nearly as costly as a transfer is valuable

    # ------------------------------------------------------------------ stage 5: success
    success = 5.0 * ctx.success.to(zero.dtype)            # per-step bonus for reaching and holding the goal

    # ------------------------------------------------------------------ constraints (always on)
    # tilting the source cup before the rims are aligned spills on the table; tilts below 0.8 rad
    # are harmless while carrying, so only the excess is penalised.
    pre_tilt_pen = -1.0 * (1.0 - aligned) * torch.tanh(2.0 * torch.clamp(ctx.src_cup_tilt - 0.8, min=0.0))
    # receiver must stay upright the whole time
    rcv_upright_pen = -1.0 * torch.tanh(3.0 * ctx.rcv_cup_tilt)
    # dropped / thrown cups: bounded speed penalty on both cups (0.3 m/s -> ~0.29 each)
    drop_pen = -0.5 * (torch.tanh(_dist(ctx.src_cup_lin_vel, torch.zeros_like(ctx.src_cup_lin_vel)) / 1.0)
                       + torch.tanh(_dist(ctx.rcv_cup_lin_vel, torch.zeros_like(ctx.rcv_cup_lin_vel)) / 1.0))
    # nested cups transfer nothing and never succeed
    nested_pen = -2.0 * ctx.cups_nested.to(zero.dtype)
    # real-robot safety: cups touching each other, hands touching anything but their own cup
    cup_collision_pen = -1.0 * torch.tanh(ctx.cup_cup_force / 5.0)
    hand_foreign_pen = -1.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                               + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    # fingertip grasp wanted: palm pressing on the cup means a shove / power grasp attempt
    palm_push_pen = -0.5 * (torch.tanh(ctx.src_palm_force / 5.0) + torch.tanh(ctx.rcv_palm_force / 5.0))

    # ------------------------------------------------------------------ regularisation
    # action rate: mean squared change, bounded by 4 since actions are in [-1, 1]; weight 0.1
    action_rate_pen = -0.1 * ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    # arm joint speed: slow, smooth motion near light cups; tanh keeps it <= 0.05 per arm
    arm_speed_pen = -0.05 * (torch.tanh((ctx.src_arm_qd ** 2).mean(dim=-1))
                             + torch.tanh((ctx.rcv_arm_qd ** 2).mean(dim=-1)))

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "tips_src": tips_src,
        "tips_rcv": tips_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "align": align,
        "tilt": tilt,
        "pour_delta": pour_delta,
        "spill_delta": spill_delta,
        "success": success,
        "pre_tilt_pen": pre_tilt_pen,
        "rcv_upright_pen": rcv_upright_pen,
        "drop_pen": drop_pen,
        "nested_pen": nested_pen,
        "cup_collision_pen": cup_collision_pen,
        "hand_foreign_pen": hand_foreign_pen,
        "palm_push_pen": palm_push_pen,
        "action_rate_pen": action_rate_pen,
        "arm_speed_pen": arm_speed_pen,
    }
    reward = zero
    for v in components.values():
        reward = reward + v
    return reward, components
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.91e-06 · max 9.77e-05
bead/spill: [4.88e-05, 0, 4.88e-05, 0, 0.000537, 0.000342, 0.000244, 0.000244, 0.000439, 0.000342]  min 0 · mean 0.0006 · max 0.00557
done/drop: [0.00195, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000241 · max 0.0107
done/mimic_runaway: [0.00293, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000147 · max 0.00293
episode_lengths/step: [114, 841, 831, 855, 815, 860, 834, 846, 777, 789]  min 114 · mean 807 · max 891
reward/action_rate_pen: [-0.102, -0.0877, -0.0764, -0.0711, -0.0705, -0.0653, -0.067, -0.0625, -0.0656, -0.0679]  min -0.103 · mean -0.0722 · max -0.0624
reward/align: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/approach_rcv: [0.178, 0.165, 0.197, 0.206, 0.189, 0.186, 0.19, 0.186, 0.206, 0.207]  min 0.124 · mean 0.192 · max 0.219
reward/approach_src: [0.178, 0.479, 0.601, 0.624, 0.644, 0.631, 0.638, 0.643, 0.636, 0.64]  min 0.173 · mean 0.595 · max 0.656
reward/arm_speed_pen: [-0.0568, -0.0213, -0.0161, -0.0123, -0.0128, -0.0115, -0.0121, -0.012, -0.0103, -0.0103]  min -0.0568 · mean -0.0145 · max -0.01
reward/cup_collision_pen: [-3.68e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00196 · mean -2.65e-05 · max 0
reward/drop_pen: [-0.0607, -0.0329, -0.044, -0.0406, -0.0446, -0.0421, -0.0482, -0.0451, -0.0495, -0.0477]  min -0.0607 · mean -0.0441 · max -0.0315
reward/grasp_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grasp_src: [0, 0, 0.0117, 0.0146, 0.0898, 0.103, 0.201, 0.308, 0.332, 0.369]  min 0 · mean 0.156 · max 0.408
reward/hand_foreign_pen: [-0.00229, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00722 · mean -0.000245 · max 0
reward/lift_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/lift_src: [0, 0, 0.000226, 0.000507, 0.00262, 0.0043, 0.00803, 0.0112, 0.0116, 0.0124]  min 0 · mean 0.00573 · max 0.0149
reward/nested_pen: [-0.00391, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0156 · mean -0.000357 · max 0
reward/palm_push_pen: [0, -0.000721, -0.00161, -0.000152, -0.000716, -0.000635, -0.0004, -0.000856, -0.000946, -0.00173]  min -0.00439 · mean -0.000725 · max 0
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0.00146, 0, 0, 0]  min -0.00146 · mean 0 · max 0.00146
reward/pre_tilt_pen: [-0.0127, 0, 0, 0, -0.000508, 0, -0.00179, 0, -0.000659, 0]  min -0.0423 · mean -0.00129 · max 0
reward/rcv_upright_pen: [-0.12, -0.00137, -0.00137, -0.00275, -0.00241, -0.00132, -0.00348, -0.00134, -0.00309, -0.00276]  min -0.12 · mean -0.00323 · max -0.00122
reward/spill_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0117 · mean -7.93e-05 · max 0.00391
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tips_rcv: [0.0103, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 6.6e-05 · max 0.0103
reward/tips_src: [0.00784, 0.143, 0.189, 0.212, 0.205, 0.221, 0.227, 0.23, 0.231, 0.234]  min 0.00455 · mean 0.201 · max 0.238
reward/total: [0.00552, 0.633, 0.851, 0.923, 0.99, 1.02, 1.13, 1.25, 1.28, 1.32]  min -0.0483 · mean 1 · max 1.37
rewards/step: [-7.66, 488, 669, 780, 791, 871, 938, 1.01e+03, 961, 1.01e+03]  min -7.66 · mean 820 · max 1.13e+03
task/aim_dist: [0.309, 0.321, 0.311, 0.309, 0.279, 0.295, 0.546, 0.273, 0.269, 0.276]  min 0.264 · mean 0.317 · max 2.89
task/cup_collision_rate: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.39e-05 · max 0.00391
task/cups_center_dist: [0.313, 0.321, 0.312, 0.309, 0.278, 0.295, 0.545, 0.271, 0.269, 0.276]  min 0.267 · mean 0.317 · max 2.89
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0.00195, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000178 · max 0.00781
task/rcv_cup_lift: [0.000394, 1.28e-05, 1.57e-05, 9.25e-06, 0.000244, 1.18e-05, 8.15e-06, 1.35e-05, 2.57e-05, 0.000828]  min -2.69e-05 · mean 9.42e-05 · max 0.019
task/rcv_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/rcv_hand_foreign_rate: [0.000977, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 5.1e-05 · max 0.00195
task/src_cup_lift: [0.000776, 0.000205, 0.000763, 0.000906, 0.00123, 0.00133, 0.0324, 0.00157, 0.00173, 0.00164]  min -0.000619 · mean 0.00519 · max 0.175
task/src_grasped: [0, 0, 0.0117, 0.0146, 0.0898, 0.103, 0.201, 0.308, 0.332, 0.369]  min 0 · mean 0.156 · max 0.408
task/src_hand_foreign_rate: [0.00195, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000258 · max 0.00781
task/src_tilt_deg: [4.06, 0.744, 2.71, 3.13, 4.98, 5.2, 6.6, 5.78, 6.5, 6.09]  min 0.475 · mean 4.85 · max 8.76
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Observations from watching the trained policy (rollout video at epoch 650, camera on the robot's right side, 4 envs) plus the training metrics. These are facts about what the policy does, not design instructions.

1. The RIGHT (source) hand does not form a usable grasp. It reaches its cup with the palm held too HIGH relative to the cup: the four fingers curl around the upper cup body, but the thumb arches over the top and its tip rests on the cup MOUTH (the rim of the opening) instead of pressing on the cup wall opposite the fingers. The operator identified this thumb-on-rim hook as the reason the grasp fails. The hand keeps this pose for the whole episode and the cup never leaves the table. `task/src_grasped` still rose from 0 to 0.36 (max 0.39), because that flag is true whenever any thumb link (the two distal segments or the tip) and any of the four fingers each press on the cup with more than 1 N, regardless of where on the cup the contact is, so the thumb resting on the rim counts as grasped. `task/src_cup_lift` stayed at about 0.002 m from epoch 360 to epoch 670 (the only large values coincide with physics blow-ups). The arm does not move upward after closing the hand.

2. The LEFT (receiver) hand never touches its cup. It hovers above and behind the receiver cup with the fingers half closed, drifting slightly, for the whole episode. `task/rcv_grasped` was 0 for the entire run and `reward/tips_rcv` was 0 after the first epochs. The average palm-to-cup distance implied by `reward/approach_rcv` (0.21) is about 0.19 m, so the receiver palm rarely comes within 0.15 m of its cup. The source palm reached about 0.06 m on average.

3. Because the receiver cup is never held, every term that requires both cups held and both lifted at least 3 cm was zero for the whole run: `reward/align`, `reward/tilt`, `reward/success` and `task/episode_success` were exactly 0, and `reward/pour_delta` stayed at or below 0.0015. Per-step reward at the end was dominated by approach_src 0.64, grasp_src 0.36, tips_src 0.23 and approach_rcv 0.21; lift_src was 0.012. `reward/total` plateaued around 1.3 from epoch 600.

4. Environment facts unchanged from the previous function: RH56F1 hands have 6 driven joints each (the other finger joints follow by mimic coupling), fingertip grasps are acceptable, the cup is a slim shaker (~0.06 m diameter, 0.134 kg). Episodes in which a hand joint blows up physically are terminated (`done/mimic_runaway`, about 0.02% of episodes). No cup–cup or hand–hand contact occurred (`task/cup_collision_rate` 0, hand_foreign rates ≤ 0.0003).

The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
