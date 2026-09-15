You are an expert in robotics, reinforcement learning and code generation.

We control a bimanual robot: two 7-DOF OpenArm arms, each carrying a 20-DOF five-finger Tesollo DG-5F hand, standing at a table. The RIGHT arm is the "source" arm (src_*), the LEFT arm is the "receiver" arm (rcv_*). Two identical cups stand on the table: the source cup (in front of the right hand, filled with 20 small beads) and the receiver cup (in front of the left hand, empty). Positions are in metres in a frame whose origin is at the robot base; +z is up, +x is forward (away from the robot), +y is to the robot's left. The table top is at z = table_z.

The action space is a normalized `Box(-1, 1, (18,), float32)`:
  actions[0:6]   = source palm 6-DoF target offset (xyz + yaw/pitch/roll) — the arm is moved by a geometric-fabrics controller toward this target
  actions[6]     = source thumb opposition command
  actions[7]     = source thumb closure command
  actions[8]     = source four-finger closure command (index, middle, ring and pinky close together; all flexion joints of a finger share this one command)
  actions[9:15]  = receiver palm 6-DoF target offset
  actions[15]    = receiver thumb opposition command
  actions[16]    = receiver thumb closure command
  actions[17]    = receiver four-finger closure command
Hand commands: -1 = open, +1 = fully closed. The hand controller stops a finger joint automatically once that finger link touches its own cup (contact freeze), so a closing command produces a wrap-around power grasp; opening is always allowed. Fingers can only close when the palm is near its cup. The environment low-pass filters the palm commands (exponential moving average) before they reach the arm controller, so switching a palm command between extremes from one step to the next barely moves the arm. `ctx.actions` / `ctx.prev_actions` are the raw policy outputs before that filter.

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
7. `ctx.success` is computed by the environment (enough beads in the receiver cup, little spill, cups close together, the receiver cup held nearly upright — `ctx.rcv_cup_tilt` at most 20° — and the cups NOT nested). You may add a bonus on it but you cannot redefine it. Beads only count as "in the receiver" once they have LEFT the source cup — pushing the source cup into the receiver cup (`ctx.cups_nested`) transfers nothing and is never a success; the beads must fall out of the tilted source cup through the air.
8. Height above the table: `ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]` is how far the source cup has been lifted (0 while it rests on the table).
9. Do not keep any state between calls (no globals, no attributes); the function must be pure.
10. Each component you return is logged separately during training and shown back to you after training, so name them meaningfully (e.g. "approach_src", "grasp_rcv", "pour_delta").
11. This policy will be deployed on the real robot, so collisions are a safety hazard: `ctx.cup_cup_force` is the contact force between the two cups, and `ctx.src_hand_foreign_force` / `ctx.rcv_hand_foreign_force` are the forces each hand exerts on anything other than its own cup (the other hand, the other cup, the table). During a correct pour the cups do not touch each other and the hands touch only their own cup; penalise these forces (bounded, e.g. `torch.tanh(f / 5.0)`) so the policy learns to keep clearance. Cup poses seen by the policy are delayed and noisy as on the real perception system.

I want it to fulfil the following task: Using both arms, grasp each cup: the right hand grasps the source cup (which contains the beads) and the left hand grasps the empty receiver cup. Lift both cups off the table, bring the mouth of the source cup over the mouth of the receiver cup, and tilt the source cup so that the beads pour into the receiver cup. Keep the receiver cup upright, do not drop either cup, and spill as few beads as possible. The task is finished when at least half of the beads are inside the receiver cup with little spill.
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
    # (N,3),(N,3) -> (N,)
    return torch.norm(a - b, dim=-1)


def _rim_low_point(mouth_pos: torch.Tensor, up: torch.Tensor, radius: float) -> torch.Tensor:
    # Lowest point of the rim circle = the "pouring lip" the beads leave from.
    # Direction = -z projected onto the rim plane (perpendicular to the cup's up axis):
    #   p = -z_hat + (u . z_hat) u ;  lip = mouth + r * p / |p|
    # For an upright cup p -> 0 and the lip collapses to the mouth centre (eps keeps it finite).
    z_hat = torch.cat([torch.zeros_like(up[:, :2]), torch.ones_like(up[:, 2:3])], dim=-1)  # (N,3)
    p = up * up[:, 2:3] - z_hat                                                             # (N,3)
    p_hat = p / (torch.norm(p, dim=-1, keepdim=True) + 1e-6)
    return mouth_pos + radius * p_hat


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    # x already normalised; clamp to [0,1] then 3x^2 - 2x^3 (smooth ends, steepest in the middle)
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _wrap_quality(finger_force: torch.Tensor, palm_force: torch.Tensor, closure: torch.Tensor,
                  palm_axes: torch.Tensor, cup_up: torch.Tensor, cup_pos: torch.Tensor,
                  palm_pos: torch.Tensor) -> torch.Tensor:
    # Power-grasp quality in [0,1]. Weighted MEAN so each ingredient keeps its own gradient.
    FINGER_ON = 0.3      # [N] a finger counts as "on the cup" above this force
    # iter_06: 2 N -> 3 N. With 2 N the post-pour collapse (palm 7 N -> 2.2 N, fingertip pinch) only moved
    # this ingredient 1.0 -> 0.8; with 3 N it moves 0.98 -> 0.63, so wrap/hold/success notice the pinch.
    PALM_SCALE = 3.0
    LEVEL_SIGMA = 0.05   # [m] cup origin within ~5 cm of palm height along the cup axis
    # closure saturates at 0.35-0.43 on this cup under contact freeze whatever the grasp type
    CLOS_LO, CLOS_HI = 0.25, 0.40
    dt = closure.dtype
    finger_frac = torch.mean((finger_force > FINGER_ON).to(dt), dim=-1)                # (N,)
    palm_c = torch.tanh(torch.clamp(palm_force, min=0.0) / PALM_SCALE)                 # (N,)
    normal = palm_axes[:, 0:3]                                                          # (N,3) palm normal
    side = 1.0 - torch.clamp(torch.abs(torch.sum(normal * cup_up, dim=-1)), 0.0, 1.0)  # (N,)
    axial = torch.sum((cup_pos - palm_pos) * cup_up, dim=-1)                            # (N,)
    level = torch.exp(-(axial / LEVEL_SIGMA) ** 2)                                       # (N,)
    clos = torch.clamp((closure - CLOS_LO) / (CLOS_HI - CLOS_LO), 0.0, 1.0)             # (N,)
    return 0.30 * finger_frac + 0.30 * palm_c + 0.20 * side + 0.10 * level + 0.10 * clos


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    N = ctx.src_palm_pos.shape[0]
    dev = ctx.src_palm_pos.device
    dt = ctx.src_palm_pos.dtype
    zeros = torch.zeros(N, device=dev, dtype=dt)

    # ------------------------------------------------------------------ constants
    LIFT_TARGET_SRC = 0.20    # [m] source cup must end up clearly higher than the receiver
    LIFT_TARGET_RCV = 0.10    # [m] receiver lifted moderately (operator requirement)
    LIFT_GATE = 0.05          # [m] minimum lift before the pour stack is rewarded
    LIP_DEADBAND = 0.01       # [m] lip within 1 cm of the receiver mouth centre counts as centred (pose noise)
    LIP_DEADBAND_LOOSE = 0.02 # [m] deadband of the loose aim gate used by the tilt-progress term
    DZ_LOW = -0.06            # [m] "above" ramp start: lip 6 cm below the receiver rim -> 0
    DZ_HIGH = 0.02            # [m] "above" ramp end: lip 2 cm above the rim -> 1
    DZ_FREE = 0.04            # [m] pouring from up to 4 cm above the rim costs nothing
    DZ_SIGMA = 0.06           # [m] beyond that, gentle decay (10 cm above -> 0.37)
    TILT_TARGET = 2.0         # [rad] beads leave past ~1.9 rad, saturate slightly above
    POUR_TILT_LO = 1.3        # [rad] start of the release band bonus (~75 deg)
    POUR_TILT_HI = 2.0        # [rad] end of the release band bonus (~115 deg)
    TILT_DIR_SCALE = 1.0      # [rad] direction term fully weighted once tilted >= 57 deg
    TILT_FREE = 1.5           # [rad] unaimed tilt below 86 deg spills nothing -> free
    ROT_SPEED_FREE = 1.5      # [rad/s] controlled pouring rotation is free; only violent flips are charged
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped / knocked off the table
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side
    NEAR_DIST = 0.30          # [m] cup-centre distance inside which approach speed is regulated
    FORCE_SCALE = 5.0         # [N] tanh scale for all contact penalties
    FOREIGN_DEADBAND = 1.0    # [N] incidental brushes are free (additive term)
    CLEAN_DEADBAND = 0.5      # [N] deadband of the multiplicative hand-foreign gate
    GRIP_FLOOR = 0.3          # pour/lift income still worth 30 % with a hook grasp, 100 % with a wrap
    MEET_RCV_FREE = 0.02      # [m] receiver cup may cross the spawn midline by 2 cm for free
    MEET_SRC_FREE = 0.05      # [m] source cup may cross the midline by 5 cm for free
    MEET_SCALE = 0.10         # [m] tanh scale of the crossing penalties
    # iter_06 receiver-upright (success now requires rcv tilt <= 20 deg = 0.349 rad)
    RCV_TILT_FREE = 0.12      # [rad] ~7 deg of receiver tilt is free (grasp/lift wobble, pose noise)
    RCV_TILT_SIGMA = 0.20     # [rad] gate: 15 deg -> 0.61, 20 deg -> 0.27, 30 deg -> 0.03
    RCV_PEN_SCALE = 0.25      # [rad] additive penalty: 10 deg -> 0.22, 20 deg -> 0.73, 46 deg -> ~1
    RCV_GATE_FLOOR = 0.3      # positioning terms (aim / bring_together) keep 30 % income while rcv is tilted
    # iter_06 post-pour hold
    POST_POUR_FRAC = 0.5      # hold terms ramp in as bead_in_target_frac goes 0 -> 0.5
    # iter_06 command smoothness (raw actions, before the env EMA)
    PALM_RATE_BASE = 0.02     # per unit of sum-of-squared palm command change, before the grasp (exploration)
    PALM_RATE_HELD = 0.18     # extra once that hand holds its cup: observed 2-dim +-0.8<->-0.8 switching ~ -0.8/step
    HAND_RATE_W = 0.01        # finger commands: small (grasp closing/opening is legitimate)

    # ------------------------------------------------------------------ clearance factors (unchanged — worked)
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_foreign = torch.maximum(ctx.src_hand_foreign_force, ctx.rcv_hand_foreign_force)
    clean = 1.0 - torch.tanh(torch.clamp(f_foreign - CLEAN_DEADBAND, min=0.0) / FORCE_SCALE)
    not_nested_f = (~ctx.cups_nested).to(dt)

    # ------------------------------------------------------------------ receiver upright gate (new)
    rcv_tilt_excess = torch.clamp(ctx.rcv_cup_tilt - RCV_TILT_FREE, min=0.0)
    rcv_up = torch.exp(-(rcv_tilt_excess / RCV_TILT_SIGMA) ** 2)                  # (N,) in [0,1]
    rcv_up_soft = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_up                 # (N,) in [0.3,1]

    # ------------------------------------------------------------------ stage 1: approach (unchanged — solved)
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)

    # ------------------------------------------------------------------ stage 2: grasp (unchanged — solved)
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    prox_src = torch.exp(-6.0 * d_src)
    prox_rcv = torch.exp(-6.0 * d_rcv)
    grasp_src = 1.0 * g_src + 0.5 * prox_src * torch.clamp(ctx.src_hand_closure, 0.0, 1.0)
    grasp_rcv = 1.0 * g_rcv + 0.5 * prox_rcv * torch.clamp(ctx.rcv_hand_closure, 0.0, 1.0)
    both_grasped_b = ctx.src_grasped & ctx.rcv_grasped
    both_grasped = 1.0 * both_grasped_b.to(dt)

    wq_src = _wrap_quality(ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                           ctx.src_palm_axes, ctx.src_cup_up, ctx.src_cup_pos, ctx.src_palm_pos)
    wq_rcv = _wrap_quality(ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                           ctx.rcv_palm_axes, ctx.rcv_cup_up, ctx.rcv_cup_pos, ctx.rcv_palm_pos)
    wrap_src = 1.5 * g_src * wq_src
    wrap_rcv = 0.5 * g_rcv * wq_rcv
    grip = GRIP_FLOOR + (1.0 - GRIP_FLOOR) * wq_src   # in [0.3,1]; multiplies lift_src / aim / tilt / pour_pose

    # ------------------------------------------------------------------ stage 3: lift
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET_SRC, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET_RCV, 0.0, 1.0)
    lift_src = 2.0 * g_src * grip * lift_frac_src
    # receiver lift now worth 50 % if the cup is carried tilted, 100 % upright: the upright habit is formed
    # at lift time, before the pour stack (which is fully gated on rcv_up) is ever reached.
    lift_rcv = 2.0 * g_rcv * lift_frac_rcv * (0.5 + 0.5 * rcv_up)

    src_lifted_b = ctx.src_grasped & (h_src > LIFT_GATE)
    both_lifted_b = src_lifted_b & ctx.rcv_grasped & (h_rcv > LIFT_GATE)
    src_lifted_f = src_lifted_b.to(dt)
    both_lifted_f = both_lifted_b.to(dt)
    both_lifted = 1.0 * both_lifted_f

    # ------------------------------------------------------------------ pouring-lip geometry
    lip = _rim_low_point(ctx.src_cup_mouth_pos, ctx.src_cup_up, ctx.cup_radius)   # (N,3)
    lip_delta = lip - ctx.rcv_cup_mouth_pos                                         # (N,3)
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)                                   # (N,) horizontal miss
    dz_lip = lip_delta[:, 2]                                                        # (N,) lip height above rcv rim
    above = torch.clamp((dz_lip - DZ_LOW) / (DZ_HIGH - DZ_LOW), 0.0, 1.0)
    high = torch.exp(-(torch.clamp(dz_lip - DZ_FREE, min=0.0) / DZ_SIGMA) ** 2)
    aim_z = above * high
    aim_xy = torch.exp(-6.0 * torch.clamp(lip_xy - LIP_DEADBAND, min=0.0))
    aim_soft = aim_xy * aim_z                                                       # precise, in [0,1]
    aim_xy_loose = torch.exp(-3.0 * torch.clamp(lip_xy - LIP_DEADBAND_LOOSE, min=0.0))
    aim_loose = aim_xy_loose * aim_z

    # ------------------------------------------------------------------ stage 4: bring cups together
    # soft receiver-upright gate (floor 0.3): positioning gradient survives while the receiver is still tilted,
    # but bringing the lip over a LEVEL receiver mouth pays >3x more — no more tilting the receiver to meet the lip.
    bring_together = 1.0 * src_lifted_f * not_nested_f * clean * torch.exp(-3.0 * lip_xy) * aim_z * rcv_up_soft
    stack_gate = both_lifted_f * not_nested_f * clear * clean
    aim = 3.0 * stack_gate * grip * aim_soft * rcv_up_soft

    # ------------------------------------------------------------------ stage 5: tilt / pour pose
    # All pour-pose income is FULLY gated on rcv_up: last round the receiver was tilted 46 deg during the pour,
    # which the new success rule rejects. At 20 deg these terms keep 27 %, at 30 deg ~0.
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_rad = ctx.src_cup_tilt
    tilt_frac = torch.clamp(tilt_rad / TILT_TARGET, 0.0, 1.0)
    tilt = 4.0 * stack_gate * grip * beads_left * aim_loose * tilt_frac * rcv_up

    to_rcv = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]                  # (N,2)
    to_rcv_n = to_rcv / (torch.norm(to_rcv, dim=-1, keepdim=True) + 1e-6)
    up_xy = ctx.src_cup_up[:, :2]                                                    # (N,2)
    up_xy_n = up_xy / (torch.norm(up_xy, dim=-1, keepdim=True) + 1e-6)
    dir_cos = torch.sum(to_rcv_n * up_xy_n, dim=-1)                                  # (N,) in [-1,1]
    tilt_amt = torch.clamp(tilt_rad / TILT_DIR_SCALE, 0.0, 1.0)
    tilt_dir = 1.0 * stack_gate * beads_left * tilt_amt * dir_cos * rcv_up

    pour_band = _smoothstep((tilt_rad - POUR_TILT_LO) / (POUR_TILT_HI - POUR_TILT_LO))
    pour_pose = 3.0 * stack_gate * grip * beads_left * aim_soft * pour_band * rcv_up

    tilt_premature = -1.0 * (1.0 - aim_soft) * torch.clamp(tilt_rad - TILT_FREE, min=0.0)

    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    rot_speed = -0.3 * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (unchanged, ungated)
    # 50 * dfrac = 2.5 per bead; -30 * dfrac = -1.5 per spilled bead. Never gated: the bead signal must survive.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    spill_delta = -30.0 * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ stage 7: hold after the pour (new)
    # Beads already in the receiver cannot be un-poured, so paying a level here does not reward stalling:
    # finishing the pour earlier just starts the hold income earlier. Weight 1.5 (src) ~ wrap_src; together with
    # wrap_src and the grip-scaled success bonus a fingertip pinch (palm 7 N -> 2.2 N) now costs ~0.7/step.
    post_pour = _smoothstep(ctx.bead_in_target_frac / POST_POUR_FRAC)
    hold_src = 1.5 * post_pour * g_src * wq_src
    hold_rcv = 0.5 * post_pour * g_rcv * wq_rcv * rcv_up    # keep the filled receiver level and firmly held

    # ------------------------------------------------------------------ workspace: meet near the midline (unchanged)
    mid_y = 0.5 * (ctx.src_cup_spawn_pos[:, 1] + ctx.rcv_cup_spawn_pos[:, 1])
    rcv_cross = mid_y - ctx.rcv_cup_pos[:, 1]
    src_cross = ctx.src_cup_pos[:, 1] - mid_y
    meet_rcv = -1.0 * both_lifted_f * torch.tanh(torch.clamp(rcv_cross - MEET_RCV_FREE, min=0.0) / MEET_SCALE)
    meet_src = -0.5 * both_lifted_f * torch.tanh(torch.clamp(src_cross - MEET_SRC_FREE, min=0.0) / MEET_SCALE)

    # ------------------------------------------------------------------ collision / clearance (unchanged)
    cup_contact = -1.0 * torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_src = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    f_rcv = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    hand_foreign_src = -1.0 * torch.tanh(f_src / FORCE_SCALE)
    hand_foreign_rcv = -1.0 * torch.tanh(f_rcv / FORCE_SCALE)
    rel = ctx.rcv_cup_pos - ctx.src_cup_pos
    cup_dist = torch.norm(rel, dim=-1)
    u = rel / (cup_dist.unsqueeze(-1) + 1e-6)
    v_rel = ctx.src_cup_lin_vel - ctx.rcv_cup_lin_vel
    v_close = torch.sum(v_rel * u, dim=-1)
    near_f = (cup_dist < NEAR_DIST).to(dt)
    closing_speed = -1.0 * near_f * torch.tanh(torch.clamp(v_close - 0.10, min=0.0) / 0.20)
    nested = -0.5 * ctx.cups_nested.to(dt)

    # ------------------------------------------------------------------ constraints
    # receiver upright: -0.3*clamp(tilt) cost only -0.24 at 46 deg and was outbid by easier aiming.
    # Now -1.0 * tanh: 10 deg -> -0.22, 20 deg -> -0.73, 46 deg -> -0.99 (plus the multiplicative gates above).
    upright_rcv = -1.0 * torch.tanh(rcv_tilt_excess / RCV_PEN_SCALE)
    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)

    # ------------------------------------------------------------------ command smoothness (reworked)
    # Old action_rate (-0.02 * mean over all dims) cost ~0.002/step while two palm-rotation commands flipped sign
    # on >80 % of steps (4-5 Hz arm vibration). The env EMA only damps that (+-1 alternation still leaves a
    # +-0.14 target ripple). Sum of squared RAW palm command changes per arm, bounded by 24*w by construction.
    # Small before the hand holds its cup (random exploration ~ -0.1/step), strong once held
    # (observed switching ~ -0.8/step; a smooth pour with |da|~0.1 costs ~ -0.01).
    da = ctx.actions - ctx.prev_actions                                              # (N,18)
    palm_sq_src = torch.sum(da[:, 0:6] ** 2, dim=-1)
    palm_sq_rcv = torch.sum(da[:, 9:15] ** 2, dim=-1)
    hand_sq = torch.sum(da[:, 6:9] ** 2, dim=-1) + torch.sum(da[:, 15:18] ** 2, dim=-1)
    palm_rate_src = -(PALM_RATE_BASE + PALM_RATE_HELD * g_src) * palm_sq_src
    palm_rate_rcv = -(PALM_RATE_BASE + PALM_RATE_HELD * g_rcv) * palm_sq_rcv
    hand_rate = -HAND_RATE_W * hand_sq

    # ------------------------------------------------------------------ success bonus
    # env success now includes rcv tilt <= 20 deg. Scaled by source grip quality (0.6..1.0): the success window
    # is the post-pour hold, so a loose fingertip pinch loses up to 4/step there.
    success = 10.0 * ctx.success.to(dt) * clear * clean * (0.6 + 0.4 * g_src * wq_src)

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "wrap_src": wrap_src,
        "wrap_rcv": wrap_rcv,
        "both_grasped": both_grasped,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "both_lifted": both_lifted,
        "bring_together": bring_together,
        "aim": aim,
        "tilt": tilt,
        "tilt_dir": tilt_dir,
        "pour_pose": pour_pose,
        "tilt_premature": tilt_premature,
        "rot_speed": rot_speed,
        "pour_delta": pour_delta,
        "spill_delta": spill_delta,
        "hold_src": hold_src,
        "hold_rcv": hold_rcv,
        "meet_rcv": meet_rcv,
        "meet_src": meet_src,
        "cup_contact": cup_contact,
        "hand_foreign_src": hand_foreign_src,
        "hand_foreign_rcv": hand_foreign_rcv,
        "closing_speed": closing_speed,
        "nested": nested,
        "upright_rcv": upright_rcv,
        "drop": drop,
        "palm_rate_src": palm_rate_src,
        "palm_rate_rcv": palm_rate_rcv,
        "hand_rate": hand_rate,
        "success": success,
    }

    reward = zeros
    for v in components.values():
        reward = reward + v

    return reward, components
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [6.1e-05, 0, 0, 1.22e-05, 0, 2.44e-05, 0, 0, 0, 0]  min 0 · mean 8.64e-06 · max 0.00022
bead/spill: [0.000671, 0.00171, 0.00291, 0.0105, 0.00612, 0.00398, 0.00372, 0.00176, 0.0027, 0.00146]  min 0.000671 · mean 0.00358 · max 0.0112
done/drop: [0.0105, 0.000732, 0.000977, 0.000488, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000473 · max 0.0105
episode_lengths/step: [112, 769, 743, 784, 871, 871, 886, 887, 895, 892]  min 99.7 · mean 802 · max 899
reward/aim: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/approach_rcv: [0.463, 0.451, 0.456, 0.427, 0.428, 0.441, 0.443, 0.456, 0.45, 0.442]  min 0.42 · mean 0.446 · max 0.488
reward/approach_src: [0.442, 0.523, 0.566, 0.646, 0.67, 0.671, 0.682, 0.682, 0.68, 0.684]  min 0.442 · mean 0.639 · max 0.687
reward/both_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.37e-05 · max 0.000977
reward/both_lifted: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/bring_together: [0, 0.00293, 0.0358, 0.0605, 0.068, 0.0704, 0.0538, 0.045, 0.0352, 0.043]  min 0 · mean 0.0428 · max 0.0766
reward/closing_speed: [-0.284, -0.00353, -0.00151, -0.000206, -0.000371, -0.000149, 0, -5.84e-05, -0.000183, -0.000279]  min -0.284 · mean -0.00472 · max 0
reward/cup_contact: [-0.0922, 0, -6.23e-05, 0, 0, 0, 0, 0, -0.000244, 0]  min -0.0951 · mean -0.00273 · max 0
reward/drop: [-0.0145, -0.000366, -0.000488, -0.000244, 0, 0, 0, 0, 0, 0]  min -0.0145 · mean -0.000525 · max 0
reward/grasp_rcv: [0.0114, 0.00572, 0.00583, 0.00204, 0.00123, 0.00142, 0.00162, 0.00238, 0.00247, 0.00427]  min 0.00115 · mean 0.00419 · max 0.0229
reward/grasp_src: [0.00958, 0.24, 0.624, 0.911, 0.989, 0.988, 0.984, 0.99, 0.976, 0.996]  min 0.00654 · mean 0.822 · max 1.02
reward/hand_foreign_rcv: [-0.0892, -0.0101, -0.00671, -0.00638, -0.00656, -0.00593, -0.008, -0.00673, -0.00651, -0.00755]  min -0.0892 · mean -0.00898 · max -0.00381
reward/hand_foreign_src: [-0.107, -0.0188, -0.0591, -0.113, -0.0565, -0.0366, -0.0217, -0.0144, -0.0178, -0.0107]  min -0.122 · mean -0.0386 · max -0.0107
reward/hand_rate: [-0.0606, -0.0588, -0.0518, -0.0489, -0.0445, -0.0381, -0.0364, -0.0332, -0.0312, -0.0336]  min -0.0617 · mean -0.042 · max -0.0295
reward/hold_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 5.74e-10 · max 1.54e-07
reward/hold_src: [0, 0, 0, 4.02e-06, 0, 2.37e-05, 0, 0, 0, 0]  min 0 · mean 1.04e-06 · max 3.32e-05
reward/lift_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000143 · max 0.00339
reward/lift_src: [0, 0.0125, 0.151, 0.808, 1.09, 1.2, 1.29, 1.36, 1.39, 1.44]  min 0 · mean 0.964 · max 1.51
reward/meet_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/meet_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/nested: [-0.005, 0, -0.000122, 0, 0, 0, 0, 0, 0, 0]  min -0.0289 · mean -0.000918 · max 0
reward/palm_rate_rcv: [-0.123, -0.106, -0.0961, -0.0918, -0.0911, -0.084, -0.0814, -0.0767, -0.0705, -0.0745]  min -0.129 · mean -0.0866 · max -0.0681
reward/palm_rate_src: [-0.124, -0.264, -0.462, -0.478, -0.419, -0.345, -0.288, -0.246, -0.2, -0.198]  min -0.508 · mean -0.297 · max -0.123
reward/pour_delta: [0.00061, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.14e-05 · max 0.00061
reward/pour_pose: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/rot_speed: [-0.107, -0.0461, -0.0653, -0.0799, -0.0619, -0.0553, -0.0504, -0.0472, -0.0509, -0.0433]  min -0.107 · mean -0.0547 · max -0.0381
reward/spill_delta: [-0.00146, 0, -0.000732, -0.00146, -0.00146, -0.000366, -0.000732, 0, -0.000366, -0.0011]  min -0.00696 · mean -0.00087 · max 0
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt_dir: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt_premature: [0, 0, -0.0013, -0.00121, -0.000831, -0.000603, -3.11e-05, -4.45e-05, -0.000158, -5.06e-06]  min -0.00205 · mean -0.000313 · max 0
reward/total: [-0.656, 0.792, 1.41, 2.77, 3.49, 3.75, 3.96, 4.11, 4.16, 4.29]  min -0.656 · mean 3.13 · max 4.47
reward/upright_rcv: [-0.551, -0.00411, -0.000636, -0.000749, -0.000889, -0.00402, -0.00277, -0.00332, -0.00417, -0.000636]  min -0.551 · mean -0.0121 · max -8.4e-05
reward/wrap_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000102 · max 0.00211
reward/wrap_src: [0, 0.105, 0.353, 0.756, 0.953, 0.975, 1.01, 1.04, 1.03, 1.08]  min 0 · mean 0.794 · max 1.12
rewards/step: [-34.3, 518, 904, 1.99e+03, 2.92e+03, 3.18e+03, 3.51e+03, 3.66e+03, 3.76e+03, 3.85e+03]  min -34.3 · mean 2.64e+03 · max 3.94e+03
task/aim_dist: [0.147, 0.352, 0.5, 0.437, 0.454, 0.531, 0.473, 0.471, 0.484, 0.467]  min 0.147 · mean 0.444 · max 0.605
task/cup_collision_rate: [0.115, 0, 0.000244, 0, 0, 0, 0, 0, 0.000244, 0]  min 0 · mean 0.00318 · max 0.115
task/cups_center_dist: [0.203, 0.344, 0.49, 0.43, 0.453, 0.533, 0.477, 0.482, 0.498, 0.475]  min 0.203 · mean 0.448 · max 0.599
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0.01, 0, 0.000244, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.00184 · max 0.0579
task/rcv_cup_lift: [0.00182, 0.000113, 5.93e-05, 4.1e-05, 3.78e-05, 0.000153, 0.000199, 0.000245, 0.000288, 8.15e-05]  min 2.06e-05 · mean 0.000741 · max 0.0597
task/rcv_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.00047 · max 0.00952
task/rcv_hand_foreign_rate: [0.175, 0.0408, 0.0234, 0.0239, 0.0291, 0.0237, 0.0332, 0.033, 0.0281, 0.0347]  min 0.0166 · mean 0.0337 · max 0.175
task/src_cup_lift: [-0.000636, 0.0187, 0.0254, 0.114, 0.138, 0.168, 0.164, 0.179, 0.195, 0.188]  min -0.000636 · mean 0.133 · max 0.219
task/src_grasped: [0, 0.167, 0.531, 0.79, 0.868, 0.871, 0.866, 0.874, 0.861, 0.879]  min 0 · mean 0.717 · max 0.905
task/src_hand_foreign_rate: [0.195, 0.0427, 0.104, 0.204, 0.106, 0.0676, 0.0447, 0.0344, 0.0391, 0.0276]  min 0.0269 · mean 0.0745 · max 0.205
task/src_tilt_deg: [28.9, 11.5, 19.7, 28.1, 36.5, 44.8, 47, 49.5, 52.3, 53.1]  min 7.61 · mean 37.9 · max 57.2
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Observations from the loop operator about the policy trained with the previous function, and one operator requirement. Facts only, apart from the explicitly marked operator requirement.

1. This round was stopped early, after 208 epochs (1.7 h), because the receiver branch never started. The environment is unchanged from the previous round: 18-dimensional action space (per hand thumb opposition, thumb closure, one four-finger closure), palm commands low-pass filtered (EMA 0.25), and `ctx.success` requires the receiver tilt ≤ 20°. The feedback table above therefore covers only these 208 epochs.

2. Source side learned normally; the receiver side never did. Comparison at the same epochs with the round before the previous function (same task, older environment, a policy that did learn the receiver), shown as earlier-run / this-run:

| metric | epoch 20 | epoch 50 | epoch 145 | epoch 208 |
|---|---|---|---|---|
| source grasped | 0.06 / 0.06 | 0.57 / 0.47 | 0.87 / 0.88 | 0.88 / 0.88 |
| source lift [m] | 0.004 / 0.005 | 0.026 / 0.025 | 0.206 / 0.163 | 0.217 / 0.196 |
| receiver grasped | 0.16 / 0.00 | 0.64 / 0.00 | 0.86 / 0.00 | 0.87 / 0.00 |
| receiver lift [m] | 0.037 / 0.010 | 0.102 / 0.000 | 0.134 / 0.000 | 0.152 / 0.000 |
| reward/approach_rcv | 0.50 / 0.45 | 0.57 / 0.46 | 0.64 / 0.44 | 0.67 / 0.45 |
| cup distance `task/aim_dist` [m] | 0.44 / 0.34 | 0.71 / 0.41 | 0.34 / 0.47 | 0.21 / 0.48 |

   In this run the receiver hand stayed almost open (`task/rcv_closure` 0.01–0.02) and the receiver arm did not move toward its cup. The source cup was tilted to 53° while the cups stayed 0.48 m apart, so no pour and no success (0.0).

3. Early epochs, receiver upright penalty and receiver cup tilt (earlier-run / this-run):

| metric | epoch 1 | epoch 5 | epoch 10 | epoch 20 | epoch 50 |
|---|---|---|---|---|---|
| reward/upright_rcv | −0.10 / −0.36 | −0.08 / −0.22 | −0.06 / −0.11 | −0.07 / −0.014 | −0.09 / −0.001 |
| `task/rcv_tilt_deg` | 19 / 15 | 15 / 10 | 13 / 5 | 13 / 0.7 | 18 / 0.1 |
| reward/grasp_rcv | 0.02 / 0.01 | 0.02 / 0.01 | 0.04 / 0.02 | 0.21 / 0.01 | 0.73 / 0.01 |

   During early random exploration both runs bumped the receiver cup (15–19° tilt). With the previous function this cost −0.36 per step at epoch 1 instead of −0.10; within 20 epochs the policy stopped touching the receiver cup at all (tilt 0.1–0.7°, i.e. the cup standing untouched on the table) and receiver grasping never began. In the previous function `upright_rcv` is applied whenever the receiver cup is tilted more than about 7°, regardless of whether the receiver cup is grasped or lifted.

4. OPERATOR REQUIREMENT: the penalty on receiver tilt must differ between phases. While the receiver hand approaches and grasps the receiver cup on the table, tilt caused by hand contact must not be penalised. The receiver-upright penalty (and any upright-dependent scaling) should apply only after the receiver cup has been grasped and lifted, and especially during the pour, where the receiver must be held with its mouth up and only slightly tilted (the environment success still requires ≤ 20°).

5. Unchanged facts from earlier rounds: with contact freeze the measured `hand_closure` saturates at about 0.35–0.43 on these cups whatever the grasp type; hand–hand and cup–cup contact remain safety hazards on the real robot; the policy is trained from scratch again.

The policy must complete the full task (wrap-grasp both cups → lift → bring together without contact → tilt the source cup → beads in the receiver, receiver held nearly upright during the pour, no drop, little spill) and hold both grasps stably after pouring.
