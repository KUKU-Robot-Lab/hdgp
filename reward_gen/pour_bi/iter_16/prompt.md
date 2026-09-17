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
    bead_fill_level: torch.Tensor       # (N,) 에피소드 시작 시 소스 컵이 부피로 얼마나 찼는지 [0,1] (정착 후 실측, 에피소드 동안 고정; 비드 양은 에피소드마다 다르다)

    # ---- 충돌 (실기 안전 — 09.14) ---------------------------------------------------------
    cup_cup_force: torch.Tensor         # (N,) 두 컵이 서로 부딪히는 접촉력 [N] (0 = 안 닿음)
    src_hand_foreign_force: torch.Tensor  # (N,) 소스 손이 **자기 컵 외**(상대 손·상대 컵·테이블)에 닿는 힘 [N]
    rcv_hand_foreign_force: torch.Tensor  # (N,) 리시버 손이 자기 컵 외에 닿는 힘 [N]

    # ---- 과제 판정 / 시간 ------------------------------------------------------------
    cups_nested: torch.Tensor           # (N,) bool 두 컵 원점 거리 < 9 cm — 소스 컵이 리시버 컵에 끼워져 있음(붓기가 아니라 성공 무효)
    premature_tilt: torch.Tensor        # (N,) bool 에피소드 래치 — **잡은** 소스 컵이 src_pour_lip_pos 가 리시버 입구 중심에서 xy 8.1 cm 보다 멀 때 premature_tilt_limit 을 넘은 적이 있음(성공 무효, 리셋 전까지 유지; 잡지 않은 채 넘어진 컵은 아님)
    src_pour_lip_pos: torch.Tensor      # (N,3) 소스 컵의 **붓는 쪽 림 점** — 비드가 나가는 지점. 방향 d̂(소스 컵 원점→리시버 입구, 수평)는 컵 자세와 무관해서 직립에서도 정의됨: 입구 중심 + r·(cos θ·d̂ − sin θ·ẑ), r = 4.1 cm, θ = src_tilt_toward_rcv
    src_tilt_toward_rcv: torch.Tensor   # (N,) 리시버 쪽으로 기운 부호 있는 각도 θ [rad] — (d̂, z) 수직면 성분만(옆으로 기운 성분은 0, 반대로 기울면 음수)
    premature_tilt_limit: torch.Tensor  # (N,) 조준 없이 허용되는 src_cup_tilt 상한 [rad] = 채움별 첫 유출각 − 20° (가득 52°, 채움 0.5 에서 69°); 에피소드 동안 일정
    success: torch.Tensor               # (N,) bool 성공 조건 충족 (env 가 판정, 보상이 바꿀 수 없음; cups_nested·premature_tilt 면 항상 False)
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
5. Cup tilt: `ctx.src_cup_tilt` is the angle between the cup's up axis and world +z (0 = upright, π/2 = horizontal). The amount of beads in the source cup changes every episode and `ctx.bead_fill_level` (0 = empty, 1 = full to the rim, measured once the beads have settled, constant during the episode) tells the policy how full the cup is by volume. Beads start leaving the cup at a tilt that depends on that fill (measured): the first bead leaves at about 70° when the cup is full and at about 90° when it holds only a few beads, and a full cup has lost a fifth of its beads by 80° and half by 86°. `ctx.src_cup_mouth_pos` and `ctx.rcv_cup_mouth_pos` are the rim centres. `ctx.src_pour_lip_pos` is the point on the source rim where the beads leave: it is built from the horizontal direction d from the source cup's origin to the receiver's mouth, which does not depend on the cup's attitude, so it is defined for an upright cup and does not jump when the cup wobbles or leans the wrong way (rim centre + r·(cos θ·d − sin θ·z), r = 4.1 cm). `ctx.src_tilt_toward_rcv` is θ, the signed tilt toward the receiver in the vertical plane through d (0 for a sideways lean, negative for a lean away). Because the rim centre itself moves toward the receiver by (cup height × sin θ) as the cup tilts, a cup that must tilt further (few beads) is aimed with its body further away — aim the LIP, not the rim centre, at the receiver's mouth.
6. Bead bookkeeping: `bead_in_target_frac` rises as beads land in the receiver cup; `bead_spill_frac` counts beads lost outside both cups (permanent). `d_in_target` and `d_spill` are this step's increments — reward INCREMENTS of beads transferred rather than the level, otherwise the policy is paid for standing still with a filled cup.
7. `ctx.success` is computed by the environment (enough beads in the receiver cup, little spill, cups close together, the receiver cup held nearly upright — `ctx.rcv_cup_tilt` at most 20° — the cups NOT nested, and NO premature tilt: `ctx.premature_tilt` latches for the rest of the episode as soon as the GRASPED source cup's tilt `ctx.src_cup_tilt` (any direction) exceeds `ctx.premature_tilt_limit` — the fill-dependent release tilt minus 20°, i.e. 52° for a full cup and 69° for a half-full one — while `ctx.src_pour_lip_pos` is still more than 8.1 cm (xy) from the receiver's mouth centre (a cup knocked over without being grasped does not latch), and a latched episode can never succeed). Below that limit the source cup may be tilted anywhere, including while it is still approaching the receiver. You may add a bonus on it but you cannot redefine it. Beads only count as "in the receiver" once they have LEFT the source cup — pushing the source cup into the receiver cup (`ctx.cups_nested`) transfers nothing and is never a success; the beads must fall out of the tilted source cup through the air.
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


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    # x already normalised; clamp to [0,1] then 3x^2 - 2x^3
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _wrap_quality(finger_force: torch.Tensor, palm_force: torch.Tensor, closure: torch.Tensor,
                  palm_axes: torch.Tensor, cup_up: torch.Tensor, cup_pos: torch.Tensor,
                  palm_pos: torch.Tensor) -> torch.Tensor:
    # Power-grasp quality in [0,1]. Weighted MEAN so each ingredient keeps its own gradient.
    FINGER_ON = 0.3      # [N]
    PALM_SCALE = 3.0     # [N]
    LEVEL_SIGMA = 0.05   # [m]
    CLOS_LO, CLOS_HI = 0.25, 0.40
    dt = closure.dtype
    finger_frac = torch.mean((finger_force > FINGER_ON).to(dt), dim=-1)
    palm_c = torch.tanh(torch.clamp(palm_force, min=0.0) / PALM_SCALE)
    normal = palm_axes[:, 0:3]
    side = 1.0 - torch.clamp(torch.abs(torch.sum(normal * cup_up, dim=-1)), 0.0, 1.0)
    axial = torch.sum((cup_pos - palm_pos) * cup_up, dim=-1)
    level = torch.exp(-(axial / LEVEL_SIGMA) ** 2)
    clos = torch.clamp((closure - CLOS_LO) / (CLOS_HI - CLOS_LO), 0.0, 1.0)
    return 0.30 * finger_frac + 0.30 * palm_c + 0.20 * side + 0.10 * level + 0.10 * clos


def _carry_phase(grasped_f: torch.Tensor, h: torch.Tensor,
                 grasp_lo: float, grasp_hi: float, free_lo: float, free_hi: float) -> torch.Tensor:
    # (N,) in [0,1]: 0 on the table, 1 once grasped and lifted; far above the table the flag is not needed
    held_ramp = grasped_f * _smoothstep((h - grasp_lo) / (grasp_hi - grasp_lo))
    high_ramp = _smoothstep((h - free_lo) / (free_hi - free_lo))
    return torch.maximum(held_ramp, high_ramp)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    N = ctx.src_palm_pos.shape[0]
    dev = ctx.src_palm_pos.device
    dt = ctx.src_palm_pos.dtype
    zeros = torch.zeros(N, device=dev, dtype=dt)

    # ------------------------------------------------------------------ constants
    LIFT_TARGET_SRC = 0.25
    LIFT_TARGET_RCV = 0.10
    RCV_LIFT_FREE = 0.15
    RCV_LIFT_SIGMA = 0.08
    LIP_DEADBAND = 0.02       # [m]
    APPR_FULL = 0.08          # [m] approach zone full up to the latch radius
    APPR_ZERO = 0.24
    # iter_15: one "safe" gate replaces `over` (3.5 -> 7 cm). Full within 4.5 cm: round 15 parked the lip at
    # 5.3 cm where `over` was 0.48 and tilt past theta_pre lost more to the corridor penalty than it earned.
    SAFE_FULL = 0.045         # [m]
    SAFE_ZERO = 0.07          # [m] 1.1 cm inside the env latch radius 8.1 cm
    LIP_Z_LO = 0.0
    LIP_Z_HI = 0.02
    LIP_Z_FREE = 0.07
    LIP_Z_SIGMA = 0.08
    HOVER_LO = 0.06
    HOVER_HI = 0.11
    HOVER_FREE = 0.18
    HOVER_SIGMA = 0.06
    RAISE_LO = -0.04
    RAISE_W = 1.0
    CONV_FAR = 0.40
    CONV_W = 3.0
    BRING_W = 2.0
    BT_K = 10.0
    AIM_K = 15.0
    MEET_RANGE = 0.30
    MEET_RCV_W = 0.75
    MEET_SRC_W = 1.0
    SRC_TILT_FREE = 0.20
    PRE_MARGIN = 0.17         # [rad] theta_pre = limit - 10 deg
    SRC_TILT_SCALE = 0.20
    SRC_TILT_SIGMA = 0.25
    SRC_UP_W_TABLE = 1.0
    SRC_UP_W_CARRY = 1.5
    LATCH_W = 0.5
    RELEASE_ABOVE_LIMIT = math.radians(20.0)
    TILT_TARGET_ABOVE = 0.45
    TILT_TARGET_CAP = 2.0
    BAND_BELOW = 0.10
    BAND_ABOVE = 0.40
    # iter_15: single continuous tilt income (was PRE_W 4 over 0 -> theta_pre + TILT_W 4 beyond it).
    # 10 x theta/tilt_target: same value at the round-15 pose (~2.2 at 32 deg), ~4.7 at theta_pre, 10 at the
    # release target. With pour_pose 3 the pour pose is worth +11 over the parked pose (vs ~20 carry income).
    TILT_W = 10.0
    PRE_NEAR_FLOOR = 0.25
    PRE_NEAR_K = 10.0
    POSE_W = 3.0
    ROT_SPEED_FREE = 1.0
    DROP_DEPTH = 0.03
    RCV_TOPPLED = 1.2
    NEAR_DIST = 0.15
    CLOSE_V_FREE = 0.15
    CLOSE_V_SCALE = 0.20
    FORCE_SCALE = 5.0
    FOREIGN_DEADBAND = 1.0
    CLEAN_DEADBAND = 0.5
    FOREIGN_W_FREE = 0.3
    FOREIGN_W_HELD = 1.5
    CLEAN_FLOOR = 0.5
    GRIP_FLOOR = 0.3
    RCV_TILT_FREE = 0.12
    RCV_TILT_SIGMA = 0.20
    RCV_PEN_SCALE = 0.25
    RCV_GATE_FLOOR = 0.3
    CARRY_GRASP_LO = 0.015
    CARRY_GRASP_HI = 0.05
    CARRY_FREE_LO = 0.06
    CARRY_FREE_HI = 0.10
    RCV_UP_W = 1.0
    RCV_UP_POUR_W = 1.0
    RCV_STILL_FREE = 0.05
    RCV_STILL_SCALE = 0.15
    RCV_STILL_W = 0.25
    RCV_STILL_NEAR_W = 0.25
    RCV_STILL_POUR_W = 0.50
    POST_POUR_FRAC = 0.8
    SETTLE_FRAC = 0.05
    POUR_DELTA_W = 100.0
    SPILL_W = 50.0
    REACH_FAR = 0.26
    REACH_NEAR = 0.14
    REACH_W = 0.5
    CLOSE_W = 1.0
    CLOSE_NORM = 0.35
    PALM_RATE_W = 0.01
    HAND_RATE_W = 0.005
    SAT_LO = 0.7
    SAT_W = 0.15

    # ------------------------------------------------------------------ clearance factors
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_foreign = torch.maximum(ctx.src_hand_foreign_force, ctx.rcv_hand_foreign_force)
    clean = 1.0 - torch.tanh(torch.clamp(f_foreign - CLEAN_DEADBAND, min=0.0) / FORCE_SCALE)
    clean_soft = CLEAN_FLOOR + (1.0 - CLEAN_FLOOR) * clean
    not_nested_f = (~ctx.cups_nested).to(dt)

    # ------------------------------------------------------------------ grasp flags / heights
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]

    # ------------------------------------------------------------------ carry phases
    rcv_carry = _carry_phase(g_rcv, h_rcv, CARRY_GRASP_LO, CARRY_GRASP_HI, CARRY_FREE_LO, CARRY_FREE_HI)
    src_carry = _carry_phase(g_src, h_src, CARRY_GRASP_LO, CARRY_GRASP_HI, CARRY_FREE_LO, CARRY_FREE_HI)
    # iter_15: held state = grasp flag OR carried. Used for every grasp-dependent income, so a thumb contact
    # that flickers while the cup rotates no longer costs ~6/step (round 15: src_grasped 0.98 -> 0.76 as tilt rose)
    held_src = torch.maximum(g_src, src_carry)
    held_rcv = torch.maximum(g_rcv, rcv_carry)
    carry_both = src_carry * rcv_carry

    # ------------------------------------------------------------------ receiver upright gate
    rcv_tilt_excess = torch.clamp(ctx.rcv_cup_tilt - RCV_TILT_FREE, min=0.0)
    rcv_up_raw = torch.exp(-(rcv_tilt_excess / RCV_TILT_SIGMA) ** 2)
    rcv_up = 1.0 - rcv_carry * (1.0 - rcv_up_raw)
    rcv_up_soft = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_up

    # ------------------------------------------------------------------ pouring-lip geometry
    lip_delta = ctx.src_pour_lip_pos - ctx.rcv_cup_mouth_pos
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)
    lip_dz = lip_delta[:, 2]
    lip_off = torch.clamp(lip_xy - LIP_DEADBAND, min=0.0)
    approach_gate = 1.0 - _smoothstep((lip_xy - APPR_FULL) / (APPR_ZERO - APPR_FULL))
    safe = 1.0 - _smoothstep((lip_xy - SAFE_FULL) / (SAFE_ZERO - SAFE_FULL))   # (N,) lip close enough to release

    # ------------------------------------------------------------------ tilt angles
    tilt_mag = ctx.src_cup_tilt
    limit = ctx.premature_tilt_limit.to(dt)
    release_tilt = limit + RELEASE_ABOVE_LIMIT
    theta_pre = torch.clamp(limit - PRE_MARGIN, min=0.0)
    tilt_target = torch.clamp(release_tilt + TILT_TARGET_ABOVE, max=TILT_TARGET_CAP)
    to_rcv = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    to_rcv_n = to_rcv / (torch.norm(to_rcv, dim=-1, keepdim=True) + 1e-6)
    a_own = torch.sum(ctx.src_cup_up[:, :2] * to_rcv_n, dim=-1)
    theta_own = torch.atan2(a_own, ctx.src_cup_up[:, 2])
    theta = torch.maximum(ctx.src_tilt_toward_rcv.to(dt), theta_own)
    theta_eff = torch.minimum(torch.clamp(theta, min=0.0), tilt_mag)

    # ------------------------------------------------------------------ source tilt corridor
    # allowed: 0.20 rad far away -> theta_pre in the approach zone -> unlimited once the lip is within 4.5 cm
    allowed = SRC_TILT_FREE + approach_gate * torch.clamp(theta_pre - SRC_TILT_FREE, min=0.0)
    tilt_over = torch.clamp(tilt_mag - allowed, min=0.0) * (1.0 - safe)
    src_up = torch.exp(-(tilt_over / SRC_TILT_SIGMA) ** 2)

    # ------------------------------------------------------------------ stage 1: approach + reach
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)
    reach_src = REACH_W * _smoothstep((REACH_FAR - d_src) / (REACH_FAR - REACH_NEAR))
    reach_rcv = REACH_W * _smoothstep((REACH_FAR - d_rcv) / (REACH_FAR - REACH_NEAR))

    # ------------------------------------------------------------------ stage 2: grasp (held state)
    prox_src = torch.exp(-6.0 * d_src)
    prox_rcv = torch.exp(-6.0 * d_rcv)
    clos_src = torch.clamp(ctx.src_hand_closure / CLOSE_NORM, 0.0, 1.0)
    clos_rcv = torch.clamp(ctx.rcv_hand_closure / CLOSE_NORM, 0.0, 1.0)
    grasp_src = 1.0 * held_src + CLOSE_W * prox_src * clos_src
    grasp_rcv = 1.0 * held_rcv + CLOSE_W * prox_rcv * clos_rcv
    both_grasped = 1.0 * held_src * held_rcv

    wq_src = _wrap_quality(ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                           ctx.src_palm_axes, ctx.src_cup_up, ctx.src_cup_pos, ctx.src_palm_pos)
    wq_rcv = _wrap_quality(ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                           ctx.rcv_palm_axes, ctx.rcv_cup_up, ctx.rcv_cup_pos, ctx.rcv_palm_pos)
    wrap_src = 1.5 * held_src * wq_src   # wq still falls if fingers slip, so wrap keeps its gradient
    wrap_rcv = 0.5 * held_rcv * wq_rcv
    grip = GRIP_FLOOR + (1.0 - GRIP_FLOOR) * wq_src

    # ------------------------------------------------------------------ stage 3: lift
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET_SRC, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET_RCV, 0.0, 1.0)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    lift_src = 2.0 * held_src * grip * lift_frac_src * src_up
    lift_rcv = 2.0 * held_rcv * lift_frac_rcv * rcv_band * (0.5 + 0.5 * rcv_up)
    both_lifted = 1.0 * carry_both

    # ------------------------------------------------------------------ heights
    dz_eq = ctx.src_cup_pos[:, 2] + ctx.cup_mouth_z - ctx.rcv_cup_mouth_pos[:, 2]
    hover_decay = torch.exp(-(torch.clamp(dz_eq - HOVER_FREE, min=0.0) / HOVER_SIGMA) ** 2)
    hover = _smoothstep((dz_eq - HOVER_LO) / (HOVER_HI - HOVER_LO)) * hover_decay
    hover_wide = _smoothstep((dz_eq - RAISE_LO) / (HOVER_HI - RAISE_LO)) * hover_decay
    lip_above = torch.clamp((lip_dz - LIP_Z_LO) / (LIP_Z_HI - LIP_Z_LO), 0.0, 1.0)
    lip_high = torch.exp(-(torch.clamp(lip_dz - LIP_Z_FREE, min=0.0) / LIP_Z_SIGMA) ** 2)
    aim_z = lip_above * lip_high
    z_ok = torch.maximum(hover, safe * aim_z)

    # ------------------------------------------------------------------ stage 4: raise, converge, bring, aim
    raise_src = RAISE_W * carry_both * not_nested_f * src_up * rcv_up_soft * hover_wide
    conv_xy = torch.clamp(1.0 - lip_off / CONV_FAR, 0.0, 1.0)
    converge = CONV_W * carry_both * not_nested_f * src_up * rcv_up_soft * hover_wide * conv_xy
    bring_together = BRING_W * src_carry * not_nested_f * clean_soft * src_up * torch.exp(-BT_K * lip_off) \
        * z_ok * rcv_up_soft
    stack_gate = carry_both * not_nested_f * clear * clean_soft
    aim_xy = torch.exp(-AIM_K * lip_off)
    aim_pose = aim_xy * z_ok
    aim = 3.0 * stack_gate * grip * aim_pose * rcv_up_soft

    # ------------------------------------------------------------------ stage 5: tilt (one continuous income) + pour pose
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    near_pre = PRE_NEAR_FLOOR + (1.0 - PRE_NEAR_FLOOR) * torch.exp(-PRE_NEAR_K * lip_off)
    tgt = torch.clamp(tilt_target, min=0.3)
    # part below theta_pre: cannot spill or latch, open in the whole approach zone
    part_pre = torch.minimum(theta_eff, theta_pre) / tgt * near_pre * approach_gate * hover_wide
    # part above theta_pre: only with the lip within the safe radius and above the receiver rim
    part_rel = torch.clamp(torch.minimum(theta_eff, tgt) - theta_pre, min=0.0) / tgt * safe * lip_above
    tilt = TILT_W * carry_both * not_nested_f * clean_soft * grip * beads_left * rcv_up_soft \
        * (part_pre + part_rel)

    pour_band = _smoothstep((theta_eff - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    pour_pose = POSE_W * stack_gate * grip * beads_left * safe * lip_above * aim_pose * pour_band * rcv_up

    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    rot_speed = -0.3 * held_src * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (increments)
    pour_delta = POUR_DELTA_W * torch.clamp(ctx.d_in_target, -1.0, 1.0)
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    spill_delta = -SPILL_W * settled * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ stage 7: hold after the pour
    post_pour = _smoothstep(ctx.bead_in_target_frac / POST_POUR_FRAC)
    hold_src = 1.5 * post_pour * held_src * wq_src
    hold_rcv = 0.5 * post_pour * held_rcv * wq_rcv * rcv_up

    # ------------------------------------------------------------------ workspace: meet near the midline
    mid_y = 0.5 * (ctx.src_cup_spawn_pos[:, 1] + ctx.rcv_cup_spawn_pos[:, 1])
    mid_x = 0.5 * (ctx.src_cup_spawn_pos[:, 0] + ctx.rcv_cup_spawn_pos[:, 0])
    meet_xy = torch.stack([mid_x, mid_y], dim=-1)
    d_meet_src = torch.norm(ctx.src_cup_pos[:, :2] - meet_xy, dim=-1)
    d_meet_rcv = torch.norm(ctx.rcv_cup_pos[:, :2] - meet_xy, dim=-1)
    meet_src = MEET_SRC_W * carry_both * not_nested_f * src_up * hover_wide \
        * torch.clamp(1.0 - d_meet_src / MEET_RANGE, 0.0, 1.0)
    meet_rcv = MEET_RCV_W * carry_both * not_nested_f * rcv_up_soft \
        * torch.clamp(1.0 - d_meet_rcv / MEET_RANGE, 0.0, 1.0)

    # ------------------------------------------------------------------ collision / clearance
    cup_contact = -1.0 * torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_src = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    f_rcv = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    w_for_src = FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_src
    w_for_rcv = FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_rcv
    hand_foreign_src = -w_for_src * torch.tanh(f_src / FORCE_SCALE)
    hand_foreign_rcv = -w_for_rcv * torch.tanh(f_rcv / FORCE_SCALE)
    rel = ctx.rcv_cup_pos - ctx.src_cup_pos
    cup_dist = torch.norm(rel, dim=-1)
    u = rel / (cup_dist.unsqueeze(-1) + 1e-6)
    v_rel = ctx.src_cup_lin_vel - ctx.rcv_cup_lin_vel
    v_close = torch.sum(v_rel * u, dim=-1)
    near_f = (cup_dist < NEAR_DIST).to(dt)
    carry_any = torch.maximum(src_carry, rcv_carry)
    closing_speed = -1.0 * carry_any * near_f \
        * torch.tanh(torch.clamp(v_close - CLOSE_V_FREE, min=0.0) / CLOSE_V_SCALE)
    nested = -0.5 * ctx.cups_nested.to(dt)

    # ------------------------------------------------------------------ constraints
    upright_src = -(SRC_UP_W_TABLE + SRC_UP_W_CARRY * src_carry) * held_src \
        * torch.tanh(tilt_over / SRC_TILT_SCALE)
    premature_latch = -LATCH_W * ctx.premature_tilt.to(dt)

    fin_active = _smoothstep((theta_eff - theta_pre) / torch.clamp(release_tilt - theta_pre, min=0.1))
    pour_phase = torch.maximum(carry_both * safe * fin_active, post_pour)
    upright_rcv = -(RCV_UP_W + RCV_UP_POUR_W * pour_phase) * rcv_carry * torch.tanh(rcv_tilt_excess / RCV_PEN_SCALE)

    v_rcv = torch.norm(ctx.rcv_cup_lin_vel, dim=-1)
    still_w = RCV_STILL_W + RCV_STILL_NEAR_W * approach_gate * src_carry + RCV_STILL_POUR_W * pour_phase
    rcv_still = -still_w * rcv_carry \
        * torch.tanh(torch.clamp(v_rcv - RCV_STILL_FREE, min=0.0) / RCV_STILL_SCALE)

    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)

    # ------------------------------------------------------------------ command smoothness + saturation
    da = ctx.actions - ctx.prev_actions
    palm_sq_src = torch.sum(da[:, 0:6] ** 2, dim=-1)
    palm_sq_rcv = torch.sum(da[:, 9:15] ** 2, dim=-1)
    hand_sq = torch.sum(da[:, 6:9] ** 2, dim=-1) + torch.sum(da[:, 15:18] ** 2, dim=-1)
    palm_rate_src = -PALM_RATE_W * palm_sq_src
    palm_rate_rcv = -PALM_RATE_W * palm_sq_rcv
    hand_rate = -HAND_RATE_W * hand_sq
    a_abs = torch.abs(ctx.actions)
    sat_src = torch.mean(_smoothstep((a_abs[:, 0:6] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    sat_rcv = torch.mean(_smoothstep((a_abs[:, 9:15] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    palm_sat_src = -SAT_W * sat_src
    palm_sat_rcv = -SAT_W * sat_rcv

    # ------------------------------------------------------------------ success bonus
    success = 10.0 * ctx.success.to(dt) * clear * clean * (0.6 + 0.4 * held_src * wq_src) \
        * (0.6 + 0.4 * torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0))

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "reach_src": reach_src,
        "reach_rcv": reach_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "wrap_src": wrap_src,
        "wrap_rcv": wrap_rcv,
        "both_grasped": both_grasped,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "both_lifted": both_lifted,
        "raise_src": raise_src,
        "converge": converge,
        "bring_together": bring_together,
        "aim": aim,
        "tilt": tilt,
        "pour_pose": pour_pose,
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
        "upright_src": upright_src,
        "premature_latch": premature_latch,
        "upright_rcv": upright_rcv,
        "rcv_still": rcv_still,
        "drop": drop,
        "palm_rate_src": palm_rate_src,
        "palm_rate_rcv": palm_rate_rcv,
        "palm_sat_src": palm_sat_src,
        "palm_sat_rcv": palm_sat_rcv,
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
bead/in_target: [0, 7.46e-05, 6.1e-05, 2.44e-05, 9.34e-05, 3.49e-05, 0.000145, 2.71e-05, 2.44e-05, 4.07e-05]  min 0 · mean 5.51e-05 · max 0.000396
bead/spill: [0.00231, 0.0153, 0.0156, 0.0178, 0.0125, 0.0176, 0.0178, 0.0171, 0.015, 0.0148]  min 0.00231 · mean 0.0157 · max 0.0189
done/drop: [0, 0.000244, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.56e-05 · max 0.000488
episode_lengths/step: [81.9, 817, 843, 831, 834, 892, 872, 843, 868, 890]  min 81.9 · mean 831 · max 899
reward/aim: [0, 2.14, 2.47, 2.36, 1.06, 2.49, 2.53, 2.38, 1.68, 2.5]  min 0 · mean 2.13 · max 2.66
reward/approach_rcv: [0.655, 0.701, 0.693, 0.671, 0.639, 0.67, 0.676, 0.674, 0.655, 0.681]  min 0.566 · mean 0.673 · max 0.704
reward/approach_src: [0.635, 0.686, 0.687, 0.683, 0.636, 0.684, 0.687, 0.683, 0.653, 0.686]  min 0.55 · mean 0.675 · max 0.692
reward/both_grasped: [0, 0.972, 0.977, 0.94, 0.673, 0.944, 0.949, 0.917, 0.721, 0.932]  min 0 · mean 0.863 · max 0.993
reward/both_lifted: [0, 0.97, 0.974, 0.938, 0.668, 0.943, 0.948, 0.915, 0.72, 0.932]  min 0 · mean 0.861 · max 0.992
reward/bring_together: [8.31e-05, 1.62, 1.78, 1.7, 0.762, 1.77, 1.8, 1.7, 1.19, 1.76]  min 8.31e-05 · mean 1.53 · max 1.9
reward/closing_speed: [0, -0.00397, -0.00367, -0.00235, -0.00444, -0.00127, -0.00157, -0.00142, -0.00349, -0.00105]  min -0.0248 · mean -0.00265 · max 0
reward/converge: [0, 2.69, 2.79, 2.68, 1.35, 2.74, 2.77, 2.65, 1.88, 2.72]  min 0 · mean 2.44 · max 2.91
reward/cup_contact: [0, -0.0423, -0.0345, -0.0391, -0.0688, -0.0226, -0.0226, -0.0321, -0.0667, -0.0119]  min -0.568 · mean -0.04 · max 0
reward/drop: [0, -0.00146, -0.000488, -0.000488, -0.000122, -0.000366, -0.000732, -0.000488, -0.000366, -0.000366]  min -0.00208 · mean -0.000399 · max 0
reward/grasp_rcv: [0.0965, 1.56, 1.55, 1.47, 1.07, 1.48, 1.49, 1.45, 1.19, 1.48]  min 0.0965 · mean 1.38 · max 1.57
reward/grasp_src: [0.108, 1.55, 1.55, 1.5, 1.09, 1.5, 1.51, 1.47, 1.18, 1.49]  min 0.108 · mean 1.39 · max 1.57
reward/hand_foreign_rcv: [-0.0289, -0.0351, -0.0317, -0.0401, -0.253, -0.017, -0.014, -0.0252, -0.0767, -0.0127]  min -0.814 · mean -0.0371 · max -0.00567
reward/hand_foreign_src: [-0.26, -0.0508, -0.0585, -0.0724, -0.143, -0.0364, -0.0371, -0.0531, -0.0742, -0.0257]  min -0.579 · mean -0.0639 · max -0.019
reward/hand_rate: [-0.0189, -0.0219, -0.0199, -0.0198, -0.0202, -0.0205, -0.0204, -0.0199, -0.02, -0.0199]  min -0.0237 · mean -0.0202 · max -0.016
reward/hold_rcv: [0, 1.97e-05, 9.37e-06, 4.52e-06, 7.93e-06, 6.56e-06, 1.03e-05, 2.73e-06, 3.42e-06, 1.28e-12]  min 0 · mean 7.63e-06 · max 0.000119
reward/hold_src: [0, 1.23e-05, 4.52e-05, 1.56e-05, 3.24e-05, 9.65e-06, 5.69e-05, 8.76e-06, 8.93e-06, 1.08e-05]  min 0 · mean 2.34e-05 · max 0.000203
reward/lift_rcv: [0, 1.77, 1.77, 1.51, 1.02, 1.69, 1.73, 1.63, 1.32, 1.78]  min 0 · mean 1.54 · max 1.85
reward/lift_src: [0.000736, 1.62, 1.72, 1.64, 0.893, 1.67, 1.7, 1.57, 1.17, 1.67]  min 0.000736 · mean 1.48 · max 1.82
reward/meet_rcv: [0, 0.37, 0.322, 0.306, 0.161, 0.252, 0.252, 0.255, 0.21, 0.27]  min 0 · mean 0.265 · max 0.394
reward/meet_src: [0, 0.194, 0.142, 0.119, 0.0548, 0.0508, 0.0443, 0.0349, 0.0546, 0.048]  min 0 · mean 0.0814 · max 0.206
reward/nested: [0, -0.00745, -0.00623, -0.0072, -0.0137, -0.00317, -0.00354, -0.00439, -0.012, -0.00232]  min -0.137 · mean -0.00644 · max 0
reward/palm_rate_rcv: [-0.017, -0.0379, -0.0354, -0.032, -0.0233, -0.0338, -0.0322, -0.0315, -0.0263, -0.0309]  min -0.0472 · mean -0.031 · max -0.0137
reward/palm_rate_src: [-0.0088, -0.0272, -0.0196, -0.015, -0.0134, -0.0125, -0.011, -0.0109, -0.00968, -0.00901]  min -0.0354 · mean -0.0141 · max -0.00722
reward/palm_sat_rcv: [-0.0513, -0.0537, -0.0516, -0.0499, -0.0718, -0.0595, -0.0598, -0.0567, -0.0596, -0.0601]  min -0.0981 · mean -0.0588 · max -0.0425
reward/palm_sat_src: [-0.0588, -0.0529, -0.0661, -0.077, -0.0708, -0.0931, -0.0945, -0.0921, -0.0806, -0.096]  min -0.106 · mean -0.0798 · max -0.0298
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00349 · mean 1.28e-05 · max 0.00488
reward/pour_pose: [0, 6.52e-11, 0, 0, 0, 4.72e-13, 0, 0, 0, 5.97e-06]  min 0 · mean 4.8e-06 · max 0.000348
reward/premature_latch: [0, -0.000977, -0.00232, -0.00269, -0.00244, -0.00232, -0.00195, -0.00195, -0.00171, -0.00537]  min -0.0142 · mean -0.00301 · max 0
reward/raise_src: [0, 0.928, 0.942, 0.905, 0.471, 0.922, 0.93, 0.892, 0.635, 0.916]  min 0 · mean 0.827 · max 0.975
reward/rcv_still: [0, -0.161, -0.155, -0.17, -0.157, -0.138, -0.14, -0.149, -0.15, -0.154]  min -0.299 · mean -0.149 · max 0
reward/reach_rcv: [0.5, 0.498, 0.498, 0.497, 0.488, 0.497, 0.497, 0.497, 0.489, 0.496]  min 0.453 · mean 0.495 · max 0.5
reward/reach_src: [0.5, 0.498, 0.499, 0.498, 0.483, 0.496, 0.497, 0.497, 0.488, 0.498]  min 0.446 · mean 0.495 · max 0.5
reward/rot_speed: [-0.0154, -0.0437, -0.0482, -0.0618, -0.0864, -0.0402, -0.0439, -0.0505, -0.0545, -0.0352]  min -0.177 · mean -0.049 · max -0.00419
reward/spill_delta: [-0.00224, -0.0031, -0.00356, 0, -0.0112, -0.00174, -0.00122, 0, -0.00345, 0]  min -0.0401 · mean -0.00174 · max 0
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.3e-06 · max 0.00186
reward/tilt: [0, 2.4, 3.13, 3.42, 1.54, 3.98, 4.19, 3.94, 2.64, 4.42]  min 0 · mean 3.17 · max 4.64
reward/total: [2.06, 22.3, 23.7, 22.8, 12.7, 23.9, 24.4, 23.2, 17.1, 24.4]  min 2.06 · mean 21.1 · max 25
reward/upright_rcv: [0, -0.0496, -0.0424, -0.0466, -0.205, -0.0362, -0.0304, -0.0409, -0.11, -0.0214]  min -0.467 · mean -0.0487 · max 0
reward/upright_src: [-0.00101, -0.00329, -0.00207, -0.00446, -0.0161, -0.00787, -0.00784, -0.0102, -0.0411, -0.0124]  min -0.0686 · mean -0.00862 · max -0.000231
reward/wrap_rcv: [0, 0.396, 0.414, 0.365, 0.231, 0.391, 0.399, 0.37, 0.285, 0.403]  min 0 · mean 0.356 · max 0.427
reward/wrap_src: [0.0347, 1.34, 1.36, 1.29, 0.745, 1.31, 1.32, 1.26, 0.94, 1.31]  min 0.0347 · mean 1.18 · max 1.39
rewards/step: [68.6, 1.37e+04, 1.69e+04, 1.68e+04, 1.79e+04, 1.93e+04, 1.89e+04, 1.85e+04, 1.94e+04, 1.99e+04]  min 68.6 · mean 1.74e+04 · max 2.04e+04
task/aim_dist: [0.297, 0.128, 0.131, 0.137, 0.187, 0.123, 0.194, 0.125, 0.198, 0.112]  min 0.102 · mean 0.144 · max 0.4
task/cup_collision_rate: [0, 0.0454, 0.0381, 0.0439, 0.101, 0.0256, 0.0244, 0.0381, 0.0876, 0.0146]  min 0 · mean 0.0464 · max 0.683
task/cups_center_dist: [0.306, 0.165, 0.174, 0.184, 0.195, 0.181, 0.257, 0.188, 0.233, 0.183]  min 0.112 · mean 0.192 · max 0.472
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.8e-06 · max 0.000244
task/nested_rate: [0, 0.0149, 0.0125, 0.0144, 0.0273, 0.00635, 0.00708, 0.00879, 0.0239, 0.00464]  min 0 · mean 0.0129 · max 0.274
task/rcv_cup_lift: [0.00422, 0.101, 0.0942, 0.0794, 0.0634, 0.0891, 0.159, 0.0887, 0.123, 0.0952]  min 0.00422 · mean 0.086 · max 0.173
task/rcv_grasped: [0, 0.939, 0.951, 0.924, 0.647, 0.921, 0.923, 0.903, 0.681, 0.92]  min 0 · mean 0.843 · max 0.977
task/rcv_hand_foreign_rate: [0.18, 0.0493, 0.0439, 0.0593, 0.289, 0.0315, 0.0251, 0.0522, 0.128, 0.0276]  min 0.00977 · mean 0.067 · max 0.851
task/src_cup_lift: [0.0021, 0.219, 0.229, 0.221, 0.134, 0.223, 0.226, 0.21, 0.165, 0.222]  min 0.0021 · mean 0.202 · max 0.365
task/src_grasped: [0.0518, 0.956, 0.967, 0.928, 0.523, 0.935, 0.945, 0.91, 0.64, 0.924]  min 0.0518 · mean 0.852 · max 0.993
task/src_hand_foreign_rate: [0.843, 0.0723, 0.0818, 0.11, 0.239, 0.0637, 0.0681, 0.0945, 0.155, 0.0549]  min 0.0293 · mean 0.118 · max 0.843
task/src_tilt_deg: [4.88, 31.7, 38.1, 42.6, 27.7, 47.3, 49.4, 47.9, 38, 51.8]  min 3.5 · mean 40.2 · max 54.1
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.8e-06 · max 0.000244

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Round 16 (iter_15, t2r_i15, 4096 env, warm-started from the round-15 checkpoint, 812 epochs / 5.9 h, ADR level 0) — measured facts only, environment unchanged. This round moved the source tilt further than any earlier round, but no bead was ever transferred: bead/in_target 0.0 and episode_success 0.0 for all 812 epochs, reward/pour_delta max 0.005, reward/pour_pose exactly 0.0, reward/hold_src and reward/hold_rcv 0.0, bead/spill 0.016.

Source tilt (env mean at tick checks, epoch: task/src_tilt_deg / task/pour_lip_dist): first 4.9 deg / 0.256 m; 261: 35.2 / 0.069; 538: 44.0 / 0.048; 812: 45.2 / 0.052 (max of the logged mean 54.1 deg). task/tilt_far_deg 22.6, task/tilt_limit_deg 60.2, so theta_pre was about 50 deg and the first-release angle about 80 deg for this fill. reward/tilt (single merged term, weight 10) rose 0 -> 3.75 (max 4.64) and was still rising, but the rise slowed sharply in the last 2 h (44.0 -> 45.2 deg).

Approach and grasp improved together with the tilt: task/aim_dist 0.126 m, task/cups_center_dist 0.186 m, task/src_grasped 0.861, task/rcv_grasped 0.860 (round 15 ended at 0.80/0.81), src cup lift 0.199 m, rcv cup lift 0.092 m. Contact indicators fell: task/cup_collision_rate 0.032 (round 15: 0.064), hand_foreign src/rcv 0.125/0.051 (round 15: 0.184/0.093), task/nested_rate 0.009. task/premature_tilt_rate 0.015 (max 0.028, up from 0.008 in round 15), task/rcv_palm_speed 0.37 m/s, bead/fill_level 0.758, dr/bead_active_hi 12.

End-of-round term values: aim 2.30, tilt 3.90, converge 2.58, bring_together 1.63, lift_src 1.56, lift_rcv 1.73, grasp_src 1.44, grasp_rcv 1.44, wrap_src 1.24, wrap_rcv 0.38, raise_src 0.87, meet_rcv 0.27, meet_src 0.06, rcv_still -0.172, palm_sat_src -0.092, upright_src -0.014, premature_latch -0.008, total 22.9.

Best so far remains iter_05 (round-3 environment, not comparable).
