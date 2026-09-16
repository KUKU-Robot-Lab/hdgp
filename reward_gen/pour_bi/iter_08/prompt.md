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
    premature_tilt: torch.Tensor        # (N,) bool 에피소드 래치 — 소스 입구가 리시버 입구에서 xy 10 cm 보다 멀 때 소스가 30° 를 넘은 적이 있음(성공 무효, 리셋 전까지 유지)
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
5. Cup tilt: `ctx.src_cup_tilt` is the angle between the cup's up axis and world +z (0 = upright, π/2 = horizontal). The amount of beads in the source cup changes every episode and `ctx.bead_fill_level` (0 = empty, 1 = full to the rim, measured once the beads have settled, constant during the episode) tells the policy how full the cup is by volume. Beads start leaving the cup at a tilt that depends on that fill (measured): the first bead leaves at about 70° when the cup is full and at about 95° when it holds only a few beads, and a full cup has lost a fifth of its beads by 80° and half by 86°. `ctx.src_cup_mouth_pos` and `ctx.rcv_cup_mouth_pos` are the rim centres.
6. Bead bookkeeping: `bead_in_target_frac` rises as beads land in the receiver cup; `bead_spill_frac` counts beads lost outside both cups (permanent). `d_in_target` and `d_spill` are this step's increments — reward INCREMENTS of beads transferred rather than the level, otherwise the policy is paid for standing still with a filled cup.
7. `ctx.success` is computed by the environment (enough beads in the receiver cup, little spill, cups close together, the receiver cup held nearly upright — `ctx.rcv_cup_tilt` at most 20° — the cups NOT nested, and NO premature tilt: `ctx.premature_tilt` latches for the rest of the episode as soon as the source cup exceeds 30° while its mouth is still more than 10 cm (xy) from the receiver's mouth, and a latched episode can never succeed). You may add a bonus on it but you cannot redefine it. Beads only count as "in the receiver" once they have LEFT the source cup — pushing the source cup into the receiver cup (`ctx.cups_nested`) transfers nothing and is never a success; the beads must fall out of the tilted source cup through the air.
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
    PALM_SCALE = 3.0     # [N] palm contact saturation (a fingertip pinch with ~2 N palm scores clearly lower)
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


def _carry_phase(grasped_f: torch.Tensor, h: torch.Tensor,
                 grasp_lo: float, grasp_hi: float, free_lo: float, free_hi: float) -> torch.Tensor:
    # (N,) in [0,1]: 0 while the cup stands on the table (approach / contact / grasp closing / bumps),
    # 1 once it is grasped and lifted. Tipping a cup on its bottom rim raises its origin only ~1 cm, so the
    # grasped ramp starts at 1.5 cm; far above the table (>= 6 cm) the cup must be carried, so the second
    # ramp ignores a flickering grasp flag and releasing the grasp mid-pour cannot switch the phase off.
    held_ramp = grasped_f * _smoothstep((h - grasp_lo) / (grasp_hi - grasp_lo))
    high_ramp = _smoothstep((h - free_lo) / (free_hi - free_lo))
    return torch.maximum(held_ramp, high_ramp)


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
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side (a lost cup, not contact wobble)
    NEAR_DIST = 0.30          # [m] cup-centre distance inside which approach speed is regulated
    FORCE_SCALE = 5.0         # [N] tanh scale for all contact penalties
    FOREIGN_DEADBAND = 1.0    # [N] incidental brushes are free (additive term)
    CLEAN_DEADBAND = 0.5      # [N] deadband of the multiplicative hand-foreign gate
    GRIP_FLOOR = 0.3          # pour/lift income still worth 30 % with a hook grasp, 100 % with a wrap
    MEET_RCV_FREE = 0.02      # [m] receiver cup may cross the spawn midline by 2 cm for free
    MEET_SRC_FREE = 0.05      # [m] source cup may cross the midline by 5 cm for free
    MEET_SCALE = 0.10         # [m] tanh scale of the crossing penalties
    # receiver-upright shape (success requires rcv tilt <= 20 deg = 0.349 rad)
    RCV_TILT_FREE = 0.12      # [rad] ~7 deg of receiver tilt is free (grasp/lift wobble, pose noise)
    RCV_TILT_SIGMA = 0.20     # [rad] gate: 15 deg -> 0.61, 20 deg -> 0.27, 30 deg -> 0.03
    RCV_PEN_SCALE = 0.25      # [rad] additive penalty: 10 deg -> 0.22, 20 deg -> 0.73, 46 deg -> ~1
    RCV_GATE_FLOOR = 0.3      # positioning terms (bring_together / aim) keep 30 % income while rcv is tilted
    # iter_07 phase-dependent receiver upright (operator requirement)
    CARRY_GRASP_LO = 0.015    # [m] grasped-cup carry ramp start (above the ~1 cm origin rise of a tipped cup)
    CARRY_GRASP_HI = 0.05     # [m] grasped-cup carry ramp end (= LIFT_GATE: pour stack starts here)
    CARRY_FREE_LO = 0.06      # [m] above this the cup is carried whatever the grasp flag says
    CARRY_FREE_HI = 0.10      # [m]
    RCV_UP_W = 1.0            # carried receiver: 20 deg -> -0.73 (same as before, but only once carried)
    RCV_UP_POUR_W = 1.0       # extra during pour / post-pour hold: 20 deg -> -1.46 total
    POUR_PHASE_TILT_LO = 0.8  # [rad] pour phase ramps in as the aimed source cup tilts 46 deg ...
    POUR_PHASE_TILT_HI = 1.6  # [rad] ... to 92 deg (beads start leaving ~1.9 rad)
    # post-pour hold
    POST_POUR_FRAC = 0.5      # hold terms ramp in as bead_in_target_frac goes 0 -> 0.5
    # command smoothness (raw actions, before the env EMA)
    PALM_RATE_BASE = 0.02     # per unit of sum-of-squared palm command change, before the grasp (exploration)
    PALM_RATE_HELD = 0.18     # extra once that hand holds its cup (suppresses 4-5 Hz command switching)
    HAND_RATE_W = 0.01        # finger commands: small (grasp closing/opening is legitimate)

    # ------------------------------------------------------------------ clearance factors (unchanged)
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_foreign = torch.maximum(ctx.src_hand_foreign_force, ctx.rcv_hand_foreign_force)
    clean = 1.0 - torch.tanh(torch.clamp(f_foreign - CLEAN_DEADBAND, min=0.0) / FORCE_SCALE)
    not_nested_f = (~ctx.cups_nested).to(dt)

    # ------------------------------------------------------------------ grasp flags / heights
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]

    # ------------------------------------------------------------------ carry phases (iter_07)
    # rcv_carry = 0 while the receiver cup is on the table: tilt from hand contact during approach and grasping is
    # neither penalised nor used to scale any income (last round a global upright penalty of -0.36/step at epoch 1
    # taught the policy never to touch the receiver cup; rcv_grasped stayed 0.000 for 208 epochs).
    rcv_carry = _carry_phase(g_rcv, h_rcv, CARRY_GRASP_LO, CARRY_GRASP_HI, CARRY_FREE_LO, CARRY_FREE_HI)
    src_carry = _carry_phase(g_src, h_src, CARRY_GRASP_LO, CARRY_GRASP_HI, CARRY_FREE_LO, CARRY_FREE_HI)

    # ------------------------------------------------------------------ receiver upright gate (phase-gated)
    rcv_tilt_excess = torch.clamp(ctx.rcv_cup_tilt - RCV_TILT_FREE, min=0.0)
    rcv_up_raw = torch.exp(-(rcv_tilt_excess / RCV_TILT_SIGMA) ** 2)              # (N,) in [0,1]
    # upright-dependent scaling acts only in proportion to the carry phase; identical to the old gate once carried
    rcv_up = 1.0 - rcv_carry * (1.0 - rcv_up_raw)                                  # (N,) in [0,1]
    rcv_up_soft = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_up                 # (N,) in [0.3,1]

    # ------------------------------------------------------------------ stage 1: approach (unchanged)
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)

    # ------------------------------------------------------------------ stage 2: grasp (unchanged)
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
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET_SRC, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET_RCV, 0.0, 1.0)
    lift_src = 2.0 * g_src * grip * lift_frac_src
    # receiver lift worth 50 % if carried tilted, 100 % upright; rcv_up is phase-gated, so the first centimetres
    # of lifting (and any contact tilt on the table) are paid in full
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
    # rcv_up_soft is phase-gated: a receiver cup bumped on the table no longer cuts this term; a CARRIED tilted
    # receiver still keeps only 30 % (no tilting the receiver to meet the lip)
    bring_together = 1.0 * src_lifted_f * not_nested_f * clean * torch.exp(-3.0 * lip_xy) * aim_z * rcv_up_soft
    stack_gate = both_lifted_f * not_nested_f * clear * clean
    aim = 3.0 * stack_gate * grip * aim_soft * rcv_up_soft

    # ------------------------------------------------------------------ stage 5: tilt / pour pose
    # stack_gate needs the receiver grasped and > 5 cm up, where rcv_carry = 1 -> exactly as strict as before
    # (20 deg receiver tilt keeps 27 % of the pour-pose income, 30 deg ~0)
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

    # ------------------------------------------------------------------ stage 7: hold after the pour (unchanged)
    post_pour = _smoothstep(ctx.bead_in_target_frac / POST_POUR_FRAC)
    hold_src = 1.5 * post_pour * g_src * wq_src
    hold_rcv = 0.5 * post_pour * g_rcv * wq_rcv * rcv_up    # keep the filled receiver level and firmly held

    # ------------------------------------------------------------------ workspace: meet near the midline (unchanged)
    mid_y = 0.5 * (ctx.src_cup_spawn_pos[:, 1] + ctx.rcv_cup_spawn_pos[:, 1])
    rcv_cross = mid_y - ctx.rcv_cup_pos[:, 1]
    src_cross = ctx.src_cup_pos[:, 1] - mid_y
    meet_rcv = -1.0 * both_lifted_f * torch.tanh(torch.clamp(rcv_cross - MEET_RCV_FREE, min=0.0) / MEET_SCALE)
    meet_src = -0.5 * both_lifted_f * torch.tanh(torch.clamp(src_cross - MEET_SRC_FREE, min=0.0) / MEET_SCALE)

    # ------------------------------------------------------------------ collision / clearance
    cup_contact = -1.0 * torch.tanh(ctx.cup_cup_force / FORCE_SCALE)   # unchanged, ungated: real cup-cup contact
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
    # iter_07: only while at least one cup is carried. Cups spawn ~0.20 m apart (< NEAR_DIST), so cups knocked on
    # the table during reaching were charged (-0.28/step at epoch 1) — another "don't touch the cups" signal.
    carry_any = torch.maximum(src_carry, rcv_carry)
    closing_speed = -1.0 * carry_any * near_f * torch.tanh(torch.clamp(v_close - 0.10, min=0.0) / 0.20)
    nested = -0.5 * ctx.cups_nested.to(dt)

    # ------------------------------------------------------------------ constraints
    # iter_07 (OPERATOR REQUIREMENT): receiver-upright penalty is phase-dependent.
    #   on the table (approach / grasp):     0 — tilt from hand contact is free
    #   grasped and lifted (carry):          -1.0 * tanh(excess/0.25): 10 deg -> -0.22, 20 deg -> -0.73
    #   during the pour / post-pour hold:    doubled: 10 deg -> -0.44, 20 deg -> -1.46
    # pour_phase needs both cups lifted AND the lip roughly over the receiver mouth AND the source tilted, or beads
    # already in the receiver; weight 2.0 at the pour is below the pour-stack income (aim 3 + tilt 4 + pour_pose 3)
    # it protects, so keeping the receiver level is cheaper than giving up the pour.
    pour_phase = torch.maximum(
        both_lifted_f * aim_loose * _smoothstep((tilt_rad - POUR_PHASE_TILT_LO) / (POUR_PHASE_TILT_HI - POUR_PHASE_TILT_LO)),
        post_pour,
    )
    upright_rcv = -(RCV_UP_W + RCV_UP_POUR_W * pour_phase) * rcv_carry * torch.tanh(rcv_tilt_excess / RCV_PEN_SCALE)
    # drop kept: a cup below its spawn height or a receiver lying on its side (> 69 deg) is a lost cup, not wobble
    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)

    # ------------------------------------------------------------------ command smoothness (unchanged)
    da = ctx.actions - ctx.prev_actions                                              # (N,18)
    palm_sq_src = torch.sum(da[:, 0:6] ** 2, dim=-1)
    palm_sq_rcv = torch.sum(da[:, 9:15] ** 2, dim=-1)
    hand_sq = torch.sum(da[:, 6:9] ** 2, dim=-1) + torch.sum(da[:, 15:18] ** 2, dim=-1)
    palm_rate_src = -(PALM_RATE_BASE + PALM_RATE_HELD * g_src) * palm_sq_src
    palm_rate_rcv = -(PALM_RATE_BASE + PALM_RATE_HELD * g_rcv) * palm_sq_rcv
    hand_rate = -HAND_RATE_W * hand_sq

    # ------------------------------------------------------------------ success bonus (unchanged)
    # env success includes rcv tilt <= 20 deg; scaled by source grip quality (0.6..1.0) for a firm post-pour hold
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

adr/progress: [0, 0, 0, 0.0333, 0.233, 0.433, 0.633, 0.833, 1, 1]  min 0 · mean 0.474 · max 1
bead/in_target: [1.22e-05, 0, 0, 0.504, 0.67, 0.691, 0.696, 0.714, 0.732, 0.728]  min 0 · mean 0.495 · max 0.755
bead/spill: [0.000281, 0.0109, 0.0133, 0.107, 0.0607, 0.0725, 0.082, 0.0792, 0.0629, 0.072]  min 0.000195 · mean 0.0545 · max 0.136
done/drop: [0.00928, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 8.05e-05 · max 0.012
episode_lengths/step: [116, 891, 895, 888, 889, 898, 897, 894, 886, 889]  min 98.1 · mean 877 · max 899
reward/aim: [0, 0.0504, 0.0957, 1.81, 1.98, 2.03, 2.03, 2.05, 2.08, 2.08]  min 0 · mean 1.51 · max 2.16
reward/approach_rcv: [0.461, 0.667, 0.659, 0.69, 0.694, 0.667, 0.673, 0.67, 0.663, 0.669]  min 0.461 · mean 0.67 · max 0.697
reward/approach_src: [0.442, 0.695, 0.692, 0.685, 0.678, 0.678, 0.683, 0.687, 0.689, 0.691]  min 0.442 · mean 0.681 · max 0.695
reward/both_grasped: [0, 0.869, 0.859, 0.866, 0.872, 0.871, 0.871, 0.88, 0.863, 0.874]  min 0 · mean 0.853 · max 0.887
reward/both_lifted: [0, 0.827, 0.839, 0.845, 0.852, 0.856, 0.853, 0.863, 0.854, 0.854]  min 0 · mean 0.823 · max 0.871
reward/bring_together: [0, 0.101, 0.159, 0.68, 0.737, 0.743, 0.744, 0.752, 0.75, 0.75]  min 0 · mean 0.578 · max 0.774
reward/closing_speed: [-0.000217, -0.000196, -0.000357, -0.0579, -0.0399, -0.0334, -0.0296, -0.0355, -0.0381, -0.0345]  min -0.0708 · mean -0.028 · max 0
reward/cup_contact: [-0.0847, -0.000241, -0.000852, -0.00207, -0.00193, -0.00603, -0.00612, -0.00491, -0.00476, -0.00474]  min -0.127 · mean -0.00405 · max 0
reward/drop: [-0.014, -0.000122, -0.000244, -0.000122, -0.000244, 0, 0, 0, -0.000122, -0.000122]  min -0.0255 · mean -0.000337 · max 0
reward/grasp_rcv: [0.0114, 1, 0.98, 0.999, 1, 0.997, 0.992, 1.01, 0.987, 1]  min 0.00916 · mean 0.984 · max 1.02
reward/grasp_src: [0.00974, 1.02, 1, 1, 1.01, 1.01, 1.01, 1.02, 1, 1.01]  min 0.00699 · mean 0.991 · max 1.02
reward/hand_foreign_rcv: [-0.0863, -0.033, -0.0114, -0.0187, -0.0191, -0.0147, -0.0173, -0.0215, -0.0157, -0.0152]  min -0.0863 · mean -0.0189 · max -0.00893
reward/hand_foreign_src: [-0.105, -0.0274, -0.0155, -0.00511, -0.00634, -0.00291, -0.00504, -0.00591, -0.00586, -0.00769]  min -0.105 · mean -0.0123 · max -0.00131
reward/hand_rate: [-0.0615, -0.029, -0.0326, -0.0278, -0.0264, -0.0266, -0.0272, -0.0286, -0.0298, -0.0296]  min -0.0615 · mean -0.03 · max -0.0225
reward/hold_rcv: [0, 0, 0, 0.243, 0.29, 0.284, 0.291, 0.294, 0.296, 0.297]  min 0 · mean 0.208 · max 0.307
reward/hold_src: [0, 0, 0, 0.826, 1, 1.05, 1.05, 1.08, 1.1, 1.1]  min 0 · mean 0.747 · max 1.12
reward/lift_rcv: [0, 1.66, 1.69, 1.72, 1.69, 1.71, 1.7, 1.73, 1.7, 1.69]  min 0 · mean 1.65 · max 1.74
reward/lift_src: [0, 1.43, 1.53, 1.56, 1.56, 1.59, 1.59, 1.61, 1.61, 1.62]  min 0 · mean 1.51 · max 1.65
reward/meet_rcv: [0, 0, 0, -0.625, -0.74, -0.704, -0.654, -0.667, -0.638, -0.628]  min -0.748 · mean -0.489 · max 0
reward/meet_src: [0, 0, 0, 0, 0, 0, -2.72e-05, -1.87e-05, 0, -1.57e-05]  min -0.000945 · mean -2.1e-05 · max 0
reward/nested: [-0.00537, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.038 · mean -0.000157 · max 0
reward/palm_rate_rcv: [-0.125, -0.281, -0.171, -0.147, -0.12, -0.107, -0.103, -0.0995, -0.0868, -0.0839]  min -0.765 · mean -0.157 · max -0.0755
reward/palm_rate_src: [-0.125, -0.257, -0.0909, -0.0952, -0.0713, -0.0472, -0.048, -0.0449, -0.0476, -0.0438]  min -0.786 · mean -0.109 · max -0.0365
reward/pour_delta: [0, 0, 0, 0.0409, 0.0446, 0.0439, 0.0586, 0.05, 0.0464, 0.0488]  min 0 · mean 0.0372 · max 0.0787
reward/pour_pose: [0, 0.00379, 0.011, 1.18, 1.72, 1.83, 1.79, 1.83, 1.86, 1.84]  min 0 · mean 1.27 · max 1.94
reward/rot_speed: [-0.105, -0.0483, -0.0392, -0.0342, -0.0261, -0.0209, -0.0239, -0.0234, -0.0252, -0.0278]  min -0.105 · mean -0.0313 · max -0.0187
reward/spill_delta: [-0.0011, -0.00146, -0.00183, -0.00549, -0.00293, -0.00439, -0.00439, -0.00366, -0.00256, -0.00513]  min -0.0128 · mean -0.00289 · max 0
reward/success: [0, 0, 0, 5.19, 6.77, 7.01, 7.03, 7.28, 7.31, 7.39]  min 0 · mean 4.99 · max 7.54
reward/tilt: [0, 0.235, 0.42, 2, 2.43, 2.57, 2.53, 2.62, 2.63, 2.6]  min 0 · mean 1.92 · max 2.73
reward/tilt_dir: [0, 0.245, 0.508, 0.647, 0.705, 0.701, 0.693, 0.698, 0.706, 0.708]  min -0.137 · mean 0.591 · max 0.738
reward/tilt_premature: [0, -0.00838, -0.0088, -0.0238, -0.0307, -0.0408, -0.0384, -0.0408, -0.0363, -0.035]  min -0.0471 · mean -0.027 · max 0
reward/total: [0.192, 9.42, 10.5, 21.4, 24.3, 25, 25, 25.6, 25.6, 25.7]  min 0.0946 · mean 20.5 · max 26.4
reward/upright_rcv: [-0.000327, -0.0788, -0.0395, -0.0261, -0.131, -0.0588, -0.0452, -0.0328, -0.0569, -0.0811]  min -0.254 · mean -0.0632 · max -0.000211
reward/wrap_rcv: [0, 0.327, 0.333, 0.351, 0.351, 0.328, 0.334, 0.33, 0.331, 0.333]  min 0 · mean 0.33 · max 0.361
reward/wrap_src: [0, 1.08, 1.13, 1.15, 1.15, 1.17, 1.17, 1.18, 1.18, 1.19]  min 0 · mean 1.12 · max 1.22
rewards/step: [16.4, 8.1e+03, 9.34e+03, 1.8e+04, 2.14e+04, 2.26e+04, 2.25e+04, 2.27e+04, 2.27e+04, 2.3e+04]  min 15.6 · mean 1.81e+04 · max 2.38e+04
task/aim_dist: [0.148, 0.536, 0.496, 0.137, 0.11, 0.279, 0.106, 0.108, 0.11, 0.109]  min 0.1 · mean 0.224 · max 1.63
task/cup_collision_rate: [0.11, 0.000488, 0.000732, 0.00195, 0.00635, 0.0115, 0.00562, 0.00684, 0.00635, 0.00708]  min 0 · mean 0.00538 · max 0.149
task/cups_center_dist: [0.204, 0.572, 0.564, 0.25, 0.237, 0.407, 0.228, 0.23, 0.234, 0.233]  min 0.204 · mean 0.326 · max 1.63
task/episode_success: [0, 0, 0, 0.541, 0.717, 0.729, 0.739, 0.766, 0.765, 0.776]  min 0 · mean 0.522 · max 0.788
task/nested_rate: [0.0107, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000314 · max 0.0759
task/rcv_cup_lift: [0.0019, 0.164, 0.195, 0.181, 0.186, 0.169, 0.168, 0.163, 0.166, 0.17]  min 0.000685 · mean 0.172 · max 0.377
task/rcv_grasped: [0, 0.887, 0.871, 0.879, 0.881, 0.881, 0.875, 0.89, 0.878, 0.888]  min 0 · mean 0.87 · max 0.904
task/rcv_hand_foreign_rate: [0.171, 0.0525, 0.0286, 0.0498, 0.0459, 0.0381, 0.0496, 0.053, 0.0391, 0.0425]  min 0.0217 · mean 0.0461 · max 0.171
task/src_cup_lift: [-0.000357, 0.258, 0.329, 0.343, 0.368, 0.343, 0.346, 0.345, 0.347, 0.352]  min -0.000357 · mean 0.322 · max 0.371
task/src_grasped: [0, 0.887, 0.881, 0.885, 0.896, 0.89, 0.889, 0.896, 0.876, 0.888]  min 0 · mean 0.872 · max 0.902
task/src_hand_foreign_rate: [0.186, 0.0781, 0.0325, 0.0178, 0.0198, 0.00977, 0.0227, 0.0249, 0.0276, 0.0286]  min 0.00684 · mean 0.0329 · max 0.186
task/src_tilt_deg: [28, 62.5, 70.2, 86.3, 97.2, 101, 100, 101, 99.4, 100]  min 5.6 · mean 87.2 · max 104
task/success_now: [0, 0, 0, 0.549, 0.726, 0.742, 0.749, 0.777, 0.773, 0.785]  min 0 · mean 0.53 · max 0.8

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Observations from the loop operator about the policy trained with the previous function, the environment changes made for this round, and the operator's requirements. Facts only, apart from the explicitly marked operator requirements.

1. The previous function was trained to convergence: 4096 environments, 21.5 h, epoch 2767, randomisation at its maximum (`adr/progress` 1.0). Step-averaged `task/episode_success` 0.77, beads in the receiver 73.7 %, spill 5.9 %, receiver tilt 5.4°, cup–cup contact 0.7 %, hand–foreign contact 2.8 % / 4.4 %, nesting 0. Replayed with the randomisation at zero (8 environments, 900 steps): success 8/8, 96.3 % of the beads in the receiver, spill 3.8 %. The three problems of the round before (right-hand fingers curling, arm vibration, receiver tilted 46°) are gone in the trajectory measurement: the four-finger joint targets move together (root 0.69 → 0.73 rad, tip 0.86 → 0.92 rad), palm contact force 7.6 → 9.0 N, no sign flips in the source palm rotation commands, receiver tilt during the pour median 3.5°.

2. What the operator saw in the video and what the trajectory shows (8 environments, mean): the source cup is tilted while it still stands on the table and keeps tilting during the lift. Source tilt exceeds 20° at step 77 with the cup on the table (lift 0.00 m) while the source mouth is 0.26 m (xy) from the receiver mouth; 45° at step 91, still on the table; 80° at step 116 at 0.21 m lift; the first bead reaches the receiver at step 160. During the pour the mouth-to-mouth xy distance is 0.021 m (median). With the previous bead load nothing spilled during the lift.

3. In the previous function the tilt terms did not oppose this: `tilt` was gated by a loose aim factor that still paid about 40 % at a mouth distance of 0.30 m, `tilt_dir` had no aim gate at all, and `tilt_premature` was zero below 86° (`TILT_FREE` 1.5 rad). So tilting early earned reward and cost nothing.

4. Environment changes for this round (all active in training; the policy is trained from scratch because the observation changed):
   - Beads are 30 mm in diameter (before: 12 mm, which filled only 3 % of the cup), 15.6 g each. The cup holds at most 20 of them (full to the rim). Each episode a random number of beads between 6 and 20 is in the source cup; the upper bound grows from 12 to 20 as the success rate rises. `ctx.bead_fill_level` (0 = empty, 1 = full) is measured after the beads settle and stays constant during the episode; the bead fractions `bead_in_target_frac` / `bead_spill_frac` / `d_in_target` / `d_spill` are relative to the beads present in that episode.
   - Measured spill onset with the new beads (scripted tilt, 8 environments): with a full cup the first bead leaves at about 72°, a fifth of the beads are gone by 80° and half by 86°; with 6 beads the first bead leaves at about 97°.
   - `ctx.success` now also requires that the source cup was never tilted more than 30° while its mouth was more than 0.10 m (xy) from the receiver's mouth. `ctx.premature_tilt` is that latch (True from the first violation until the episode resets). The previous policy would have violated it in every episode (item 2).
   - New feedback-table metrics: `task/premature_tilt_rate` (fraction of environments with the latch set), `task/tilt_far_deg` (mean source tilt while the mouths are more than 0.10 m apart), `task/rcv_palm_speed` (receiver palm speed while the receiver cup is lifted), `bead/fill_level`, `bead/n_active`, `task/rcv_tilt_deg`.

5. OPERATOR REQUIREMENTS:
   a. The source cup must stay upright while it is grasped, lifted and carried; it may tilt only once its mouth is close to the receiver's mouth (the environment invalidates success beyond 30° while farther than 0.10 m). The pour has to be an aimed, precise tilt, whatever the fill level.
   b. The receiver cup is lifted and then held at a steady position during the pour. A small receiver tilt is acceptable; the less the receiver moves, the better (`task/rcv_palm_speed`).
   c. All beads are to be transferred regardless of how full the source cup is; the fill level is available to the policy so that the tilt can be adapted to it.

6. Unchanged facts from earlier rounds: with contact freeze the measured `hand_closure` saturates at about 0.35–0.43 on these cups whatever the grasp type; hand–hand and cup–cup contact remain safety hazards on the real robot; the receiver must be held nearly upright (success requires ≤ 20°).

The policy must complete the full task (wrap-grasp both cups → lift both, source upright → bring the mouths together → tilt the source cup only then → beads in the receiver, receiver held steady and nearly upright, no drop, little spill) and hold both grasps stably after pouring.
