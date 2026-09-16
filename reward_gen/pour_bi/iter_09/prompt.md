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
    premature_tilt: torch.Tensor        # (N,) bool 에피소드 래치 — **잡은** 소스 컵이 입구 xy 거리 10 cm 보다 멀 때 30° 를 넘은 적이 있음(성공 무효, 리셋 전까지 유지; 잡지 않은 채 넘어진 컵은 아님)
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
5. Cup tilt: `ctx.src_cup_tilt` is the angle between the cup's up axis and world +z (0 = upright, π/2 = horizontal). The amount of beads in the source cup changes every episode and `ctx.bead_fill_level` (0 = empty, 1 = full to the rim, measured once the beads have settled, constant during the episode) tells the policy how full the cup is by volume. Beads start leaving the cup at a tilt that depends on that fill (measured): the first bead leaves at about 70° when the cup is full and at about 90° when it holds only a few beads, and a full cup has lost a fifth of its beads by 80° and half by 86°. `ctx.src_cup_mouth_pos` and `ctx.rcv_cup_mouth_pos` are the rim centres.
6. Bead bookkeeping: `bead_in_target_frac` rises as beads land in the receiver cup; `bead_spill_frac` counts beads lost outside both cups (permanent). `d_in_target` and `d_spill` are this step's increments — reward INCREMENTS of beads transferred rather than the level, otherwise the policy is paid for standing still with a filled cup.
7. `ctx.success` is computed by the environment (enough beads in the receiver cup, little spill, cups close together, the receiver cup held nearly upright — `ctx.rcv_cup_tilt` at most 20° — the cups NOT nested, and NO premature tilt: `ctx.premature_tilt` latches for the rest of the episode as soon as the GRASPED source cup exceeds 30° while its mouth is still more than 10 cm (xy) from the receiver's mouth (a cup knocked over without being grasped does not latch), and a latched episode can never succeed). You may add a bonus on it but you cannot redefine it. Beads only count as "in the receiver" once they have LEFT the source cup — pushing the source cup into the receiver cup (`ctx.cups_nested`) transfers nothing and is never a success; the beads must fall out of the tilted source cup through the air.
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
    DZ_LOW = -0.06            # [m] "above" ramp start: lip 6 cm below the receiver rim -> 0
    DZ_HIGH = 0.02            # [m] "above" ramp end: lip 2 cm above the rim -> 1
    DZ_FREE = 0.04            # [m] pouring from up to 4 cm above the rim costs nothing
    DZ_SIGMA = 0.06           # [m] beyond that, gentle decay (10 cm above -> 0.37)
    # iter_08: upright-carry hover band (SOURCE MOUTH above the receiver rim). 11 cm keeps the origins > 9 cm
    # apart (no nesting flag) and the source bottom above the receiver rim; decays above 16 cm.
    HOVER_LO = 0.06           # [m] mouth 6 cm above the receiver rim -> 0 (an upright cup here would be nested)
    HOVER_HI = 0.11           # [m] mouth 11 cm above the rim -> 1
    HOVER_FREE = 0.16         # [m] free up to 16 cm ...
    HOVER_SIGMA = 0.06        # [m] ... then Gaussian decay (22 cm -> 0.37)
    # iter_08: near gate on MOUTH-TO-MOUTH xy distance, the quantity the env latch uses (threshold 0.10 m)
    NEAR_FULL = 0.05          # [m] mouths within 5 cm (xy): tilting is fully allowed / fully paid
    NEAR_ZERO = 0.09          # [m] beyond 9 cm (1 cm inside the env latch): no tilt income, full upright penalty
    # iter_08: source upright while far (OPERATOR REQUIREMENT a)
    SRC_TILT_FREE = 0.20      # [rad] 11.5 deg of source tilt is free (wrap-grasp / carry wobble, pose noise)
    SRC_TILT_SCALE = 0.20     # [rad] tanh scale of the penalty: 20 deg -> 0.63, 30 deg -> 0.92 of the weight
    SRC_TILT_SIGMA = 0.25     # [rad] Gaussian income factor: 20 deg -> 0.70, 30 deg -> 0.19, 45 deg -> 0
    SRC_UP_W_TABLE = 1.0      # penalty weight while grasped on the table (30 deg -> -0.92)
    SRC_UP_W_CARRY = 1.5      # extra once carried: total 2.5 (20 deg -> -1.6, 30 deg -> -2.3)
    LATCH_W = 0.5             # per-step penalty once the env's premature-tilt latch is set (success now impossible)
    # iter_08: fill-adaptive release angle (measured: first bead ~72 deg full, ~97 deg with 6 beads)
    RELEASE_FULL = 1.25       # [rad] release tilt for a full cup (72 deg)
    RELEASE_SPAN = 0.60       # [rad] + (1 - fill) * span: fill 0.3 -> 1.67 rad (96 deg)
    BAND_BELOW = 0.10         # [rad] pour_pose band starts 6 deg below the release tilt ...
    BAND_ABOVE = 0.45         # [rad] ... and saturates 26 deg above it
    TILT_TARGET_ABOVE = 0.55  # [rad] linear tilt term saturates 32 deg past the release tilt (empties the cup)
    TILT_DIR_SCALE = 1.0      # [rad] direction term fully weighted once tilted >= 57 deg
    ROT_SPEED_FREE = 1.0      # [rad/s] controlled pouring rotation is free (was 1.5); 30 mm beads need a slower pour
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
    # carry phases (iter_07)
    CARRY_GRASP_LO = 0.015    # [m] grasped-cup carry ramp start (above the ~1 cm origin rise of a tipped cup)
    CARRY_GRASP_HI = 0.05     # [m] grasped-cup carry ramp end (= LIFT_GATE: pour stack starts here)
    CARRY_FREE_LO = 0.06      # [m] above this the cup is carried whatever the grasp flag says
    CARRY_FREE_HI = 0.10      # [m]
    RCV_UP_W = 1.0            # carried receiver: 20 deg -> -0.73
    RCV_UP_POUR_W = 1.0       # extra during pour / post-pour hold: 20 deg -> -1.46 total
    POUR_PHASE_TILT_LO = 0.8  # [rad] pour phase ramps in as the aimed source cup tilts 46 deg ...
    POUR_PHASE_TILT_HI = 1.6  # [rad] ... to 92 deg
    # iter_08: receiver held still (OPERATOR REQUIREMENT b)
    RCV_STILL_FREE = 0.03     # [m/s] receiver cup speed below 3 cm/s is free (a slow 10 cm lift costs nothing)
    RCV_STILL_SCALE = 0.15    # [m/s] tanh scale: 0.10 m/s -> 0.44 of the weight, 0.25 m/s -> 0.9
    RCV_STILL_W = 0.3         # while carried (lift / meet)
    RCV_STILL_POUR_W = 0.7    # extra while the source mouth is near or beads are already in: total 1.0
    # post-pour hold ramps in over 0 -> 0.8 of the beads (was 0.5): emptying the cup keeps paying
    POST_POUR_FRAC = 0.8
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

    # ------------------------------------------------------------------ carry phases (iter_07, unchanged)
    rcv_carry = _carry_phase(g_rcv, h_rcv, CARRY_GRASP_LO, CARRY_GRASP_HI, CARRY_FREE_LO, CARRY_FREE_HI)
    src_carry = _carry_phase(g_src, h_src, CARRY_GRASP_LO, CARRY_GRASP_HI, CARRY_FREE_LO, CARRY_FREE_HI)

    # ------------------------------------------------------------------ receiver upright gate (phase-gated, unchanged)
    rcv_tilt_excess = torch.clamp(ctx.rcv_cup_tilt - RCV_TILT_FREE, min=0.0)
    rcv_up_raw = torch.exp(-(rcv_tilt_excess / RCV_TILT_SIGMA) ** 2)              # (N,) in [0,1]
    rcv_up = 1.0 - rcv_carry * (1.0 - rcv_up_raw)                                  # (N,) in [0,1]
    rcv_up_soft = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_up                 # (N,) in [0.3,1]

    # ------------------------------------------------------------------ iter_08: near gate (mouth-to-mouth xy)
    # The env latches premature_tilt when the source exceeds 30 deg while the mouths are > 0.10 m apart (xy).
    # near = 1 within 5 cm, 0 beyond 9 cm. It multiplies EVERY tilt income and releases the upright penalty.
    mouth_xy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)   # (N,)
    near = 1.0 - _smoothstep((mouth_xy - NEAR_FULL) / (NEAR_ZERO - NEAR_FULL))                     # (N,) in [0,1]
    far = 1.0 - near

    # ------------------------------------------------------------------ iter_08: source upright while far
    # src_up scales the lift / carry income (like rcv_up does for the receiver); upright_src is the additive
    # penalty. Both are gated on the cup being HELD (grasp flag or carried), so an early random policy that knocks
    # the source cup over is not taught to avoid the cup (the iter_07 receiver lesson); a knocked-over cup is
    # already charged by drop / spill_delta. Once near (mouths within 5 cm) both vanish and tilting is free.
    src_tilt_excess = torch.clamp(ctx.src_cup_tilt - SRC_TILT_FREE, min=0.0)
    src_up_raw = torch.exp(-(src_tilt_excess / SRC_TILT_SIGMA) ** 2)              # (N,) in [0,1]
    src_up = 1.0 - far * (1.0 - src_up_raw)                                        # (N,) in [0,1]
    held_src = torch.maximum(g_src, src_carry)

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
    # iter_08: the source lift is paid in full only while the cup is upright (far); 30 deg far -> 19 %
    lift_src = 2.0 * g_src * grip * lift_frac_src * src_up
    lift_rcv = 2.0 * g_rcv * lift_frac_rcv * (0.5 + 0.5 * rcv_up)

    src_lifted_b = ctx.src_grasped & (h_src > LIFT_GATE)
    both_lifted_b = src_lifted_b & ctx.rcv_grasped & (h_rcv > LIFT_GATE)
    src_lifted_f = src_lifted_b.to(dt)
    both_lifted_f = both_lifted_b.to(dt)
    both_lifted = 1.0 * both_lifted_f

    # ------------------------------------------------------------------ pouring-lip geometry (unchanged)
    lip = _rim_low_point(ctx.src_cup_mouth_pos, ctx.src_cup_up, ctx.cup_radius)   # (N,3)
    lip_delta = lip - ctx.rcv_cup_mouth_pos                                         # (N,3)
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)                                   # (N,) horizontal miss
    dz_lip = lip_delta[:, 2]                                                        # (N,) lip height above rcv rim
    above = torch.clamp((dz_lip - DZ_LOW) / (DZ_HIGH - DZ_LOW), 0.0, 1.0)
    high = torch.exp(-(torch.clamp(dz_lip - DZ_FREE, min=0.0) / DZ_SIGMA) ** 2)
    aim_z = above * high
    aim_xy = torch.exp(-6.0 * torch.clamp(lip_xy - LIP_DEADBAND, min=0.0))
    aim_soft = aim_xy * aim_z                                                       # precise, in [0,1]

    # ------------------------------------------------------------------ stage 4: bring cups together
    # iter_08: the old lip-height factor rewarded an UPRIGHT cup standing beside the receiver at the same height
    # (from where only an early tilt can reach the receiver). Now the upright carry is steered to a hover: source
    # mouth 11-16 cm above the receiver rim with the mouths converging in xy; once near, the lip-based aim_z takes
    # over so lowering the lip during the pour does not lose this income. src_up: approaching tilted is not paid.
    dz_mouth = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    hover = _smoothstep((dz_mouth - HOVER_LO) / (HOVER_HI - HOVER_LO)) \
        * torch.exp(-(torch.clamp(dz_mouth - HOVER_FREE, min=0.0) / HOVER_SIGMA) ** 2)
    bt_z = torch.maximum(hover, near * aim_z)
    bring_together = 1.0 * src_lifted_f * not_nested_f * clean * src_up * torch.exp(-3.0 * mouth_xy) * bt_z * rcv_up_soft
    stack_gate = both_lifted_f * not_nested_f * clear * clean
    # aim is gated by near as well: no income for hovering the lip somewhere beside the receiver
    aim = 3.0 * stack_gate * grip * near * aim_soft * rcv_up_soft

    # ------------------------------------------------------------------ stage 5: tilt / pour pose (iter_08: all x near)
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_rad = ctx.src_cup_tilt
    # fill-adaptive release angle: full -> 1.25 rad (72 deg), fill 0.3 (6 beads) -> 1.67 rad (96 deg)
    fill = torch.clamp(ctx.bead_fill_level, 0.0, 1.0)
    release_tilt = RELEASE_FULL + RELEASE_SPAN * (1.0 - fill)                        # (N,)
    tilt_target = release_tilt + TILT_TARGET_ABOVE                                   # (N,) full 1.80, 6 beads 2.22
    tilt_frac = torch.clamp(tilt_rad / tilt_target, 0.0, 1.0)
    # weight 4.0 x near: worth nothing until the mouths are within 5 cm; aim_z (lip height) gives the gradient
    # from the upright hover (lip ~11 cm above the rim -> 0.26) down to the pouring lip at the rim (1.0)
    tilt = 4.0 * stack_gate * grip * beads_left * near * aim_z * tilt_frac * rcv_up

    to_rcv = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]                  # (N,2)
    to_rcv_n = to_rcv / (torch.norm(to_rcv, dim=-1, keepdim=True) + 1e-6)
    up_xy = ctx.src_cup_up[:, :2]                                                    # (N,2)
    up_xy_n = up_xy / (torch.norm(up_xy, dim=-1, keepdim=True) + 1e-6)
    dir_cos = torch.sum(to_rcv_n * up_xy_n, dim=-1)                                  # (N,) in [-1,1]
    tilt_amt = torch.clamp(tilt_rad / TILT_DIR_SCALE, 0.0, 1.0)
    tilt_dir = 1.0 * stack_gate * beads_left * near * tilt_amt * dir_cos * rcv_up    # was ungated by distance

    pour_band = _smoothstep((tilt_rad - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    pour_pose = 3.0 * stack_gate * grip * beads_left * near * aim_soft * pour_band * rcv_up

    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    rot_speed = -0.3 * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (unchanged, ungated)
    # fractions are per-episode: 20 beads -> 2.5 per bead, 6 beads -> 8.3 per bead. Never gated.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    spill_delta = -30.0 * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ stage 7: hold after the pour
    post_pour = _smoothstep(ctx.bead_in_target_frac / POST_POUR_FRAC)   # 50 % -> 0.68, 75 % -> 0.99
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
    carry_any = torch.maximum(src_carry, rcv_carry)
    closing_speed = -1.0 * carry_any * near_f * torch.tanh(torch.clamp(v_close - 0.10, min=0.0) / 0.20)
    nested = -0.5 * ctx.cups_nested.to(dt)

    # ------------------------------------------------------------------ constraints
    # iter_08 (OPERATOR REQUIREMENT a): source stays upright until its mouth is near the receiver's mouth.
    #   held on the table, far:   -1.0 * tanh(excess/0.2): 20 deg -> -0.63, 30 deg -> -0.92
    #   carried, far:             -2.5 * tanh(excess/0.2): 20 deg -> -1.6,  30 deg -> -2.3, 45 deg -> -2.5
    #   near (mouths <= 5 cm):    0 — the pour is free, and it is the only place any tilt income exists
    # 2.5 exceeds the largest income tilting far could still touch (lift_src 2.0, itself scaled down by src_up).
    upright_src = -(SRC_UP_W_TABLE + SRC_UP_W_CARRY * src_carry) * held_src * far \
        * torch.tanh(src_tilt_excess / SRC_TILT_SCALE)
    # once the env latch is set the episode can no longer succeed; a constant -0.5/step from that moment makes the
    # value drop at the violating step even before the policy has ever seen the success bonus
    premature_latch = -LATCH_W * ctx.premature_tilt.to(dt)

    # receiver-upright penalty is phase-dependent (iter_07); pour_phase now uses the near gate instead of aim_loose
    pour_phase = torch.maximum(
        both_lifted_f * near * aim_z
        * _smoothstep((tilt_rad - POUR_PHASE_TILT_LO) / (POUR_PHASE_TILT_HI - POUR_PHASE_TILT_LO)),
        post_pour,
    )
    upright_rcv = -(RCV_UP_W + RCV_UP_POUR_W * pour_phase) * rcv_carry * torch.tanh(rcv_tilt_excess / RCV_PEN_SCALE)

    # iter_08 (OPERATOR REQUIREMENT b): the carried receiver is held still — 0.3 while carried (a slow lift is
    # inside the 3 cm/s deadband), 1.0 while the source mouth is near or beads are already in (pour / hold):
    # 0.10 m/s -> -0.44, 0.25 m/s -> -0.9 during the pour
    v_rcv = torch.norm(ctx.rcv_cup_lin_vel, dim=-1)
    still_phase = torch.maximum(near, post_pour)
    rcv_still = -(RCV_STILL_W + RCV_STILL_POUR_W * still_phase) * rcv_carry \
        * torch.tanh(torch.clamp(v_rcv - RCV_STILL_FREE, min=0.0) / RCV_STILL_SCALE)

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

    # ------------------------------------------------------------------ success bonus
    # env success now also requires no premature tilt. Scaled by source grip (0.6..1.0) and, iter_08, by the
    # transferred fraction (0.6..1.0; OPERATOR REQUIREMENT c): 50 % of the beads -> 8.0/step, all of them -> 10/step
    success = 10.0 * ctx.success.to(dt) * clear * clean * (0.6 + 0.4 * g_src * wq_src) \
        * (0.6 + 0.4 * torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0))

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
bead/in_target: [0, 0, 2.22e-05, 0, 0, 0, 0, 4.07e-05, 0, 0]  min 0 · mean 7.74e-06 · max 7.1e-05
bead/spill: [0.00139, 0.00371, 0.00613, 0.00697, 0.0071, 0.00753, 0.00782, 0.00825, 0.00805, 0.00792]  min 0.00139 · mean 0.00679 · max 0.0089
done/drop: [0, 0.00439, 0.00293, 0.000244, 0, 0, 0.000244, 0.000244, 0, 0]  min 0 · mean 0.00119 · max 0.0154
episode_lengths/step: [123, 251, 480, 655, 738, 768, 731, 757, 747, 749]  min 97.6 · mean 625 · max 794
reward/aim: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/approach_rcv: [0.52, 0.513, 0.583, 0.612, 0.627, 0.632, 0.645, 0.65, 0.653, 0.659]  min 0.473 · mean 0.614 · max 0.66
reward/approach_src: [0.503, 0.453, 0.446, 0.441, 0.441, 0.442, 0.443, 0.441, 0.439, 0.438]  min 0.434 · mean 0.445 · max 0.503
reward/both_grasped: [0, 0, 0.000244, 0, 0, 0, 0, 0.000244, 0, 0]  min 0 · mean 2.7e-05 · max 0.000732
reward/both_lifted: [0, 0, 0.000244, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.26e-06 · max 0.000244
reward/bring_together: [0, 0.000109, 1.09e-05, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.38e-06 · max 0.000204
reward/closing_speed: [-0.000407, -0.000537, -0.00111, -0.000312, -0.000479, 0, 0, 0, 0, -0.000347]  min -0.00192 · mean -0.000348 · max 0
reward/cup_contact: [-0.0258, -0.00856, -0.0014, -0.000987, -0.00146, -0.000759, -0.000271, -0.000206, -5.5e-05, 0]  min -0.114 · mean -0.00632 · max 0
reward/drop: [0, -0.0116, -0.0118, -0.00476, -0.00525, -0.00256, -0.00281, -0.00256, -0.00232, -0.00122]  min -0.0292 · mean -0.00585 · max 0
reward/grasp_rcv: [0.00965, 0.237, 0.649, 0.796, 0.847, 0.846, 0.893, 0.893, 0.891, 0.891]  min 0.00301 · mean 0.737 · max 0.916
reward/grasp_src: [0.00906, 0.00804, 0.0116, 0.0105, 0.0106, 0.0124, 0.0128, 0.0152, 0.0147, 0.0167]  min 0.00377 · mean 0.0126 · max 0.0185
reward/hand_foreign_rcv: [-0.0848, -0.0404, -0.0462, -0.0758, -0.0799, -0.0696, -0.0836, -0.0837, -0.0855, -0.0642]  min -0.0934 · mean -0.0702 · max -0.0138
reward/hand_foreign_src: [-0.0829, -0.0264, -0.0133, -0.0087, -0.0127, -0.00965, -0.0101, -0.00894, -0.00708, -0.00768]  min -0.0829 · mean -0.0134 · max -0.00499
reward/hand_rate: [-0.0623, -0.0604, -0.0576, -0.0509, -0.0491, -0.0488, -0.0462, -0.044, -0.0424, -0.0413]  min -0.0623 · mean -0.0489 · max -0.0405
reward/hold_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/hold_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/lift_rcv: [0, 0.0439, 0.161, 0.279, 0.549, 1.1, 1.31, 1.37, 1.41, 1.43]  min 0 · mean 0.839 · max 1.47
reward/lift_src: [0, 9.11e-05, 0.000129, 0, 0, 0, 0, 5.27e-05, 0, 0]  min 0 · mean 2.58e-05 · max 0.000492
reward/meet_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/meet_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/nested: [0, -0.00464, -0.000732, -0.000244, -0.000488, -0.000488, -0.000366, -0.000122, -0.000122, 0]  min -0.0226 · mean -0.00165 · max 0
reward/palm_rate_rcv: [-0.125, -0.322, -0.622, -0.587, -0.555, -0.497, -0.439, -0.383, -0.35, -0.323]  min -0.667 · mean -0.43 · max -0.124
reward/palm_rate_src: [-0.123, -0.114, -0.109, -0.1, -0.0961, -0.0926, -0.0912, -0.0876, -0.0846, -0.0831]  min -0.125 · mean -0.0954 · max -0.0765
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/pour_pose: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/premature_latch: [-0.0989, -0.00806, -0.00391, -0.000732, -0.000366, -0.000122, -0.000366, -0.000244, -0.000244, -0.000244]  min -0.0989 · mean -0.004 · max 0
reward/rcv_still: [-0.000122, -0.00674, -0.0289, -0.0537, -0.108, -0.177, -0.194, -0.194, -0.191, -0.192]  min -0.198 · mean -0.124 · max -0.000122
reward/rot_speed: [-0.134, -0.0114, -0.00474, -0.00236, -0.00203, -0.00166, -0.00159, -0.00184, -0.00145, -0.00145]  min -0.134 · mean -0.0066 · max -0.0011
reward/spill_delta: [-0.00259, -0.00238, -0.00105, -0.000666, -0.00387, -0.00227, -0.00308, -0.000916, -0.000732, -0.00275]  min -0.00578 · mean -0.00157 · max 0
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt_dir: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/total: [0.3, 0.568, 0.991, 1.31, 1.58, 2.16, 2.46, 2.65, 2.76, 2.84]  min -0.0917 · mean 1.87 · max 2.9
reward/upright_rcv: [-0.000402, -0.0163, -0.0686, -0.121, -0.179, -0.174, -0.196, -0.14, -0.117, -0.117]  min -0.196 · mean -0.118 · max -0.000311
reward/upright_src: [-0.00113, -3.91e-06, -0.0019, -0.000611, 0, 0, -4.33e-06, -6.91e-06, 0, 0]  min -0.00285 · mean -0.00038 · max 0
reward/wrap_rcv: [0, 0.053, 0.17, 0.226, 0.249, 0.253, 0.271, 0.276, 0.282, 0.287]  min 0 · mean 0.221 · max 0.296
reward/wrap_src: [0, 0.000296, 0.000334, 0, 0, 0, 0, 0.000132, 0, 0]  min 0 · mean 0.000143 · max 0.00259
rewards/step: [8.21, 130, 399, 770, 1.06e+03, 1.5e+03, 1.78e+03, 1.92e+03, 2.04e+03, 2.13e+03]  min 3.22 · mean 1.28e+03 · max 2.28e+03
task/aim_dist: [0.209, 0.717, 1.57, 0.866, 0.666, 0.867, 0.609, 0.589, 0.507, 0.619]  min 0.209 · mean 0.682 · max 1.82
task/cup_collision_rate: [0.0325, 0.00903, 0.00171, 0.000977, 0.00146, 0.000732, 0.000244, 0.000244, 0, 0]  min 0 · mean 0.00709 · max 0.134
task/cups_center_dist: [0.255, 0.722, 1.56, 0.871, 0.671, 0.867, 0.605, 0.589, 0.505, 0.616]  min 0.246 · mean 0.683 · max 1.82
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0.00928, 0.00146, 0.000488, 0.000977, 0.000977, 0.000732, 0.000244, 0.000244, 0]  min 0 · mean 0.00329 · max 0.0452
task/rcv_cup_lift: [0.00424, 0.154, 0.392, 0.146, 0.0899, 0.297, 0.198, 0.183, 0.173, 0.224]  min -0.000505 · mean 0.176 · max 0.431
task/rcv_grasped: [0, 0.199, 0.568, 0.692, 0.735, 0.732, 0.773, 0.77, 0.768, 0.767]  min 0 · mean 0.637 · max 0.791
task/rcv_hand_foreign_rate: [0.153, 0.0862, 0.0872, 0.126, 0.128, 0.109, 0.119, 0.119, 0.125, 0.102]  min 0.0283 · mean 0.114 · max 0.153
task/src_cup_lift: [0.00299, 0.000401, 0.000318, 0.00355, 0.000103, 6.69e-05, 7.62e-05, 9.19e-05, 4.92e-05, 5.2e-05]  min 3.33e-05 · mean 0.000228 · max 0.00355
task/src_grasped: [0, 0.000488, 0.000488, 0, 0, 0, 0, 0.000244, 0, 0]  min 0 · mean 0.000213 · max 0.00366
task/src_hand_foreign_rate: [0.149, 0.0654, 0.0359, 0.0286, 0.0342, 0.0337, 0.0337, 0.0281, 0.0232, 0.0232]  min 0.0154 · mean 0.0359 · max 0.149
task/src_tilt_deg: [16.6, 1.23, 0.457, 0.217, 0.194, 0.136, 0.126, 0.183, 0.117, 0.144]  min 0.0981 · mean 0.781 · max 16.6
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Observations from the loop operator about the policy trained with the previous function, one environment fix made for this round, and the operator's requirements. Facts only, apart from the explicitly marked operator requirements.

1. The previous function was stopped early, after 159 epochs (1.3 h), because the source hand never grasped its cup. The feedback table above therefore covers only these epochs. Comparison at epoch 160 with the two earlier converged runs (this run / round-8 run / round-6 run):

| metric | this run | round 8 | round 6 |
|---|---|---|---|
| source grasped | 0.000 | 0.872 | 0.871 |
| receiver grasped | 0.775 | 0.875 | 0.866 |
| source cup lift [m] | 0.000 | 0.166 | 0.207 |
| receiver cup lift [m] | 0.222 | 0.192 | 0.143 |
| source tilt [deg] | 0.1 | 42.6 | 33.3 |
| receiver hand–foreign contact | 10.8 % | 4.0 % | 2.8 % |
| reward/total | 2.8 | 8.0 | 10.2 |

   The receiver side learned normally (grasp 0.78, lift 0.22 m, receiver held 8.9° from upright). The source hand approached its cup (`reward/approach_src` 0.44) but did not close (`task/src_closure` 0.12) and never grasped; the source cup stayed untouched and upright.

2. Why (measured, first epoch → epoch 159): at epoch 1 the premature-tilt latch fired in 19.8 % of environments while source grasping was 0.000 — random exploration knocked the source cup over without grasping it, and the environment latched those episodes (the latch had no grasp condition). The previous function paid `premature_latch` −0.5 per step for the rest of every latched episode. The policy removed the latch by avoiding the source cup: latch rate 19.8 % → 0.06 %, source hand–foreign contact 14.9 % → 2.0 %, source tilt 16.6° → 0.1°, source grasp 0.000 throughout. This is the same mechanism as the round-7 receiver avoidance (always-on receiver-upright penalty), now on the source side.

3. Environment fix for this round: `ctx.premature_tilt` now latches only when the source cup is GRASPED (thumb and another finger in contact) at the moment it exceeds 30° while its mouth is more than 0.10 m (xy) from the receiver's mouth. A cup knocked over without being grasped no longer latches. Tilting a grasped cup on the table before bringing the mouths together — what the round-8 policy did (20° at a mouth distance of 0.26 m with the cup still on the table) — still latches and still invalidates success. Everything else is unchanged from the previous round: beads 30 mm, 6–20 per episode (upper bound 12 → 20 with success), `ctx.bead_fill_level`, receiver ≤ 20° for success. The policy is trained from scratch.

4. Spill of 0.8 % appeared although the source cup was never moved: it is the bead-spawn/settling floor of this environment (measured 0.6–6 % in probes), not a policy behaviour.

5. OPERATOR REQUIREMENTS (unchanged):
   a. The source cup stays upright while grasped, lifted and carried; it may tilt only once its mouth is close to the receiver's mouth (success invalid beyond 30° while farther than 0.10 m, when grasped).
   b. The receiver cup is lifted and held at a steady position during the pour; small tilt acceptable, less movement better (`task/rcv_palm_speed`).
   c. All beads are to be transferred whatever the fill level.

6. Unchanged facts from earlier rounds: with contact freeze the measured `hand_closure` saturates at about 0.35–0.43 on these cups; hand–hand and cup–cup contact remain safety hazards on the real robot; measured spill onset with the 30 mm beads is about 70° for a full cup and about 90° for a few beads.

The policy must complete the full task (wrap-grasp both cups → lift both, source upright → bring the mouths together → tilt the source cup only then → beads in the receiver, receiver held steady and nearly upright, no drop, little spill) and hold both grasps stably after pouring.
