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
    # iter_09: the hover band needs h_src = h_rcv + 0.11..0.16; with the receiver in its 0.10-0.15 m band that is
    # 0.21-0.31 m, so the absolute source lift keeps its gradient up to 25 cm (was 20)
    LIFT_TARGET_SRC = 0.25    # [m]
    LIFT_TARGET_RCV = 0.10    # [m] receiver lifted moderately (operator requirement)
    RCV_LIFT_FREE = 0.15      # [m] iter_09: receiver lift income is full up to 15 cm ...
    RCV_LIFT_SIGMA = 0.08     # [m] ... then decays (20 cm -> 0.68, 22 cm -> 0.46, 25 cm -> 0.21); it sat at 22 cm
    LIFT_GATE = 0.05          # [m] minimum lift before the pour stack is rewarded
    LIP_DEADBAND = 0.01       # [m] lip within 1 cm of the receiver mouth centre counts as centred (pose noise)
    DZ_LOW = -0.06            # [m] "above" ramp start: lip 6 cm below the receiver rim -> 0
    DZ_HIGH = 0.02            # [m] "above" ramp end: lip 2 cm above the rim -> 1
    DZ_FREE = 0.04            # [m] pouring from up to 4 cm above the rim costs nothing
    DZ_SIGMA = 0.06           # [m] beyond that, gentle decay (10 cm above -> 0.37)
    # upright-carry hover band (SOURCE MOUTH above the receiver rim). 11 cm keeps the origins > 9 cm apart
    # (no nesting flag) and the source bottom above the receiver rim; decays above 16 cm.
    HOVER_LO = 0.06           # [m] mouth 6 cm above the receiver rim -> 0 (an upright cup here would be nested)
    HOVER_HI = 0.11           # [m] mouth 11 cm above the rim -> 1
    HOVER_FREE = 0.16         # [m] free up to 16 cm ...
    HOVER_SIGMA = 0.06        # [m] ... then Gaussian decay (22 cm -> 0.37)
    # iter_09: relative-height ramp, NOT multiplied by the xy distance, so "source mouth above receiver rim" has a
    # gradient from 4 cm below the rim upward even while the cups are still 30 cm apart
    RAISE_LO = -0.04          # [m] ramp start (mouth 4 cm below the receiver rim -> 0); ends at HOVER_HI -> 1
    RAISE_W = 1.0             # weight: below bring_together (1.5) so converging still pays more than height alone
    BRING_W = 1.5             # iter_09: was 1.0 — the xy transport is the only road to the near-gated pour income
    # near gate on MOUTH-TO-MOUTH xy distance, the quantity the env latch uses (threshold 0.10 m)
    NEAR_FULL = 0.05          # [m] mouths within 5 cm (xy): tilting is fully allowed / fully paid
    NEAR_ZERO = 0.09          # [m] beyond 9 cm (1 cm inside the env latch): no tilt income, full upright penalty
    # source upright while far (OPERATOR REQUIREMENT a) — all gated on the cup being HELD
    SRC_TILT_FREE = 0.20      # [rad] 11.5 deg of source tilt is free (wrap-grasp / carry wobble, pose noise)
    SRC_TILT_SCALE = 0.20     # [rad] tanh scale of the penalty: 20 deg -> 0.63, 30 deg -> 0.92 of the weight
    SRC_TILT_SIGMA = 0.25     # [rad] Gaussian income factor: 20 deg -> 0.70, 30 deg -> 0.19, 45 deg -> 0
    SRC_UP_W_TABLE = 1.0      # penalty weight while grasped on the table (30 deg -> -0.92)
    SRC_UP_W_CARRY = 1.5      # extra once carried: total 2.5 (20 deg -> -1.6, 30 deg -> -2.3)
    LATCH_W = 0.5             # per-step penalty once the env's premature-tilt latch is set (GRASPED cups only)
    # fill-adaptive release angle (measured: first bead ~72 deg full, ~97 deg with 6 beads)
    RELEASE_FULL = 1.25       # [rad] release tilt for a full cup (72 deg)
    RELEASE_SPAN = 0.60       # [rad] + (1 - fill) * span: fill 0.3 -> 1.67 rad (96 deg)
    BAND_BELOW = 0.10         # [rad] pour_pose band starts 6 deg below the release tilt ...
    BAND_ABOVE = 0.45         # [rad] ... and saturates 26 deg above it
    TILT_TARGET_ABOVE = 0.55  # [rad] linear tilt term saturates 32 deg past the release tilt (empties the cup)
    TILT_DIR_SCALE = 1.0      # [rad] direction term fully weighted once tilted >= 57 deg
    ROT_SPEED_FREE = 1.0      # [rad/s] controlled pouring rotation is free; 30 mm beads need a slow pour
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped / knocked off the table
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side (a lost cup, not contact wobble)
    NEAR_DIST = 0.30          # [m] cup-centre distance inside which approach speed is regulated
    FORCE_SCALE = 5.0         # [N] tanh scale for all contact penalties
    FOREIGN_DEADBAND = 1.0    # [N] incidental brushes are free (additive term)
    CLEAN_DEADBAND = 0.5      # [N] deadband of the multiplicative hand-foreign gate
    # iter_10: the hand-foreign penalty is phase-weighted. At the first checkpoint the fumbling source hand paid
    # -0.088/step for touching the table and learnt to back away from its cup instead of grasping it; a hand that
    # is NOT holding its cup pays 30 %, a hand holding/carrying its cup (the real-robot hazard) pays 100 %.
    FOREIGN_W_FREE = 0.3
    FOREIGN_W_HELD = 1.0
    GRIP_FLOOR = 0.3          # pour/lift income still worth 30 % with a hook grasp, 100 % with a wrap
    MEET_RCV_FREE = 0.02      # [m] receiver cup may cross the spawn midline by 2 cm for free
    MEET_SRC_FREE = 0.05      # [m] source cup may cross the midline by 5 cm for free
    MEET_SCALE = 0.10         # [m] tanh scale of the crossing penalties
    # receiver-upright shape (success requires rcv tilt <= 20 deg = 0.349 rad)
    RCV_TILT_FREE = 0.12      # [rad] ~7 deg of receiver tilt is free (grasp/lift wobble, pose noise)
    RCV_TILT_SIGMA = 0.20     # [rad] gate: 15 deg -> 0.61, 20 deg -> 0.27, 30 deg -> 0.03
    RCV_PEN_SCALE = 0.25      # [rad] additive penalty: 10 deg -> 0.22, 20 deg -> 0.73, 46 deg -> ~1
    RCV_GATE_FLOOR = 0.3      # positioning terms keep 30 % income while rcv is tilted
    # carry phases
    CARRY_GRASP_LO = 0.015    # [m] grasped-cup carry ramp start (above the ~1 cm origin rise of a tipped cup)
    CARRY_GRASP_HI = 0.05     # [m] grasped-cup carry ramp end (= LIFT_GATE: pour stack starts here)
    CARRY_FREE_LO = 0.06      # [m] above this the cup is carried whatever the grasp flag says
    CARRY_FREE_HI = 0.10      # [m]
    RCV_UP_W = 1.0            # carried receiver: 20 deg -> -0.73
    RCV_UP_POUR_W = 1.0       # extra during pour / post-pour hold: 20 deg -> -1.46 total
    POUR_PHASE_TILT_LO = 0.8  # [rad] pour phase ramps in as the aimed source cup tilts 46 deg ...
    POUR_PHASE_TILT_HI = 1.6  # [rad] ... to 92 deg
    # receiver held still (OPERATOR REQUIREMENT b)
    RCV_STILL_FREE = 0.03     # [m/s] receiver cup speed below 3 cm/s is free (a slow 10 cm lift costs nothing)
    RCV_STILL_SCALE = 0.15    # [m/s] tanh scale: 0.10 m/s -> 0.44 of the weight, 0.25 m/s -> 0.9
    RCV_STILL_W = 0.5         # while carried (0.10 m/s -> -0.22)
    RCV_STILL_POUR_W = 0.5    # extra while the source mouth is near or beads are already in: total 1.0
    # post-pour hold ramps in over 0 -> 0.8 of the beads: emptying the cup keeps paying
    POST_POUR_FRAC = 0.8
    # bead spawn/settling drops 0.6-6 % of the beads in the first moments of every episode whatever the policy
    # does; the spill penalty ramps in over the first 5 % of the episode so that floor is not charged
    SETTLE_FRAC = 0.05
    # iter_10: reach ramp across the 0.22 m closing radius. Round 9 parked the source palm at 0.21-0.25 m
    # (approach_src still paid 0.44 there vs 0.63 grasped) so the hand could never close; the ramp pays 0 at
    # 26 cm and 1.0 at 14 cm (a wrap grasp sits at ~0.115 m): from the 0.165 m start pose it is worth 0.89
    # and backing off to 0.25 m loses ~0.87/step — far more than any pre-grasp contact penalty.
    REACH_FAR = 0.26          # [m]
    REACH_NEAR = 0.14         # [m]
    REACH_W = 1.0
    # iter_10: closing at the cup is paid 1.0 x exp(-6d) x closure/0.35 (0.5 at the grasp distance; was
    # 0.5 x exp(-6d) x raw closure ~ 0.1). Closure is normalised by its contact-freeze saturation (0.35-0.43).
    CLOSE_W = 1.0
    CLOSE_NORM = 0.35
    # iter_10: command smoothness on RAW actions is a tax on exploration noise and paid the policy to pin the
    # palm commands at +-1 (noise clipped -> zero rate): palm_rate_rcv -0.53 -> -0.12 by saturating, and once the
    # source was held the 0.18 extra cost -0.31/step, more than lift_src (+0.29) paid. The env EMA already removes
    # command chatter physically, so only a small flat rate remains (0.02 + 0.18 held -> 0.01 flat).
    PALM_RATE_W = 0.01        # per unit of sum-of-squared palm command change (both arms, all phases)
    HAND_RATE_W = 0.005       # finger commands: grasp closing/opening is legitimate (was 0.01)
    # iter_10: saturation penalty — palm commands beyond |0.7| ramp to a penalty (all six pinned at +-1 ->
    # -0.15/step per arm). Pinned commands drove the arm to its workspace limit (target-actual error 0.10 m,
    # 144 N cup-cup peaks) and the source palm out of the closing radius. A legitimate 25 cm lift needs far less.
    SAT_LO = 0.7
    SAT_W = 0.15

    # ------------------------------------------------------------------ clearance factors
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_foreign = torch.maximum(ctx.src_hand_foreign_force, ctx.rcv_hand_foreign_force)
    clean = 1.0 - torch.tanh(torch.clamp(f_foreign - CLEAN_DEADBAND, min=0.0) / FORCE_SCALE)
    not_nested_f = (~ctx.cups_nested).to(dt)

    # ------------------------------------------------------------------ grasp flags / heights
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]

    # ------------------------------------------------------------------ carry phases
    rcv_carry = _carry_phase(g_rcv, h_rcv, CARRY_GRASP_LO, CARRY_GRASP_HI, CARRY_FREE_LO, CARRY_FREE_HI)
    src_carry = _carry_phase(g_src, h_src, CARRY_GRASP_LO, CARRY_GRASP_HI, CARRY_FREE_LO, CARRY_FREE_HI)
    held_src = torch.maximum(g_src, src_carry)      # (N,) source cup held (grasp flag or carried)
    held_rcv = torch.maximum(g_rcv, rcv_carry)      # (N,) receiver cup held

    # ------------------------------------------------------------------ receiver upright gate (phase-gated)
    rcv_tilt_excess = torch.clamp(ctx.rcv_cup_tilt - RCV_TILT_FREE, min=0.0)
    rcv_up_raw = torch.exp(-(rcv_tilt_excess / RCV_TILT_SIGMA) ** 2)              # (N,) in [0,1]
    rcv_up = 1.0 - rcv_carry * (1.0 - rcv_up_raw)                                  # (N,) in [0,1]
    rcv_up_soft = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_up                 # (N,) in [0.3,1]

    # ------------------------------------------------------------------ near gate (mouth-to-mouth xy)
    # The env latches premature_tilt when the GRASPED source exceeds 30 deg while the mouths are > 0.10 m apart
    # (xy). near = 1 within 5 cm, 0 beyond 9 cm. It multiplies EVERY tilt income and releases the upright penalty.
    mouth_xy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)   # (N,)
    near = 1.0 - _smoothstep((mouth_xy - NEAR_FULL) / (NEAR_ZERO - NEAR_FULL))                     # (N,) in [0,1]
    far = 1.0 - near

    # ------------------------------------------------------------------ source upright while far
    # src_up scales the lift / carry income; upright_src is the additive penalty. Both are gated on the cup being
    # HELD (grasp flag or carried) so a random policy that knocks the source cup over is not taught to avoid the cup.
    src_tilt_excess = torch.clamp(ctx.src_cup_tilt - SRC_TILT_FREE, min=0.0)
    src_up_raw = torch.exp(-(src_tilt_excess / SRC_TILT_SIGMA) ** 2)              # (N,) in [0,1]
    src_up = 1.0 - far * (1.0 - src_up_raw)                                        # (N,) in [0,1]

    # ------------------------------------------------------------------ stage 1: approach + reach
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)
    # iter_10: steep ramp across the closing radius (0 at 26 cm, 1 at 14 cm) — the anchor that keeps the palm
    # inside the 0.22 m radius where the hand is allowed to close
    reach_src = REACH_W * _smoothstep((REACH_FAR - d_src) / (REACH_FAR - REACH_NEAR))
    reach_rcv = REACH_W * _smoothstep((REACH_FAR - d_rcv) / (REACH_FAR - REACH_NEAR))

    # ------------------------------------------------------------------ stage 2: grasp
    prox_src = torch.exp(-6.0 * d_src)
    prox_rcv = torch.exp(-6.0 * d_rcv)
    clos_src = torch.clamp(ctx.src_hand_closure / CLOSE_NORM, 0.0, 1.0)   # wrapped hand under contact freeze -> 1
    clos_rcv = torch.clamp(ctx.rcv_hand_closure / CLOSE_NORM, 0.0, 1.0)
    grasp_src = 1.0 * g_src + CLOSE_W * prox_src * clos_src
    grasp_rcv = 1.0 * g_rcv + CLOSE_W * prox_rcv * clos_rcv
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
    # receiver income band — full up to 15 cm, decaying above (income shaping, never negative)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    # the source lift is paid in full only while the cup is upright (far); 30 deg far -> 19 %
    lift_src = 2.0 * g_src * grip * lift_frac_src * src_up
    lift_rcv = 2.0 * g_rcv * lift_frac_rcv * rcv_band * (0.5 + 0.5 * rcv_up)

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

    # ------------------------------------------------------------------ stage 4: raise, then bring cups together
    # Upright carry is steered to a hover: source mouth 11-16 cm above the receiver rim with the mouths converging
    # in xy; once near, the lip-based aim_z takes over so lowering the lip during the pour does not lose income.
    dz_mouth = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    hover_decay = torch.exp(-(torch.clamp(dz_mouth - HOVER_FREE, min=0.0) / HOVER_SIGMA) ** 2)
    hover = _smoothstep((dz_mouth - HOVER_LO) / (HOVER_HI - HOVER_LO)) * hover_decay
    # relative height paid on its own (no xy factor): lowering the receiver (free above its 0.10 m target) or
    # raising the source both climb this ramp
    hover_wide = _smoothstep((dz_mouth - RAISE_LO) / (HOVER_HI - RAISE_LO)) * hover_decay
    raise_src = RAISE_W * both_lifted_f * not_nested_f * src_up * rcv_up_soft * hover_wide
    bt_z = torch.maximum(hover, near * aim_z)
    bring_together = BRING_W * src_lifted_f * not_nested_f * clean * src_up * torch.exp(-3.0 * mouth_xy) \
        * bt_z * rcv_up_soft
    stack_gate = both_lifted_f * not_nested_f * clear * clean
    # aim is gated by near as well: no income for hovering the lip somewhere beside the receiver
    aim = 3.0 * stack_gate * grip * near * aim_soft * rcv_up_soft

    # ------------------------------------------------------------------ stage 5: tilt / pour pose (all x near)
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
    tilt_dir = 1.0 * stack_gate * beads_left * near * tilt_amt * dir_cos * rcv_up

    pour_band = _smoothstep((tilt_rad - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    pour_pose = 3.0 * stack_gate * grip * beads_left * near * aim_soft * pour_band * rcv_up

    # iter_10: gated on the source being HELD — a cup knocked over by an exploring hand spins fast and used to
    # charge -0.134/step at the first checkpoint, another "avoid the cup" signal; a held cup's rotation is the pour
    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    rot_speed = -0.3 * held_src * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (ungated income)
    # fractions are per-episode: 20 beads -> 2.5 per bead, 6 beads -> 8.3 per bead. Never gated.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    # spill charged only once the beads have settled (first 5 % of the episode ramps 0 -> 1)
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    spill_delta = -30.0 * settled * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ stage 7: hold after the pour
    post_pour = _smoothstep(ctx.bead_in_target_frac / POST_POUR_FRAC)   # 50 % -> 0.68, 75 % -> 0.99
    hold_src = 1.5 * post_pour * g_src * wq_src
    hold_rcv = 0.5 * post_pour * g_rcv * wq_rcv * rcv_up    # keep the filled receiver level and firmly held

    # ------------------------------------------------------------------ workspace: meet near the midline
    mid_y = 0.5 * (ctx.src_cup_spawn_pos[:, 1] + ctx.rcv_cup_spawn_pos[:, 1])
    rcv_cross = mid_y - ctx.rcv_cup_pos[:, 1]
    src_cross = ctx.src_cup_pos[:, 1] - mid_y
    meet_rcv = -1.0 * both_lifted_f * torch.tanh(torch.clamp(rcv_cross - MEET_RCV_FREE, min=0.0) / MEET_SCALE)
    meet_src = -0.5 * both_lifted_f * torch.tanh(torch.clamp(src_cross - MEET_SRC_FREE, min=0.0) / MEET_SCALE)

    # ------------------------------------------------------------------ collision / clearance
    cup_contact = -1.0 * torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_src = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    f_rcv = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    # iter_10: 0.3 while the hand is not holding its cup (table brushes during exploration), 1.0 while it holds it
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
    closing_speed = -1.0 * carry_any * near_f * torch.tanh(torch.clamp(v_close - 0.10, min=0.0) / 0.20)
    nested = -0.5 * ctx.cups_nested.to(dt)

    # ------------------------------------------------------------------ constraints
    # (OPERATOR REQUIREMENT a): source stays upright until its mouth is near the receiver's mouth.
    #   held on the table, far:   -1.0 * tanh(excess/0.2): 20 deg -> -0.63, 30 deg -> -0.92
    #   carried, far:             -2.5 * tanh(excess/0.2): 20 deg -> -1.6,  30 deg -> -2.3, 45 deg -> -2.5
    #   near (mouths <= 5 cm):    0 — the pour is free, and it is the only place any tilt income exists
    # 2.5 exceeds the largest income tilting far could still touch (lift_src 2.0, itself scaled down by src_up).
    upright_src = -(SRC_UP_W_TABLE + SRC_UP_W_CARRY * src_carry) * held_src * far \
        * torch.tanh(src_tilt_excess / SRC_TILT_SCALE)
    # The env latch fires only on a GRASPED cup; a constant -0.5/step from the latch moment makes the value drop at
    # the violating step, while the held-cup income (grasp 1.0 + wrap <= 1.5 + lift <= 2.0) still exceeds it.
    premature_latch = -LATCH_W * ctx.premature_tilt.to(dt)

    # receiver-upright penalty is phase-dependent; pour_phase uses the near gate
    pour_phase = torch.maximum(
        both_lifted_f * near * aim_z
        * _smoothstep((tilt_rad - POUR_PHASE_TILT_LO) / (POUR_PHASE_TILT_HI - POUR_PHASE_TILT_LO)),
        post_pour,
    )
    upright_rcv = -(RCV_UP_W + RCV_UP_POUR_W * pour_phase) * rcv_carry * torch.tanh(rcv_tilt_excess / RCV_PEN_SCALE)

    # (OPERATOR REQUIREMENT b): the carried receiver is held still — 0.5 while carried (a slow lift is inside the
    # 3 cm/s deadband), 1.0 while the source mouth is near or beads are already in (pour / hold):
    # 0.10 m/s -> -0.22 carried / -0.44 during the pour, 0.25 m/s -> -0.45 / -0.9
    v_rcv = torch.norm(ctx.rcv_cup_lin_vel, dim=-1)
    still_phase = torch.maximum(near, post_pour)
    rcv_still = -(RCV_STILL_W + RCV_STILL_POUR_W * still_phase) * rcv_carry \
        * torch.tanh(torch.clamp(v_rcv - RCV_STILL_FREE, min=0.0) / RCV_STILL_SCALE)

    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)

    # ------------------------------------------------------------------ command smoothness + saturation
    da = ctx.actions - ctx.prev_actions                                              # (N,18)
    palm_sq_src = torch.sum(da[:, 0:6] ** 2, dim=-1)
    palm_sq_rcv = torch.sum(da[:, 9:15] ** 2, dim=-1)
    hand_sq = torch.sum(da[:, 6:9] ** 2, dim=-1) + torch.sum(da[:, 15:18] ** 2, dim=-1)
    palm_rate_src = -PALM_RATE_W * palm_sq_src
    palm_rate_rcv = -PALM_RATE_W * palm_sq_rcv
    hand_rate = -HAND_RATE_W * hand_sq
    # iter_10: palm commands pinned at +-1 (mean of the six per arm, ramp from |0.7| to |1.0|)
    a_abs = torch.abs(ctx.actions)                                                   # (N,18)
    sat_src = torch.mean(_smoothstep((a_abs[:, 0:6] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)    # (N,) in [0,1]
    sat_rcv = torch.mean(_smoothstep((a_abs[:, 9:15] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)   # (N,) in [0,1]
    palm_sat_src = -SAT_W * sat_src
    palm_sat_rcv = -SAT_W * sat_rcv

    # ------------------------------------------------------------------ success bonus
    # env success requires no premature tilt, no nesting, receiver <= 20 deg. Scaled by source grip (0.6..1.0) and
    # by the transferred fraction (0.6..1.0; OPERATOR REQUIREMENT c): 50 % of the beads -> 8.0/step, all -> 10/step
    success = 10.0 * ctx.success.to(dt) * clear * clean * (0.6 + 0.4 * g_src * wq_src) \
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
bead/in_target: [0, 0.000705, 0.000943, 0.00141, 0.000783, 0.00114, 0.00219, 0.00219, 0.0037, 0.00401]  min 0 · mean 0.00196 · max 0.00444
bead/spill: [0.00144, 0.0174, 0.0196, 0.0205, 0.0198, 0.0243, 0.0259, 0.0271, 0.0252, 0.0252]  min 0.00144 · mean 0.0224 · max 0.0298
done/drop: [0.000244, 0, 0, 0, 0, 0.000244, 0, 0, 0, 0]  min 0 · mean 0.000157 · max 0.012
episode_lengths/step: [123, 884, 846, 884, 879, 882, 883, 898, 893, 880]  min 102 · mean 857 · max 898
reward/aim: [0, 1.39e-06, 8.99e-05, 2e-05, 1.14e-05, 1.88e-06, 8.08e-11, 3.92e-08, 1.03e-10, 1.04e-05]  min 0 · mean 8.52e-06 · max 0.000188
reward/approach_rcv: [0.521, 0.665, 0.67, 0.679, 0.685, 0.686, 0.686, 0.689, 0.691, 0.692]  min 0.478 · mean 0.677 · max 0.695
reward/approach_src: [0.503, 0.67, 0.678, 0.677, 0.677, 0.677, 0.678, 0.68, 0.684, 0.682]  min 0.477 · mean 0.671 · max 0.686
reward/both_grasped: [0, 0.745, 0.8, 0.798, 0.802, 0.808, 0.821, 0.821, 0.815, 0.822]  min 0 · mean 0.771 · max 0.84
reward/both_lifted: [0, 0.00928, 0.129, 0.488, 0.712, 0.765, 0.783, 0.8, 0.783, 0.793]  min 0 · mean 0.565 · max 0.819
reward/bring_together: [0, 0.00193, 0.00537, 0.113, 0.218, 0.238, 0.267, 0.365, 0.425, 0.426]  min 0 · mean 0.232 · max 0.499
reward/closing_speed: [-4.98e-07, -0.0143, -0.0192, -0.0224, -0.00553, -0.00556, -0.00256, -0.00489, -0.0126, -0.0106]  min -0.0308 · mean -0.0118 · max -4.98e-07
reward/cup_contact: [-0.0241, -0.356, -0.118, -0.106, -0.0902, -0.0854, -0.0676, -0.0619, -0.0807, -0.0731]  min -0.664 · mean -0.133 · max -0.0241
reward/drop: [-0.000122, -0.00159, -0.000732, -0.000977, -0.000854, -0.000488, -0.00061, -0.00061, -0.000488, -0.000732]  min -0.0293 · mean -0.00175 · max 0
reward/grasp_rcv: [0.0552, 1.29, 1.34, 1.34, 1.36, 1.35, 1.36, 1.36, 1.36, 1.36]  min 0.0165 · mean 1.31 · max 1.39
reward/grasp_src: [0.0519, 1.33, 1.36, 1.35, 1.35, 1.38, 1.37, 1.38, 1.39, 1.39]  min 0.0181 · mean 1.33 · max 1.41
reward/hand_foreign_rcv: [-0.0251, -0.115, -0.0456, -0.0537, -0.054, -0.058, -0.0373, -0.0363, -0.0481, -0.0352]  min -0.192 · mean -0.056 · max -0.00532
reward/hand_foreign_src: [-0.0256, -0.148, -0.0437, -0.0531, -0.0575, -0.0583, -0.0466, -0.0416, -0.052, -0.0407]  min -0.195 · mean -0.0606 · max -0.00655
reward/hand_rate: [-0.0311, -0.0233, -0.0181, -0.0169, -0.0146, -0.0159, -0.0158, -0.0185, -0.019, -0.0187]  min -0.0311 · mean -0.0185 · max -0.0144
reward/hold_rcv: [0, 6.94e-05, 0.000118, 0.000179, 0.000104, 0.000149, 0.000299, 0.000341, 0.000574, 0.000598]  min 0 · mean 0.000288 · max 0.000779
reward/hold_src: [0, 0.000267, 0.000451, 0.000698, 0.000317, 0.000525, 0.000965, 0.00104, 0.00183, 0.00193]  min 0 · mean 0.000927 · max 0.00244
reward/lift_rcv: [0, 0.36, 0.538, 0.888, 1.3, 1.42, 1.51, 1.55, 1.52, 1.53]  min 0 · mean 1.14 · max 1.6
reward/lift_src: [0, 0.0888, 0.227, 0.736, 1.14, 1.25, 1.27, 1.31, 1.33, 1.34]  min 0 · mean 0.944 · max 1.42
reward/meet_rcv: [0, -0.00181, -0.00917, -0.00944, -0.0126, -0.0144, -0.0153, -0.0133, -0.0176, -0.0128]  min -0.0312 · mean -0.0115 · max 0
reward/meet_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -1.74e-05 · mean -7.52e-08 · max 0
reward/nested: [0, -0.094, -0.0177, -0.02, -0.0161, -0.0239, -0.0171, -0.0154, -0.0183, -0.0157]  min -0.169 · mean -0.0309 · max 0
reward/palm_rate_rcv: [-0.0625, -0.0489, -0.0416, -0.0385, -0.0356, -0.0341, -0.033, -0.0326, -0.0306, -0.0293]  min -0.0625 · mean -0.0368 · max -0.0279
reward/palm_rate_src: [-0.0613, -0.0459, -0.0419, -0.0375, -0.03, -0.0271, -0.0256, -0.027, -0.0285, -0.0276]  min -0.0614 · mean -0.0335 · max -0.0252
reward/palm_sat_rcv: [-0.0598, -0.0647, -0.0664, -0.0646, -0.0623, -0.061, -0.0597, -0.0583, -0.0579, -0.054]  min -0.0677 · mean -0.0609 · max -0.0532
reward/palm_sat_src: [-0.0586, -0.0686, -0.068, -0.0669, -0.0726, -0.0735, -0.0726, -0.0633, -0.0568, -0.0563]  min -0.0739 · mean -0.0654 · max -0.0501
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0.00111, 0, 0, 0]  min 0 · mean 0.00014 · max 0.00276
reward/pour_pose: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/premature_latch: [0, -0.0389, -0.0151, -0.00598, -0.00537, -0.027, -0.0125, -0.0125, -0.00854, -0.00574]  min -0.142 · mean -0.0195 · max 0
reward/raise_src: [0, 0.000787, 0.0231, 0.267, 0.61, 0.672, 0.72, 0.758, 0.747, 0.76]  min 0 · mean 0.497 · max 0.791
reward/rcv_still: [-0.000116, -0.124, -0.183, -0.275, -0.312, -0.319, -0.305, -0.291, -0.261, -0.26]  min -0.326 · mean -0.246 · max 0
reward/reach_rcv: [0.849, 0.985, 0.989, 0.989, 0.991, 0.991, 0.99, 0.991, 0.991, 0.992]  min 0.676 · mean 0.984 · max 0.993
reward/reach_src: [0.773, 0.991, 0.991, 0.991, 0.992, 0.99, 0.991, 0.991, 0.991, 0.991]  min 0.67 · mean 0.984 · max 0.993
reward/rot_speed: [-2.84e-06, -0.114, -0.0943, -0.0893, -0.0824, -0.0732, -0.0647, -0.0542, -0.0479, -0.0447]  min -0.12 · mean -0.0719 · max -2.84e-06
reward/spill_delta: [-0.00226, -0.00264, -0.00472, -0.00166, -0.00166, -0.00317, -0.00173, -0.00262, -0.00061, -0.0024]  min -0.0088 · mean -0.00226 · max 0
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 3.59e-07, 1.42e-05, 1.74e-06, 2.4e-06, 1.72e-07, 3.95e-12, 7.49e-09, 1.02e-12, 1.6e-06]  min 0 · mean 1.3e-06 · max 4.25e-05
reward/tilt_dir: [0, 9.16e-07, 1.55e-05, 1.74e-06, 2.63e-06, -8.67e-06, 4.51e-12, 6.2e-09, 1.72e-12, 1.28e-05]  min -3.54e-05 · mean 1.42e-06 · max 3.75e-05
reward/total: [2.4, 6.57, 7.88, 9.29, 10.9, 11.3, 11.7, 12.1, 12.2, 12.3]  min 0.688 · mean 10.2 · max 12.7
reward/upright_rcv: [-0.000336, -0.131, -0.18, -0.253, -0.199, -0.178, -0.124, -0.0737, -0.066, -0.0604]  min -0.276 · mean -0.127 · max 0
reward/upright_src: [-2.31e-05, -0.193, -0.0713, -0.0843, -0.0579, -0.0485, -0.043, -0.0448, -0.0342, -0.0403]  min -0.297 · mean -0.0718 · max -2.31e-05
reward/wrap_rcv: [0, 0.258, 0.305, 0.313, 0.324, 0.337, 0.34, 0.359, 0.362, 0.364]  min 0 · mean 0.319 · max 0.375
reward/wrap_src: [0, 0.883, 0.991, 0.984, 0.951, 0.987, 0.989, 1.03, 1.06, 1.06]  min 0 · mean 0.96 · max 1.11
rewards/step: [142, 5.51e+03, 6.51e+03, 8.06e+03, 9.41e+03, 9.82e+03, 1.03e+04, 1.08e+04, 1.11e+04, 1.09e+04]  min 105 · mean 8.86e+03 · max 1.14e+04
task/aim_dist: [0.209, 0.275, 0.302, 0.349, 0.407, 0.416, 0.407, 0.353, 0.308, 0.319]  min 0.126 · mean 0.34 · max 0.498
task/cup_collision_rate: [0.0283, 0.374, 0.129, 0.115, 0.0977, 0.0906, 0.0737, 0.0676, 0.0854, 0.0791]  min 0.0283 · mean 0.141 · max 0.686
task/cups_center_dist: [0.255, 0.276, 0.292, 0.337, 0.4, 0.408, 0.405, 0.353, 0.306, 0.319]  min 0.136 · mean 0.336 · max 0.49
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0.188, 0.0354, 0.04, 0.0322, 0.0479, 0.0342, 0.0308, 0.0366, 0.0315]  min 0 · mean 0.0618 · max 0.338
task/rcv_cup_lift: [0.0042, 0.024, 0.0357, 0.0596, 0.0833, 0.0942, 0.0971, 0.0939, 0.0938, 0.0894]  min 6.04e-05 · mean 0.0728 · max 0.11
task/rcv_grasped: [0, 0.785, 0.827, 0.823, 0.833, 0.826, 0.836, 0.829, 0.822, 0.83]  min 0 · mean 0.799 · max 0.85
task/rcv_hand_foreign_rate: [0.154, 0.246, 0.113, 0.127, 0.127, 0.154, 0.11, 0.103, 0.118, 0.0945]  min 0.0344 · mean 0.141 · max 0.481
task/src_cup_lift: [0.003, 0.0174, 0.0359, 0.113, 0.178, 0.193, 0.195, 0.197, 0.196, 0.197]  min -4.16e-05 · mean 0.143 · max 0.208
task/src_grasped: [0, 0.819, 0.839, 0.838, 0.837, 0.857, 0.856, 0.861, 0.863, 0.863]  min 0 · mean 0.823 · max 0.881
task/src_hand_foreign_rate: [0.159, 0.3, 0.116, 0.129, 0.139, 0.126, 0.106, 0.0977, 0.106, 0.0908]  min 0.0396 · mean 0.139 · max 0.472
task/src_tilt_deg: [16.9, 14.2, 8.01, 8.1, 7.97, 7.53, 7.57, 7.77, 7.87, 7.96]  min 5.91 · mean 9.23 · max 26.5
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Round 11 (iter_10, t2r_i10, 4096 env, 712 epochs / 5.8 h, ADR level 0) — measured facts only. Both cups grasped (src 0.87, rcv 0.84 by epoch 150 — no source avoidance this round) and lifted (src 0.20 m, rcv 0.09 m). episode_success 0.0, bead/in_target 0.0035. task/aim_dist 0.29 m (0.35 at ep215, 0.39 at ep466, 0.29 at ep712): the mouths never came closer than ~0.29 m. task/src_tilt_deg 7.7 deg, task/premature_tilt_rate 0.010, task/tilt_far_deg 7.0 (latch rarely trips because the cup is never tilted). task/cup_collision_rate 0.074, task/nested_rate 0.037, hand_foreign src/rcv 0.087/0.093, bead/spill 0.027 (rising from 0.020 at ep215 without any pour), task/rcv_palm_speed 0.13 m/s, bead/fill_level 0.76, dr/bead_active_hi 12. Reward log at ep712 (per-step means): total 12.44 = lift_rcv 1.55 + grasp_src 1.38 + lift_src 1.37 + grasp_rcv 1.37 + wrap_src 1.08 + reach_rcv 0.99 + reach_src 0.99 + both_grasped 0.82 + both_lifted 0.80 + raise_src 0.77 + approach_rcv 0.69 + approach_src 0.68 + bring_together 0.48 + wrap_rcv 0.37; the pouring-stage terms aim, tilt, tilt_dir, pour_pose, meet_src, success are exactly 0.0 for the whole run and pour_delta 0.0003 — their gates never opened. Largest penalties: rcv_still -0.245, cup_contact -0.071, palm_sat_rcv/src -0.057/-0.051. reach_src/reach_rcv saturated at 0.99 from epoch 0 (first 0.77/0.85), so they add no gradient. The policy collects ~12/step from grasp-lift-hold terms and has no measured incentive to close the last 0.29 m and tilt; bring_together at 0.48 is the only partially-earned approach term. This is the last round (round 12 of 12).
