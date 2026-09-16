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
    LATCH_W = 0.5             # per-step penalty once the env's premature-tilt latch is set (now GRASPED cups only)
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
    # receiver held still (OPERATOR REQUIREMENT b). iter_09: carry weight 0.3 -> 0.5 — with 0.3 the carried
    # receiver averaged ~0.14 m/s (rcv_still -0.19) and its height swung 0.09-0.39 m between checkpoints
    RCV_STILL_FREE = 0.03     # [m/s] receiver cup speed below 3 cm/s is free (a slow 10 cm lift costs nothing)
    RCV_STILL_SCALE = 0.15    # [m/s] tanh scale: 0.10 m/s -> 0.44 of the weight, 0.25 m/s -> 0.9
    RCV_STILL_W = 0.5         # while carried (0.10 m/s -> -0.22)
    RCV_STILL_POUR_W = 0.5    # extra while the source mouth is near or beads are already in: total 1.0
    # post-pour hold ramps in over 0 -> 0.8 of the beads: emptying the cup keeps paying
    POST_POUR_FRAC = 0.8
    # iter_09: bead spawn/settling drops 0.6-6 % of the beads in the first moments of every episode whatever the
    # policy does; the spill penalty ramps in over the first 5 % of the episode so that floor is not charged
    SETTLE_FRAC = 0.05
    # command smoothness (raw actions, before the env EMA)
    PALM_RATE_BASE = 0.02     # per unit of sum-of-squared palm command change, before the grasp (exploration)
    PALM_RATE_HELD = 0.18     # extra once that hand holds its cup (suppresses 4-5 Hz command switching)
    HAND_RATE_W = 0.01        # finger commands: small (grasp closing/opening is legitimate)

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
    # HELD (grasp flag or carried) so a random policy that knocks the source cup over is not taught to avoid the
    # cup — the exact mechanism that killed the previous run through the old ungated latch.
    src_tilt_excess = torch.clamp(ctx.src_cup_tilt - SRC_TILT_FREE, min=0.0)
    src_up_raw = torch.exp(-(src_tilt_excess / SRC_TILT_SIGMA) ** 2)              # (N,) in [0,1]
    src_up = 1.0 - far * (1.0 - src_up_raw)                                        # (N,) in [0,1]
    held_src = torch.maximum(g_src, src_carry)

    # ------------------------------------------------------------------ stage 1: approach
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)

    # ------------------------------------------------------------------ stage 2: grasp
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
    # iter_09: receiver income band — full up to 15 cm, decaying above (income shaping, never negative)
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
    # iter_09: relative height paid on its own (no xy factor) — the previous run parked the receiver at 0.22 m,
    # where a 0.20 m source lift can never reach the hover band and bring_together would stay identically zero.
    # Lowering the receiver (free above its 0.10 m target) or raising the source both climb this ramp.
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

    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    rot_speed = -0.3 * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (ungated income)
    # fractions are per-episode: 20 beads -> 2.5 per bead, 6 beads -> 8.3 per bead. Never gated.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    # iter_09: spill charged only once the beads have settled (first 5 % of the episode ramps 0 -> 1)
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
    # (OPERATOR REQUIREMENT a): source stays upright until its mouth is near the receiver's mouth.
    #   held on the table, far:   -1.0 * tanh(excess/0.2): 20 deg -> -0.63, 30 deg -> -0.92
    #   carried, far:             -2.5 * tanh(excess/0.2): 20 deg -> -1.6,  30 deg -> -2.3, 45 deg -> -2.5
    #   near (mouths <= 5 cm):    0 — the pour is free, and it is the only place any tilt income exists
    # 2.5 exceeds the largest income tilting far could still touch (lift_src 2.0, itself scaled down by src_up).
    upright_src = -(SRC_UP_W_TABLE + SRC_UP_W_CARRY * src_carry) * held_src * far \
        * torch.tanh(src_tilt_excess / SRC_TILT_SCALE)
    # The env latch now fires only on a GRASPED cup, so this can no longer be escaped by staying away from the cup;
    # a constant -0.5/step from the latch moment makes the value drop at the violating step, while the held-cup
    # income (grasp 1.0 + wrap <= 1.5 + lift <= 2.0) still exceeds it, so holding on stays net positive.
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

    # ------------------------------------------------------------------ command smoothness
    da = ctx.actions - ctx.prev_actions                                              # (N,18)
    palm_sq_src = torch.sum(da[:, 0:6] ** 2, dim=-1)
    palm_sq_rcv = torch.sum(da[:, 9:15] ** 2, dim=-1)
    hand_sq = torch.sum(da[:, 6:9] ** 2, dim=-1) + torch.sum(da[:, 15:18] ** 2, dim=-1)
    palm_rate_src = -(PALM_RATE_BASE + PALM_RATE_HELD * g_src) * palm_sq_src
    palm_rate_rcv = -(PALM_RATE_BASE + PALM_RATE_HELD * g_rcv) * palm_sq_rcv
    hand_rate = -HAND_RATE_W * hand_sq

    # ------------------------------------------------------------------ success bonus
    # env success requires no premature tilt, no nesting, receiver <= 20 deg. Scaled by source grip (0.6..1.0) and
    # by the transferred fraction (0.6..1.0; OPERATOR REQUIREMENT c): 50 % of the beads -> 8.0/step, all -> 10/step
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
bead/in_target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.07e-06 · max 0.000141
bead/spill: [0.00158, 0.00742, 0.00676, 0.00728, 0.00851, 0.0097, 0.00997, 0.00886, 0.00945, 0.00983]  min 0.00158 · mean 0.00873 · max 0.0119
done/drop: [0, 0.000244, 0.000244, 0.000244, 0.000488, 0.00122, 0.000244, 0.000244, 0, 0]  min 0 · mean 0.000544 · max 0.0146
episode_lengths/step: [124, 704, 741, 698, 712, 698, 804, 853, 850, 861]  min 103 · mean 743 · max 887
reward/aim: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/approach_rcv: [0.521, 0.623, 0.647, 0.662, 0.665, 0.67, 0.674, 0.678, 0.684, 0.685]  min 0.478 · mean 0.66 · max 0.691
reward/approach_src: [0.503, 0.433, 0.425, 0.433, 0.459, 0.508, 0.562, 0.576, 0.578, 0.585]  min 0.419 · mean 0.511 · max 0.634
reward/both_grasped: [0, 0.000244, 0, 0.000488, 0.0132, 0.113, 0.47, 0.615, 0.641, 0.693]  min 0 · mean 0.3 · max 0.799
reward/both_lifted: [0, 0, 0, 0, 0, 0, 0.000244, 0.000244, 0.0151, 0.0625]  min 0 · mean 0.0274 · max 0.458
reward/bring_together: [0, 0, 0, 0, 5.11e-05, 2.5e-05, 0.00025, 0, 0.000294, 0.000628]  min 0 · mean 0.000292 · max 0.00377
reward/closing_speed: [-0.00015, -0.00175, -0.000941, -3.71e-06, -0.00034, -0.00016, -0.000114, -9.39e-05, 0, -0.000184]  min -0.00262 · mean -0.00049 · max 0
reward/cup_contact: [-0.0261, -0.000857, -0.000811, -0.000446, -0.000981, -0.000171, 0, -0.000286, 0, 0]  min -0.0974 · mean -0.00211 · max 0
reward/drop: [0, -0.00647, -0.00232, -0.00281, -0.00342, -0.00488, -0.00439, -0.00208, -0.000977, -0.00122]  min -0.0315 · mean -0.00369 · max 0
reward/grasp_rcv: [0.00971, 0.787, 0.834, 0.789, 0.829, 0.857, 0.882, 0.874, 0.908, 0.896]  min 0.00269 · mean 0.826 · max 0.94
reward/grasp_src: [0.009, 0.00708, 0.0107, 0.0103, 0.035, 0.168, 0.585, 0.771, 0.778, 0.842]  min 0.00309 · mean 0.375 · max 0.962
reward/hand_foreign_rcv: [-0.0879, -0.0666, -0.0937, -0.0818, -0.101, -0.146, -0.12, -0.091, -0.0895, -0.0758]  min -0.158 · mean -0.0941 · max -0.01
reward/hand_foreign_src: [-0.0878, -0.0157, -0.0125, -0.0267, -0.0243, -0.024, -0.0354, -0.0262, -0.025, -0.0275]  min -0.0878 · mean -0.0238 · max -0.00948
reward/hand_rate: [-0.0623, -0.055, -0.0481, -0.0454, -0.0432, -0.0393, -0.0394, -0.0405, -0.0405, -0.0399]  min -0.0623 · mean -0.0442 · max -0.0374
reward/hold_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 6.28e-08 · max 7.75e-06
reward/hold_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.84e-07 · max 3.86e-05
reward/lift_rcv: [0, 0.176, 0.308, 0.466, 0.761, 0.945, 0.836, 1.01, 1.19, 1.24]  min 0 · mean 0.779 · max 1.35
reward/lift_src: [0, 7.91e-06, 0, 1.95e-06, 0.000324, 0.00191, 0.00704, 0.00982, 0.0209, 0.0471]  min 0 · mean 0.0212 · max 0.323
reward/meet_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -6.64e-05 · mean -1.21e-07 · max 0
reward/meet_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -6.45e-05 · mean -2.98e-07 · max 0
reward/nested: [0, -0.00061, -0.00061, 0, -0.000488, 0, 0, -0.000122, -0.000122, 0]  min -0.022 · mean -0.000587 · max 0
reward/palm_rate_rcv: [-0.125, -0.53, -0.398, -0.29, -0.263, -0.212, -0.173, -0.146, -0.138, -0.124]  min -0.654 · mean -0.252 · max -0.107
reward/palm_rate_src: [-0.122, -0.0952, -0.0766, -0.0712, -0.0761, -0.109, -0.224, -0.226, -0.207, -0.204]  min -0.315 · mean -0.151 · max -0.0693
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.86e-06 · max 0.00102
reward/pour_pose: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/premature_latch: [0, -0.000122, -0.000122, -0.000244, -0.000854, -0.00159, -0.00391, -0.00366, -0.00342, -0.00696]  min -0.00781 · mean -0.00209 · max 0
reward/raise_src: [0, 0, 0, 0, 0, 0, 0, 1.29e-06, 0.000346, 0.00303]  min 0 · mean 0.00172 · max 0.0627
reward/rcv_still: [-0.000279, -0.0407, -0.0932, -0.147, -0.213, -0.256, -0.203, -0.23, -0.258, -0.259]  min -0.276 · mean -0.187 · max -0.000207
reward/rot_speed: [-0.134, -0.00304, -0.00227, -0.00254, -0.00784, -0.0179, -0.0303, -0.0277, -0.0311, -0.0371]  min -0.134 · mean -0.0208 · max -0.00151
reward/spill_delta: [-0.00183, -0.00311, -0.000916, -0.000916, -0.00173, 0, -0.000916, -0.00122, -0.000916, 0]  min -0.00639 · mean -0.00156 · max 0
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt_dir: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/total: [0.393, 1.31, 1.59, 1.78, 2.13, 2.52, 3.54, 4.28, 4.56, 4.89]  min 0.0214 · mean 3.01 · max 6.04
reward/upright_rcv: [-0.000494, -0.0623, -0.101, -0.111, -0.106, -0.186, -0.138, -0.0798, -0.109, -0.104]  min -0.2 · mean -0.104 · max -0.000376
reward/upright_src: [0, -0.000978, 0, 0, -0.0025, -0.0108, -0.0203, -0.0222, -0.0323, -0.0325]  min -0.0984 · mean -0.0162 · max 0
reward/wrap_rcv: [0, 0.221, 0.248, 0.248, 0.255, 0.253, 0.262, 0.264, 0.275, 0.274]  min 0 · mean 0.249 · max 0.3
reward/wrap_src: [0, 0.000298, 0, 0.000241, 0.0106, 0.0731, 0.301, 0.421, 0.459, 0.523]  min 0 · mean 0.216 · max 0.703
rewards/step: [20.7, 901, 1.15e+03, 1.22e+03, 1.46e+03, 1.76e+03, 2.78e+03, 3.65e+03, 3.9e+03, 4.16e+03]  min 13.9 · mean 2.36e+03 · max 5.3e+03
task/aim_dist: [0.211, 1.6, 0.717, 1.02, 0.572, 0.853, 0.56, 0.419, 0.425, 0.443]  min 0.211 · mean 0.612 · max 1.85
task/cup_collision_rate: [0.0308, 0.00122, 0.000732, 0.000488, 0.000977, 0.000244, 0, 0.000488, 0, 0]  min 0 · mean 0.00238 · max 0.115
task/cups_center_dist: [0.256, 1.6, 0.713, 1.01, 0.57, 0.846, 0.556, 0.417, 0.423, 0.441]  min 0.247 · mean 0.609 · max 1.84
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0.00122, 0.00122, 0, 0.000977, 0, 0, 0.000244, 0.000244, 0]  min 0 · mean 0.00118 · max 0.0439
task/rcv_cup_lift: [0.0043, 0.438, 0.0955, 0.222, 0.14, 0.163, 0.113, 0.0747, 0.0968, 0.101]  min -0.000593 · mean 0.127 · max 0.438
task/rcv_grasped: [0, 0.691, 0.72, 0.679, 0.718, 0.738, 0.764, 0.758, 0.786, 0.777]  min 0 · mean 0.715 · max 0.819
task/rcv_hand_foreign_rate: [0.168, 0.126, 0.165, 0.156, 0.189, 0.263, 0.223, 0.186, 0.204, 0.184]  min 0.0203 · mean 0.185 · max 0.275
task/src_cup_lift: [0.00289, 0.000194, 8.73e-05, 0.000112, 0.00042, 0.00154, 0.00303, 0.00282, 0.00505, 0.0101]  min 4.82e-05 · mean 0.00509 · max 0.0611
task/src_grasped: [0, 0.000488, 0, 0.000488, 0.0151, 0.124, 0.515, 0.693, 0.697, 0.76]  min 0 · mean 0.33 · max 0.867
task/src_hand_foreign_rate: [0.156, 0.0383, 0.0354, 0.0647, 0.0601, 0.0579, 0.0886, 0.0801, 0.0754, 0.074]  min 0.0271 · mean 0.0625 · max 0.156
task/src_tilt_deg: [16.4, 0.25, 0.218, 0.258, 0.871, 3.22, 5.8, 6.4, 6.37, 6.08]  min 0.149 · mean 3.57 · max 16.4
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Observations from the loop operator about the policy trained with the previous function. Facts only, apart from the explicitly marked operator requirements. The environment is unchanged from the previous round (premature-tilt latch only for a grasped source cup, beads 30 mm and 6–20 per episode, `ctx.bead_fill_level`, receiver ≤ 20° for success).

1. This round ran 546 epochs (4.4 h) and was advanced by the loop rule (success 0). Timeline of the source grasp: 0.000 at epoch 44, 0.002 at epoch 177, 0.20 at epoch 292, 0.86 at epoch 546. The receiver grasped from the start (0.65 at epoch 44, 0.81 at epoch 546). At epoch 546: source cup lift 0.056 m, receiver cup lift 0.098 m, source tilt 7.6°, mouth distance 0.44 m, beads in receiver 0, spill 1.1 %, premature-tilt latch 0.4 %, receiver hand–foreign contact 19 %, source hand–foreign 3 %. Comparison at the same epoch with the two earlier converged runs (this run / round-8 run at epoch 558 / round-6 run at epoch 600): source lift 0.056 / 0.333 / 0.274 m, receiver lift 0.098 / 0.195 / 0.137 m, source tilt 7.6° / 70° / 78°, reward/total 6.0 / 10.3 / 20.8.

2. What the trajectory showed while the source grasp was still 0 (checkpoint at epoch 150, 8 environments, randomisation at zero): the source palm commands were saturated at the outer limit of the palm workspace (mean lateral command −0.99), the source palm moved from 0.165 m to 0.25 m away from its cup, i.e. beyond the 0.22 m distance at which the hand is allowed to close, so the source hand never closed (closure 0.08, contact force 0). The source cup stayed on the table untouched (moved 3 mm, upright). The receiver hand grasped its cup (0.70) and lifted it 6 cm, but the receiver arm was driven hard: palm target–actual error 0.10 m, receiver lateral command +0.97, rotation command +0.95, cup–cup contact peaks of 144 N (round-8 run at the same stage: 42 N), receiver hand–foreign contact 16 %. Neither `premature_latch` nor `upright_src` paid anything during this phase (both 0.000, because both are gated on the source grasp). This outward saturation is the same signature the round-8 policy showed between epochs 200 and 558.

3. Reward components at epoch 546 (mean per step): lift_rcv +1.34, grasp_src +0.96, grasp_rcv +0.93, both_grasped +0.79, approach_rcv +0.69, wrap_src +0.68, approach_src +0.63, both_lifted +0.43, lift_src +0.29, wrap_rcv +0.30; penalties palm_rate_src −0.31, rcv_still −0.27, palm_rate_rcv −0.11, upright_rcv −0.11. `approach_src` paid +0.44 while the source palm hovered 0.21–0.25 m from the cup and +0.63 once grasped.

4. OPERATOR REQUIREMENTS (unchanged): (a) the source cup stays upright while grasped, lifted and carried, and tilts only once its mouth is close to the receiver's mouth; (b) the receiver cup is lifted and held at a steady position during the pour, less movement is better (`task/rcv_palm_speed`); (c) all beads are to be transferred whatever the fill level.

5. Unchanged facts: with contact freeze the measured `hand_closure` saturates at about 0.35–0.43 on these cups; hand–hand and cup–cup contact remain safety hazards on the real robot; spill onset with the 30 mm beads is about 70° for a full cup and about 90° for a few beads; the policy is trained from scratch.

The policy must complete the full task (wrap-grasp both cups → lift both, source upright → bring the mouths together → tilt the source cup only then → beads in the receiver, receiver held steady and nearly upright, no drop, little spill) and hold both grasps stably after pouring.
