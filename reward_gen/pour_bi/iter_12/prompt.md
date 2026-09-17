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
    LIFT_TARGET_SRC = 0.25    # [m] source lift keeps its gradient up to 25 cm (hover needs h_src ~ h_rcv + 0.11..0.16)
    LIFT_TARGET_RCV = 0.10    # [m] receiver lifted moderately (operator requirement)
    RCV_LIFT_FREE = 0.15      # [m] receiver lift income is full up to 15 cm ...
    RCV_LIFT_SIGMA = 0.08     # [m] ... then decays (20 cm -> 0.68, 25 cm -> 0.21)
    LIFT_GATE = 0.05          # [m] minimum lift before the pour stack is rewarded
    LIP_DEADBAND = 0.01       # [m] lip within 1 cm of the receiver mouth centre counts as centred (pose noise)
    DZ_LOW = -0.06            # [m] "above" ramp start: lip 6 cm below the receiver rim -> 0
    DZ_HIGH = 0.02            # [m] "above" ramp end: lip 2 cm above the rim -> 1
    DZ_FREE = 0.04            # [m] pouring from up to 4 cm above the rim costs nothing
    DZ_SIGMA = 0.06           # [m] beyond that, gentle decay
    # upright-carry hover band (source mouth above the receiver rim)
    HOVER_LO = 0.06           # [m] mouth 6 cm above the receiver rim -> 0
    HOVER_HI = 0.11           # [m] mouth 11 cm above the rim -> 1
    HOVER_FREE = 0.16         # [m] free up to 16 cm ...
    HOVER_SIGMA = 0.06        # [m] ... then Gaussian decay
    RAISE_LO = -0.04          # [m] relative-height ramp start (mouth 4 cm below the rim)
    RAISE_W = 1.0
    BRING_W = 1.5
    # iter_11: dedicated xy convergence ramp (mouth-to-mouth). Round 11 parked at 0.29 m: bring_together's
    # exp(-3d) paid 0.42 there vs 0.86 at 5 cm (0.65/step for the whole gap on a 12/step income). This ramp
    # pays 0.72 at 0.29 m, 2.0 at 0.17 m, 2.9 at 0.10 m and 3.0 at 6 cm: ~2.3/step for closing the gap.
    CONV_FAR = 0.40           # [m]
    CONV_NEAR = 0.06          # [m]
    CONV_W = 3.0
    # near gate on MOUTH-TO-MOUTH xy distance (env premature-tilt latch is at 0.10 m)
    NEAR_FULL = 0.05          # [m] mouths within 5 cm (xy): tilt income fully open
    NEAR_ZERO = 0.09          # [m] beyond 9 cm: closed
    # source upright while far (OPERATOR REQUIREMENT a)
    SRC_TILT_FREE = 0.20      # [rad] 11.5 deg of source tilt is free (carry wobble, pose noise)
    SRC_TILT_SCALE = 0.20     # [rad] tanh scale of the penalty
    SRC_TILT_SIGMA = 0.25     # [rad] Gaussian income factor
    SRC_UP_W_TABLE = 1.0      # penalty weight while grasped on the table
    SRC_UP_W_CARRY = 1.5      # extra once carried: total 2.5 exceeds any far income tilting could touch
    LATCH_W = 0.5             # per-step penalty once the env's premature_tilt has latched
    # fill-adaptive release angle (measured: first bead ~72 deg full, ~90 deg with a few beads)
    RELEASE_FULL = 1.25       # [rad] release tilt for a full cup
    RELEASE_SPAN = 0.60       # [rad] + (1 - fill) * span
    BAND_BELOW = 0.10         # [rad] pour_pose band starts 6 deg below release ...
    BAND_ABOVE = 0.45         # [rad] ... and saturates 26 deg above it
    TILT_TARGET_ABOVE = 0.55  # [rad] linear tilt term saturates 32 deg past release
    TILT_DIR_SCALE = 1.0      # [rad] direction term fully weighted at 57 deg
    ROT_SPEED_FREE = 1.0      # [rad/s] controlled pouring rotation is free below this
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped
    RCV_TOPPLED = 1.2         # [rad] receiver lying on its side
    # iter_11: closing-speed guard moved inside 15 cm (was 30 cm: the policy parked exactly on that boundary) and
    # the free closing speed raised to 0.15 m/s; cup_contact still charges any actual collision
    NEAR_DIST = 0.15          # [m] cup-centre distance inside which fast closing is charged
    CLOSE_V_FREE = 0.15       # [m/s]
    CLOSE_V_SCALE = 0.20      # [m/s]
    FORCE_SCALE = 5.0         # [N] tanh scale for all contact penalties
    FOREIGN_DEADBAND = 1.0    # [N] incidental brushes are free
    CLEAN_DEADBAND = 0.5      # [N] deadband of the multiplicative clean gate
    FOREIGN_W_FREE = 0.3      # hand-foreign weight while the hand does not hold its cup
    FOREIGN_W_HELD = 1.0      # ... and while it holds / carries it
    GRIP_FLOOR = 0.3          # pour/lift income still worth 30 % with a poor wrap
    # iter_11: meet bands widened (receiver 6 cm, source 10 cm across the midline) and lighter, so the hover xy
    # target is reachable without paying the crossing tax; weights 0.5 / 0.3 (were 1.0 / 0.5)
    MEET_RCV_FREE = 0.06      # [m]
    MEET_SRC_FREE = 0.10      # [m]
    MEET_SCALE = 0.10         # [m] tanh scale of the crossing excess
    MEET_RCV_W = 0.5
    MEET_SRC_W = 0.3
    # receiver-upright shape (success requires rcv tilt <= 20 deg)
    RCV_TILT_FREE = 0.12      # [rad] ~7 deg of receiver tilt is free
    RCV_TILT_SIGMA = 0.20     # [rad] gate: 15 deg -> 0.61, 20 deg -> 0.37
    RCV_PEN_SCALE = 0.25      # [rad] additive penalty scale
    RCV_GATE_FLOOR = 0.3      # positioning terms keep 30 % income under a tipped receiver
    # carry phases
    CARRY_GRASP_LO = 0.015    # [m]
    CARRY_GRASP_HI = 0.05     # [m]
    CARRY_FREE_LO = 0.06      # [m]
    CARRY_FREE_HI = 0.10      # [m]
    RCV_UP_W = 1.0            # carried receiver upright penalty
    RCV_UP_POUR_W = 1.0       # extra during pour / post-pour
    POUR_PHASE_TILT_LO = 0.8  # [rad]
    POUR_PHASE_TILT_HI = 1.6  # [rad]
    # receiver held still (OPERATOR REQUIREMENT b). iter_11: the carry-far weight is halved (0.5 -> 0.25) because
    # -0.245/step was the largest penalty and it paid the left arm to freeze 0.29 m from the source; during the
    # pour / post-pour the total stays 1.0 (0.25 + 0.75)
    RCV_STILL_FREE = 0.03     # [m/s]
    RCV_STILL_SCALE = 0.15    # [m/s] tanh scale
    RCV_STILL_W = 0.25        # while carried far (0.10 m/s -> -0.11)
    RCV_STILL_POUR_W = 0.75   # extra while the source mouth is near or beads are in (0.10 m/s -> -0.44)
    POST_POUR_FRAC = 0.8      # post-pour hold ramps in over 0 -> 0.8 of the beads
    SETTLE_FRAC = 0.05        # spill penalty ramps in over the first 5 % of the episode (bead settling)
    # reach ramp across the 0.22 m closing radius: 0 at 26 cm, 1 at 14 cm. Saturated at 0.99 from epoch 0 in
    # round 11, so iter_11 halves it (anchor only, less standing income)
    REACH_FAR = 0.26          # [m]
    REACH_NEAR = 0.14         # [m]
    REACH_W = 0.5
    CLOSE_W = 1.0             # closing at the cup: 1.0 x exp(-6d) x normalised closure
    CLOSE_NORM = 0.35
    PALM_RATE_W = 0.01        # per unit of sum-of-squared palm command change
    HAND_RATE_W = 0.005       # finger commands
    SAT_LO = 0.7              # palm commands pinned beyond |0.7| are taxed ...
    SAT_W = 0.15              # ... up to -0.15/step per arm

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
    mouth_xy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)   # (N,)
    near = 1.0 - _smoothstep((mouth_xy - NEAR_FULL) / (NEAR_ZERO - NEAR_FULL))                     # (N,) in [0,1]
    far = 1.0 - near

    # ------------------------------------------------------------------ source upright while far
    src_tilt_excess = torch.clamp(ctx.src_cup_tilt - SRC_TILT_FREE, min=0.0)
    src_up_raw = torch.exp(-(src_tilt_excess / SRC_TILT_SIGMA) ** 2)              # (N,) in [0,1]
    src_up = 1.0 - far * (1.0 - src_up_raw)                                        # (N,) in [0,1]

    # ------------------------------------------------------------------ stage 1: approach + reach
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)
    reach_src = REACH_W * _smoothstep((REACH_FAR - d_src) / (REACH_FAR - REACH_NEAR))
    reach_rcv = REACH_W * _smoothstep((REACH_FAR - d_rcv) / (REACH_FAR - REACH_NEAR))

    # ------------------------------------------------------------------ stage 2: grasp
    prox_src = torch.exp(-6.0 * d_src)
    prox_rcv = torch.exp(-6.0 * d_rcv)
    clos_src = torch.clamp(ctx.src_hand_closure / CLOSE_NORM, 0.0, 1.0)
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
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
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

    # ------------------------------------------------------------------ stage 4: raise, converge, bring together
    dz_mouth = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    hover_decay = torch.exp(-(torch.clamp(dz_mouth - HOVER_FREE, min=0.0) / HOVER_SIGMA) ** 2)
    hover = _smoothstep((dz_mouth - HOVER_LO) / (HOVER_HI - HOVER_LO)) * hover_decay
    hover_wide = _smoothstep((dz_mouth - RAISE_LO) / (HOVER_HI - RAISE_LO)) * hover_decay
    raise_src = RAISE_W * both_lifted_f * not_nested_f * src_up * rcv_up_soft * hover_wide

    # iter_11: the missing gradient — xy convergence of the mouths from 40 cm down to 6 cm, weight 3.0, gated on
    # both cups lifted, no nesting, source upright (far), receiver upright-ish and the source mouth above the
    # receiver rim (hover_wide) so the only way to earn it is the correct carry geometry.
    conv_xy = _smoothstep((CONV_FAR - mouth_xy) / (CONV_FAR - CONV_NEAR))          # (N,) 0 at 40 cm, 1 at 6 cm
    converge = CONV_W * both_lifted_f * not_nested_f * clear * src_up * rcv_up_soft * hover_wide * conv_xy

    # iter_11: xy factor sharpened exp(-3d) -> exp(-5d): 0.29 m -> 0.23, 0.10 m -> 0.61, 0.05 m -> 0.78
    bt_z = torch.maximum(hover, near * aim_z)
    bring_together = BRING_W * src_lifted_f * not_nested_f * clean * src_up * torch.exp(-5.0 * mouth_xy) \
        * bt_z * rcv_up_soft
    stack_gate = both_lifted_f * not_nested_f * clear * clean
    aim = 3.0 * stack_gate * grip * near * aim_soft * rcv_up_soft

    # ------------------------------------------------------------------ stage 5: tilt / pour pose (all x near)
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_rad = ctx.src_cup_tilt
    fill = torch.clamp(ctx.bead_fill_level, 0.0, 1.0)
    release_tilt = RELEASE_FULL + RELEASE_SPAN * (1.0 - fill)                        # (N,)
    tilt_target = release_tilt + TILT_TARGET_ABOVE                                   # (N,)
    tilt_frac = torch.clamp(tilt_rad / tilt_target, 0.0, 1.0)
    # weight 4.0 x near: worth nothing until the mouths are within 5 cm; aim_z gives the lip-height gradient
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
    rot_speed = -0.3 * held_src * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (ungated income)
    # increments, never the level: 20 beads -> 2.5 per bead, 6 beads -> 8.3 per bead
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    spill_delta = -30.0 * settled * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ stage 7: hold after the pour
    post_pour = _smoothstep(ctx.bead_in_target_frac / POST_POUR_FRAC)
    hold_src = 1.5 * post_pour * g_src * wq_src
    hold_rcv = 0.5 * post_pour * g_rcv * wq_rcv * rcv_up

    # ------------------------------------------------------------------ workspace: meet near the midline
    mid_y = 0.5 * (ctx.src_cup_spawn_pos[:, 1] + ctx.rcv_cup_spawn_pos[:, 1])
    rcv_cross = mid_y - ctx.rcv_cup_pos[:, 1]
    src_cross = ctx.src_cup_pos[:, 1] - mid_y
    meet_rcv = -MEET_RCV_W * both_lifted_f * torch.tanh(torch.clamp(rcv_cross - MEET_RCV_FREE, min=0.0) / MEET_SCALE)
    meet_src = -MEET_SRC_W * both_lifted_f * torch.tanh(torch.clamp(src_cross - MEET_SRC_FREE, min=0.0) / MEET_SCALE)

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
    # (OPERATOR REQUIREMENT a): source stays upright until its mouth is near the receiver's mouth.
    upright_src = -(SRC_UP_W_TABLE + SRC_UP_W_CARRY * src_carry) * held_src * far \
        * torch.tanh(src_tilt_excess / SRC_TILT_SCALE)
    premature_latch = -LATCH_W * ctx.premature_tilt.to(dt)

    pour_phase = torch.maximum(
        both_lifted_f * near * aim_z
        * _smoothstep((tilt_rad - POUR_PHASE_TILT_LO) / (POUR_PHASE_TILT_HI - POUR_PHASE_TILT_LO)),
        post_pour,
    )
    upright_rcv = -(RCV_UP_W + RCV_UP_POUR_W * pour_phase) * rcv_carry * torch.tanh(rcv_tilt_excess / RCV_PEN_SCALE)

    # (OPERATOR REQUIREMENT b): the carried receiver is held still — light while carrying far, full near / post-pour
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
    a_abs = torch.abs(ctx.actions)                                                   # (N,18)
    sat_src = torch.mean(_smoothstep((a_abs[:, 0:6] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)    # (N,) in [0,1]
    sat_rcv = torch.mean(_smoothstep((a_abs[:, 9:15] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)   # (N,) in [0,1]
    palm_sat_src = -SAT_W * sat_src
    palm_sat_rcv = -SAT_W * sat_rcv

    # ------------------------------------------------------------------ success bonus
    # (OPERATOR REQUIREMENT c): 50 % of the beads -> 8.0/step, all -> 10/step, scaled by source grip
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
        "converge": converge,
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
bead/in_target: [0, 0.00078, 0.000297, 0.000419, 0.000376, 0.000163, 0.000183, 0.000128, 2.22e-05, 0.000101]  min 0 · mean 0.000213 · max 0.000915
bead/spill: [0.00136, 0.0157, 0.023, 0.0201, 0.0175, 0.0189, 0.0171, 0.0163, 0.019, 0.0175]  min 0.00136 · mean 0.0169 · max 0.0257
done/drop: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.00022 · max 0.01
episode_lengths/step: [124, 856, 873, 854, 855, 866, 845, 870, 872, 877]  min 98.8 · mean 829 · max 888
reward/aim: [0, 4.7e-05, 5.59e-05, 1.36e-05, 7.45e-05, 3.96e-05, 3.61e-07, 4.57e-05, 8.51e-05, 2.46e-05]  min 0 · mean 5.51e-05 · max 0.000543
reward/approach_rcv: [0.521, 0.642, 0.654, 0.651, 0.667, 0.682, 0.685, 0.688, 0.688, 0.686]  min 0.475 · mean 0.667 · max 0.69
reward/approach_src: [0.503, 0.655, 0.666, 0.662, 0.673, 0.679, 0.677, 0.68, 0.681, 0.681]  min 0.478 · mean 0.667 · max 0.684
reward/both_grasped: [0, 0.714, 0.765, 0.789, 0.792, 0.81, 0.818, 0.826, 0.831, 0.838]  min 0 · mean 0.763 · max 0.846
reward/both_lifted: [0, 0.0969, 0.465, 0.73, 0.774, 0.804, 0.814, 0.823, 0.828, 0.836]  min 0 · mean 0.661 · max 0.844
reward/bring_together: [0, 0.00338, 0.00192, 0.0119, 0.12, 0.219, 0.316, 0.436, 0.485, 0.529]  min 0 · mean 0.242 · max 0.583
reward/closing_speed: [0, -0.0081, -0.00496, -0.00261, -0.00362, -0.00219, -0.00548, -0.002, -0.00266, -0.00382]  min -0.0139 · mean -0.00416 · max 0
reward/converge: [0, 0.000878, 0.00901, 0.0317, 0.152, 0.508, 1.19, 1.58, 1.78, 1.95]  min 0 · mean 0.825 · max 2.06
reward/cup_contact: [-0.0271, -0.369, -0.158, -0.0912, -0.0741, -0.0675, -0.0544, -0.0379, -0.0447, -0.0245]  min -0.62 · mean -0.117 · max -0.0161
reward/drop: [0, -0.00366, -0.000732, -0.000854, -0.00061, -0.000122, -0.000366, -0.000244, -0.000122, 0]  min -0.0287 · mean -0.00193 · max 0
reward/grasp_rcv: [0.0551, 1.26, 1.28, 1.3, 1.32, 1.35, 1.36, 1.36, 1.38, 1.38]  min 0.017 · mean 1.29 · max 1.39
reward/grasp_src: [0.0518, 1.27, 1.33, 1.34, 1.36, 1.38, 1.37, 1.38, 1.38, 1.39]  min 0.0169 · mean 1.31 · max 1.4
reward/hand_foreign_rcv: [-0.0263, -0.128, -0.0668, -0.0357, -0.0355, -0.0297, -0.0332, -0.0257, -0.0245, -0.0203]  min -0.213 · mean -0.0498 · max -0.0032
reward/hand_foreign_src: [-0.0249, -0.137, -0.098, -0.0686, -0.0557, -0.0534, -0.0504, -0.0386, -0.0414, -0.0361]  min -0.217 · mean -0.0675 · max -0.00641
reward/hand_rate: [-0.0307, -0.026, -0.0212, -0.0217, -0.021, -0.021, -0.0211, -0.0209, -0.0209, -0.0214]  min -0.0309 · mean -0.0221 · max -0.02
reward/hold_rcv: [0, 6.8e-05, 2.4e-05, 3.76e-05, 3.74e-05, 1.31e-05, 9.43e-06, 7.21e-06, 2.18e-06, 1.46e-05]  min 0 · mean 1.97e-05 · max 7.97e-05
reward/hold_src: [0, 0.000302, 6.34e-05, 0.000164, 0.00015, 6.5e-05, 2.34e-05, 1.78e-05, 8.9e-06, 4.91e-05]  min 0 · mean 7.44e-05 · max 0.000334
reward/lift_rcv: [0, 0.722, 1.22, 1.37, 1.46, 1.54, 1.58, 1.6, 1.62, 1.66]  min 0 · mean 1.34 · max 1.67
reward/lift_src: [0, 0.108, 0.362, 0.799, 1.22, 1.35, 1.33, 1.45, 1.48, 1.47]  min 0 · mean 1.05 · max 1.52
reward/meet_rcv: [0, -0.000296, -0.00016, -0.000806, -0.00253, -0.00535, -0.0241, -0.0415, -0.077, -0.108]  min -0.135 · mean -0.0315 · max 0
reward/meet_src: [0, 0, 0, 0, 0, 0, -8.36e-05, 0, 0, 0]  min -8.36e-05 · mean -2.54e-07 · max 0
reward/nested: [0, -0.0826, -0.0394, -0.0206, -0.0133, -0.0125, -0.0155, -0.00708, -0.00769, -0.00439]  min -0.133 · mean -0.0262 · max 0
reward/palm_rate_rcv: [-0.0615, -0.0506, -0.046, -0.0433, -0.0416, -0.0416, -0.0384, -0.0374, -0.0373, -0.0365]  min -0.0619 · mean -0.042 · max -0.0349
reward/palm_rate_src: [-0.0618, -0.0524, -0.0487, -0.0454, -0.0433, -0.041, -0.0423, -0.0418, -0.0426, -0.0412]  min -0.0621 · mean -0.0451 · max -0.039
reward/palm_sat_rcv: [-0.0592, -0.0631, -0.0605, -0.0609, -0.0594, -0.0574, -0.06, -0.0629, -0.0629, -0.0616]  min -0.0647 · mean -0.0612 · max -0.0565
reward/palm_sat_src: [-0.0596, -0.0603, -0.0606, -0.0619, -0.0616, -0.0608, -0.0571, -0.0591, -0.0569, -0.0578]  min -0.0639 · mean -0.0597 · max -0.0563
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.73e-05 · max 0.00244
reward/pour_pose: [0, 6.39e-17, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.44e-10 · max 1.92e-07
reward/premature_latch: [0, -0.0765, -0.109, -0.016, -0.0116, -0.0387, -0.0291, -0.0529, -0.0219, -0.0569]  min -0.13 · mean -0.0447 · max 0
reward/raise_src: [0, 0.00558, 0.0196, 0.241, 0.658, 0.74, 0.742, 0.781, 0.791, 0.808]  min 0 · mean 0.521 · max 0.823
reward/rcv_still: [0, -0.148, -0.173, -0.165, -0.168, -0.172, -0.163, -0.144, -0.14, -0.136]  min -0.183 · mean -0.148 · max 0
reward/reach_rcv: [0.424, 0.489, 0.492, 0.491, 0.492, 0.494, 0.494, 0.495, 0.496, 0.495]  min 0.331 · mean 0.49 · max 0.496
reward/reach_src: [0.387, 0.495, 0.495, 0.494, 0.494, 0.495, 0.495, 0.494, 0.496, 0.496]  min 0.337 · mean 0.491 · max 0.496
reward/rot_speed: [-8.37e-05, -0.125, -0.129, -0.1, -0.0765, -0.0666, -0.0523, -0.0423, -0.0369, -0.0351]  min -0.134 · mean -0.0707 · max -8.37e-05
reward/spill_delta: [-0.000916, -0.003, -0.00372, -0.000666, -0.00208, -0.00177, -0.00061, -0.000732, -0.000732, -0.000666]  min -0.00854 · mean -0.00152 · max 0
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 8.03e-06, 8.83e-06, 1.51e-06, 1.77e-05, 1.04e-05, 9.77e-08, 1.02e-05, 1.27e-05, 3.54e-06]  min 0 · mean 9.34e-06 · max 8.3e-05
reward/tilt_dir: [0, 8.85e-05, 4.27e-06, 8.84e-07, 4.09e-05, 7.23e-06, 1.51e-08, 7.96e-06, 1.15e-05, 2.5e-06]  min -3.93e-05 · mean 7.99e-06 · max 0.000117
reward/total: [1.58, 5.66, 7.45, 9.08, 10.5, 11.4, 12.3, 13.2, 13.6, 13.9]  min 0.443 · mean 10.5 · max 14.2
reward/upright_rcv: [0, -0.212, -0.175, -0.109, -0.0861, -0.0789, -0.0833, -0.072, -0.0528, -0.0292]  min -0.223 · mean -0.0926 · max 0
reward/upright_src: [-0.00109, -0.21, -0.231, -0.113, -0.149, -0.12, -0.0921, -0.0552, -0.0373, -0.0273]  min -0.345 · mean -0.113 · max -0.000192
reward/wrap_rcv: [0, 0.249, 0.267, 0.276, 0.3, 0.331, 0.347, 0.356, 0.363, 0.363]  min 0 · mean 0.308 · max 0.371
reward/wrap_src: [0, 0.788, 0.931, 0.942, 0.98, 1.01, 1.02, 1.07, 1.09, 1.08]  min 0 · mean 0.96 · max 1.11
rewards/step: [95.2, 4.74e+03, 6.47e+03, 8.02e+03, 8.71e+03, 9.61e+03, 1.07e+04, 1.12e+04, 1.17e+04, 1.19e+04]  min 71.4 · mean 8.87e+03 · max 1.25e+04
task/aim_dist: [0.212, 0.261, 0.336, 0.387, 0.364, 0.314, 0.244, 0.476, 0.206, 0.198]  min 0.138 · mean 0.278 · max 0.476
task/cup_collision_rate: [0.0325, 0.38, 0.165, 0.0979, 0.0811, 0.0732, 0.0591, 0.043, 0.0486, 0.0278]  min 0.0183 · mean 0.124 · max 0.643
task/cups_center_dist: [0.257, 0.271, 0.333, 0.383, 0.354, 0.301, 0.229, 0.462, 0.196, 0.191]  min 0.15 · mean 0.272 · max 0.462
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0.165, 0.0789, 0.0413, 0.0266, 0.0249, 0.031, 0.0142, 0.0154, 0.00879]  min 0 · mean 0.0525 · max 0.267
task/rcv_cup_lift: [0.00422, 0.049, 0.109, 0.116, 0.109, 0.118, 0.114, 0.121, 0.12, 0.12]  min -7.12e-05 · mean 0.106 · max 0.155
task/rcv_grasped: [0, 0.784, 0.794, 0.815, 0.813, 0.824, 0.833, 0.837, 0.847, 0.849]  min 0 · mean 0.792 · max 0.854
task/rcv_hand_foreign_rate: [0.167, 0.247, 0.147, 0.0933, 0.0903, 0.0828, 0.0872, 0.0774, 0.0676, 0.0635]  min 0.0239 · mean 0.12 · max 0.465
task/src_cup_lift: [0.00305, 0.0231, 0.0619, 0.125, 0.192, 0.214, 0.207, 0.284, 0.219, 0.219]  min 0.00011 · mean 0.165 · max 0.304
task/src_grasped: [0, 0.784, 0.828, 0.84, 0.847, 0.858, 0.852, 0.857, 0.862, 0.866]  min 0 · mean 0.815 · max 0.875
task/src_hand_foreign_rate: [0.155, 0.282, 0.216, 0.156, 0.135, 0.124, 0.111, 0.0923, 0.0979, 0.0862]  min 0.0422 · mean 0.152 · max 0.487
task/src_tilt_deg: [16, 15.3, 10.7, 8.87, 10.1, 10.3, 9.77, 8.71, 7.07, 6.32]  min 5.79 · mean 10.2 · max 26.6
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Round 12 (iter_11, t2r_i11, 4096 env, 555 epochs / 4.4 h, ADR level 0) — measured facts only. Both cups grasped (src 0.86, rcv 0.84) and lifted (src cup lift 0.225 m, rcv 0.119 m) from about epoch 300 on. episode_success 0.0, bead/in_target 0.0001, bead/spill 0.017. Between epoch 490 and 555 task/aim_dist stayed at 0.21 m (0.211 -> 0.206) and task/cups_center_dist at 0.20 m (0.201 -> 0.199): the mouths never came closer. src tilt 6.2 deg, rcv tilt 6.1 deg, task/tilt_far_deg 6.1 (the source cup is never tilted). task/premature_tilt_rate 0.066-0.086, task/nested_rate 0.008, task/cup_collision_rate 0.029, hand_foreign src/rcv 0.089/0.061, task/rcv_palm_speed 0.14 m/s, bead/fill_level 0.76, dr/bead_active_hi 12. Reward per step (recent mean): total 14.06 = grasp_src 1.38 + grasp_rcv 1.36 + lift_src 1.47 + lift_rcv 1.63 + wrap_src 1.08 + wrap_rcv 0.35 + both_grasped 0.83 + both_lifted 0.83 + raise_src 0.80 + converge 2.02 + bring_together 0.58 + approach 0.68+0.68 + reach 0.50+0.49; aim, tilt, tilt_dir, pour_pose, pour_delta, hold_src, hold_rcv, meet_src, success were exactly 0.0 for the whole round, so their gates never opened. Penalties: rcv_still -0.134, meet_rcv -0.105, palm_sat -0.059/-0.063, palm_rate -0.042/-0.036. converge rose 1.85 -> 2.02 and bring_together 0.51 -> 0.58 over the last 65 epochs while the measured distances did not change. This is the same plateau as round 11 (mouths 0.2-0.3 m apart, pouring-stage terms at zero). The next run is warm-started from this policy's checkpoint (grasp and lift already learned).
