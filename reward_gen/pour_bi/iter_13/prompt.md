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
    DZ_SIGMA = 0.08           # [m] beyond that, gentle decay (iter_12: 0.06 -> 0.08, closes the dip between hover and lip bands)
    # upright-carry hover band (source mouth above the receiver rim)
    HOVER_LO = 0.06           # [m] mouth 6 cm above the receiver rim -> 0
    HOVER_HI = 0.11           # [m] mouth 11 cm above the rim -> 1
    HOVER_FREE = 0.16         # [m] free up to 16 cm ...
    HOVER_SIGMA = 0.06        # [m] ... then Gaussian decay
    RAISE_LO = -0.04          # [m] relative-height ramp start (mouth 4 cm below the rim)
    RAISE_W = 1.0
    BRING_W = 3.0             # iter_12: 1.5 -> 3.0, the last 15 cm must out-earn the standing carry income
    # iter_11: dedicated xy convergence ramp (mouth-to-mouth). Round 11 parked at 0.29 m: bring_together's
    # exp(-3d) paid 0.42 there vs 0.86 at 5 cm (0.65/step for the whole gap on a 12/step income). This ramp
    # pays 0.72 at 0.29 m, 2.0 at 0.17 m, 2.9 at 0.10 m and 3.0 at 6 cm: ~2.3/step for closing the gap.
    CONV_FAR = 0.40           # [m]
    CONV_NEAR = 0.06          # [m]
    CONV_W = 3.0
    # iter_12: converge is LINEAR in mouth xy (constant 7.5/m all the way to 0, no flat end) and no longer
    # multiplied by the cup-contact gate (a brush zeroed 2.0/step of income: the policy learned to stay 0.2 m away;
    # the additive cup_contact penalty -1.0 still charges every contact)
    BT_K = 10.0               # [1/m] bring_together kernel exp(-10 d): 0.20 m -> 0.14, 0.10 -> 0.37, 0.05 -> 0.61
    # iter_12: aim opens on a WIDE mouth-xy gate (does not pay tilt, so it cannot cause a premature latch)
    AIM_FULL = 0.06           # [m]
    AIM_ZERO = 0.22           # [m]
    # iter_12: per-arm meeting-point income replaces the meet_* crossing taxes (each arm gets its own gradient,
    # independent of what the other arm does). Meeting point = xy midpoint of the two spawn positions.
    MEET_RANGE = 0.30         # [m] linear ramp 0 at 30 cm from the meeting point, 1 on it
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
    MEET_RCV_W = 0.75         # iter_12: now POSITIVE per-arm income weights (see MEET_RANGE)
    MEET_SRC_W = 1.0
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
    RCV_STILL_FREE = 0.05     # [m/s] iter_12: 0.03 -> 0.05 (measured carry jitter 0.14 m/s)
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
    conv_xy = torch.clamp(1.0 - mouth_xy / CONV_FAR, 0.0, 1.0)                     # iter_12: linear, 0 at 40 cm, 1 at 0
    converge = CONV_W * both_lifted_f * not_nested_f * src_up * rcv_up_soft * hover_wide * conv_xy

    # iter_11: xy factor sharpened exp(-3d) -> exp(-5d): 0.29 m -> 0.23, 0.10 m -> 0.61, 0.05 m -> 0.78
    bt_z = torch.maximum(hover, near * aim_z)
    bring_together = BRING_W * src_lifted_f * not_nested_f * clean * src_up * torch.exp(-BT_K * mouth_xy) \
        * bt_z * rcv_up_soft
    stack_gate = both_lifted_f * not_nested_f * clear * clean
    # iter_12: an UPRIGHT source hovering 11-16 cm above the rim scored aim_z = 0.26, so entering the near zone
    # paid ~0.4 while rcv_still charged ~0.47 there. Height factor is now max(hover, lip band): the hover pose
    # is a full-value aim pose and tilting from it pays at once.
    near_wide = 1.0 - _smoothstep((mouth_xy - AIM_FULL) / (AIM_ZERO - AIM_FULL))
    aim_pose = aim_xy * bt_z
    aim = 3.0 * stack_gate * grip * near_wide * aim_pose * rcv_up_soft

    # ------------------------------------------------------------------ stage 5: tilt / pour pose (all x near)
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_rad = ctx.src_cup_tilt
    fill = torch.clamp(ctx.bead_fill_level, 0.0, 1.0)
    release_tilt = RELEASE_FULL + RELEASE_SPAN * (1.0 - fill)                        # (N,)
    tilt_target = release_tilt + TILT_TARGET_ABOVE                                   # (N,)
    tilt_frac = torch.clamp(tilt_rad / tilt_target, 0.0, 1.0)
    # weight 4.0 x near: worth nothing until the mouths are within 5 cm; aim_z gives the lip-height gradient
    tilt = 4.0 * stack_gate * grip * beads_left * near * bt_z * tilt_frac * rcv_up

    to_rcv = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]                  # (N,2)
    to_rcv_n = to_rcv / (torch.norm(to_rcv, dim=-1, keepdim=True) + 1e-6)
    up_xy = ctx.src_cup_up[:, :2]                                                    # (N,2)
    up_xy_n = up_xy / (torch.norm(up_xy, dim=-1, keepdim=True) + 1e-6)
    dir_cos = torch.sum(to_rcv_n * up_xy_n, dim=-1)                                  # (N,) in [-1,1]
    tilt_amt = torch.clamp(tilt_rad / TILT_DIR_SCALE, 0.0, 1.0)
    tilt_dir = 1.0 * stack_gate * beads_left * near * tilt_amt * dir_cos * rcv_up

    pour_band = _smoothstep((tilt_rad - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    pour_pose = 3.0 * stack_gate * grip * beads_left * near * aim_pose * pour_band * rcv_up

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
    mid_x = 0.5 * (ctx.src_cup_spawn_pos[:, 0] + ctx.rcv_cup_spawn_pos[:, 0])
    meet_xy = torch.stack([mid_x, mid_y], dim=-1)                                    # (N,2)
    d_meet_src = torch.norm(ctx.src_cup_mouth_pos[:, :2] - meet_xy, dim=-1)          # (N,)
    d_meet_rcv = torch.norm(ctx.rcv_cup_mouth_pos[:, :2] - meet_xy, dim=-1)          # (N,)
    meet_src = MEET_SRC_W * both_lifted_f * not_nested_f * src_up * hover_wide \
        * torch.clamp(1.0 - d_meet_src / MEET_RANGE, 0.0, 1.0)
    meet_rcv = MEET_RCV_W * both_lifted_f * not_nested_f * rcv_up_soft \
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
    # (OPERATOR REQUIREMENT a): source stays upright until its mouth is near the receiver's mouth.
    upright_src = -(SRC_UP_W_TABLE + SRC_UP_W_CARRY * src_carry) * held_src * far \
        * torch.tanh(src_tilt_excess / SRC_TILT_SCALE)
    premature_latch = -LATCH_W * ctx.premature_tilt.to(dt)

    pour_phase = torch.maximum(
        both_lifted_f * near * bt_z
        * _smoothstep((tilt_rad - POUR_PHASE_TILT_LO) / (POUR_PHASE_TILT_HI - POUR_PHASE_TILT_LO)),
        post_pour,
    )
    upright_rcv = -(RCV_UP_W + RCV_UP_POUR_W * pour_phase) * rcv_carry * torch.tanh(rcv_tilt_excess / RCV_PEN_SCALE)

    # (OPERATOR REQUIREMENT b): the carried receiver is held still — light while carrying far, full near / post-pour
    v_rcv = torch.norm(ctx.rcv_cup_lin_vel, dim=-1)
    # iter_12: full weight only while actually pouring / after the pour (was: whenever near -> a -0.47 wall at 9 cm)
    still_phase = pour_phase
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
bead/in_target: [0, 2.22e-05, 4.93e-05, 4.25e-05, 0.000107, 6.29e-05, 6.51e-05, 0.00022, 0.000129, 2.03e-05]  min 0 · mean 8.06e-05 · max 0.000277
bead/spill: [0.00158, 0.0121, 0.0135, 0.013, 0.0147, 0.015, 0.017, 0.0149, 0.0159, 0.0168]  min 0.00158 · mean 0.0134 · max 0.0183
done/drop: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.99e-05 · max 0.000488
episode_lengths/step: [84.2, 643, 726, 722, 745, 777, 824, 834, 825, 835]  min 84.2 · mean 763 · max 899
reward/aim: [0, 0.0465, 0.652, 0.853, 0.948, 1.13, 1.23, 1.39, 1.48, 1.62]  min 0 · mean 0.908 · max 1.65
reward/approach_rcv: [0.652, 0.655, 0.687, 0.694, 0.697, 0.7, 0.701, 0.7, 0.698, 0.699]  min 0.562 · mean 0.687 · max 0.706
reward/approach_src: [0.667, 0.642, 0.664, 0.683, 0.688, 0.692, 0.694, 0.696, 0.694, 0.696]  min 0.557 · mean 0.68 · max 0.702
reward/both_grasped: [0, 0.329, 0.841, 0.921, 0.95, 0.961, 0.958, 0.96, 0.956, 0.957]  min 0 · mean 0.836 · max 0.98
reward/both_lifted: [0, 0.322, 0.839, 0.919, 0.948, 0.959, 0.957, 0.958, 0.954, 0.955]  min 0 · mean 0.833 · max 0.979
reward/bring_together: [0, 0.108, 0.683, 0.781, 0.829, 0.911, 0.902, 0.971, 1.02, 1.1]  min 0 · mean 0.751 · max 1.1
reward/closing_speed: [0, -0.0052, -0.00453, -0.00308, -0.0036, -0.00149, -0.00352, -0.00194, -0.00126, -0.000422]  min -0.0165 · mean -0.00187 · max 0
reward/converge: [0, 0.384, 1.68, 1.88, 1.95, 2.05, 2.04, 2.09, 2.11, 2.17]  min 0 · mean 1.72 · max 2.17
reward/cup_contact: [0, -0.133, -0.079, -0.0408, -0.0284, -0.0181, -0.0235, -0.0156, -0.0205, -0.0162]  min -0.296 · mean -0.0334 · max 0
reward/drop: [0, -0.00061, -0.000244, 0, -0.000122, -0.000122, -0.000366, -0.000244, -0.000122, -0.000244]  min -0.00122 · mean -0.00023 · max 0
reward/grasp_rcv: [0.107, 0.878, 1.4, 1.5, 1.53, 1.54, 1.54, 1.54, 1.54, 1.54]  min 0.107 · mean 1.37 · max 1.58
reward/grasp_src: [0.248, 1.16, 1.4, 1.5, 1.53, 1.54, 1.54, 1.54, 1.54, 1.54]  min 0.232 · mean 1.39 · max 1.57
reward/hand_foreign_rcv: [-0.0479, -0.128, -0.0373, -0.0231, -0.0157, -0.0119, -0.0164, -0.0134, -0.0122, -0.0106]  min -0.159 · mean -0.0194 · max -0.00233
reward/hand_foreign_src: [-0.297, -0.119, -0.0568, -0.0438, -0.0283, -0.0343, -0.0379, -0.0361, -0.0295, -0.0361]  min -0.297 · mean -0.0436 · max -0.00473
reward/hand_rate: [-0.0145, -0.0151, -0.0205, -0.0212, -0.0211, -0.0218, -0.0214, -0.0221, -0.0222, -0.0218]  min -0.0241 · mean -0.021 · max -0.0128
reward/hold_rcv: [0, 7.9e-07, 1e-05, 3.5e-06, 1.3e-05, 1.11e-05, 9.91e-06, 4.29e-05, 8.78e-06, 3.66e-06]  min 0 · mean 8.68e-06 · max 4.95e-05
reward/hold_src: [0, 6.61e-06, 1.78e-05, 5.04e-06, 1.83e-05, 2.92e-05, 2.64e-05, 5.86e-05, 2.37e-05, 1.11e-05]  min 0 · mean 2.79e-05 · max 0.000215
reward/lift_rcv: [0, 0.638, 1.66, 1.82, 1.86, 1.89, 1.88, 1.87, 1.85, 1.84]  min 0 · mean 1.64 · max 1.94
reward/lift_src: [0.00219, 0.556, 1.31, 1.53, 1.62, 1.65, 1.66, 1.6, 1.62, 1.72]  min 0.00219 · mean 1.45 · max 1.8
reward/meet_rcv: [0, 0.107, 0.388, 0.454, 0.486, 0.481, 0.49, 0.485, 0.479, 0.457]  min 0 · mean 0.411 · max 0.531
reward/meet_src: [0, 0.0594, 0.223, 0.275, 0.308, 0.316, 0.329, 0.338, 0.335, 0.331]  min 0 · mean 0.248 · max 0.347
reward/nested: [0, -0.0483, -0.00891, -0.00281, -0.00208, -0.00281, -0.00317, -0.00195, -0.00281, -0.00159]  min -0.0483 · mean -0.00368 · max 0
reward/palm_rate_rcv: [-0.0211, -0.033, -0.0364, -0.0382, -0.0387, -0.0409, -0.0417, -0.0442, -0.0464, -0.0469]  min -0.0502 · mean -0.0404 · max -0.0172
reward/palm_rate_src: [-0.0243, -0.0368, -0.0401, -0.0411, -0.0419, -0.0431, -0.0424, -0.0427, -0.0438, -0.042]  min -0.0482 · mean -0.0409 · max -0.0204
reward/palm_sat_rcv: [-0.076, -0.0654, -0.0682, -0.0671, -0.0636, -0.064, -0.061, -0.0581, -0.0564, -0.0573]  min -0.0894 · mean -0.0627 · max -0.0547
reward/palm_sat_src: [-0.0688, -0.0569, -0.0603, -0.0593, -0.0587, -0.0587, -0.0573, -0.0529, -0.0564, -0.059]  min -0.0888 · mean -0.0607 · max -0.0438
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.58e-06 · max 0.00136
reward/pour_pose: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/premature_latch: [0, -0.0294, -0.0599, -0.0463, -0.0328, -0.0316, -0.0319, -0.0267, -0.0298, -0.0197]  min -0.0642 · mean -0.0317 · max 0
reward/raise_src: [0, 0.181, 0.797, 0.892, 0.927, 0.944, 0.936, 0.942, 0.942, 0.942]  min 0 · mean 0.813 · max 0.964
reward/rcv_still: [-6.1e-05, -0.167, -0.113, -0.113, -0.105, -0.0944, -0.0951, -0.0918, -0.0864, -0.089]  min -0.218 · mean -0.0905 · max -6.1e-05
reward/reach_rcv: [0.5, 0.496, 0.499, 0.499, 0.5, 0.499, 0.499, 0.499, 0.499, 0.499]  min 0.451 · mean 0.496 · max 0.5
reward/reach_src: [0.499, 0.496, 0.499, 0.499, 0.499, 0.499, 0.499, 0.499, 0.498, 0.499]  min 0.447 · mean 0.495 · max 0.5
reward/rot_speed: [-0.038, -0.128, -0.0473, -0.0354, -0.0256, -0.02, -0.0211, -0.0222, -0.0206, -0.0183]  min -0.143 · mean -0.0259 · max -0.00516
reward/spill_delta: [0, -1.48e-05, 0, -0.00105, 0, 0, -0.00105, -0.000666, 0, 0]  min -0.0103 · mean -0.000808 · max 0
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 9.01e-05, 0.00104, 0.000163, 0.000121, 0.000455, 0.000909, 0.0031, 0.0064, 0.0255]  min 0 · mean 0.00495 · max 0.0352
reward/tilt_dir: [0, 0.000199, 0.001, 0.000182, 0.000134, 0.000371, 0.000543, 0.00169, 0.00336, 0.0124]  min -2.51e-05 · mean 0.00249 · max 0.0178
reward/total: [2.17, 6.26, 14.7, 16.5, 17.3, 17.8, 18, 18.3, 18.4, 18.8]  min 2.17 · mean 15.5 · max 18.8
reward/upright_rcv: [-6.53e-05, -0.305, -0.0336, -0.0331, -0.0425, -0.027, -0.0312, -0.0228, -0.0198, -0.024]  min -0.305 · mean -0.0306 · max -6.53e-05
reward/upright_src: [-0.00867, -0.0841, -0.0644, -0.0256, -0.018, -0.0131, -0.0141, -0.0175, -0.0188, -0.0149]  min -0.281 · mean -0.0248 · max -0.0022
reward/wrap_rcv: [0, 0.112, 0.375, 0.402, 0.417, 0.423, 0.423, 0.426, 0.424, 0.431]  min 0 · mean 0.366 · max 0.439
reward/wrap_src: [0.103, 0.513, 0.904, 1.06, 1.14, 1.16, 1.21, 1.22, 1.23, 1.28]  min 0.103 · mean 1.06 · max 1.31
rewards/step: [81.7, 8.74e+03, 1.06e+04, 1.07e+04, 1.11e+04, 1.17e+04, 1.27e+04, 1.32e+04, 1.32e+04, 1.36e+04]  min 81.7 · mean 1.15e+04 · max 1.47e+04
task/aim_dist: [0.283, 0.172, 0.163, 0.203, 0.172, 0.172, 0.17, 0.281, 0.245, 0.176]  min 0.145 · mean 0.203 · max 0.385
task/cup_collision_rate: [0, 0.147, 0.0923, 0.0452, 0.0312, 0.0195, 0.0251, 0.0176, 0.0222, 0.0193]  min 0 · mean 0.0379 · max 0.335
task/cups_center_dist: [0.299, 0.16, 0.162, 0.202, 0.169, 0.17, 0.17, 0.284, 0.249, 0.18]  min 0.133 · mean 0.204 · max 0.382
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0.0967, 0.0178, 0.00562, 0.00415, 0.00562, 0.00635, 0.00391, 0.00562, 0.00317]  min 0 · mean 0.00737 · max 0.0967
task/rcv_cup_lift: [0.00411, 0.105, 0.128, 0.139, 0.141, 0.133, 0.131, 0.11, 0.105, 0.104]  min 0.00411 · mean 0.11 · max 0.206
task/rcv_grasped: [0, 0.419, 0.857, 0.934, 0.96, 0.965, 0.966, 0.966, 0.966, 0.962]  min 0 · mean 0.846 · max 0.989
task/rcv_hand_foreign_rate: [0.333, 0.356, 0.116, 0.0674, 0.0427, 0.0339, 0.0422, 0.0352, 0.0303, 0.0278]  min 0.0083 · mean 0.0636 · max 0.562
task/src_cup_lift: [0.00431, 0.141, 0.228, 0.27, 0.259, 0.257, 0.245, 0.292, 0.274, 0.243]  min 0.00431 · mean 0.225 · max 0.294
task/src_grasped: [0.14, 0.705, 0.892, 0.948, 0.965, 0.974, 0.971, 0.971, 0.966, 0.968]  min 0.115 · mean 0.867 · max 0.991
task/src_hand_foreign_rate: [0.851, 0.279, 0.145, 0.094, 0.0635, 0.0796, 0.0872, 0.0889, 0.0681, 0.0903]  min 0.0146 · mean 0.113 · max 0.851
task/src_tilt_deg: [7.99, 6.71, 6.79, 4.97, 4.56, 4.51, 5.59, 6.31, 6.87, 7.44]  min 0.996 · mean 5.7 · max 13.6
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Round 13 (iter_12, t2r_i12, 4096 env, warm-started from the round-12 checkpoint, 294 epochs / 2.1 h, ADR level 0) — measured facts only. The run was stopped by the operator to change the environment, not because it had plateaued. Both cups stayed grasped (src 0.94, rcv 0.93) and lifted (src cup lift 0.233 m, rcv 0.100 m). The mouths came closer during the round: task/aim_dist 0.283 -> 0.174 m and task/cups_center_dist 0.299 -> 0.178 m (first -> recent), still falling at the stop. The source cup was not tilted: src tilt 7.2 deg, task/tilt_far_deg 7.1. episode_success 0.0, bead/in_target 0.0003, bead/spill 0.017. task/premature_tilt_rate 0.054 (max 0.128), task/nested_rate 0.008, task/cup_collision_rate 0.043 (max 0.335 early), hand_foreign src/rcv 0.153/0.050, task/rcv_palm_speed 0.38 m/s (was 0.14 in round 12), bead/fill_level 0.77, dr/bead_active_hi 12. Pouring-stage terms (tilt, tilt_dir, pour_pose, pour_delta, hold_src, hold_rcv, success) were 0.0 for the whole round. In iter_12 every tilt-paying term is multiplied by `near`, a gate on mouth-to-mouth xy distance that is 1 inside 0.05 m and exactly 0 beyond 0.09 m; the measured mouth distance never went below 0.17 m, so tilt income had no gradient anywhere the policy visited.

Environment change for this round (all active in training; the policy is warm-started from the round-13 checkpoint, observation and action sizes unchanged). The premature-tilt latch no longer uses a fixed 30 deg / 0.10 m rule. It now latches when the GRASPED source cup's tilt exceeds `ctx.premature_tilt_limit` while `ctx.src_pour_lip_pos` is outside the receiver mouth in xy (farther than 0.081 m from the receiver mouth centre = inner radius 0.041 + 0.04). `ctx.premature_tilt_limit` (N,) [rad] is per episode: first-release angle of that episode's fill minus 20 deg, i.e. (72 + 34*(1 - fill) - 20) deg — 52 deg for a full cup, 69 deg at fill 0.5. `ctx.src_pour_lip_pos` (N,3) is the point on the source rim, on the side facing the receiver mouth (direction frozen while the cup origin is within 0.04 m of the receiver mouth in xy), where beads leave; it does not depend on how the cup is currently tilted, so a wrong-way tilt does not move it over the receiver. `ctx.src_tilt_toward_rcv` (N,) [rad] is the signed tilt of the source cup toward the receiver mouth (positive = toward, negative = away; sideways tilt gives about 0). Scripted probe (64 env): with 12 active beads (fill 0.94) the limit read 54.1 deg, with 6 beads (fill 0.58) 66.6 deg; a grasp-and-lift with up to 9.3 deg of cup wobble never latched; a tilt to 82 deg made 0.20 m from the receiver latched 100 % of environments. So tilting up to roughly 50-65 deg (depending on fill) is now free at any distance, and the latch only punishes reaching the release angle with the pouring rim away from the receiver mouth.
