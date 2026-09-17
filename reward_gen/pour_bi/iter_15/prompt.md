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
    LIFT_TARGET_SRC = 0.25    # [m] source lift keeps its gradient up to 25 cm (hover needs h_src ~ h_rcv + 0.11..0.18)
    LIFT_TARGET_RCV = 0.10    # [m] receiver lifted moderately (operator requirement)
    RCV_LIFT_FREE = 0.15      # [m] receiver lift income is full up to 15 cm ...
    RCV_LIFT_SIGMA = 0.08     # [m] ... then decays (20 cm -> 0.68, 25 cm -> 0.21)
    LIFT_GATE = 0.05          # [m] minimum lift for the both_lifted flag
    # --- lip geometry (iter_13: everything is measured on ctx.src_pour_lip_pos, the point the env latch uses)
    LIP_DEADBAND = 0.02       # [m] lip within 2 cm (xy) of the receiver mouth centre counts as centred
    APPR_FULL = 0.08          # [m] iter_14: 0.10 -> 0.08, full zone now ends at the env latch radius (8.1 cm);
                              #     round 14 parked the lip at 9.9 cm, exactly on the old edge
    APPR_ZERO = 0.24          # [m] ... closed beyond 24 cm (lip xy at the spawn separation is ~26 cm -> 0)
    OVER_FULL = 0.035         # [m] lip inside the receiver mouth (inner radius 4.1 cm): release tilt fully paid
    OVER_ZERO = 0.07          # [m] ... nothing beyond 7 cm, 1.1 cm before the env latch radius 8.1 cm
    LIP_Z_LO = 0.0            # [m] lip at receiver rim height -> pour income 0 (it would hit the rim)
    LIP_Z_HI = 0.02           # [m] lip 2 cm above the rim -> 1
    LIP_Z_FREE = 0.07         # [m] lip up to 7 cm above the rim costs nothing ...
    LIP_Z_SIGMA = 0.08        # [m] ... then gentle decay (9 cm -> 0.94, 14 cm -> 0.47)
    # --- tilt-invariant carry height: dz_eq = source origin z + cup_mouth_z - receiver rim z. For an upright cup
    # this equals the old mouth-height signal; it does not fall by ~5 cm when the cup tilts.
    HOVER_LO = 0.06           # [m]
    HOVER_HI = 0.11           # [m] origin ~5 cm above the receiver rim: lip stays above the rim even at 100 deg
    HOVER_FREE = 0.18         # [m] iter_13: 0.16 -> 0.18, a little more bottom clearance is free
    HOVER_SIGMA = 0.06        # [m] Gaussian decay above that
    RAISE_LO = -0.04          # [m] wide ramp start
    RAISE_W = 1.0
    CONV_FAR = 0.40           # [m] linear lip-xy convergence: 0 at 40 cm, 1 inside the dead band (7.5/m at weight 3)
    CONV_W = 3.0
    BRING_W = 2.0             # iter_13: 3 -> 2; the lip target already pays more at the same pose (exp(-10 * 0.034))
    BT_K = 10.0               # [1/m]
    AIM_K = 15.0              # [1/m] precise lip term: 5.4 cm -> 0.60, 2.4 cm -> 0.94 (tilting in place raises it)
    MEET_RANGE = 0.30         # [m] per-arm meeting-point income, linear ramp
    MEET_RCV_W = 0.75
    MEET_SRC_W = 1.0
    # --- source tilt corridor (OPERATOR REQUIREMENT a + new env latch rule)
    SRC_TILT_FREE = 0.20      # [rad] 11.5 deg free far from the receiver (carry wobble, pose noise)
    PRE_MARGIN = 0.17         # [rad] pre-tilt stops 10 deg below ctx.premature_tilt_limit (measured wobble 9.3 deg)
    SRC_TILT_SCALE = 0.20     # [rad] tanh scale of the penalty on tilt beyond the corridor
    SRC_TILT_SIGMA = 0.25     # [rad] Gaussian income factor on tilt beyond the corridor
    SRC_UP_W_TABLE = 1.0      # penalty weight while grasped on the table
    SRC_UP_W_CARRY = 1.5      # extra once carried: total 2.5 exceeds anything tilting outside the corridor could earn
    LATCH_W = 0.5             # per-step penalty once the env's premature_tilt has latched
    # --- tilt income, two segments
    RELEASE_ABOVE_LIMIT = math.radians(20.0)  # env definition: limit = first-release angle - 20 deg
    TILT_TARGET_ABOVE = 0.45  # [rad] release tilt income saturates 26 deg past the first-release angle
    TILT_TARGET_CAP = 2.0     # [rad] never ask for more than 115 deg
    BAND_BELOW = 0.10         # [rad] pour_pose band starts 6 deg below release ...
    BAND_ABOVE = 0.40         # [rad] ... and saturates 23 deg above it
    PRE_W = 4.0               # iter_14: 3 -> 4; pre-tilt (0 -> limit - 10 deg): cannot spill, cannot latch
    PRE_NEAR_FLOOR = 0.25     # iter_14: pre-tilt keeps 25 % income with the lip far, full only over the mouth
    PRE_NEAR_K = 10.0         # [1/m] lip 10 cm -> 0.59, 5 cm -> 0.81, 2 cm -> 1.0
    TILT_W = 4.0              # release tilt (limit - 10 deg -> release + 26 deg) only with the lip over the mouth
    POSE_W = 3.0              # standing income of the full pour pose; 3 + 4 + 3 = 10 on top of ~20 carry income
    ROT_SPEED_FREE = 1.0      # [rad/s] controlled pouring rotation is free below this
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped
    RCV_TOPPLED = 1.2         # [rad] receiver lying on its side
    NEAR_DIST = 0.15          # [m] cup-centre distance inside which fast closing is charged
    CLOSE_V_FREE = 0.15       # [m/s]
    CLOSE_V_SCALE = 0.20      # [m/s]
    FORCE_SCALE = 5.0         # [N] tanh scale for all contact penalties
    FOREIGN_DEADBAND = 1.0    # [N] incidental brushes are free
    CLEAN_DEADBAND = 0.5      # [N] deadband of the multiplicative clean gate
    FOREIGN_W_FREE = 0.3      # hand-foreign weight while the hand does not hold its cup
    FOREIGN_W_HELD = 1.5      # iter_14: 1.0 -> 1.5 while held; contact is now charged additively (same cost
                              #     at any distance) instead of wiping out aim/bring income near the receiver
    CLEAN_FLOOR = 0.5         # iter_14: multiplicative clean gate keeps 50 % (was 0 %)
    GRIP_FLOOR = 0.3          # pour/lift income still worth 30 % with a poor wrap
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
    # receiver held still (OPERATOR REQUIREMENT b). Round 13 measured the receiver palm at 0.38 m/s (0.14 before),
    # so the weight comes back in steps: 0.25 far, +0.25 once the lip is in the approach zone, +0.5 while pouring
    # and after the pour (total 1.0). Only the receiver's own cup speed is charged, which costs nothing to reduce.
    RCV_STILL_FREE = 0.05     # [m/s]
    RCV_STILL_SCALE = 0.15    # [m/s] tanh scale
    RCV_STILL_W = 0.25
    RCV_STILL_NEAR_W = 0.25
    RCV_STILL_POUR_W = 0.50
    POST_POUR_FRAC = 0.8      # post-pour hold ramps in over 0 -> 0.8 of the beads
    SETTLE_FRAC = 0.05        # spill penalty ramps in over the first 5 % of the episode (bead settling)
    POUR_DELTA_W = 100.0      # per unit of transferred fraction: 12 beads -> 8.3 per bead, 6 beads -> 16.7
    SPILL_W = 50.0            # a spilled bead costs half of what a transferred bead earns
    REACH_FAR = 0.26          # [m]
    REACH_NEAR = 0.14         # [m]
    REACH_W = 0.5             # anchor only (solved stage)
    CLOSE_W = 1.0             # closing at the cup: 1.0 x exp(-6d) x normalised closure
    CLOSE_NORM = 0.35
    PALM_RATE_W = 0.01        # per unit of sum-of-squared palm command change (round 13: sampling noise, left as is)
    HAND_RATE_W = 0.005       # finger commands
    SAT_LO = 0.7              # palm commands pinned beyond |0.7| are taxed ...
    SAT_W = 0.15              # ... up to -0.15/step per arm (0.025 per pinned axis: negligible against the pour stack)

    # ------------------------------------------------------------------ clearance factors
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_foreign = torch.maximum(ctx.src_hand_foreign_force, ctx.rcv_hand_foreign_force)
    clean = 1.0 - torch.tanh(torch.clamp(f_foreign - CLEAN_DEADBAND, min=0.0) / FORCE_SCALE)
    # iter_14: soft version for the positioning / pour stack. With the hard gate every touch near the receiver
    # cost ~4/step of aim+bring income but nothing far away, which pushed the cups apart as the tilt grew.
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
    held_src = torch.maximum(g_src, src_carry)      # (N,) source cup held (grasp flag or carried)
    held_rcv = torch.maximum(g_rcv, rcv_carry)      # (N,) receiver cup held
    # iter_13: the pour stack is gated by the carry phases, not by the boolean grasp flags, so a flickering
    # thumb contact while the hand rotates does not zero the whole stack
    carry_both = src_carry * rcv_carry              # (N,) in [0,1]

    # ------------------------------------------------------------------ receiver upright gate (phase-gated)
    rcv_tilt_excess = torch.clamp(ctx.rcv_cup_tilt - RCV_TILT_FREE, min=0.0)
    rcv_up_raw = torch.exp(-(rcv_tilt_excess / RCV_TILT_SIGMA) ** 2)              # (N,) in [0,1]
    rcv_up = 1.0 - rcv_carry * (1.0 - rcv_up_raw)                                  # (N,) in [0,1]
    rcv_up_soft = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_up                 # (N,) in [0.3,1]

    # ------------------------------------------------------------------ pouring-lip geometry (env lip point)
    lip_delta = ctx.src_pour_lip_pos - ctx.rcv_cup_mouth_pos                        # (N,3)
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)                                   # (N,) horizontal miss of the lip
    lip_dz = lip_delta[:, 2]                                                        # (N,) lip height above the rcv rim
    lip_off = torch.clamp(lip_xy - LIP_DEADBAND, min=0.0)                           # (N,)
    approach_gate = 1.0 - _smoothstep((lip_xy - APPR_FULL) / (APPR_ZERO - APPR_FULL))   # (N,) wide zone
    over = 1.0 - _smoothstep((lip_xy - OVER_FULL) / (OVER_ZERO - OVER_FULL))            # (N,) lip over the mouth

    # ------------------------------------------------------------------ tilt angles
    tilt_mag = ctx.src_cup_tilt                                                     # (N,) any direction (latch uses this)
    limit = ctx.premature_tilt_limit.to(dt)                                         # (N,)
    release_tilt = limit + RELEASE_ABOVE_LIMIT                                      # (N,) first-release angle of this fill
    theta_pre = torch.clamp(limit - PRE_MARGIN, min=0.0)                            # (N,) safe pre-tilt angle
    tilt_target = torch.clamp(release_tilt + TILT_TARGET_ABOVE, max=TILT_TARGET_CAP)  # (N,)
    # toward-receiver tilt: env signal, backed by the same angle from fields of ctx (no folding above 90 deg);
    # never larger than the real tilt, never negative (leaning away or sideways earns nothing)
    to_rcv = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]                  # (N,2)
    to_rcv_n = to_rcv / (torch.norm(to_rcv, dim=-1, keepdim=True) + 1e-6)
    a_own = torch.sum(ctx.src_cup_up[:, :2] * to_rcv_n, dim=-1)                     # (N,) up . d
    theta_own = torch.atan2(a_own, ctx.src_cup_up[:, 2])                            # (N,)
    theta = torch.maximum(ctx.src_tilt_toward_rcv.to(dt), theta_own)                # (N,)
    theta_eff = torch.minimum(torch.clamp(theta, min=0.0), tilt_mag)                # (N,)

    # ------------------------------------------------------------------ source tilt corridor
    # allowed tilt: 0.20 rad far away -> theta_pre inside the approach zone -> unlimited with the lip over the mouth
    allowed = SRC_TILT_FREE + approach_gate * torch.clamp(theta_pre - SRC_TILT_FREE, min=0.0)
    tilt_over = torch.clamp(tilt_mag - allowed, min=0.0) * (1.0 - over)             # (N,) tilt beyond the corridor
    src_up = torch.exp(-(tilt_over / SRC_TILT_SIGMA) ** 2)                          # (N,) in [0,1]

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
    wrap_src = 1.5 * g_src * wq_src     # the source grasp has to survive a 90 deg rotation: 3x the receiver weight
    wrap_rcv = 0.5 * g_rcv * wq_rcv
    grip = GRIP_FLOOR + (1.0 - GRIP_FLOOR) * wq_src   # in [0.3,1]; multiplies lift_src / aim / pretilt / tilt / pour_pose

    # ------------------------------------------------------------------ stage 3: lift
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET_SRC, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET_RCV, 0.0, 1.0)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    lift_src = 2.0 * g_src * grip * lift_frac_src * src_up
    lift_rcv = 2.0 * g_rcv * lift_frac_rcv * rcv_band * (0.5 + 0.5 * rcv_up)

    src_lifted_b = ctx.src_grasped & (h_src > LIFT_GATE)
    both_lifted_b = src_lifted_b & ctx.rcv_grasped & (h_rcv > LIFT_GATE)
    both_lifted = 1.0 * both_lifted_b.to(dt)

    # ------------------------------------------------------------------ heights
    dz_eq = ctx.src_cup_pos[:, 2] + ctx.cup_mouth_z - ctx.rcv_cup_mouth_pos[:, 2]  # (N,) tilt-invariant
    hover_decay = torch.exp(-(torch.clamp(dz_eq - HOVER_FREE, min=0.0) / HOVER_SIGMA) ** 2)
    hover = _smoothstep((dz_eq - HOVER_LO) / (HOVER_HI - HOVER_LO)) * hover_decay
    hover_wide = _smoothstep((dz_eq - RAISE_LO) / (HOVER_HI - RAISE_LO)) * hover_decay
    lip_above = torch.clamp((lip_dz - LIP_Z_LO) / (LIP_Z_HI - LIP_Z_LO), 0.0, 1.0)  # lip not below the rcv rim
    lip_high = torch.exp(-(torch.clamp(lip_dz - LIP_Z_FREE, min=0.0) / LIP_Z_SIGMA) ** 2)
    aim_z = lip_above * lip_high
    z_ok = torch.maximum(hover, over * aim_z)       # carry height band OR a good lip height over the mouth

    # ------------------------------------------------------------------ stage 4: raise, converge, bring together, aim
    raise_src = RAISE_W * carry_both * not_nested_f * src_up * rcv_up_soft * hover_wide
    # linear in LIP xy down to the 2 cm dead band. For an upright cup lip_xy = mouth_xy - 4.1 cm, so the target no
    # longer asks to stack the source body over the receiver; tilting in place also moves the lip inward and is paid.
    conv_xy = torch.clamp(1.0 - lip_off / CONV_FAR, 0.0, 1.0)
    converge = CONV_W * carry_both * not_nested_f * src_up * rcv_up_soft * hover_wide * conv_xy
    bring_together = BRING_W * src_carry * not_nested_f * clean_soft * src_up * torch.exp(-BT_K * lip_off) \
        * z_ok * rcv_up_soft
    stack_gate = carry_both * not_nested_f * clear * clean_soft
    aim_xy = torch.exp(-AIM_K * lip_off)
    aim_pose = aim_xy * z_ok
    aim = 3.0 * stack_gate * grip * aim_pose * rcv_up_soft

    # ------------------------------------------------------------------ stage 5: pre-tilt, release tilt, pour pose
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    # segment 1: 0 -> theta_pre, open in the whole approach zone (gradient exists at the round-13 parking pose).
    # No contact gate here: the additive contact penalties still charge every touch.
    pre_frac = torch.clamp(theta_eff / (theta_pre + 1e-3), 0.0, 1.0)
    # iter_14: tilting pays most with the lip over the mouth, so tilting in place no longer competes with closing in
    near_pre = PRE_NEAR_FLOOR + (1.0 - PRE_NEAR_FLOOR) * torch.exp(-PRE_NEAR_K * lip_off)
    pretilt = PRE_W * carry_both * not_nested_f * grip * beads_left * approach_gate * hover_wide \
        * pre_frac * near_pre * rcv_up_soft
    # segment 2: theta_pre -> release + 26 deg, only with the lip over the receiver mouth and above its rim
    fin_den = torch.clamp(tilt_target - theta_pre, min=0.2)
    fin_frac = torch.clamp((theta_eff - theta_pre) / fin_den, 0.0, 1.0)
    tilt = TILT_W * stack_gate * grip * beads_left * over * lip_above * z_ok * fin_frac * rcv_up

    pour_band = _smoothstep((theta_eff - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    pour_pose = POSE_W * stack_gate * grip * beads_left * over * lip_above * aim_pose * pour_band * rcv_up

    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    rot_speed = -0.3 * held_src * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (ungated income)
    # increments, never the level. Symmetric: a bead that bounces out again is charged, so in/out chatter nets 0.
    pour_delta = POUR_DELTA_W * torch.clamp(ctx.d_in_target, -1.0, 1.0)
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    spill_delta = -SPILL_W * settled * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ stage 7: hold after the pour
    post_pour = _smoothstep(ctx.bead_in_target_frac / POST_POUR_FRAC)
    hold_src = 1.5 * post_pour * g_src * wq_src
    hold_rcv = 0.5 * post_pour * g_rcv * wq_rcv * rcv_up

    # ------------------------------------------------------------------ workspace: meet near the midline
    mid_y = 0.5 * (ctx.src_cup_spawn_pos[:, 1] + ctx.rcv_cup_spawn_pos[:, 1])
    mid_x = 0.5 * (ctx.src_cup_spawn_pos[:, 0] + ctx.rcv_cup_spawn_pos[:, 0])
    meet_xy = torch.stack([mid_x, mid_y], dim=-1)                                    # (N,2)
    # cup ORIGINS (same as the mouths for upright cups, no drift while the source tilts)
    d_meet_src = torch.norm(ctx.src_cup_pos[:, :2] - meet_xy, dim=-1)                # (N,)
    d_meet_rcv = torch.norm(ctx.rcv_cup_pos[:, :2] - meet_xy, dim=-1)                # (N,)
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
    # (OPERATOR REQUIREMENT a): the source stays inside the tilt corridor: upright far away, at most
    # (limit - 10 deg) in the approach zone, free only with the lip over the receiver mouth. The penalty reaches
    # about -1.7 at the latch limit itself, so the policy is warned before the env latch can fire.
    upright_src = -(SRC_UP_W_TABLE + SRC_UP_W_CARRY * src_carry) * held_src \
        * torch.tanh(tilt_over / SRC_TILT_SCALE)
    premature_latch = -LATCH_W * ctx.premature_tilt.to(dt)

    fin_active = _smoothstep((theta_eff - theta_pre) / torch.clamp(release_tilt - theta_pre, min=0.1))
    pour_phase = torch.maximum(carry_both * over * fin_active, post_pour)           # (N,) in [0,1]
    upright_rcv = -(RCV_UP_W + RCV_UP_POUR_W * pour_phase) * rcv_carry * torch.tanh(rcv_tilt_excess / RCV_PEN_SCALE)

    # (OPERATOR REQUIREMENT b): the carried receiver is held still
    v_rcv = torch.norm(ctx.rcv_cup_lin_vel, dim=-1)
    still_w = RCV_STILL_W + RCV_STILL_NEAR_W * approach_gate * src_carry + RCV_STILL_POUR_W * pour_phase
    rcv_still = -still_w * rcv_carry \
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
    # (OPERATOR REQUIREMENT c): 50 % of the beads -> 8.0/step, all -> 10/step, scaled by source grip.
    # 10/step on top of the 10/step pour stack doubles the carry income (~20/step): pouring dominates standing.
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
        "pretilt": pretilt,
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
bead/in_target: [0, 3.49e-05, 3.05e-05, 7.71e-05, 5.52e-05, 6.97e-05, 6.1e-05, 8.42e-05, 0.00019, 5.76e-05]  min 0 · mean 8.37e-05 · max 0.000366
bead/spill: [0.00111, 0.0135, 0.0153, 0.0175, 0.0184, 0.0182, 0.0169, 0.0182, 0.0174, 0.0154]  min 0.00111 · mean 0.0157 · max 0.021
done/drop: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000244]  min 0 · mean 2.32e-05 · max 0.000488
episode_lengths/step: [81.9, 873, 805, 842, 838, 847, 843, 858, 830, 832]  min 81.9 · mean 794 · max 899
reward/aim: [0, 2.07, 2.07, 2.18, 2.36, 2.45, 2.44, 2.36, 2.02, 1.75]  min 0 · mean 1.87 · max 2.53
reward/approach_rcv: [0.671, 0.701, 0.696, 0.695, 0.69, 0.693, 0.692, 0.686, 0.674, 0.659]  min 0.55 · mean 0.677 · max 0.703
reward/approach_src: [0.668, 0.689, 0.69, 0.689, 0.686, 0.686, 0.687, 0.682, 0.677, 0.666]  min 0.545 · mean 0.675 · max 0.695
reward/both_grasped: [0.000488, 0.968, 0.901, 0.898, 0.922, 0.939, 0.931, 0.906, 0.838, 0.726]  min 0.000488 · mean 0.82 · max 0.989
reward/both_lifted: [0, 0.96, 0.887, 0.893, 0.919, 0.92, 0.907, 0.89, 0.812, 0.716]  min 0 · mean 0.785 · max 0.977
reward/bring_together: [4.38e-06, 1.59, 1.61, 1.66, 1.74, 1.78, 1.75, 1.69, 1.48, 1.3]  min 4.38e-06 · mean 1.39 · max 1.83
reward/closing_speed: [0, -0.00189, -0.00215, -0.00283, -0.00238, -0.00187, -0.0018, -0.0019, -0.00382, -0.00247]  min -0.0196 · mean -0.00272 · max 0
reward/converge: [0, 2.73, 2.69, 2.71, 2.74, 2.76, 2.73, 2.67, 2.43, 2.15]  min 0 · mean 2.33 · max 2.89
reward/cup_contact: [0, -0.0223, -0.0239, -0.0197, -0.0311, -0.0349, -0.0423, -0.0458, -0.0578, -0.0638]  min -0.295 · mean -0.046 · max 0
reward/drop: [0, -0.00061, 0, -0.000854, -0.0011, -0.000366, -0.0011, -0.000366, -0.000854, -0.000244]  min -0.00134 · mean -0.000376 · max 0
reward/grasp_rcv: [0.111, 1.56, 1.49, 1.48, 1.5, 1.52, 1.51, 1.48, 1.38, 1.24]  min 0.111 · mean 1.35 · max 1.57
reward/grasp_src: [0.267, 1.55, 1.54, 1.53, 1.52, 1.52, 1.51, 1.47, 1.39, 1.24]  min 0.151 · mean 1.37 · max 1.58
reward/hand_foreign_rcv: [-0.0688, -0.0155, -0.018, -0.0235, -0.0194, -0.0209, -0.0326, -0.0388, -0.0576, -0.0883]  min -0.477 · mean -0.0438 · max -0.00157
reward/hand_foreign_src: [-0.36, -0.02, -0.0313, -0.0453, -0.0448, -0.0553, -0.0837, -0.102, -0.148, -0.132]  min -0.368 · mean -0.0876 · max -0.00508
reward/hand_rate: [-0.016, -0.0217, -0.0241, -0.0238, -0.0234, -0.0223, -0.0208, -0.0211, -0.0211, -0.0205]  min -0.0254 · mean -0.0211 · max -0.0133
reward/hold_rcv: [0, 5.84e-06, 0, 0, 1.14e-06, 8.72e-06, 0, 5.11e-06, 3.34e-05, 0]  min 0 · mean 8.94e-06 · max 7.99e-05
reward/hold_src: [0, 9.01e-06, 0, 1.32e-05, 2.36e-05, 1.45e-05, 0, 0, 0.000112, 1.83e-05]  min 0 · mean 2.82e-05 · max 0.000141
reward/lift_rcv: [0.000296, 1.62, 1.6, 1.61, 1.62, 1.37, 1.35, 1.44, 1.32, 1.27]  min 0.000296 · mean 1.31 · max 1.73
reward/lift_src: [0.00267, 1.7, 1.65, 1.73, 1.75, 1.71, 1.65, 1.6, 1.38, 1.23]  min 0.00267 · mean 1.39 · max 1.83
reward/meet_rcv: [0, 0.427, 0.378, 0.374, 0.342, 0.325, 0.301, 0.292, 0.282, 0.249]  min 0 · mean 0.306 · max 0.522
reward/meet_src: [0, 0.248, 0.204, 0.195, 0.185, 0.185, 0.162, 0.153, 0.136, 0.117]  min 0 · mean 0.157 · max 0.284
reward/nested: [0, -0.00281, -0.00354, -0.00378, -0.00549, -0.00623, -0.0072, -0.00806, -0.00769, -0.00647]  min -0.0818 · mean -0.00677 · max 0
reward/palm_rate_rcv: [-0.0147, -0.0417, -0.0402, -0.0394, -0.0394, -0.0375, -0.0334, -0.033, -0.0315, -0.0278]  min -0.0429 · mean -0.0324 · max -0.00981
reward/palm_rate_src: [-0.0134, -0.0375, -0.0357, -0.0313, -0.0285, -0.0227, -0.0193, -0.0182, -0.0187, -0.0175]  min -0.0391 · mean -0.0229 · max -0.0111
reward/palm_sat_rcv: [-0.066, -0.0576, -0.0607, -0.0606, -0.0641, -0.0582, -0.0589, -0.0594, -0.0559, -0.0605]  min -0.104 · mean -0.06 · max -0.0346
reward/palm_sat_src: [-0.0465, -0.0564, -0.0622, -0.0716, -0.0723, -0.0677, -0.07, -0.0699, -0.063, -0.0677]  min -0.104 · mean -0.0646 · max -0.0256
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00349 · mean 5.12e-05 · max 0.00407
reward/pour_pose: [0, 0, 0, 3.2e-07, 6.4e-07, 0, 0, 0, 0, 0]  min 0 · mean 6.44e-07 · max 0.000104
reward/premature_latch: [0, -0.00293, -0.016, -0.00806, -0.0106, -0.00757, -0.00562, -0.00574, -0.00647, -0.00281]  min -0.0184 · mean -0.0064 · max 0
reward/pretilt: [0, 1.67, 1.76, 1.98, 2.06, 2.23, 2.44, 2.34, 1.89, 1.67]  min 0 · mean 1.75 · max 2.69
reward/raise_src: [0, 0.957, 0.931, 0.935, 0.931, 0.929, 0.92, 0.898, 0.826, 0.733]  min 0 · mean 0.803 · max 0.984
reward/rcv_still: [0, -0.125, -0.121, -0.123, -0.13, -0.144, -0.161, -0.179, -0.174, -0.164]  min -0.285 · mean -0.145 · max 0
reward/reach_rcv: [0.5, 0.499, 0.498, 0.497, 0.497, 0.498, 0.498, 0.498, 0.497, 0.494]  min 0.447 · mean 0.495 · max 0.5
reward/reach_src: [0.5, 0.499, 0.498, 0.498, 0.498, 0.498, 0.498, 0.497, 0.496, 0.494]  min 0.441 · mean 0.495 · max 0.5
reward/rot_speed: [-0.0416, -0.03, -0.0311, -0.0331, -0.0335, -0.0379, -0.0422, -0.0498, -0.0545, -0.0582]  min -0.194 · mean -0.0457 · max -0.00181
reward/spill_delta: [0, 0, 0, 0, -0.00136, 0, 0, 0, -0.00174, -0.00349]  min -0.0486 · mean -0.00201 · max 0
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.11e-07 · max 6.43e-05
reward/tilt: [0, 1.1e-08, 2.99e-05, 5.39e-05, 0.000113, 0.000206, 0.00207, 0.00126, 0.000253, 0.000632]  min 0 · mean 0.000719 · max 0.00844
reward/total: [2.19, 21.7, 21.2, 21.7, 22.1, 22.2, 22.1, 21.5, 19.2, 17.1]  min 2.19 · mean 18.8 · max 23
reward/upright_rcv: [0, -0.0408, -0.0359, -0.0311, -0.0344, -0.0221, -0.0324, -0.0335, -0.0409, -0.0505]  min -0.319 · mean -0.0379 · max 0
reward/upright_src: [-0.0021, -0.00875, -0.0122, -0.00607, -0.00635, -0.00366, -0.00563, -0.00533, -0.00823, -0.0119]  min -0.034 · mean -0.00599 · max -0.000152
reward/wrap_rcv: [0.000734, 0.399, 0.356, 0.364, 0.379, 0.408, 0.411, 0.401, 0.36, 0.302]  min 0.000734 · mean 0.348 · max 0.44
reward/wrap_src: [0.104, 1.34, 1.28, 1.3, 1.3, 1.32, 1.32, 1.27, 1.17, 1]  min 0.0626 · mean 1.15 · max 1.4
rewards/step: [73.5, 1.5e+04, 1.18e+04, 1.44e+04, 1.51e+04, 1.61e+04, 1.61e+04, 1.68e+04, 1.58e+04, 1.65e+04]  min 73.5 · mean 1.47e+04 · max 1.81e+04
task/aim_dist: [0.251, 0.16, 0.159, 0.421, 0.313, 0.158, 0.147, 0.18, 0.144, 0.198]  min 0.133 · mean 0.171 · max 0.475
task/cup_collision_rate: [0, 0.0254, 0.0266, 0.0229, 0.0332, 0.0374, 0.0464, 0.05, 0.0657, 0.0742]  min 0 · mean 0.0524 · max 0.372
task/cups_center_dist: [0.279, 0.189, 0.189, 0.455, 0.345, 0.189, 0.181, 0.214, 0.172, 0.224]  min 0.125 · mean 0.199 · max 0.507
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 8.75e-07 · max 0.000244
task/nested_rate: [0, 0.00562, 0.00708, 0.00757, 0.011, 0.0125, 0.0144, 0.0161, 0.0154, 0.0129]  min 0 · mean 0.0136 · max 0.164
task/rcv_cup_lift: [0.00561, 0.0872, 0.0928, 0.125, 0.0901, 0.0725, 0.072, 0.0761, 0.0702, 0.0715]  min 0.00504 · mean 0.0727 · max 0.131
task/rcv_grasped: [0.00269, 0.975, 0.911, 0.906, 0.932, 0.948, 0.944, 0.926, 0.864, 0.772]  min 0.00269 · mean 0.839 · max 0.991
task/rcv_hand_foreign_rate: [0.351, 0.0198, 0.0269, 0.0278, 0.0269, 0.0327, 0.0459, 0.061, 0.0955, 0.136]  min 0.00269 · mean 0.0759 · max 0.712
task/src_cup_lift: [0.00352, 0.234, 0.239, 0.333, 0.328, 0.232, 0.221, 0.224, 0.188, 0.178]  min 0.00352 · mean 0.199 · max 0.345
task/src_grasped: [0.154, 0.985, 0.973, 0.968, 0.962, 0.962, 0.951, 0.929, 0.87, 0.763]  min 0.0552 · mean 0.856 · max 0.999
task/src_hand_foreign_rate: [0.873, 0.032, 0.0554, 0.0657, 0.0696, 0.0913, 0.113, 0.148, 0.217, 0.217]  min 0.0117 · mean 0.149 · max 0.873
task/src_tilt_deg: [9.18, 26.6, 28.9, 31.4, 31.5, 32.4, 35.1, 34.4, 30.1, 28.5]  min 2.04 · mean 29.1 · max 37.9
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 8.75e-07 · max 0.000244

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Round 15 (iter_14, t2r_i14, 4096 env, warm-started from the round-14 checkpoint, 557 epochs / 4.1 h, ADR level 0) — measured facts only, environment unchanged. episode_success 0.0 and bead/in_target 0.0 for the whole round; bead/spill 0.017; reward/pour_delta, reward/pour_pose, reward/hold_src, reward/hold_rcv, reward/success were 0.0.

Grasp/lift: task/src_grasped 0.80, task/rcv_grasped 0.81 (round 14 ended at 0.83/0.82), src cup lift 0.193 m, rcv cup lift 0.081 m, task/rcv_tilt_deg 6.8.

Approach and tilt (env mean, epoch: task/pour_lip_dist / task/cups_center_dist / task/src_tilt_deg): first epoch 0.211 / 0.279 / 9.2 deg; 288: 0.060 / 0.193 / 26.2 deg; 557: 0.053 / 0.184 / 32.2 deg (max of the logged mean 37.5 deg). task/aim_dist 0.149 m. Unlike round 14, the lip stayed inside the 0.081 m latch radius on average and did not drift back out. task/tilt_limit_deg averaged 60.5, so iter_14's theta_pre (limit - 10 deg) is about 50 deg; the mean source tilt stayed about 18 deg below it for the whole round.

Tilt terms: reward/pretilt 0 -> 2.07 (max 2.65, weight 4.0), reward/tilt 0.004 (max 0.006, weight 4.0). In iter_14 code `tilt` pays only for theta_eff above theta_pre and multiplies over * lip_above * z_ok * stack_gate; `pretilt` pays linearly 0 -> theta_pre.

Contact and other indicators: task/cup_collision_rate 0.064 (round 14 end 0.037), hand_foreign src/rcv 0.184/0.093 (round 14 end 0.132/0.081), task/nested_rate 0.015, task/premature_tilt_rate 0.008 (max 0.037), task/tilt_far_deg 22.7, task/rcv_palm_speed 0.39 m/s, bead/fill_level 0.75, dr/bead_active_hi 12. Penalty terms at the end: cup_contact -0.056, hand_foreign_src -0.115, hand_foreign_rcv -0.060, rcv_still -0.158, rot_speed -0.053.

Best so far remains iter_05 (round-3 environment, not comparable).
