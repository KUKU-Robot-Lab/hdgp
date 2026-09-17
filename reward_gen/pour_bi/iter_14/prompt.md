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
    APPR_FULL = 0.10          # [m] approach zone fully open inside 10 cm of lip xy (parking pose of round 13: 5.4 cm)
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
    PRE_W = 3.0               # pre-tilt (0 -> limit - 10 deg) in the approach zone: cannot spill, cannot latch
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
    FOREIGN_W_HELD = 1.0      # ... and while it holds / carries it
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
    bring_together = BRING_W * src_carry * not_nested_f * clean * src_up * torch.exp(-BT_K * lip_off) \
        * z_ok * rcv_up_soft
    stack_gate = carry_both * not_nested_f * clear * clean
    aim_xy = torch.exp(-AIM_K * lip_off)
    aim_pose = aim_xy * z_ok
    aim = 3.0 * stack_gate * grip * aim_pose * rcv_up_soft

    # ------------------------------------------------------------------ stage 5: pre-tilt, release tilt, pour pose
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    # segment 1: 0 -> theta_pre, open in the whole approach zone (gradient exists at the round-13 parking pose).
    # No contact gate here: the additive contact penalties still charge every touch.
    pre_frac = torch.clamp(theta_eff / (theta_pre + 1e-3), 0.0, 1.0)
    pretilt = PRE_W * carry_both * not_nested_f * grip * beads_left * approach_gate * hover_wide \
        * pre_frac * rcv_up_soft
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
bead/in_target: [0, 0.000127, 0.000234, 2.71e-05, 0, 2.44e-05, 2.03e-05, 5.43e-05, 2.44e-05, 0]  min 0 · mean 5.86e-05 · max 0.000287
bead/spill: [0.00124, 0.0132, 0.0126, 0.0157, 0.0138, 0.0162, 0.0166, 0.0153, 0.0172, 0.0128]  min 0.00124 · mean 0.0143 · max 0.0181
done/drop: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.18e-05 · max 0.000244
episode_lengths/step: [75.9, 702, 832, 812, 831, 835, 869, 856, 888, 869]  min 75.9 · mean 811 · max 899
reward/aim: [0, 1.73, 1.14, 2.16, 1.84, 2.22, 1.63, 2.34, 2.26, 1.79]  min 0 · mean 1.82 · max 2.47
reward/approach_rcv: [0.674, 0.702, 0.671, 0.695, 0.676, 0.684, 0.683, 0.689, 0.685, 0.673]  min 0.565 · mean 0.68 · max 0.705
reward/approach_src: [0.667, 0.701, 0.657, 0.702, 0.681, 0.693, 0.679, 0.692, 0.685, 0.669]  min 0.561 · mean 0.68 · max 0.704
reward/both_grasped: [0, 0.962, 0.711, 0.937, 0.839, 0.918, 0.897, 0.895, 0.905, 0.692]  min 0 · mean 0.832 · max 0.988
reward/both_lifted: [0, 0.944, 0.701, 0.893, 0.828, 0.915, 0.892, 0.893, 0.903, 0.689]  min 0 · mean 0.817 · max 0.987
reward/bring_together: [0, 1.46, 1.06, 1.71, 1.49, 1.69, 1.36, 1.7, 1.66, 1.29]  min 0 · mean 1.42 · max 1.76
reward/closing_speed: [0, -0.000886, -0.00418, -0.00199, -0.00133, -0.00142, -0.00198, -0.00141, -0.00108, -0.00111]  min -0.0142 · mean -0.00169 · max 0
reward/converge: [0, 2.63, 1.97, 2.7, 2.43, 2.7, 2.48, 2.72, 2.68, 2.06]  min 0 · mean 2.38 · max 2.78
reward/cup_contact: [0, -0.0139, -0.0928, -0.0329, -0.0579, -0.0246, -0.0312, -0.0155, -0.0252, -0.0168]  min -0.193 · mean -0.0334 · max 0
reward/drop: [0, -0.000244, -0.000488, -0.000854, -0.000244, -0.000488, 0, -0.000122, -0.000488, -0.000366]  min -0.00122 · mean -0.000281 · max 0
reward/grasp_rcv: [0.111, 1.55, 1.19, 1.52, 1.36, 1.48, 1.45, 1.45, 1.46, 1.14]  min 0.111 · mean 1.36 · max 1.58
reward/grasp_src: [0.203, 1.57, 1.22, 1.54, 1.4, 1.51, 1.46, 1.5, 1.48, 1.17]  min 0.196 · mean 1.39 · max 1.58
reward/hand_foreign_rcv: [-0.0798, -0.0114, -0.0348, -0.0126, -0.0203, -0.0145, -0.014, -0.00882, -0.0148, -0.0216]  min -0.162 · mean -0.0184 · max -0.00273
reward/hand_foreign_src: [-0.269, -0.0166, -0.105, -0.0189, -0.0518, -0.0237, -0.037, -0.0226, -0.0303, -0.0647]  min -0.269 · mean -0.0395 · max -0.0049
reward/hand_rate: [-0.0091, -0.0218, -0.0196, -0.0223, -0.0197, -0.0198, -0.0188, -0.02, -0.0188, -0.0192]  min -0.0248 · mean -0.02 · max -0.0091
reward/hold_rcv: [0, 1.41e-05, 1.11e-05, 5.61e-06, 0, 4.07e-06, 1.03e-07, 2.06e-05, 0, 0]  min 0 · mean 6.89e-06 · max 4.57e-05
reward/hold_src: [0, 4.23e-05, 7.12e-05, 0, 0, 7.22e-06, 8.21e-06, 4.13e-05, 0, 0]  min 0 · mean 2e-05 · max 0.000119
reward/lift_rcv: [0.000289, 1.42, 1.22, 1.26, 1.3, 1.55, 1.64, 1.68, 1.65, 1.2]  min 0.000289 · mean 1.36 · max 1.9
reward/lift_src: [0.00162, 1.66, 1.09, 1.61, 1.41, 1.65, 1.45, 1.72, 1.66, 1.27]  min 0.00162 · mean 1.44 · max 1.77
reward/meet_rcv: [0, 0.407, 0.317, 0.372, 0.36, 0.407, 0.47, 0.42, 0.419, 0.302]  min 0 · mean 0.368 · max 0.571
reward/meet_src: [0, 0.25, 0.233, 0.247, 0.254, 0.27, 0.304, 0.27, 0.28, 0.201]  min 0 · mean 0.241 · max 0.352
reward/nested: [0, -0.00256, -0.00757, -0.00293, -0.00488, -0.00415, -0.00354, -0.00183, -0.00159, -0.00195]  min -0.0405 · mean -0.00348 · max 0
reward/palm_rate_rcv: [-0.0149, -0.0583, -0.0416, -0.0511, -0.0378, -0.0448, -0.0359, -0.0391, -0.034, -0.0276]  min -0.0671 · mean -0.0407 · max -0.0149
reward/palm_rate_src: [-0.0191, -0.0463, -0.0355, -0.044, -0.0354, -0.0417, -0.0336, -0.0388, -0.0321, -0.0265]  min -0.0521 · mean -0.0364 · max -0.0165
reward/palm_sat_rcv: [-0.0916, -0.0601, -0.0571, -0.0554, -0.0504, -0.052, -0.0469, -0.0532, -0.0509, -0.065]  min -0.0916 · mean -0.0558 · max -0.044
reward/palm_sat_src: [-0.0542, -0.0633, -0.0586, -0.0596, -0.0566, -0.0582, -0.0455, -0.0547, -0.0519, -0.0569]  min -0.095 · mean -0.0561 · max -0.0419
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.1e-05 · max 0.00552
reward/pour_pose: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.05e-08 · max 9.76e-06
reward/premature_latch: [0, -0.000244, 0, -0.000366, -0.000366, -0.00134, -0.000977, -0.000732, -0.00171, -0.000977]  min -0.00366 · mean -0.00076 · max 0
reward/pretilt: [0, 0.368, 0.269, 0.632, 0.661, 0.862, 0.879, 1.01, 1.13, 0.988]  min 0 · mean 0.729 · max 1.57
reward/raise_src: [0, 0.949, 0.689, 0.923, 0.828, 0.921, 0.881, 0.926, 0.913, 0.699]  min 0 · mean 0.825 · max 0.982
reward/rcv_still: [-6.1e-05, -0.143, -0.17, -0.145, -0.181, -0.157, -0.199, -0.147, -0.152, -0.106]  min -0.368 · mean -0.147 · max -6.1e-05
reward/reach_rcv: [0.499, 0.499, 0.495, 0.499, 0.497, 0.497, 0.498, 0.498, 0.498, 0.493]  min 0.453 · mean 0.496 · max 0.5
reward/reach_src: [0.499, 0.499, 0.494, 0.499, 0.497, 0.497, 0.497, 0.498, 0.497, 0.492]  min 0.447 · mean 0.495 · max 0.5
reward/rot_speed: [-0.0264, -0.0195, -0.0405, -0.0308, -0.0413, -0.0264, -0.0409, -0.0252, -0.0331, -0.0307]  min -0.145 · mean -0.0328 · max -0.00468
reward/spill_delta: [0, 0, -0.00296, 0, 0, -0.00174, 0, -0.00153, -0.00378, -0.00122]  min -0.0361 · mean -0.00147 · max 0
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 5.81e-11, 8.37e-06, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.47e-07 · max 5.94e-05
reward/total: [2.15, 19.5, 14.5, 20, 18.1, 20.6, 19.1, 21.1, 20.9, 16.5]  min 2.15 · mean 18.2 · max 21.7
reward/upright_rcv: [-0.00021, -0.0245, -0.0327, -0.0202, -0.0234, -0.0225, -0.0469, -0.016, -0.0135, -0.0135]  min -0.271 · mean -0.0251 · max -0.00021
reward/upright_src: [-0.000814, -0.00379, -0.00256, -0.00287, -0.00156, -0.00787, -0.00225, -0.00154, -0.00253, -0.00326]  min -0.0111 · mean -0.00285 · max 0
reward/wrap_rcv: [0.000797, 0.409, 0.321, 0.407, 0.347, 0.393, 0.373, 0.375, 0.392, 0.293]  min 0.000797 · mean 0.357 · max 0.446
reward/wrap_src: [0.073, 1.26, 0.816, 1.24, 1.06, 1.24, 1.14, 1.29, 1.25, 0.993]  min 0.073 · mean 1.11 · max 1.31
rewards/step: [63.3, 1.1e+04, 1.43e+04, 1.41e+04, 1.53e+04, 1.5e+04, 1.6e+04, 1.61e+04, 1.7e+04, 1.66e+04]  min 63.3 · mean 1.46e+04 · max 1.75e+04
task/aim_dist: [0.265, 0.297, 0.183, 0.181, 0.173, 0.171, 0.157, 0.166, 0.164, 0.174]  min 0.134 · mean 0.188 · max 0.401
task/cup_collision_rate: [0, 0.0156, 0.116, 0.0449, 0.0732, 0.0276, 0.0366, 0.0178, 0.0286, 0.0198]  min 0 · mean 0.0402 · max 0.228
task/cups_center_dist: [0.289, 0.303, 0.189, 0.191, 0.186, 0.188, 0.181, 0.188, 0.187, 0.202]  min 0.126 · mean 0.203 · max 0.422
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0.00513, 0.0151, 0.00586, 0.00977, 0.0083, 0.00708, 0.00366, 0.00317, 0.00391]  min 0 · mean 0.00696 · max 0.0811
task/rcv_cup_lift: [0.00468, 0.137, 0.0743, 0.0681, 0.0697, 0.0832, 0.0895, 0.0919, 0.0871, 0.0648]  min 0.00468 · mean 0.0769 · max 0.154
task/rcv_grasped: [0.00269, 0.966, 0.72, 0.945, 0.849, 0.928, 0.905, 0.899, 0.914, 0.698]  min 0.00269 · mean 0.841 · max 0.994
task/rcv_hand_foreign_rate: [0.431, 0.0269, 0.13, 0.0271, 0.0649, 0.033, 0.0479, 0.0254, 0.0366, 0.0854]  min 0.00879 · mean 0.0582 · max 0.578
task/src_cup_lift: [0.00316, 0.236, 0.175, 0.228, 0.207, 0.233, 0.21, 0.238, 0.23, 0.174]  min 0.00316 · mean 0.206 · max 0.314
task/src_grasped: [0.0986, 0.988, 0.76, 0.964, 0.873, 0.947, 0.929, 0.946, 0.93, 0.731]  min 0.0935 · mean 0.869 · max 0.993
task/src_hand_foreign_rate: [0.846, 0.0298, 0.275, 0.0461, 0.136, 0.0542, 0.0942, 0.0608, 0.073, 0.195]  min 0.0132 · mean 0.102 · max 0.846
task/src_tilt_deg: [7.76, 8.18, 10.5, 13.1, 15.4, 17.5, 19.5, 19.3, 22.1, 21.8]  min 1.16 · mean 15.9 · max 29.9
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Round 14 (iter_13, t2r_i13, 4096 env, warm-started from the round-13 checkpoint, 640 epochs / 4.7 h, ADR level 0) — measured facts only, environment unchanged from the round-14 description (fill-scaled premature_tilt_limit, pour lip point). Grasp and lift held all round: task/src_grasped 0.83, task/rcv_grasped 0.82 (both were 0.91-0.93 at epoch 365), src cup lift 0.209 m, rcv cup lift 0.070 m. episode_success 0.0, bead/in_target 0.0 for the whole round, bead/spill 0.016.

Approach and tilt over time (env-mean values at tick checks, epoch: task/pour_lip_dist / task/cups_center_dist / task/src_tilt_deg): 69: 0.108 / 0.233 / 7.7 deg; 141: 0.044 / 0.195 / 9.6 deg; 212: 0.051 / 0.197 / 13.3 deg; 365: 0.091 / 0.249 / 18.7 deg; 640: 0.099 / 0.239 / 26.0 deg (max of the logged mean 29.9 deg). task/aim_dist ended at 0.215 m. So the lip first came inside the receiver mouth radius band (< 0.081 m) around epoch 140-210, then the cups moved apart again while the source tilt kept rising.

Tilt-related terms: reward/pretilt rose 0 -> 1.27 (max 1.57). reward/tilt, reward/pour_pose, reward/hold_src, reward/hold_rcv, reward/success were exactly 0.0 for all 640 epochs; reward/pour_delta max 0.005. In iter_13 code, `tilt` and `pour_pose` pay only when the toward-receiver tilt exceeds premature_tilt_limit - 10 deg (theta_pre; task/tilt_limit_deg averaged 60.2, so about 50 deg) and the lip is within 0.07 m of the receiver mouth in xy; `pretilt` pays 0 -> theta_pre linearly inside the approach zone. The logged mean source tilt never reached 30 deg.

Safety/quality indicators: task/premature_tilt_rate 0.002 (max 0.007), task/tilt_far_deg 18.7, task/nested_rate 0.007, task/cup_collision_rate 0.037 (0.059 at epoch 212, max 0.228 early), hand_foreign src/rcv 0.132/0.081 (rising from 0.044/0.033 at epoch 141), task/rcv_tilt_deg 5.6, task/rcv_palm_speed 0.36 m/s, bead/fill_level 0.76, dr/bead_active_hi 12. reward/rcv_still -0.13, reward/cup_contact -0.03.

Best so far remains iter_05 (t2r_i05, episode_success 0.81, in_target 0.78) but it was trained in the round-3 environment (12 mm beads, fixed volume, no premature-tilt latch, 42-dim hand action); its numbers are not comparable to this environment.
