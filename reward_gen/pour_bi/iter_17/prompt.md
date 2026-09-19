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
    cup_hit: torch.Tensor               # (N,) bool 에피소드 래치 — hold 이후 두 컵이 한 번이라도 부딪혔다(cup_cup_force > 0.5 N). 성공 무효
    pour_dir_xy: torch.Tensor           # (N,2) 붓는 방향 d̂(소스 컵 원점→리시버 입구, 수평 단위벡터). x>0 = 몸 바깥, 로봇 몸통은 −x. x > 0.3 이면 성공 무효
    rcv_side_margin: torch.Tensor       # (N,) 리시버 컵이 자기 팔 쪽으로 중심선에서 떨어진 거리 [m] (스폰 ≈ +0.16). < 0.03 이면(상대 팔 쪽으로 넘어감) 성공 무효
    src_wrap_count: torch.Tensor        # (N,) 소스 손에서 중간/원위 마디가 컵에 닿은 손가락 수 [0,5] — 팁만 닿은 손가락은 0(손끝 집기 ≠ 감싸 쥐기)
    rcv_wrap_count: torch.Tensor        # (N,) 리시버 손의 같은 값
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
7. `ctx.success` is computed by the environment (enough beads in the receiver cup, little spill, cups close together, the receiver cup held nearly upright — `ctx.rcv_cup_tilt` at most 20° — the cups NOT nested, and NO premature tilt: `ctx.premature_tilt` latches for the rest of the episode as soon as the GRASPED source cup's tilt `ctx.src_cup_tilt` (any direction) exceeds `ctx.premature_tilt_limit` — the fill-dependent release tilt minus 20°, i.e. 52° for a full cup and 69° for a half-full one — while `ctx.src_pour_lip_pos` is still more than 8.1 cm (xy) from the receiver's mouth centre (a cup knocked over without being grasped does not latch), and a latched episode can never succeed). Below that limit the source cup may be tilted anywhere, including while it is still approaching the receiver. You may add a bonus on it but you cannot redefine it. Beads only count as "in the receiver" once they have LEFT the source cup — pushing the source cup into the receiver cup (`ctx.cups_nested`) transfers nothing and is never a success; the beads must fall out of the tilted source cup through the air. Three more conditions void success: `ctx.cup_hit` (episode latch: the two cups touched each other with more than 0.5 N at any time after the settle phase — keep the cups apart on the table, while lifting and while pouring), the pouring direction `ctx.pour_dir_xy[:, 0] > 0.3` (the robot body is at -x; the source cup must pour sideways or toward the body, not away from it, so the receiver mouth must NOT be placed further from the body than the source cup), and `ctx.rcv_side_margin < 0.03` (the receiver cup must stay on its own arm's side of the centre line, about where it spawned at +0.16; the SOURCE arm brings its cup over). `ctx.src_wrap_count` / `ctx.rcv_wrap_count` count the fingers whose middle or distal link touches the cup (fingertip-only contact counts 0): an enveloping grasp has 3 or more, a fingertip pinch has 0-1 and collapses when the cup is tilted.
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


# --- geometry constant: the pouring lip sits this far from the rim centre (see ctx doc, r = 4.1 cm) ----
_LIP_R = 0.041


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
    LIFT_TARGET_SRC = 0.28    # iter_16: the lip must clear the receiver rim at ~80 deg -> ask for more lift
    LIFT_TARGET_RCV = 0.10
    RCV_LIFT_FREE = 0.15
    RCV_LIFT_SIGMA = 0.08
    LIP_DEADBAND = 0.02       # [m]
    APPR_FULL = 0.08          # [m] approach zone full up to the latch radius
    APPR_ZERO = 0.24
    # iter_16: widen the release gate a little (round 16 parked the lip at 5.2 cm, `safe` only ~0.8).
    SAFE_FULL = 0.050         # [m]
    SAFE_ZERO = 0.075         # [m] still 0.6 cm inside the env latch radius 8.1 cm
    LIP_Z_LO = 0.0
    LIP_Z_HI = 0.02
    # iter_16 (main fix a): the height budget GROWS with the tilt. At 80 deg the lip is r*sin(theta) = 4 cm
    # below the rim centre, so the cup must ride ~6 cm higher than at 45 deg; the old fixed decay taxed
    # exactly that motion and made the release pose unreachable (pour_pose was exactly 0.0 for 812 epochs).
    LIP_Z_FREE0 = 0.07
    LIP_Z_SIGMA = 0.10
    LIP_ABOVE_FLOOR = 0.35    # soft gate: tilt income no longer collapses the instant the lip dips
    HOVER_LO = 0.06
    HOVER_HI = 0.11
    HOVER_FREE0 = 0.10
    HOVER_SIGMA = 0.08
    RAISE_LO = -0.04
    # iter_16: carry plateau roughly halved (was ~20 of the 22.9 total) so the pour channel can dominate
    RAISE_W = 0.8
    CONV_FAR = 0.40
    CONV_W = 2.0
    BRING_W = 1.5
    BT_K = 10.0
    AIM_K = 15.0
    AIM_W = 2.5
    MEET_RANGE = 0.30
    MEET_RCV_W = 0.5
    MEET_SRC_W = 0.5
    APPROACH_W = 0.6
    GRASP_W = 0.8
    WRAP_SRC_W = 1.0
    WRAP_RCV_W = 0.4
    LIFT_W = 1.5
    BOTH_W = 0.8
    SRC_TILT_FREE = 0.20
    PRE_MARGIN = 0.17         # [rad] theta_pre = limit - 10 deg
    SRC_TILT_SCALE = 0.20
    SRC_TILT_SIGMA = 0.25
    SRC_UP_W_TABLE = 1.0
    SRC_UP_W_CARRY = 1.5
    LATCH_W = 1.0             # iter_16: tilt is pushed much harder, so latching must cost more
    RELEASE_ABOVE_LIMIT = math.radians(20.0)
    BAND_BELOW = 0.10
    BAND_ABOVE = 0.40
    # iter_16 (main fix b): split the tilt income again, with the weight on the FAR side. Round 16 paid
    # ~5.4/rad uniformly, so the last 35 deg (45 -> 80) was worth only ~+3.3 against a ~20 carry plateau.
    # Now 0 -> theta_pre is worth 6 and theta_pre -> release another 20, i.e. ~+7.6 for the last 30 deg,
    # plus a 2.0 overshoot ramp so the policy keeps turning once beads start falling.
    TILT_PRE_W = 6.0
    TILT_REL_W = 20.0
    TILT_OVER_W = 2.0
    TILT_OVER_RANGE = 0.45    # [rad] extra rotation past the release angle that still earns
    PRE_NEAR_FLOOR = 0.25
    PRE_NEAR_K = 10.0
    POSE_W = 8.0              # iter_16: was 3.0 and logged exactly 0.0 — the pose it asked for was unreachable
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
    GRIP_FLOOR = 0.5          # iter_16: 0.3 -> 0.5; wrap quality dips while the cup rotates and was taxing
                              # every carry income at exactly the moment we want rotation
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
    POST_POUR_FRAC = 0.5      # iter_16: task succeeds at half the beads, so "after the pour" starts there
    SETTLE_FRAC = 0.05
    POUR_DELTA_W = 120.0      # increments stay far above the level term below
    TRANSFER_W = 5.0          # iter_16: new, level term — holding a partly filled receiver is worth something
    SPILL_W = 50.0
    REACH_FAR = 0.26
    REACH_NEAR = 0.14
    REACH_W = 0.3
    CLOSE_W = 0.8
    CLOSE_NORM = 0.35
    PALM_RATE_W = 0.01
    HAND_RATE_W = 0.005
    SAT_LO = 0.7
    SAT_W = 0.15
    SUCCESS_W = 20.0          # iter_16: 10 -> 20, the plateau shrank so the terminal payoff must grow

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
    # held state = grasp flag OR carried, so a thumb contact that flickers while the cup rotates is not fatal
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
    safe = 1.0 - _smoothstep((lip_xy - SAFE_FULL) / (SAFE_ZERO - SAFE_FULL))   # lip close enough to release

    # ------------------------------------------------------------------ tilt angles
    tilt_mag = ctx.src_cup_tilt
    limit = ctx.premature_tilt_limit.to(dt)
    release_tilt = limit + RELEASE_ABOVE_LIMIT
    theta_pre = torch.clamp(limit - PRE_MARGIN, min=0.0)
    to_rcv = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    to_rcv_n = to_rcv / (torch.norm(to_rcv, dim=-1, keepdim=True) + 1e-6)
    a_own = torch.sum(ctx.src_cup_up[:, :2] * to_rcv_n, dim=-1)
    theta_own = torch.atan2(a_own, ctx.src_cup_up[:, 2])
    theta = torch.maximum(ctx.src_tilt_toward_rcv.to(dt), theta_own)
    theta_eff = torch.minimum(torch.clamp(theta, min=0.0), tilt_mag)

    # ------------------------------------------------------------------ source tilt corridor
    # allowed: 0.20 rad far away -> theta_pre in the approach zone -> unlimited once the lip is within 5 cm
    allowed = SRC_TILT_FREE + approach_gate * torch.clamp(theta_pre - SRC_TILT_FREE, min=0.0)
    tilt_over = torch.clamp(tilt_mag - allowed, min=0.0) * (1.0 - safe)
    src_up = torch.exp(-(tilt_over / SRC_TILT_SIGMA) ** 2)

    # ------------------------------------------------------------------ stage 1: approach + reach
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = APPROACH_W * torch.exp(-4.0 * d_src)
    approach_rcv = APPROACH_W * torch.exp(-4.0 * d_rcv)
    reach_src = REACH_W * _smoothstep((REACH_FAR - d_src) / (REACH_FAR - REACH_NEAR))
    reach_rcv = REACH_W * _smoothstep((REACH_FAR - d_rcv) / (REACH_FAR - REACH_NEAR))

    # ------------------------------------------------------------------ stage 2: grasp (held state)
    prox_src = torch.exp(-6.0 * d_src)
    prox_rcv = torch.exp(-6.0 * d_rcv)
    clos_src = torch.clamp(ctx.src_hand_closure / CLOSE_NORM, 0.0, 1.0)
    clos_rcv = torch.clamp(ctx.rcv_hand_closure / CLOSE_NORM, 0.0, 1.0)
    grasp_src = GRASP_W * held_src + CLOSE_W * prox_src * clos_src
    grasp_rcv = GRASP_W * held_rcv + CLOSE_W * prox_rcv * clos_rcv
    both_grasped = BOTH_W * held_src * held_rcv

    wq_src = _wrap_quality(ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                           ctx.src_palm_axes, ctx.src_cup_up, ctx.src_cup_pos, ctx.src_palm_pos)
    wq_rcv = _wrap_quality(ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                           ctx.rcv_palm_axes, ctx.rcv_cup_up, ctx.rcv_cup_pos, ctx.rcv_palm_pos)
    wrap_src = WRAP_SRC_W * held_src * wq_src   # wq still falls if fingers slip, so wrap keeps its gradient
    wrap_rcv = WRAP_RCV_W * held_rcv * wq_rcv
    grip = GRIP_FLOOR + (1.0 - GRIP_FLOOR) * wq_src

    # ------------------------------------------------------------------ stage 3: lift
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET_SRC, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET_RCV, 0.0, 1.0)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    lift_src = LIFT_W * held_src * grip * lift_frac_src * src_up
    lift_rcv = LIFT_W * held_rcv * lift_frac_rcv * rcv_band * (0.5 + 0.5 * rcv_up)
    both_lifted = BOTH_W * carry_both

    # ------------------------------------------------------------------ heights (budget grows with tilt)
    # iter_16 main fix (a): every height allowance is shifted up by the lip drop r*sin(theta), so rotating
    # and raising together is rewarded instead of taxed.
    sin_t = torch.sin(torch.clamp(theta_eff, 0.0, 0.5 * math.pi))
    hover_free = HOVER_FREE0 + _LIP_R * sin_t
    lip_free = LIP_Z_FREE0 + _LIP_R * sin_t
    dz_eq = ctx.src_cup_pos[:, 2] + ctx.cup_mouth_z - ctx.rcv_cup_mouth_pos[:, 2]
    hover_decay = torch.exp(-(torch.clamp(dz_eq - hover_free, min=0.0) / HOVER_SIGMA) ** 2)
    hover = _smoothstep((dz_eq - HOVER_LO) / (HOVER_HI - HOVER_LO)) * hover_decay
    hover_wide = _smoothstep((dz_eq - RAISE_LO) / (HOVER_HI - RAISE_LO)) * hover_decay
    lip_above_raw = torch.clamp((lip_dz - LIP_Z_LO) / (LIP_Z_HI - LIP_Z_LO), 0.0, 1.0)
    lip_high = torch.exp(-(torch.clamp(lip_dz - lip_free, min=0.0) / LIP_Z_SIGMA) ** 2)
    # soft version used to GATE income: keeps a floor so the tilt reward survives the moment the lip dips
    lip_above = LIP_ABOVE_FLOOR + (1.0 - LIP_ABOVE_FLOOR) * lip_above_raw
    aim_z = lip_above_raw * lip_high
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
    aim = AIM_W * stack_gate * grip * aim_pose * rcv_up_soft

    # ------------------------------------------------------------------ stage 5: tilt + pour pose
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    near_pre = PRE_NEAR_FLOOR + (1.0 - PRE_NEAR_FLOOR) * torch.exp(-PRE_NEAR_K * lip_off)
    tgt = torch.clamp(release_tilt, min=0.5)
    tilt_gate = carry_both * not_nested_f * clean_soft * grip * beads_left * rcv_up_soft
    # part below theta_pre: cannot spill or latch, open in the whole approach zone. Weight 6.
    part_pre = TILT_PRE_W * (torch.minimum(theta_eff, theta_pre) / tgt) * near_pre * approach_gate * hover_wide
    # part above theta_pre: only with the lip inside the safe radius. Weight 20 — this is the stage that
    # round 16 never entered, and it must outweigh the risk of losing the carry income.
    part_rel = TILT_REL_W * (torch.clamp(torch.minimum(theta_eff, tgt) - theta_pre, min=0.0) / tgt) \
        * safe * lip_above
    # keep turning past the release angle (beads keep coming out)
    part_over = TILT_OVER_W * torch.clamp((theta_eff - tgt) / TILT_OVER_RANGE, 0.0, 1.0) * safe * lip_above
    tilt = tilt_gate * (part_pre + part_rel + part_over)

    pour_band = _smoothstep((theta_eff - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    # iter_16: gates softened (aim_xy + lip_above instead of aim_pose + lip_above_raw) so the release pose is
    # actually attainable; weight 8 makes holding it clearly better than parking at 45 deg.
    pour_pose = POSE_W * stack_gate * grip * beads_left * safe * lip_above * aim_xy * pour_band * rcv_up

    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    rot_speed = -0.3 * held_src * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer
    pour_delta = POUR_DELTA_W * torch.clamp(ctx.d_in_target, -1.0, 1.0)   # increments dominate the level term
    transferred = TRANSFER_W * torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0) * not_nested_f * rcv_up
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
    success = SUCCESS_W * ctx.success.to(dt) * clear * clean * (0.6 + 0.4 * held_src * wq_src) \
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
        "transferred": transferred,
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

adr/progress: [0, 0, 0.2, 0.6, 1, 1, 1, 1, 1, 1]  min 0 · mean 0.725 · max 1
bead/in_target: [0, 0.0285, 0.354, 0.393, 0.503, 0.563, 0.589, 0.602, 0.62, 0.607]  min 0 · mean 0.461 · max 0.63
bead/spill: [0.00106, 0.0267, 0.0504, 0.0362, 0.056, 0.051, 0.0411, 0.0454, 0.0406, 0.0467]  min 0.00106 · mean 0.0411 · max 0.0748
done/drop: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.88e-05 · max 0.000732
episode_lengths/step: [77.7, 889, 840, 864, 840, 873, 827, 876, 874, 867]  min 77.7 · mean 854 · max 899
reward/aim: [0, 2.02, 1.24, 1.29, 1.29, 1.4, 1.41, 1.48, 1.51, 1.49]  min 0 · mean 1.47 · max 2.27
reward/approach_rcv: [0.391, 0.408, 0.39, 0.38, 0.38, 0.393, 0.392, 0.394, 0.398, 0.392]  min 0.328 · mean 0.392 · max 0.42
reward/approach_src: [0.392, 0.404, 0.382, 0.384, 0.375, 0.377, 0.384, 0.383, 0.387, 0.384]  min 0.324 · mean 0.385 · max 0.419
reward/both_grasped: [0.000586, 0.763, 0.717, 0.687, 0.669, 0.685, 0.681, 0.695, 0.69, 0.681]  min 0.000586 · mean 0.687 · max 0.798
reward/both_lifted: [0, 0.762, 0.716, 0.687, 0.658, 0.673, 0.668, 0.678, 0.68, 0.672]  min 0 · mean 0.68 · max 0.798
reward/bring_together: [0, 1.34, 0.982, 0.992, 0.989, 1.05, 1.05, 1.09, 1.09, 1.08]  min 0 · mean 1.07 · max 1.42
reward/closing_speed: [0, -0.00101, -0.00413, -0.00401, -0.00469, -0.00189, -0.00269, -0.00234, -0.00149, -0.00213]  min -0.025 · mean -0.00248 · max 0
reward/converge: [0, 1.8, 1.53, 1.49, 1.38, 1.42, 1.41, 1.39, 1.41, 1.39]  min 0 · mean 1.46 · max 1.95
reward/cup_contact: [0, -0.0166, -0.103, -0.0901, -0.0746, -0.0584, -0.0511, -0.0484, -0.0474, -0.0459]  min -0.471 · mean -0.0532 · max 0
reward/drop: [0, -0.000244, -0.000732, -0.0011, -0.00122, -0.000488, -0.000977, -0.000977, -0.000977, -0.00061]  min -0.00256 · mean -0.000803 · max 0
reward/grasp_rcv: [0.0878, 1.2, 1.11, 1.06, 1.05, 1.08, 1.08, 1.1, 1.1, 1.07]  min 0.0878 · mean 1.08 · max 1.26
reward/grasp_src: [0.173, 1.2, 1.1, 1.07, 1.04, 1.06, 1.07, 1.08, 1.08, 1.07]  min 0.102 · mean 1.07 · max 1.26
reward/hand_foreign_rcv: [-0.0253, -0.0164, -0.0789, -0.087, -0.0971, -0.0549, -0.0618, -0.0654, -0.0696, -0.0632]  min -0.529 · mean -0.0661 · max -0.00124
reward/hand_foreign_src: [-0.291, -0.0545, -0.218, -0.227, -0.194, -0.185, -0.195, -0.177, -0.19, -0.171]  min -0.357 · mean -0.173 · max -0.00563
reward/hand_rate: [-0.0192, -0.0196, -0.02, -0.0222, -0.0228, -0.0213, -0.0214, -0.0215, -0.0224, -0.0232]  min -0.025 · mean -0.0217 · max -0.0166
reward/hold_rcv: [0, 0.0193, 0.205, 0.198, 0.23, 0.259, 0.268, 0.267, 0.265, 0.251]  min 0 · mean 0.212 · max 0.295
reward/hold_src: [0, 0.0504, 0.537, 0.542, 0.608, 0.69, 0.729, 0.752, 0.766, 0.756]  min 0 · mean 0.583 · max 0.776
reward/lift_rcv: [0.000259, 1.23, 1.18, 1.16, 0.97, 0.951, 0.905, 0.938, 0.923, 1.02]  min 0.000259 · mean 1.02 · max 1.48
reward/lift_src: [0.00124, 1.09, 0.916, 0.922, 0.862, 0.894, 0.89, 0.924, 0.923, 0.965]  min 0.00124 · mean 0.921 · max 1.22
reward/meet_rcv: [0, 0.185, 0.176, 0.161, 0.182, 0.205, 0.205, 0.188, 0.2, 0.201]  min 0 · mean 0.19 · max 0.251
reward/meet_src: [0, 0.0372, 0.0604, 0.0494, 0.0766, 0.0839, 0.0736, 0.0635, 0.0692, 0.0915]  min 0 · mean 0.0684 · max 0.12
reward/nested: [0, -0.00159, -0.00574, -0.00818, -0.0135, -0.00659, -0.00671, -0.00562, -0.00818, -0.00696]  min -0.0739 · mean -0.0065 · max 0
reward/palm_rate_rcv: [-0.0132, -0.0222, -0.0123, -0.0117, -0.0156, -0.0132, -0.0117, -0.0128, -0.013, -0.0141]  min -0.0398 · mean -0.015 · max -0.00957
reward/palm_rate_src: [-0.00876, -0.00506, -0.00556, -0.00517, -0.00497, -0.00573, -0.00448, -0.00605, -0.00719, -0.0074]  min -0.0303 · mean -0.00597 · max -0.0035
reward/palm_sat_rcv: [-0.0694, -0.0501, -0.0703, -0.0737, -0.0754, -0.0708, -0.0724, -0.0715, -0.0726, -0.0757]  min -0.0972 · mean -0.0695 · max -0.0421
reward/palm_sat_src: [-0.0406, -0.1, -0.0893, -0.0839, -0.0814, -0.0847, -0.0896, -0.0882, -0.0874, -0.0829]  min -0.126 · mean -0.0856 · max -0.0221
reward/pour_delta: [0, 0.0135, 0.1, 0.0631, 0.0913, 0.115, 0.0981, 0.146, 0.109, 0.119]  min -0.00488 · mean 0.095 · max 0.215
reward/pour_pose: [0, 0.541, 1.19, 1.45, 2.06, 2.62, 2.61, 2.86, 2.97, 2.99]  min 0 · mean 2.08 · max 3.22
reward/premature_latch: [0, -0.0518, -0.0925, -0.0764, -0.155, -0.0918, -0.0625, -0.0603, -0.0356, -0.0549]  min -0.199 · mean -0.0709 · max 0
reward/raise_src: [0, 0.725, 0.633, 0.607, 0.563, 0.577, 0.575, 0.565, 0.568, 0.56]  min 0 · mean 0.591 · max 0.787
reward/rcv_still: [0, -0.21, -0.248, -0.261, -0.307, -0.279, -0.248, -0.253, -0.218, -0.227]  min -0.327 · mean -0.243 · max 0
reward/reach_rcv: [0.3, 0.298, 0.297, 0.294, 0.294, 0.297, 0.295, 0.297, 0.296, 0.296]  min 0.268 · mean 0.296 · max 0.3
reward/reach_src: [0.3, 0.299, 0.297, 0.296, 0.295, 0.295, 0.296, 0.296, 0.297, 0.296]  min 0.264 · mean 0.296 · max 0.3
reward/rot_speed: [-0.0329, -0.0292, -0.0559, -0.0492, -0.048, -0.0442, -0.0423, -0.0405, -0.0397, -0.0429]  min -0.253 · mean -0.0443 · max -0.00267
reward/spill_delta: [0, -0.00102, -0.0108, -0.00634, -0.00475, -0.00658, -0.00416, -0.00401, -0.00646, -0.00285]  min -0.0413 · mean -0.00466 · max 0
reward/success: [0, 0.0287, 4.84, 5.98, 7.02, 8.39, 9.36, 9.74, 10.4, 9.94]  min 0 · mean 7.15 · max 10.5
reward/tilt: [0, 7.53, 5.04, 5.42, 5.64, 6.21, 6.44, 6.73, 6.79, 6.71]  min 0 · mean 6.11 · max 7.53
reward/total: [1.19, 22.6, 25, 26.6, 28.7, 32.2, 33.6, 34.8, 35.8, 35.2]  min 1.19 · mean 30 · max 36.5
reward/transferred: [0, 0.142, 1.76, 1.93, 2.49, 2.78, 2.93, 2.98, 3.07, 3]  min 0 · mean 2.28 · max 3.13
reward/upright_rcv: [0, -0.0306, -0.114, -0.164, -0.112, -0.0881, -0.0942, -0.1, -0.116, -0.124]  min -0.682 · mean -0.101 · max 0
reward/upright_src: [-0.000157, -0.0172, -0.149, -0.0611, -0.0754, -0.0526, -0.0375, -0.0426, -0.0314, -0.0256]  min -0.413 · mean -0.0465 · max -0.000157
reward/wrap_rcv: [0.000553, 0.322, 0.278, 0.25, 0.25, 0.265, 0.266, 0.264, 0.259, 0.245]  min 0.000553 · mean 0.264 · max 0.363
reward/wrap_src: [0.0473, 0.787, 0.612, 0.574, 0.543, 0.573, 0.585, 0.599, 0.6, 0.592]  min 0.0263 · mean 0.604 · max 0.949
rewards/step: [32.6, 1.7e+04, 2.11e+04, 2.29e+04, 2.36e+04, 2.84e+04, 2.71e+04, 2.96e+04, 3.08e+04, 3.14e+04]  min 32.6 · mean 2.56e+04 · max 3.25e+04
task/aim_dist: [0.258, 0.088, 0.0907, 0.306, 0.111, 0.0986, 0.147, 0.0958, 0.0942, 0.096]  min 0.0798 · mean 0.111 · max 0.528
task/cup_collision_rate: [0, 0.019, 0.15, 0.124, 0.0884, 0.0735, 0.0615, 0.0579, 0.0564, 0.0544]  min 0 · mean 0.0665 · max 0.566
task/cups_center_dist: [0.284, 0.184, 0.185, 0.396, 0.203, 0.2, 0.252, 0.197, 0.198, 0.2]  min 0.127 · mean 0.207 · max 0.632
task/episode_success: [0, 0.00269, 0.331, 0.388, 0.443, 0.532, 0.575, 0.597, 0.632, 0.603]  min 0 · mean 0.445 · max 0.636
task/nested_rate: [0, 0.00317, 0.0115, 0.0164, 0.0271, 0.0132, 0.0134, 0.0112, 0.0164, 0.0139]  min 0 · mean 0.013 · max 0.148
task/rcv_cup_lift: [0.0057, 0.0851, 0.0883, 0.117, 0.0779, 0.0736, 0.0661, 0.069, 0.0675, 0.0814]  min 0.0035 · mean 0.0801 · max 0.252
task/rcv_grasped: [0.00293, 0.938, 0.847, 0.812, 0.808, 0.837, 0.832, 0.847, 0.836, 0.827]  min 0.00293 · mean 0.83 · max 0.988
task/rcv_hand_foreign_rate: [0.149, 0.0283, 0.112, 0.144, 0.152, 0.113, 0.111, 0.104, 0.121, 0.11]  min 0.0022 · mean 0.109 · max 0.597
task/src_cup_lift: [0.00252, 0.225, 0.213, 0.212, 0.203, 0.206, 0.216, 0.209, 0.207, 0.22]  min 0.00252 · mean 0.21 · max 0.342
task/src_grasped: [0.114, 0.949, 0.865, 0.831, 0.787, 0.833, 0.829, 0.837, 0.819, 0.819]  min 0.0359 · mean 0.829 · max 0.997
task/src_hand_foreign_rate: [0.783, 0.139, 0.346, 0.327, 0.281, 0.276, 0.277, 0.257, 0.265, 0.249]  min 0.0129 · mean 0.264 · max 0.783
task/src_tilt_deg: [5.56, 70.1, 72.3, 67.3, 71.3, 75.5, 75, 76.1, 76.4, 76.3]  min 2.82 · mean 71.1 · max 78.6
task/success_now: [0, 0.00269, 0.34, 0.396, 0.452, 0.544, 0.585, 0.606, 0.641, 0.618]  min 0 · mean 0.453 · max 0.645

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Round 17 (t2r_i16, 4096 env, HOLD run to epoch 5711, ADR 30/30): episode_success recent 0.61, in_target 0.61, src_tilt 76 deg. The numbers passed but the USER REJECTED the play video (09.20). Four defects, each measured on a 64-env deterministic trace (trace_i16_ep5600_adr30_64env.npz):
1. Cups collide after grasp: around step 98 the two cup origins are 0.082-0.099 m apart in all 64 envs — the policy pushes the cups together on the table before lifting.
2. Receiver (left) arm crosses far into the source arm's side: receiver cup y median -0.13 during the pour (spawn +0.16).
3. Grasp is a fingertip pinch, not an envelope, and collapses under tilt: on the receiver hand only the thumb has middle/distal-link contact; the four fingers touch with tips only (1-4 N).
4. Pour direction points away from the body: pour_dir_xy mean (0.90, 0.35) over the tilted segment, +x on every step (receiver placed ~10 cm further from the body than the source cup).
ENV CHANGE (commit 1ef50870): success is now void on ctx.cup_hit (latch, cup_cup_force > 0.5 N after settle), ctx.pour_dir_xy[:,0] > 0.3, ctx.rcv_side_margin < 0.03. New ctx fields: cup_hit, pour_dir_xy, rcv_side_margin, src_wrap_count, rcv_wrap_count.
REQUIRED in the new reward: (a) a penalty on cup_cup_force at all phases and a one-time charge on cup_hit; no term may pay for bringing the cups closer than ~0.12 m origin distance before both are lifted. (b) pay the receiver for holding its cup lifted near its own spawn xy (rcv_side_margin >= 0.10) and standing still; all approach travel must come from the source arm. (c) gate lift/carry/tilt income on wrap counts (smooth factor of wrap_count/3 for each hand) so a fingertip pinch earns little. (d) shape pour_dir_xy[:,0] toward <= 0 (source cup at larger x than, or level with, the receiver mouth); tilt income multiplied by a smooth factor that is ~0 at x >= 0.3. Keep the iter_16 tilt/pour_pose/pour_delta structure, which did produce pouring.
