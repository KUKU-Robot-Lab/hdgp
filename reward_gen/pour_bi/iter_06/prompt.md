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

    # ---- 충돌 (실기 안전 — 09.14) ---------------------------------------------------------
    cup_cup_force: torch.Tensor         # (N,) 두 컵이 서로 부딪히는 접촉력 [N] (0 = 안 닿음)
    src_hand_foreign_force: torch.Tensor  # (N,) 소스 손이 **자기 컵 외**(상대 손·상대 컵·테이블)에 닿는 힘 [N]
    rcv_hand_foreign_force: torch.Tensor  # (N,) 리시버 손이 자기 컵 외에 닿는 힘 [N]

    # ---- 과제 판정 / 시간 ------------------------------------------------------------
    cups_nested: torch.Tensor           # (N,) bool 두 컵 원점 거리 < 9 cm — 소스 컵이 리시버 컵에 끼워져 있음(붓기가 아니라 성공 무효)
    success: torch.Tensor               # (N,) bool 성공 조건 충족 (env 가 판정, 보상이 바꿀 수 없음; cups_nested 면 항상 False)
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
5. Cup tilt: `ctx.src_cup_tilt` is the angle between the cup's up axis and world +z (0 = upright, π/2 = horizontal). Beads leave the source cup once it is tilted past roughly 110° (1.9 rad) with its mouth over the receiver cup's mouth. `ctx.src_cup_mouth_pos` and `ctx.rcv_cup_mouth_pos` are the rim centres.
6. Bead bookkeeping: `bead_in_target_frac` rises as beads land in the receiver cup; `bead_spill_frac` counts beads lost outside both cups (permanent). `d_in_target` and `d_spill` are this step's increments — reward INCREMENTS of beads transferred rather than the level, otherwise the policy is paid for standing still with a filled cup.
7. `ctx.success` is computed by the environment (enough beads in the receiver cup, little spill, cups close together, the receiver cup held nearly upright — `ctx.rcv_cup_tilt` at most 20° — and the cups NOT nested). You may add a bonus on it but you cannot redefine it. Beads only count as "in the receiver" once they have LEFT the source cup — pushing the source cup into the receiver cup (`ctx.cups_nested`) transfers nothing and is never a success; the beads must fall out of the tilted source cup through the air.
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
    PALM_SCALE = 2.0     # [N] palm force tanh scale: 2 N -> 0.76
    LEVEL_SIGMA = 0.05   # [m] cup origin within ~5 cm of palm height along the cup axis
    # iter_05: closure saturates at 0.35-0.43 on this cup under contact freeze whatever the grasp type
    # (operator correction). Old ramp 0.35-0.65 capped a real wrap at ~0.2 on this ingredient; 0.25-0.40
    # lets a real wrap score ~1. It no longer tries to separate hook from wrap (forces/palm/side do that).
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


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    N = ctx.src_palm_pos.shape[0]
    dev = ctx.src_palm_pos.device
    dt = ctx.src_palm_pos.dtype
    zeros = torch.zeros(N, device=dev, dtype=dt)

    # ------------------------------------------------------------------ constants
    LIFT_TARGET_SRC = 0.20    # [m] source cup must end up clearly higher than the receiver
    LIFT_TARGET_RCV = 0.10    # [m]
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
    TILT_FREE = 1.5           # [rad] unaimed tilt below 86 deg spills nothing -> free (was 0.8)
    ROT_SPEED_FREE = 1.5      # [rad/s] controlled pouring rotation is free; only violent flips are charged
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped / knocked off the table
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side
    NEAR_DIST = 0.30          # [m] cup-centre distance inside which approach speed is regulated
    FORCE_SCALE = 5.0         # [N] tanh scale for all contact penalties
    FOREIGN_DEADBAND = 1.0    # [N] incidental brushes are free (additive term)
    CLEAN_DEADBAND = 0.5      # [N] deadband of the multiplicative hand-foreign gate
    GRIP_FLOOR = 0.3          # pour/lift income still worth 30 % with a hook grasp, 100 % with a wrap
    MEET_RCV_FREE = 0.02      # [m] receiver cup may cross the spawn midline by 2 cm for free
    MEET_SRC_FREE = 0.05      # [m] source cup may cross the midline by 5 cm for free
    MEET_SCALE = 0.10         # [m] tanh scale of the crossing penalties

    # ------------------------------------------------------------------ clearance factors (unchanged — worked)
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_foreign = torch.maximum(ctx.src_hand_foreign_force, ctx.rcv_hand_foreign_force)
    clean = 1.0 - torch.tanh(torch.clamp(f_foreign - CLEAN_DEADBAND, min=0.0) / FORCE_SCALE)
    not_nested_f = (~ctx.cups_nested).to(dt)

    # ------------------------------------------------------------------ stage 1: approach (unchanged — solved)
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)

    # ------------------------------------------------------------------ stage 2: grasp (unchanged — solved)
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
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
    wrap_src = 1.5 * g_src * wq_src   # weights unchanged; the wrap is achieved (operator note 1)
    wrap_rcv = 0.5 * g_rcv * wq_rcv
    grip = GRIP_FLOOR + (1.0 - GRIP_FLOOR) * wq_src   # in [0.3,1]; multiplies lift_src / aim / tilt / pour_pose

    # ------------------------------------------------------------------ stage 3: lift (unchanged — solved)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET_SRC, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET_RCV, 0.0, 1.0)
    lift_src = 2.0 * g_src * grip * lift_frac_src
    lift_rcv = 2.0 * g_rcv * lift_frac_rcv

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
    # loose aim (new): 12 cm miss -> 0.74 (precise gives 0.50). Used only by the tilt-progress term so the
    # wrist can start rotating before the lip is perfectly centred; beads then judge the precision.
    aim_xy_loose = torch.exp(-3.0 * torch.clamp(lip_xy - LIP_DEADBAND_LOOSE, min=0.0))
    aim_loose = aim_xy_loose * aim_z

    # ------------------------------------------------------------------ stage 4: bring cups together (unchanged)
    bring_together = 1.0 * src_lifted_f * not_nested_f * clean * torch.exp(-3.0 * lip_xy) * aim_z
    stack_gate = both_lifted_f * not_nested_f * clear * clean
    aim = 3.0 * stack_gate * grip * aim_soft

    # ------------------------------------------------------------------ stage 5: tilt / pour pose (reworked)
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_rad = ctx.src_cup_tilt
    tilt_frac = torch.clamp(tilt_rad / TILT_TARGET, 0.0, 1.0)
    # weight 3 -> 4 and loose aim gate: slope per rad at today's pose roughly doubles (0.5 -> ~1.1 /step/rad),
    # still below aim (3.0, precise) so centring the lip is never traded for raw tilt.
    tilt = 4.0 * stack_gate * grip * beads_left * aim_loose * tilt_frac

    # direction (new): the cup's up axis, projected on the table plane, should point from the source cup toward
    # the receiver mouth — that is the side the lip must swing to. Signed: tilting away is charged.
    # Scaled by tilt amount so an upright cup (ill-defined horizontal axis) contributes nothing.
    to_rcv = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]                  # (N,2)
    to_rcv_n = to_rcv / (torch.norm(to_rcv, dim=-1, keepdim=True) + 1e-6)
    up_xy = ctx.src_cup_up[:, :2]                                                    # (N,2)
    up_xy_n = up_xy / (torch.norm(up_xy, dim=-1, keepdim=True) + 1e-6)
    dir_cos = torch.sum(to_rcv_n * up_xy_n, dim=-1)                                  # (N,) in [-1,1]
    tilt_amt = torch.clamp(tilt_rad / TILT_DIR_SCALE, 0.0, 1.0)
    # weight 1.0: small next to tilt/pour_pose, but a direct wrist-rotation gradient that the lip geometry
    # only provides through a 12-cm-wide exponential.
    tilt_dir = 1.0 * stack_gate * beads_left * tilt_amt * dir_cos

    # release-band bonus (new): extra slope exactly where beads start to leave (75-115 deg), tied to PRECISE aim.
    # weight 3.0: at the ideal pose tilt + pour_pose ~ 6/step vs ~1.2/step at today's 47 deg plateau.
    pour_band = _smoothstep((tilt_rad - POUR_TILT_LO) / (POUR_TILT_HI - POUR_TILT_LO))
    pour_pose = 3.0 * stack_gate * grip * beads_left * aim_soft * pour_band

    # unaimed tilt is harmless below the release angle -> free up to 1.5 rad; beyond, charge harder (1.0)
    tilt_premature = -1.0 * (1.0 - aim_soft) * torch.clamp(tilt_rad - TILT_FREE, min=0.0)

    # rotation speed (replaces pour_rate): no longer scaled by aim (it taxed rotation most where rotation is
    # needed). Only violent flips above 1.5 rad/s are charged — real-robot safety, weight 0.3.
    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    rot_speed = -0.3 * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (unchanged, ungated)
    # 50 * dfrac = 2.5 per bead; -30 * dfrac = -1.5 per spilled bead. Never gated: the bead signal must survive.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    spill_delta = -30.0 * torch.clamp(ctx.d_spill, min=0.0)

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
    closing_speed = -1.0 * near_f * torch.tanh(torch.clamp(v_close - 0.10, min=0.0) / 0.20)
    nested = -0.5 * ctx.cups_nested.to(dt)

    # ------------------------------------------------------------------ constraints (unchanged)
    upright_rcv = -0.3 * torch.clamp(ctx.rcv_cup_tilt, 0.0, 1.0)
    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)
    action_rate = -0.02 * torch.mean((ctx.actions - ctx.prev_actions) ** 2, dim=-1)

    # ------------------------------------------------------------------ success bonus (unchanged)
    success = 10.0 * ctx.success.to(dt) * clear * clean

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
        "meet_rcv": meet_rcv,
        "meet_src": meet_src,
        "cup_contact": cup_contact,
        "hand_foreign_src": hand_foreign_src,
        "hand_foreign_rcv": hand_foreign_rcv,
        "closing_speed": closing_speed,
        "nested": nested,
        "upright_rcv": upright_rcv,
        "drop": drop,
        "action_rate": action_rate,
        "success": success,
    }

    reward = zeros
    for v in components.values():
        reward = reward + v

    return reward, components
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0.0333, 0.267, 0.467, 0.667, 0.9, 1, 1, 1]  min 0 · mean 0.585 · max 1
bead/in_target: [6.1e-05, 0, 0.445, 0.68, 0.744, 0.748, 0.762, 0.765, 0.76, 0.767]  min 0 · mean 0.593 · max 0.79
bead/spill: [0.000208, 0.000183, 0.0406, 0.0522, 0.0366, 0.048, 0.0478, 0.0436, 0.0544, 0.0556]  min 6.1e-05 · mean 0.0391 · max 0.0721
done/drop: [0.00269, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 9.09e-05 · max 0.00952
episode_lengths/step: [121, 898, 886, 888, 836, 824, 897, 893, 877, 884]  min 103 · mean 869 · max 899
reward/action_rate: [-0.0206, -0.015, -0.0121, -0.0133, -0.0148, -0.0165, -0.0179, -0.0187, -0.019, -0.0194]  min -0.0206 · mean -0.0166 · max -0.0118
reward/aim: [0, 1.61, 1.96, 1.85, 1.97, 2.16, 2.22, 2.23, 2.24, 2.17]  min 0 · mean 1.95 · max 2.3
reward/approach_rcv: [0.483, 0.677, 0.684, 0.677, 0.667, 0.662, 0.663, 0.669, 0.656, 0.647]  min 0.473 · mean 0.663 · max 0.69
reward/approach_src: [0.48, 0.691, 0.688, 0.651, 0.641, 0.657, 0.667, 0.668, 0.67, 0.664]  min 0.479 · mean 0.663 · max 0.699
reward/both_grasped: [0, 0.869, 0.874, 0.871, 0.879, 0.879, 0.88, 0.875, 0.876, 0.869]  min 0 · mean 0.857 · max 0.892
reward/both_lifted: [0, 0.865, 0.871, 0.869, 0.879, 0.878, 0.879, 0.874, 0.875, 0.869]  min 0 · mean 0.85 · max 0.892
reward/bring_together: [0, 0.65, 0.75, 0.763, 0.805, 0.807, 0.809, 0.81, 0.811, 0.795]  min 0 · mean 0.741 · max 0.826
reward/closing_speed: [-0.285, -0.0669, -0.0538, -0.0659, -0.0602, -0.0525, -0.0433, -0.039, -0.0409, -0.0418]  min -0.285 · mean -0.0501 · max -0.0159
reward/cup_contact: [-0.214, -0.00315, -0.0028, -0.00472, -0.00481, -0.00312, -0.00665, -0.00584, -0.00497, -0.0104]  min -0.214 · mean -0.007 · max 0
reward/drop: [-0.00745, -0.000244, -0.000122, 0, 0, -0.000122, -0.000122, -0.000122, 0, -0.000122]  min -0.0184 · mean -0.000328 · max 0
reward/grasp_rcv: [0.0212, 0.974, 0.982, 0.981, 0.984, 0.982, 0.984, 0.983, 0.98, 0.981]  min 0.0209 · mean 0.966 · max 0.999
reward/grasp_src: [0.0211, 1, 1.02, 0.989, 0.996, 1, 1, 0.995, 0.998, 0.993]  min 0.0211 · mean 0.981 · max 1.03
reward/hand_foreign_rcv: [-0.101, -0.00466, -0.00566, -0.00517, -0.0078, -0.0121, -0.00743, -0.00821, -0.0105, -0.0164]  min -0.101 · mean -0.01 · max -0.00274
reward/hand_foreign_src: [-0.0931, -0.00666, -0.0105, -0.00788, -0.00779, -0.00437, -0.00497, -0.00734, -0.00501, -0.00751]  min -0.0931 · mean -0.00861 · max -0.00273
reward/lift_rcv: [0, 1.74, 1.75, 1.76, 1.77, 1.77, 1.77, 1.76, 1.77, 1.77]  min 0 · mean 1.72 · max 1.8
reward/lift_src: [6.39e-06, 1.61, 1.57, 1.46, 1.45, 1.59, 1.62, 1.61, 1.61, 1.6]  min 6.39e-06 · mean 1.53 · max 1.67
reward/meet_rcv: [0, -0.0115, -0.0108, -0.0261, -0.0592, -0.0626, -0.0219, -0.00965, -0.0165, -0.0188]  min -0.18 · mean -0.0246 · max 0
reward/meet_src: [0, 0, 0, 0, -7.36e-05, -1.41e-05, -2.51e-05, 0, 0, -0.000216]  min -0.00415 · mean -0.000126 · max 0
reward/nested: [-0.0133, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0591 · mean -0.000118 · max 0
reward/pour_delta: [0.00061, 0, 0.0586, 0.0623, 0.0568, 0.0507, 0.0513, 0.0452, 0.0494, 0.0494]  min 0 · mean 0.0441 · max 0.0787
reward/pour_pose: [0, 0.00113, 0.984, 1.44, 1.74, 1.92, 2.03, 2.07, 2.05, 2]  min 0 · mean 1.52 · max 2.14
reward/rot_speed: [-0.119, -0.0319, -0.0507, -0.0337, -0.0299, -0.0284, -0.0257, -0.0243, -0.0241, -0.0226]  min -0.119 · mean -0.0325 · max -0.0193
reward/spill_delta: [-0.000732, 0, -0.00366, -0.00439, -0.000732, -0.00439, -0.00439, -0.00366, -0.0033, -0.00293]  min -0.0103 · mean -0.00304 · max 0
reward/success: [0, 0, 4.71, 7.05, 7.63, 7.8, 7.97, 7.97, 8.02, 7.96]  min 0 · mean 6.17 · max 8.2
reward/tilt: [0, 1.19, 2.11, 2.29, 2.49, 2.7, 2.81, 2.85, 2.83, 2.77]  min 0 · mean 2.35 · max 2.93
reward/tilt_dir: [0, 0.668, 0.755, 0.786, 0.801, 0.793, 0.793, 0.795, 0.783, 0.77]  min -0.0773 · mean 0.728 · max 0.817
reward/tilt_premature: [0, -1.28e-06, -0.0138, -0.0387, -0.0229, -0.0244, -0.0281, -0.0238, -0.0258, -0.0349]  min -0.0496 · mean -0.0217 · max 0
reward/total: [-0.000684, 13.7, 20.9, 23.5, 24.7, 25.7, 26.3, 26.4, 26.3, 26]  min -0.000684 · mean 22.8 · max 26.9
reward/upright_rcv: [-0.141, -0.0851, -0.104, -0.121, -0.133, -0.155, -0.159, -0.17, -0.163, -0.161]  min -0.188 · mean -0.138 · max -0.0564
reward/wrap_rcv: [0, 0.276, 0.284, 0.336, 0.321, 0.315, 0.322, 0.328, 0.311, 0.313]  min 0 · mean 0.307 · max 0.352
reward/wrap_src: [0.000102, 1.19, 1.14, 1, 0.991, 1.13, 1.17, 1.16, 1.16, 1.15]  min 0.000102 · mean 1.1 · max 1.23
rewards/step: [11, 1.21e+04, 1.84e+04, 2.11e+04, 2.03e+04, 2.08e+04, 2.32e+04, 2.36e+04, 2.31e+04, 2.29e+04]  min 11 · mean 2e+04 · max 2.42e+04
task/aim_dist: [0.122, 0.159, 0.116, 0.107, 0.102, 0.104, 0.0985, 0.102, 0.125, 0.0975]  min 0.0947 · mean 0.131 · max 0.954
task/cup_collision_rate: [0.265, 0.00415, 0.00391, 0.00586, 0.0061, 0.00439, 0.00903, 0.00757, 0.00732, 0.0134]  min 0 · mean 0.00915 · max 0.265
task/cups_center_dist: [0.178, 0.232, 0.233, 0.248, 0.241, 0.244, 0.239, 0.244, 0.266, 0.239]  min 0.178 · mean 0.253 · max 0.942
task/episode_success: [0, 0, 0.466, 0.701, 0.763, 0.781, 0.797, 0.796, 0.803, 0.804]  min 0 · mean 0.618 · max 0.827
task/nested_rate: [0.0266, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000237 · max 0.118
task/rcv_cup_lift: [0.00459, 0.131, 0.134, 0.151, 0.152, 0.134, 0.15, 0.148, 0.148, 0.147]  min 0.00114 · mean 0.142 · max 0.2
task/rcv_grasped: [0, 0.876, 0.884, 0.881, 0.888, 0.888, 0.888, 0.884, 0.887, 0.89]  min 0 · mean 0.871 · max 0.901
task/rcv_hand_foreign_rate: [0.19, 0.0188, 0.0146, 0.0144, 0.0215, 0.0337, 0.0254, 0.0303, 0.0374, 0.0474]  min 0.00854 · mean 0.0295 · max 0.19
task/src_cup_lift: [0.00269, 0.225, 0.274, 0.312, 0.322, 0.306, 0.322, 0.322, 0.338, 0.323]  min 0.000999 · mean 0.294 · max 0.389
task/src_grasped: [0.000244, 0.88, 0.89, 0.884, 0.895, 0.892, 0.889, 0.882, 0.883, 0.88]  min 0.000244 · mean 0.869 · max 0.905
task/src_hand_foreign_rate: [0.178, 0.0215, 0.0247, 0.0203, 0.0188, 0.0146, 0.0154, 0.0208, 0.0193, 0.0227]  min 0.00977 · mean 0.0241 · max 0.178
task/src_tilt_deg: [26.5, 45.5, 78.5, 93.9, 99.1, 100, 104, 105, 105, 106]  min 7.43 · mean 89.8 · max 108
task/success_now: [0, 0, 0.474, 0.712, 0.773, 0.791, 0.807, 0.808, 0.815, 0.812]  min 0 · mean 0.626 · max 0.836

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Observations from the loop operator: rollout of the policy trained with the previous function (final checkpoint, 8 environments, per-step trace), the operator's own viewing of the video, and the environment changes made for this round. Facts only, no design instructions.

1. What the previous policy achieved: step-averaged `task/episode_success` about 0.81 with the physics/perception randomisation at its maximum (`adr/progress` 1.0). In the recorded rollout about 19 of 20 beads ended in the receiver cup (in-target 95.6–97.5 %), spill 2.5–4.4 %, all 8 environments succeeded. Hand–hand contact stayed at 2–4 %, cup–cup contact about 1 %, no nesting. The right hand formed a wrap grasp and the pour was a real pour through the air.

2. Right (source) hand grasp collapses while holding after the pour (operator saw it in the video; measured from step 300 to step 880 of the rollout, mean over envs, index/middle/ring): the command for the proximal flexion joint `_2` went down from 1.07 to 0.49 rad target while the command for the middle/distal joints went up (target 1.26 → 1.48 rad). The finger straightened at the knuckle and curled at the tip, so the cup was pressed by the fingertips: palm contact force fell from 7.0 N to 2.2 N and the distal-link force rose from about 7 N to 11 N. The left hand kept its grasp (palm force 4.0 → 3.1 N).

3. Arm vibration (operator saw it; measured): the policy switched the sign of two source palm rotation commands on 84 % and 81 % of consecutive steps (about +0.8 ↔ −0.5…−1.0), while the third rotation command stayed at −1. The palm target jumped about 68° per step, the arm joints vibrated with a dominant frequency of 4–5 Hz. The receiver arm did not do this.

4. Receiver cup tilt (operator saw it; measured): while the source cup was tilted past 80°, the receiver cup was tilted by a median of 46° and up to 55°. The operator's requirement: lift the receiver cup moderately, keep its mouth pointing to the sky and tilt it only slightly.

5. Environment changes for this round (all active in training; the policy is trained from scratch because the action space changed):
   - The action space is now 18-dimensional with the layout given in the robot description: per hand one thumb-opposition command, one thumb-closure command and ONE four-finger closure command that drives all flexion joints of index, middle, ring and pinky together. A finger can no longer straighten its knuckle while curling its tip.
   - The palm 6-DoF commands are low-pass filtered by the environment before the arm controller (exponential moving average, new = 0.25·command + 0.75·previous, at 60 Hz). `ctx.actions` / `ctx.prev_actions` are the raw policy outputs before this filter.
   - `ctx.success` now additionally requires the receiver cup tilt `ctx.rcv_cup_tilt` ≤ 20°. Holding the receiver at 46° as before is no longer a success.

6. How to read the feedback table above: it was recorded with the old 42-dimensional action space and the old success definition (no receiver-tilt condition). Under the new definition the previous policy's success would have been much lower, because the receiver was tilted about 46° during the pour. `task/rcv_tilt_deg` in the table is a step average over the whole episode (including the time before the pour).

7. Reminder of an earlier correction that still holds: with contact freeze the measured `hand_closure` saturates at about 0.35–0.43 on these cups whatever the grasp type, so closure above ~0.45 is not reachable and does not distinguish a hook from a wrap.

The policy must complete the full task (wrap-grasp both cups → lift → bring together without contact → tilt the source cup → beads in the receiver, receiver held nearly upright, no drop, little spill) and hold the grasp stably after pouring.
