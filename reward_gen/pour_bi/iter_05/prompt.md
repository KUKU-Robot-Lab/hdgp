You are an expert in robotics, reinforcement learning and code generation.

We control a bimanual robot: two 7-DOF OpenArm arms, each carrying a 20-DOF five-finger Tesollo DG-5F hand, standing at a table. The RIGHT arm is the "source" arm (src_*), the LEFT arm is the "receiver" arm (rcv_*). Two identical cups stand on the table: the source cup (in front of the right hand, filled with 20 small beads) and the receiver cup (in front of the left hand, empty). Positions are in metres in a frame whose origin is at the robot base; +z is up, +x is forward (away from the robot), +y is to the robot's left. The table top is at z = table_z.

The action space is a normalized `Box(-1, 1, (42,), float32)`:
  actions[0:6]   = source palm 6-DoF target offset (xyz + yaw/pitch/roll) — the arm is moved by a geometric-fabrics controller toward this target
  actions[6:21]  = source hand closure commands (5 fingers × 3 channels, 0 = open, 1 = closed)
  actions[21:27] = receiver palm 6-DoF target offset
  actions[27:42] = receiver hand closure commands
The hand controller stops a finger joint automatically once that finger link touches its own cup (contact freeze), so a closing command produces a wrap-around power grasp; opening is always allowed. Fingers can only close when the palm is near its cup.

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
7. `ctx.success` is computed by the environment (enough beads in the receiver cup, little spill, cups close together, and the cups NOT nested). You may add a bonus on it but you cannot redefine it. Beads only count as "in the receiver" once they have LEFT the source cup — pushing the source cup into the receiver cup (`ctx.cups_nested`) transfers nothing and is never a success; the beads must fall out of the tilted source cup through the air.
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


def _wrap_quality(finger_force: torch.Tensor, palm_force: torch.Tensor, closure: torch.Tensor,
                  palm_axes: torch.Tensor, cup_up: torch.Tensor, cup_pos: torch.Tensor,
                  palm_pos: torch.Tensor) -> torch.Tensor:
    # Power-grasp quality in [0,1] (new in iter_04). iter_03's right hand hooked the rim from above with
    # two fingertips (closure 0.38, palm force 0, palm normal pointing down the cup axis) and satisfied
    # the two-finger `grasped` flag. A wrap has: most fingers pressing, the cup body pressed into the
    # palm, the palm facing the cup SIDE, and the cup origin at palm height. Weighted MEAN (not product)
    # so every ingredient carries its own gradient from the hook posture.
    FINGER_ON = 0.3      # [N] a finger counts as "on the cup" above this force
    PALM_SCALE = 2.0     # [N] palm force tanh scale: 2 N -> 0.76
    LEVEL_SIGMA = 0.05   # [m] cup origin within ~5 cm of palm height along the cup axis
    CLOS_LO, CLOS_HI = 0.35, 0.65   # closure ramp: the measured hook (0.38) -> 0.1, a wrap -> 1
    dt = closure.dtype
    finger_frac = torch.mean((finger_force > FINGER_ON).to(dt), dim=-1)                # (N,)
    palm_c = torch.tanh(torch.clamp(palm_force, min=0.0) / PALM_SCALE)                 # (N,)
    normal = palm_axes[:, 0:3]                                                          # (N,3) palm normal
    # side: 1 when the palm normal is perpendicular to the cup axis (grasp from the side), 0 when the
    # palm sits on top of / under the cup. Rotates with the cup, so tilting does not change it.
    side = 1.0 - torch.clamp(torch.abs(torch.sum(normal * cup_up, dim=-1)), 0.0, 1.0)  # (N,)
    axial = torch.sum((cup_pos - palm_pos) * cup_up, dim=-1)                            # (N,) origin height vs palm
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
    DZ_LOW = -0.06            # [m] "above" ramp start: lip 6 cm below the receiver rim -> 0
    DZ_HIGH = 0.02            # [m] "above" ramp end: lip 2 cm above the rim -> 1
    DZ_FREE = 0.04            # [m] pouring from up to 4 cm above the rim costs nothing
    DZ_SIGMA = 0.06           # [m] beyond that, gentle decay (10 cm above -> 0.37)
    TILT_TARGET = 2.0         # [rad] beads leave past ~1.9 rad, saturate slightly above
    TILT_FREE = 0.8           # [rad] grasp posture alone tilts the cup 20-35 deg; only charge beyond 46 deg
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped / knocked off the table
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side
    NEAR_DIST = 0.30          # [m] cup-centre distance inside which approach speed is regulated
    FORCE_SCALE = 5.0         # [N] tanh scale for all contact penalties (knowledge item 11)
    FOREIGN_DEADBAND = 1.0    # [N] incidental brushes (finger on table while grasping) are free (additive term)
    CLEAN_DEADBAND = 0.5      # [N] deadband of the multiplicative hand-foreign gate (pour stage only)
    GRIP_FLOOR = 0.3          # pour/lift income still worth 30 % with a hook grasp, 100 % with a wrap
    MEET_RCV_FREE = 0.02      # [m] receiver cup may cross the spawn midline by 2 cm for free
    MEET_SRC_FREE = 0.05      # [m] source cup may cross the midline by 5 cm for free
    MEET_SCALE = 0.10         # [m] tanh scale of the crossing penalties (15 cm over -> 0.9)

    # ------------------------------------------------------------------ clearance factors
    # clear in [0,1]: 1 when the cups do not touch, -> 0 as the cup-cup force grows (unchanged, worked:
    # cup_collision_rate 0.28 -> 0.005).
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    # clean in [0,1] (new): same idea for hand-foreign force. iter_03 poured by resting the tilted source cup
    # on the left hand's fingers (rcv_hand_foreign_rate 0.44 while success rose) because the additive
    # 0.5-weight penalty was ~2 % of the pour-stack income. Now touching the other hand / cup / table with
    # either hand forfeits bring_together + aim + tilt + the success bonus, exactly like a cup-cup bump.
    f_foreign = torch.maximum(ctx.src_hand_foreign_force, ctx.rcv_hand_foreign_force)
    clean = 1.0 - torch.tanh(torch.clamp(f_foreign - CLEAN_DEADBAND, min=0.0) / FORCE_SCALE)
    not_nested_f = (~ctx.cups_nested).to(dt)

    # ------------------------------------------------------------------ stage 1: approach (unchanged — solved)
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)

    # ------------------------------------------------------------------ stage 2: grasp (flag part unchanged)
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    prox_src = torch.exp(-6.0 * d_src)
    prox_rcv = torch.exp(-6.0 * d_rcv)
    grasp_src = 1.0 * g_src + 0.5 * prox_src * torch.clamp(ctx.src_hand_closure, 0.0, 1.0)
    grasp_rcv = 1.0 * g_rcv + 0.5 * prox_rcv * torch.clamp(ctx.rcv_hand_closure, 0.0, 1.0)
    both_grasped_b = ctx.src_grasped & ctx.rcv_grasped
    both_grasped = 1.0 * both_grasped_b.to(dt)

    # wrap quality (new): hook ~0.15, full power grasp ~1.0
    wq_src = _wrap_quality(ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                           ctx.src_palm_axes, ctx.src_cup_up, ctx.src_cup_pos, ctx.src_palm_pos)
    wq_rcv = _wrap_quality(ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                           ctx.rcv_palm_axes, ctx.rcv_cup_up, ctx.rcv_cup_pos, ctx.rcv_palm_pos)
    # wrap_src weight 1.5: bigger than the 1.0 grasp flag so that, once the flag is banked, the next thing
    # worth doing on the table is enveloping the cup. wrap_rcv 0.5: the left hand already wraps; keep it.
    wrap_src = 1.5 * g_src * wq_src
    wrap_rcv = 0.5 * g_rcv * wq_rcv
    # grip in [0.3,1]: multiplies lift_src, aim and tilt. With the hook the whole downstream stack is worth
    # ~40 %, with a wrap 100 % -> ~+5/step at the pour pose for wrapping, felt already at grasp time.
    grip = GRIP_FLOOR + (1.0 - GRIP_FLOOR) * wq_src

    # ------------------------------------------------------------------ stage 3: lift
    # weight 2.0 per arm; asymmetric targets unchanged. lift_src additionally scaled by grip (see above).
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

    # ------------------------------------------------------------------ pouring-lip geometry (unchanged — worked)
    lip = _rim_low_point(ctx.src_cup_mouth_pos, ctx.src_cup_up, ctx.cup_radius)   # (N,3)
    lip_delta = lip - ctx.rcv_cup_mouth_pos                                         # (N,3)
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)                                   # (N,) horizontal miss
    dz_lip = lip_delta[:, 2]                                                        # (N,) lip height above rcv rim
    above = torch.clamp((dz_lip - DZ_LOW) / (DZ_HIGH - DZ_LOW), 0.0, 1.0)
    high = torch.exp(-(torch.clamp(dz_lip - DZ_FREE, min=0.0) / DZ_SIGMA) ** 2)
    aim_z = above * high
    aim_xy = torch.exp(-6.0 * torch.clamp(lip_xy - LIP_DEADBAND, min=0.0))
    aim_soft = aim_xy * aim_z                                                       # (N,) in [0,1]

    # ------------------------------------------------------------------ stage 4: bring cups together
    # weights unchanged (1.0 / 3.0); new gates: clean (hand-foreign) and grip (wrap quality).
    bring_together = 1.0 * src_lifted_f * not_nested_f * clean * torch.exp(-3.0 * lip_xy) * aim_z
    stack_gate = both_lifted_f * not_nested_f * clear * clean
    aim = 3.0 * stack_gate * grip * aim_soft

    # ------------------------------------------------------------------ stage 5: tilt (weights unchanged)
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_frac = torch.clamp(ctx.src_cup_tilt / TILT_TARGET, 0.0, 1.0)
    tilt = 3.0 * stack_gate * grip * beads_left * aim_soft * tilt_frac
    tilt_premature = -0.5 * (1.0 - aim_soft) * torch.clamp(ctx.src_cup_tilt - TILT_FREE, min=0.0)
    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    pour_rate = -0.3 * aim_soft * torch.tanh(src_ang_speed / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (unchanged, ungated)
    # 50 * dfrac = 2.5 per bead; -30 * dfrac = -1.5 per spilled bead. Never gated: the bead signal must survive.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    spill_delta = -30.0 * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ workspace: meet near the midline (new)
    # iter_03 carried the receiver cup all the way into the right hand's home region (y < -0.15 m) and the
    # right arm ran out of room. Midline from the spawn positions (robust to spawn randomisation).
    # meet_rcv weight 1.0: receiver cup crossing into the source half beyond 2 cm; 15 cm over -> -0.9/step.
    # meet_src weight 0.5: mirror image with a 5 cm allowance (the tilted source body naturally sits ~6-8 cm
    # to the RIGHT of the receiver, so the target pose "receiver at mid, source at mid-0.06" is free).
    mid_y = 0.5 * (ctx.src_cup_spawn_pos[:, 1] + ctx.rcv_cup_spawn_pos[:, 1])
    rcv_cross = mid_y - ctx.rcv_cup_pos[:, 1]          # >0: receiver moved toward the source side (-y)
    src_cross = ctx.src_cup_pos[:, 1] - mid_y          # >0: source moved toward the receiver side (+y)
    meet_rcv = -1.0 * both_lifted_f * torch.tanh(torch.clamp(rcv_cross - MEET_RCV_FREE, min=0.0) / MEET_SCALE)
    meet_src = -0.5 * both_lifted_f * torch.tanh(torch.clamp(src_cross - MEET_SRC_FREE, min=0.0) / MEET_SCALE)

    # ------------------------------------------------------------------ collision / clearance (real-robot safety)
    cup_contact = -1.0 * torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    # hand_foreign_* 0.5 -> 1.0 each, 1 N deadband kept: still small versus the ~3.5/step each arm earns from
    # grasp+lift (no own-cup avoidance, iter_00), while the multiplicative `clean` gate does the real work.
    f_src = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    f_rcv = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    hand_foreign_src = -1.0 * torch.tanh(f_src / FORCE_SCALE)
    hand_foreign_rcv = -1.0 * torch.tanh(f_rcv / FORCE_SCALE)
    rel = ctx.rcv_cup_pos - ctx.src_cup_pos                                   # (N,3)
    cup_dist = torch.norm(rel, dim=-1)                                        # (N,)
    u = rel / (cup_dist.unsqueeze(-1) + 1e-6)
    v_rel = ctx.src_cup_lin_vel - ctx.rcv_cup_lin_vel                         # (N,3)
    v_close = torch.sum(v_rel * u, dim=-1)                                    # >0 = approaching
    near_f = (cup_dist < NEAR_DIST).to(dt)
    closing_speed = -1.0 * near_f * torch.tanh(torch.clamp(v_close - 0.10, min=0.0) / 0.20)
    nested = -0.5 * ctx.cups_nested.to(dt)

    # ------------------------------------------------------------------ constraints (unchanged)
    upright_rcv = -0.3 * torch.clamp(ctx.rcv_cup_tilt, 0.0, 1.0)
    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)
    action_rate = -0.02 * torch.mean((ctx.actions - ctx.prev_actions) ** 2, dim=-1)

    # ------------------------------------------------------------------ success bonus
    # weight 10.0, withheld while the cups touch (clear) OR a hand touches something foreign (clean, new):
    # a pour that leans on the other hand is not a deployable success.
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
        "tilt_premature": tilt_premature,
        "pour_rate": pour_rate,
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

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [6.1e-05, 0, 0, 0, 0, 0, 0, 8.54e-05, 7.32e-05, 0.00022]  min 0 · mean 6.7e-05 · max 0.00134
bead/spill: [0.000549, 0.00245, 0.00219, 0.000952, 0.000146, 0.000208, 0.000183, 0.00072, 0.000635, 0.000366]  min 0.000122 · mean 0.000692 · max 0.00314
done/drop: [0.00391, 0.000488, 0, 0.000244, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.00045 · max 0.0105
episode_lengths/step: [120, 644, 833, 842, 883, 897, 884, 889, 888, 873]  min 102 · mean 812 · max 899
reward/action_rate: [-0.0207, -0.0195, -0.0182, -0.0165, -0.0163, -0.016, -0.0149, -0.0153, -0.0153, -0.0152]  min -0.0207 · mean -0.0165 · max -0.0149
reward/aim: [0, 0.000454, 0.00766, 0.0726, 0.161, 0.415, 1.29, 1.49, 1.69, 1.91]  min 0 · mean 0.813 · max 2.03
reward/approach_rcv: [0.479, 0.441, 0.421, 0.609, 0.634, 0.655, 0.638, 0.644, 0.646, 0.656]  min 0.404 · mean 0.593 · max 0.661
reward/approach_src: [0.479, 0.56, 0.659, 0.666, 0.674, 0.685, 0.684, 0.687, 0.684, 0.685]  min 0.479 · mean 0.657 · max 0.688
reward/both_grasped: [0, 0.0381, 0.0823, 0.8, 0.864, 0.867, 0.862, 0.875, 0.86, 0.866]  min 0 · mean 0.662 · max 0.882
reward/both_lifted: [0, 0.0061, 0.053, 0.762, 0.843, 0.849, 0.852, 0.865, 0.853, 0.863]  min 0 · mean 0.644 · max 0.877
reward/bring_together: [0, 0.0439, 0.025, 0.101, 0.188, 0.307, 0.577, 0.615, 0.675, 0.732]  min 0 · mean 0.368 · max 0.771
reward/closing_speed: [-0.29, -0.0132, -0.00482, -0.00982, -0.00844, -0.0593, -0.0846, -0.0884, -0.0699, -0.0659]  min -0.29 · mean -0.0459 · max -0.00259
reward/cup_contact: [-0.208, 0, 0, 0, 0, 0, -0.00951, -0.00273, -0.00478, -0.00969]  min -0.208 · mean -0.00583 · max 0
reward/drop: [-0.00818, -0.00146, -0.000244, -0.00134, -0.000244, -0.000366, -0.000244, -0.000122, -0.000122, -0.000122]  min -0.021 · mean -0.00108 · max 0
reward/grasp_rcv: [0.0208, 0.0943, 0.109, 0.893, 0.954, 0.967, 0.965, 0.979, 0.966, 0.977]  min 0.0182 · mean 0.752 · max 0.992
reward/grasp_src: [0.021, 0.734, 0.977, 0.981, 1, 1.02, 1.01, 1.02, 0.994, 0.996]  min 0.021 · mean 0.919 · max 1.03
reward/hand_foreign_rcv: [-0.0929, -0.0221, -0.0227, -0.0158, -0.0109, -0.00749, -0.0109, -0.00725, -0.00798, -0.0123]  min -0.0929 · mean -0.0161 · max -0.00597
reward/hand_foreign_src: [-0.0848, -0.0362, -0.0143, -0.0123, -0.00608, -0.00669, -0.00712, -0.00624, -0.0052, -0.00875]  min -0.0848 · mean -0.0136 · max -0.0038
reward/lift_rcv: [0, 0.0596, 0.11, 1.57, 1.7, 1.72, 1.72, 1.75, 1.72, 1.74]  min 0 · mean 1.31 · max 1.76
reward/lift_src: [1.36e-05, 0.186, 1.17, 1.22, 1.47, 1.53, 1.53, 1.58, 1.55, 1.58]  min 1.36e-05 · mean 1.26 · max 1.61
reward/meet_rcv: [0, 0, -0.000727, -0.000232, -0.000143, -0.00373, -0.182, -0.0849, -0.0632, -0.0683]  min -0.19 · mean -0.0405 · max 0
reward/meet_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -6.54e-06 · mean -3.35e-08 · max 0
reward/nested: [-0.0133, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0559 · mean -0.000578 · max 0
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.24e-05 · max 0.00122
reward/pour_rate: [-0.0725, -0.0135, -0.00492, -0.00992, -0.0158, -0.0347, -0.1, -0.108, -0.118, -0.13]  min -0.137 · mean -0.0622 · max -0.00441
reward/spill_delta: [-0.00146, -0.00183, -0.0011, -0.000366, 0, 0, 0, 0, 0, 0]  min -0.00256 · mean -0.000194 · max 0
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00244]  min 0 · mean 0.000469 · max 0.0143
reward/tilt: [0, 4.15e-05, 0.00145, 0.013, 0.0412, 0.113, 0.531, 0.629, 0.78, 0.872]  min 0 · mean 0.351 · max 1.01
reward/tilt_premature: [-0.0176, -0.00574, -0.00313, -0.00148, -0.00467, -0.00187, -0.00828, -0.00853, -0.009, -0.00468]  min -0.0176 · mean -0.00498 · max -0.000938
reward/total: [0.0372, 2.42, 4.42, 8.66, 9.66, 10.3, 11.5, 12.1, 12.4, 12.8]  min 0.0372 · mean 9.15 · max 13.2
reward/upright_rcv: [-0.141, -0.0224, -0.0168, -0.0955, -0.0715, -0.0573, -0.0548, -0.0643, -0.0695, -0.0881]  min -0.141 · mean -0.0637 · max -0.0088
reward/wrap_rcv: [0, 0.0111, 0.0158, 0.199, 0.216, 0.231, 0.233, 0.231, 0.23, 0.247]  min 0 · mean 0.176 · max 0.264
reward/wrap_src: [0.000118, 0.432, 0.904, 0.979, 1.08, 1.12, 1.1, 1.15, 1.13, 1.15]  min 0.000118 · mean 0.962 · max 1.17
rewards/step: [19.2, 1.4e+03, 3.55e+03, 7.3e+03, 8.51e+03, 9.04e+03, 9.98e+03, 1.08e+04, 1.1e+04, 1.12e+04]  min 13.7 · mean 7.85e+03 · max 1.18e+04
task/aim_dist: [0.123, 0.366, 0.416, 0.427, 0.407, 0.308, 0.176, 0.16, 0.144, 0.125]  min 0.114 · mean 0.287 · max 0.792
task/cup_collision_rate: [0.258, 0, 0, 0, 0, 0, 0.0122, 0.00317, 0.00635, 0.0117]  min 0 · mean 0.00733 · max 0.258
task/cups_center_dist: [0.179, 0.349, 0.416, 0.42, 0.413, 0.311, 0.233, 0.223, 0.217, 0.2]  min 0.179 · mean 0.317 · max 0.78
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000244]  min 0 · mean 4.77e-05 · max 0.00146
task/nested_rate: [0.0266, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.00116 · max 0.112
task/rcv_cup_lift: [0.00429, 0.00649, 0.0112, 0.17, 0.171, 0.187, 0.171, 0.159, 0.155, 0.165]  min 0.0026 · mean 0.135 · max 0.192
task/rcv_grasped: [0, 0.0642, 0.0835, 0.813, 0.87, 0.87, 0.87, 0.881, 0.867, 0.874]  min 0 · mean 0.674 · max 0.887
task/rcv_hand_foreign_rate: [0.18, 0.0488, 0.0483, 0.0383, 0.0286, 0.0208, 0.0234, 0.0242, 0.0312, 0.0352]  min 0.0166 · mean 0.0378 · max 0.18
task/src_cup_lift: [0.00302, 0.0317, 0.175, 0.167, 0.215, 0.234, 0.254, 0.252, 0.246, 0.255]  min 0.00109 · mean 0.196 · max 0.263
task/src_grasped: [0.000244, 0.653, 0.865, 0.865, 0.885, 0.896, 0.884, 0.894, 0.875, 0.878]  min 0.000244 · mean 0.809 · max 0.901
task/src_hand_foreign_rate: [0.167, 0.0791, 0.0322, 0.0322, 0.0178, 0.0186, 0.0173, 0.0178, 0.0142, 0.0195]  min 0.0115 · mean 0.0317 · max 0.167
task/src_tilt_deg: [26, 20, 19.5, 16.9, 25.7, 25.4, 38.1, 40.1, 43.8, 44.1]  min 9.21 · mean 30 · max 49.4
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000244]  min 0 · mean 4.9e-05 · max 0.00146

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Observations from the loop operator (rollout video of the policy trained with the previous function, epoch 500, camera on the robot's right side) and one correction. Facts only, no design instructions.

1. The RIGHT (source) hand now forms the intended envelope grasp: it approaches the source cup from the side, the palm rests against the cup body and the four fingers wrap around the cup at mid-height with the thumb opposite; the cup bottom protrudes below the hand. The cup stays in the hand without slipping while it is lifted and tilted. The LEFT (receiver) hand wraps the receiver cup the same way. The operator's grasp requirement from the previous round is met by this policy.

2. CORRECTION of the previous observation note: it claimed a wrapped grasp "closes much further" than 0.38. That was wrong. With contact freeze each finger joint stops as soon as its link touches the cup, so on this cup the measured `hand_closure` saturates at about 0.35–0.42 whatever the grasp type: the left hand's visually correct wrap has measured 0.35–0.38 in every run so far, and the right hand's visually confirmed wrap in this run measures 0.40–0.43. Closure values above ~0.45 are not physically reachable on this cup, and closure does not distinguish a hook from a wrap. In the feedback table `task/src_closure` staying at ~0.41 is the wrapped grasp, not a failure to wrap.

3. The pour stalls. With both cups wrapped and lifted (source ~0.26 m, receiver ~0.17 m), the policy brings the source cup next to the receiver cup (aim distance 0.12 m) and holds it tilted at about 47° (`task/src_tilt_deg` max 47.7 over the whole run); beads only leave the source cup past roughly 110°. `bead/in_target` 0.0005, `task/episode_success` 0.0004. Hand–hand contact stays low (`task/*_hand_foreign_rate` ~2 %), cup–cup contact ~1 %, no nesting.

4. Environment facts unchanged: cup poses seen by the policy are delayed/noisy; cup mass, joint gains and cup friction are randomised as success grows (`adr/progress` still 0); `ctx.cup_cup_force`, `ctx.src_hand_foreign_force`, `ctx.rcv_hand_foreign_force` are available; the policy will run on the real robot, so hand–hand and cup–cup contact remain safety hazards. The cups spawn symmetric about the robot midline (source y ≈ −0.16 m, receiver y ≈ +0.16 m, x ≈ 0.36 m).

The policy must complete the full task (grasp both with a wrap → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
