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


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    N = ctx.src_palm_pos.shape[0]
    dev = ctx.src_palm_pos.device
    dt = ctx.src_palm_pos.dtype
    zeros = torch.zeros(N, device=dev, dtype=dt)

    # ------------------------------------------------------------------ constants
    LIFT_TARGET_SRC = 0.20    # [m] source cup must end up clearly higher than the receiver: its tilted lip
    LIFT_TARGET_RCV = 0.10    # [m] hangs ~5-8 cm below its origin and the receiver rim is above the receiver origin
    LIFT_GATE = 0.05          # [m] minimum lift before the pour stack is rewarded
    LIP_DEADBAND = 0.01       # [m] lip within 1 cm of the receiver mouth centre counts as centred (pose noise)
    DZ_LOW = -0.06            # [m] "above" ramp start: lip 6 cm below the receiver rim -> 0
    DZ_HIGH = 0.02            # [m] "above" ramp end: lip 2 cm above the rim -> 1 (wide so the gradient is felt early)
    DZ_FREE = 0.04            # [m] pouring from up to 4 cm above the rim costs nothing
    DZ_SIGMA = 0.06           # [m] beyond that, gentle decay (10 cm above -> 0.37): high drop-pours spill
    TILT_TARGET = 2.0         # [rad] beads leave past ~1.9 rad, saturate slightly above
    TILT_FREE = 0.8           # [rad] grasp posture alone tilts the cup 20-35 deg; only charge beyond 46 deg
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped / knocked off the table
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side
    NEAR_DIST = 0.30          # [m] cup-centre distance inside which approach speed is regulated
    FORCE_SCALE = 5.0         # [N] tanh scale for all contact penalties (knowledge item 11)
    FOREIGN_DEADBAND = 1.0    # [N] incidental brushes (e.g. finger on table while grasping) are free

    # ------------------------------------------------------------------ clearance factors
    # clear in [0,1]: 1 when the cups do not touch, -> 0 as the cup-cup force grows. Multiplies the whole
    # pour stack (aim + tilt + success) so a bump costs ~6-16/step instead of a small additive penalty.
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    not_nested_f = (~ctx.cups_nested).to(dt)

    # ------------------------------------------------------------------ stage 1: approach (unchanged — solved)
    # weight 1.0 per arm, exp(-4d): wide basin, saturates ~0.68 in the grasp posture.
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)

    # ------------------------------------------------------------------ stage 2: grasp (unchanged — solved)
    # 1.0 for the contact-established flag + 0.5 * closure weighted by proximity exp(-6d).
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    prox_src = torch.exp(-6.0 * d_src)
    prox_rcv = torch.exp(-6.0 * d_rcv)
    grasp_src = 1.0 * g_src + 0.5 * prox_src * torch.clamp(ctx.src_hand_closure, 0.0, 1.0)
    grasp_rcv = 1.0 * g_rcv + 0.5 * prox_rcv * torch.clamp(ctx.rcv_hand_closure, 0.0, 1.0)
    # weight 1.0 bonus for holding BOTH cups: the second grasp is worth more than the first.
    both_grasped_b = ctx.src_grasped & ctx.rcv_grasped
    both_grasped = 1.0 * both_grasped_b.to(dt)

    # ------------------------------------------------------------------ stage 3: lift
    # weight 2.0 per arm — larger than grasp so holding on and lifting beats holding on the table.
    # ASYMMETRIC targets (new): the source saturates at 0.20 m, the receiver at 0.10 m, so the lift stage
    # already produces most of the height difference the pour pose needs (iter_02 lifted both to ~0.14 m).
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET_SRC, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET_RCV, 0.0, 1.0)
    lift_src = 2.0 * g_src * lift_frac_src
    lift_rcv = 2.0 * g_rcv * lift_frac_rcv

    src_lifted_b = ctx.src_grasped & (h_src > LIFT_GATE)
    both_lifted_b = src_lifted_b & ctx.rcv_grasped & (h_rcv > LIFT_GATE)
    src_lifted_f = src_lifted_b.to(dt)
    both_lifted_f = both_lifted_b.to(dt)
    # weight 1.0 bonus for both cups in the air
    both_lifted = 1.0 * both_lifted_f

    # ------------------------------------------------------------------ pouring-lip geometry (new)
    # iter_02 measured alignment between MOUTH CENTRES; with a near-upright cup that is only reachable by
    # nesting / bumping the cups, so the tilt gate never opened and the policy parked 13 cm away.
    # The point that matters is the lowest rim point ("lip"): it swings over the receiver by TILTING,
    # while the cup body stays outside the receiver. Aim terms are defined on the lip.
    lip = _rim_low_point(ctx.src_cup_mouth_pos, ctx.src_cup_up, ctx.cup_radius)   # (N,3)
    lip_delta = lip - ctx.rcv_cup_mouth_pos                                         # (N,3)
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)                                   # (N,) horizontal miss
    dz_lip = lip_delta[:, 2]                                                        # (N,) lip height above rcv rim
    # above: wide ramp -6 cm -> +2 cm (gradient toward "source higher" is felt long before the pose is right).
    above = torch.clamp((dz_lip - DZ_LOW) / (DZ_HIGH - DZ_LOW), 0.0, 1.0)
    # high: free up to 4 cm above the rim, then gentle decay — a 10 cm drop-pour (iter_01: 7 % spill) pays 0.37.
    high = torch.exp(-(torch.clamp(dz_lip - DZ_FREE, min=0.0) / DZ_SIGMA) ** 2)
    aim_z = above * high
    # aim_xy: exp(-6 d) with a 1 cm deadband — 5 cm miss pays 0.79, 13 cm (the iter_02 fixed point) pays 0.49.
    aim_xy = torch.exp(-6.0 * torch.clamp(lip_xy - LIP_DEADBAND, min=0.0))
    aim_soft = aim_xy * aim_z                                                       # (N,) in [0,1]

    # ------------------------------------------------------------------ stage 4: bring cups together
    # bring_together, weight 1.0: once the SOURCE cup is lifted, pull its lip over the receiver rim with a wide
    # basin exp(-3d); aim_z included so hovering upright 20 cm above the receiver is not a free 1.0.
    bring_together = 1.0 * src_lifted_f * not_nested_f * torch.exp(-3.0 * lip_xy) * aim_z
    # aim, weight 3.0 (replaces align): both lifted, not nested, cups clear. No hard xy / z gate any more —
    # the soft product gives a monotone path from "side by side, 30 deg" to "lip over the rim, 110 deg".
    # Must beat lift (2+2 banked) + bring_together (1.0).
    aim = 3.0 * both_lifted_f * not_nested_f * clear * aim_soft

    # ------------------------------------------------------------------ stage 5: tilt
    # beads_left in [0,1]: beads still in play (in source or already in receiver). Spilled beads shrink the
    # tilt reward permanently, so "dump the beads on the table, then keep tilting" earns nothing.
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_frac = torch.clamp(ctx.src_cup_tilt / TILT_TARGET, 0.0, 1.0)
    # weight 3.0, scaled by the SOFT aim factor instead of a hard aligned flag: at the iter_02 fixed point
    # (aim ~0.4) tilting from 30 to 60 deg already pays ~+0.4/step and moves the lip closer, so the
    # chicken-and-egg deadlock is gone. At the pour pose aim -> ~0.9 and this pays ~2.7/step.
    tilt = 3.0 * both_lifted_f * not_nested_f * clear * beads_left * aim_soft * tilt_frac
    # premature tilt (weight 0.5): tilting well past grasp posture while the lip is NOT over the receiver.
    # Weighted by (1 - aim_soft) so it vanishes as the lip comes over the rim; spills are priced by spill_delta.
    tilt_premature = -0.5 * (1.0 - aim_soft) * torch.clamp(ctx.src_cup_tilt - TILT_FREE, min=0.0)
    # pour_rate, weight 0.3: gentle cost on fast tilting while aimed — a slow pour spills less.
    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    pour_rate = -0.3 * aim_soft * torch.tanh(src_ang_speed / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer (unchanged)
    # 50 * dfrac = 2.5 per bead (20 beads) — one-shot payments, must dominate the shaping stack.
    # 30 * dfrac = -1.5 per spilled bead: one landed bead still outweighs one lost.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    spill_delta = -30.0 * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ collision / clearance (unchanged — real-robot safety)
    # cup_contact, weight 1.0: bounded price on cup-cup force on top of the lost-stage gate.
    cup_contact = -1.0 * torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    # hand_foreign_*, weight 0.5 each with a 1 N deadband — small versus the ~3.5/step each arm earns from
    # grasp+lift, so the hand is not taught to avoid its own cup (iter_00 failure mode).
    f_src = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    f_rcv = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    hand_foreign_src = -0.5 * torch.tanh(f_src / FORCE_SCALE)
    hand_foreign_rcv = -0.5 * torch.tanh(f_rcv / FORCE_SCALE)
    # closing_speed, weight 1.0: rate at which the two cups approach each other inside 30 cm
    # (10 cm/s deadband, 30 cm/s costs ~0.76).
    rel = ctx.rcv_cup_pos - ctx.src_cup_pos                                   # (N,3)
    cup_dist = torch.norm(rel, dim=-1)                                        # (N,)
    u = rel / (cup_dist.unsqueeze(-1) + 1e-6)
    v_rel = ctx.src_cup_lin_vel - ctx.rcv_cup_lin_vel                         # (N,3)
    v_close = torch.sum(v_rel * u, dim=-1)                                    # >0 = approaching
    near_f = (cup_dist < NEAR_DIST).to(dt)
    closing_speed = -1.0 * near_f * torch.tanh(torch.clamp(v_close - 0.10, min=0.0) / 0.20)
    # nested, weight 0.5: small explicit price; the gates above already remove all pour income.
    nested = -0.5 * ctx.cups_nested.to(dt)

    # ------------------------------------------------------------------ constraints (unchanged)
    # Receiver-cup penalties stay SMALL (0.3 + 0.5 = 0.8/step worst case) versus the ~7/step the receiver
    # branch earns — larger values taught the left hand to avoid the cup (iter_00).
    upright_rcv = -0.3 * torch.clamp(ctx.rcv_cup_tilt, 0.0, 1.0)
    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)

    # action-rate regularisation (weight 0.02 — tiny, must never compete with any stage term)
    action_rate = -0.02 * torch.mean((ctx.actions - ctx.prev_actions) ** 2, dim=-1)

    # ------------------------------------------------------------------ success bonus
    # weight 10.0, withheld while the cups touch: success is defined by the env, the bonus is not.
    success = 10.0 * ctx.success.to(dt) * clear

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
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

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0333]  min 0 · mean 0.0065 · max 0.0667
bead/in_target: [7.32e-05, 0, 1.22e-05, 0, 0, 0, 0, 0.000549, 0.00229, 0.521]  min 0 · mean 0.0812 · max 0.657
bead/spill: [0.000476, 0.00126, 0.0414, 0.00829, 0.000781, 0.000366, 0.000439, 0.000427, 0.000757, 0.0404]  min 0.000134 · mean 0.0115 · max 0.0944
done/drop: [0.0022, 0.00122, 0, 0, 0, 0.000244, 0, 0, 0, 0]  min 0 · mean 0.000402 · max 0.0107
episode_lengths/step: [120, 566, 843, 883, 880, 883, 892, 895, 899, 895]  min 104 · mean 820 · max 899
reward/action_rate: [-0.0206, -0.0191, -0.0169, -0.0165, -0.016, -0.0155, -0.0159, -0.0162, -0.0162, -0.0156]  min -0.0206 · mean -0.0166 · max -0.0152
reward/aim: [0, 0.0226, 0.136, 0.215, 0.392, 1.46, 2.13, 2.42, 2.4, 2.45]  min 0 · mean 1.28 · max 2.51
reward/approach_rcv: [0.482, 0.544, 0.633, 0.636, 0.619, 0.625, 0.638, 0.646, 0.662, 0.664]  min 0.472 · mean 0.626 · max 0.673
reward/approach_src: [0.478, 0.564, 0.624, 0.625, 0.637, 0.649, 0.669, 0.673, 0.666, 0.649]  min 0.478 · mean 0.633 · max 0.677
reward/both_grasped: [0, 0.402, 0.826, 0.847, 0.85, 0.855, 0.863, 0.89, 0.862, 0.894]  min 0 · mean 0.774 · max 0.897
reward/both_lifted: [0, 0.189, 0.788, 0.837, 0.843, 0.849, 0.859, 0.884, 0.856, 0.889]  min 0 · mean 0.751 · max 0.894
reward/bring_together: [0, 0.048, 0.157, 0.22, 0.309, 0.583, 0.744, 0.816, 0.807, 0.823]  min 0 · mean 0.492 · max 0.839
reward/closing_speed: [-0.286, -0.0115, -0.00544, -0.0127, -0.0375, -0.0838, -0.0638, -0.0499, -0.0362, -0.0301]  min -0.286 · mean -0.0387 · max -0.00215
reward/cup_contact: [-0.225, -0.000523, -0.000244, 0, 0, -0.0033, -0.0022, -0.00159, -0.0017, -0.00441]  min -0.225 · mean -0.00371 · max 0
reward/drop: [-0.00745, -0.00513, -0.000732, -0.00061, -0.00134, -0.0011, -0.000488, -0.000366, -0.000366, -0.000122]  min -0.0183 · mean -0.00149 · max 0
reward/grasp_rcv: [0.0211, 0.625, 0.955, 0.957, 0.951, 0.969, 0.965, 0.991, 0.964, 1]  min 0.0211 · mean 0.893 · max 1
reward/grasp_src: [0.0207, 0.647, 0.949, 0.969, 0.975, 0.981, 0.99, 1.01, 0.98, 1.01]  min 0.0202 · mean 0.901 · max 1.02
reward/hand_foreign_rcv: [-0.0469, -0.0673, -0.0113, -0.00777, -0.00622, -0.0307, -0.0484, -0.043, -0.106, -0.109]  min -0.156 · mean -0.0482 · max -0.00462
reward/hand_foreign_src: [-0.0392, -0.032, -0.00611, -0.00287, -0.00226, -0.0113, -0.015, -0.0137, -0.0104, -0.00409]  min -0.0824 · mean -0.013 · max -0.00226
reward/lift_rcv: [0, 0.732, 1.65, 1.68, 1.7, 1.72, 1.73, 1.77, 1.72, 1.79]  min 0 · mean 1.54 · max 1.79
reward/lift_src: [0, 0.56, 1.61, 1.72, 1.76, 1.76, 1.76, 1.8, 1.75, 1.8]  min 0 · mean 1.55 · max 1.8
reward/nested: [-0.011, -0.000488, -0.000122, 0, 0, 0, 0, 0, 0, 0]  min -0.0579 · mean -0.000531 · max 0
reward/pour_delta: [0.00122, 0, 0, 0, 0, 0, 0, 0, 0.00183, 0.0568]  min 0 · mean 0.0091 · max 0.083
reward/pour_rate: [-0.0723, -0.00835, -0.0127, -0.018, -0.0292, -0.103, -0.143, -0.156, -0.155, -0.147]  min -0.159 · mean -0.0856 · max -0.00822
reward/spill_delta: [-0.00256, -0.000732, -0.00696, -0.00293, 0, 0, 0, 0, 0, -0.0022]  min -0.0198 · mean -0.0017 · max 0
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0.00244, 5.49]  min 0 · mean 0.835 · max 6.74
reward/tilt: [0, 0.00757, 0.0756, 0.122, 0.208, 0.838, 1.26, 1.58, 1.66, 1.95]  min 0 · mean 0.871 · max 2.1
reward/tilt_premature: [-0.0172, -0.00952, -0.155, -0.133, -0.0824, -0.0388, -0.0184, -0.0125, -0.00963, -0.0232]  min -0.196 · mean -0.0505 · max -0.00184
reward/total: [0.123, 4.04, 8.08, 8.55, 8.97, 10.9, 12.2, 13.1, 12.9, 19]  min 0.123 · mean 10.8 · max 20.2
reward/upright_rcv: [-0.14, -0.0883, -0.0746, -0.061, -0.0604, -0.0835, -0.0704, -0.0753, -0.0664, -0.0975]  min -0.14 · mean -0.0765 · max -0.0526
rewards/step: [20.6, 2.06e+03, 6.83e+03, 7.51e+03, 7.86e+03, 9.28e+03, 1.09e+04, 1.15e+04, 1.18e+04, 1.64e+04]  min 15.5 · mean 9.3e+03 · max 1.79e+04
task/aim_dist: [0.119, 0.477, 0.42, 0.384, 0.325, 0.184, 0.132, 0.11, 0.111, 0.11]  min 0.0962 · mean 0.256 · max 0.916
task/cup_collision_rate: [0.28, 0.000488, 0.000244, 0, 0, 0.00488, 0.00293, 0.0022, 0.00269, 0.00586]  min 0 · mean 0.00458 · max 0.28
task/cups_center_dist: [0.176, 0.461, 0.399, 0.382, 0.346, 0.252, 0.219, 0.21, 0.213, 0.231]  min 0.176 · mean 0.309 · max 0.905
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0.000244, 0.543]  min 0 · mean 0.0821 · max 0.667
task/nested_rate: [0.022, 0.000977, 0.000244, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.00106 · max 0.116
task/rcv_cup_lift: [0.00442, 0.0829, 0.167, 0.162, 0.17, 0.179, 0.176, 0.182, 0.184, 0.197]  min 0.00137 · mean 0.163 · max 0.208
task/rcv_grasped: [0, 0.551, 0.859, 0.857, 0.859, 0.868, 0.869, 0.894, 0.865, 0.9]  min 0 · mean 0.801 · max 0.902
task/rcv_hand_foreign_rate: [0.187, 0.192, 0.0491, 0.0369, 0.0271, 0.103, 0.181, 0.181, 0.413, 0.438]  min 0.022 · mean 0.184 · max 0.554
task/src_cup_lift: [0.003, 0.0884, 0.256, 0.262, 0.268, 0.285, 0.296, 0.312, 0.315, 0.354]  min 0.00102 · mean 0.261 · max 0.384
task/src_grasped: [0, 0.573, 0.856, 0.884, 0.896, 0.892, 0.893, 0.909, 0.885, 0.91]  min 0 · mean 0.813 · max 0.913
task/src_hand_foreign_rate: [0.161, 0.113, 0.0332, 0.0208, 0.0156, 0.0547, 0.075, 0.0669, 0.0535, 0.0256]  min 0.0156 · mean 0.0576 · max 0.268
task/src_tilt_deg: [26, 20.9, 56.5, 56.1, 50.6, 54.7, 58.4, 66.5, 68.5, 86]  min 8.05 · mean 57 · max 90.7
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0.000244, 0.551]  min 0 · mean 0.0838 · max 0.675

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Observations from watching the trained policy (rollout video at epoch 550, plus the operator's notes). These are facts about what the policy does, not design instructions.

1. The RIGHT (source) hand never forms a proper power grasp. It closes on the cup from above and hooks the rim/upper body with its fingertips, so the cup hangs below the fingers with its body OUTSIDE the palm. Measured `task/src_closure` stays at 0.38 for the whole run (a wrapped grasp closes much further), yet `task/src_grasped` is 0.90 because that flag only needs the thumb AND one other finger to touch the cup — a two-finger hook satisfies it and `grasp_src` (1.0 flag + 0.5 × closure) is nearly saturated at 1.0. The LEFT (receiver) hand, by contrast, wraps the cup body with the palm against it and lifts it cleanly. The operator states this grasp is the root problem: tilting must happen with the source cup fully enveloped by the right hand.

2. Because the source cup is not held firmly, during the tilt the right hand does not keep the cup by itself — it lowers the tilted cup until the right hand (and the cup) rests on the LEFT hand's fingers / the receiver cup rim and uses that as a support. This is `task/rcv_hand_foreign_rate` rising to 0.52 (max 0.55) while success rises: the two are the same behaviour. The pour itself is real (beads fall through the air into the receiver, in_target 0.62, spill 4%), but the cups end up touching hand-to-hand.

3. The meeting point drifts to the extreme RIGHT side of the workspace. The cups spawn symmetric about the robot's midline (source at y ≈ −0.16 m in front of the right hand, receiver at y ≈ +0.16 m in front of the left hand, x ≈ 0.36 m, table midline y = 0). The left arm carries the receiver cup all the way across to the right hand's home region (y < −0.15 m) and the pour happens there, where the right arm has almost no free space (elbow folded, wrist near its limits) — the operator notes the left arm's transport is easy but the right arm loses its working room when the receiver cup comes that far right. The source cup is lifted ~0.35 m, the receiver ~0.19 m.

4. Per-finger contact forces on the own cup are available: `ctx.src_finger_force` / `ctx.rcv_finger_force` (N,5) with index 0 = thumb, `ctx.src_palm_force` / `ctx.rcv_palm_force` (N,), and `ctx.src_hand_closure` / `ctx.rcv_hand_closure` (0 = open, 1 = fully closed). `ctx.src_grasped` remains the two-finger flag described above.

5. Environment facts unchanged from the previous round: the policy sees delayed/noisy cup poses; cup mass, joint gains and cup friction are randomised as success grows (`adr/progress` reached 0.07); `ctx.cup_cup_force`, `ctx.src_hand_foreign_force`, `ctx.rcv_hand_foreign_force` are available; the policy will run on the real robot, so hand–hand and cup–cup contact is a safety hazard.

The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
