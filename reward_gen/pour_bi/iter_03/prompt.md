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


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    N = ctx.src_palm_pos.shape[0]
    dev = ctx.src_palm_pos.device
    dt = ctx.src_palm_pos.dtype
    zeros = torch.zeros(N, device=dev, dtype=dt)

    # ------------------------------------------------------------------ constants
    LIFT_TARGET = 0.10        # [m] cup height above spawn that counts as fully lifted
    LIFT_GATE = 0.05          # [m] minimum lift before alignment / bring_together are rewarded
    ALIGN_XY_GATE = 1.5 * ctx.cup_radius   # [m] rim-over-rim tolerance for the tilt gate
    DZ_MIN = 0.01             # [m] source rim must stay above the receiver rim (nesting guard)
    DZ_AIM = 0.03             # [m] preferred rim clearance — a low, gentle pour
    DZ_SIGMA = 0.03           # [m] tolerance around DZ_AIM (perception noise is a few cm)
    DZ_TILT_MAX = 0.08        # [m] no tilt reward when pouring from higher than this
    TILT_TARGET = 2.0         # [rad] beads leave past ~1.9 rad, so saturate slightly above
    TILT_FREE = 0.8           # [rad] grasp posture alone tilts the cup 20-35 deg; only charge beyond 46 deg
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped / knocked off the table
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side
    NEAR_DIST = 0.30          # [m] cup-centre distance inside which approach speed is regulated
    FORCE_SCALE = 5.0         # [N] tanh scale for all contact penalties (knowledge item 11)
    FOREIGN_DEADBAND = 1.0    # [N] incidental brushes (e.g. finger on table while grasping) are free

    # ------------------------------------------------------------------ clearance factors
    # clear in [0,1]: 1 when the cups do not touch, -> 0 as the cup-cup force grows.
    # Multiplies align, tilt and the success bonus: a bump costs the whole pour stack
    # (~16/step) instead of a small additive penalty that the level rewards would outrun.
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    not_nested_f = (~ctx.cups_nested).to(dt)

    # ------------------------------------------------------------------ stage 1: approach
    # weight 1.0 per arm, exp(-4d): wide basin, saturates ~0.68 in the grasp posture.
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)

    # ------------------------------------------------------------------ stage 2: grasp
    # 1.0 for the contact-established flag + 0.5 * closure weighted by a smooth proximity
    # factor (exp(-6d), softened from -8d: a few cm of cup-pose noise must not zero it).
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
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET, 0.0, 1.0)
    lift_src = 2.0 * g_src * lift_frac_src
    lift_rcv = 2.0 * g_rcv * lift_frac_rcv

    src_lifted_b = ctx.src_grasped & (h_src > LIFT_GATE)
    both_lifted_b = src_lifted_b & ctx.rcv_grasped & (h_rcv > LIFT_GATE)
    src_lifted_f = src_lifted_b.to(dt)
    both_lifted_f = both_lifted_b.to(dt)
    # weight 1.0 bonus for both cups in the air
    both_lifted = 1.0 * both_lifted_f

    # ------------------------------------------------------------------ stage 4: bring cups together / align rims
    mouth_delta = ctx.src_cup_mouth_pos - ctx.rcv_cup_mouth_pos          # (N,3)
    d_xy = torch.norm(mouth_delta[:, :2], dim=-1)                          # (N,)
    dz = mouth_delta[:, 2]                                                 # (N,)
    # xy: exp(-5d) (was -8d) — 2 cm error still pays 0.90, 5 cm pays 0.78; tolerant to noise.
    align_xy = torch.exp(-5.0 * d_xy)
    # z: PEAKED at DZ_AIM (3 cm rim clearance) instead of flat up to 12 cm. iter_01 poured from
    # ~10 cm above the receiver rim (src lift 0.30 vs rcv 0.14) and spilled 7 %; this pulls the
    # source rim down to a few cm. Below the rim (dz < DZ_MIN) the gate below zeroes everything.
    align_z = torch.exp(-((dz - DZ_AIM) / DZ_SIGMA) ** 2)
    above_rcv = (dz > DZ_MIN).to(dt)

    # bring_together, weight 1.0: once the SOURCE cup is lifted, pull its rim horizontally over
    # the receiver rim while staying above it and NOT nested (explicit nesting gate replaces the
    # old 3 cm height margin, so a low pour is allowed without reopening the nesting exploit).
    bring_together = 1.0 * src_lifted_f * above_rcv * not_nested_f * torch.exp(-4.0 * d_xy)

    # align, weight 3.0 — full alignment only with both cups lifted, above, not nested and with
    # the cups NOT touching (clear). Must beat lift (2+2 banked) + bring_together (1.0).
    align = 3.0 * both_lifted_f * above_rcv * not_nested_f * align_xy * align_z * clear

    aligned_b = (both_lifted_b & (~ctx.cups_nested)
                 & (d_xy < ALIGN_XY_GATE) & (dz > 0.0) & (dz < DZ_TILT_MAX))
    aligned_f = aligned_b.to(dt)

    # ------------------------------------------------------------------ stage 5: tilt
    # weight 3.0, only while aligned (now also: not from higher than 8 cm) and cups clear.
    tilt_frac = torch.clamp(ctx.src_cup_tilt / TILT_TARGET, 0.0, 1.0)
    tilt = 3.0 * aligned_f * tilt_frac * clear
    # premature tilt (weight 0.5): tilting well past grasp posture when the rim is NOT over the
    # receiver. Real spills are priced by spill_delta, so this stays small.
    tilt_premature = -0.5 * (1.0 - aligned_f) * torch.clamp(ctx.src_cup_tilt - TILT_FREE, min=0.0)
    # pour_rate, weight 0.3: gentle cost on fast tilting while aligned — a slow pour spills less.
    # tanh(|w|/1.5): 1 rad/s costs 0.17, saturates at 0.3, never competes with tilt (3.0).
    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    pour_rate = -0.3 * aligned_f * torch.tanh(src_ang_speed / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer
    # 50 * Δfrac = 2.5 per bead (20 beads) — one-shot payments, must dominate the shaping stack.
    # 30 * Δfrac = -1.5 per spilled bead (was -1.0): one landed bead still outweighs one lost,
    # but a high drop-pour that loses 1 bead in 2 is no longer profitable.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    spill_delta = -30.0 * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ collision / clearance (real-robot safety)
    # cup_contact, weight 1.0: bounded price on cup-cup force on top of the lost-stage gate.
    cup_contact = -1.0 * torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    # hand_foreign_*, weight 0.5 each with a 1 N deadband: hand on the other hand / other cup /
    # table. 0.5 << the ~3.5/step that arm earns from grasp+lift, so the hand is not taught to
    # avoid its own cup (the iter_00 failure mode of heavy receiver-cup penalties).
    f_src = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    f_rcv = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    hand_foreign_src = -0.5 * torch.tanh(f_src / FORCE_SCALE)
    hand_foreign_rcv = -0.5 * torch.tanh(f_rcv / FORCE_SCALE)
    # closing_speed, weight 1.0: rate at which the two cups approach each other inside 30 cm.
    # The level rewards pay ~21/step at the pour pose, so speed pressure is inherent; this term
    # only shapes the final approach to be slow (10 cm/s deadband, 30 cm/s costs ~0.76).
    rel = ctx.rcv_cup_pos - ctx.src_cup_pos                                   # (N,3)
    cup_dist = torch.norm(rel, dim=-1)                                        # (N,)
    u = rel / (cup_dist.unsqueeze(-1) + 1e-6)
    v_rel = ctx.src_cup_lin_vel - ctx.rcv_cup_lin_vel                         # (N,3)
    v_close = torch.sum(v_rel * u, dim=-1)                                    # >0 = approaching
    near_f = (cup_dist < NEAR_DIST).to(dt)
    closing_speed = -1.0 * near_f * torch.tanh(torch.clamp(v_close - 0.10, min=0.0) / 0.20)
    # nested, weight 0.5: small explicit price; the gates above already remove all pour income.
    nested = -0.5 * ctx.cups_nested.to(dt)

    # ------------------------------------------------------------------ constraints
    # Receiver-cup penalties stay SMALL (0.3 + 0.5 = 0.8/step worst case) versus the ~7/step
    # the receiver branch earns — larger values taught the left hand to avoid the cup (iter_00).
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
        "align": align,
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

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [4.88e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.7e-06 · max 0.000269
bead/spill: [0.000476, 0.00293, 0.0035, 0.000867, 0.000403, 0.000281, 0.000244, 0.000195, 0.000244, 0.000134]  min 7.32e-05 · mean 0.000865 · max 0.00557
done/drop: [0.00391, 0.00171, 0.000244, 0, 0, 0.000244, 0, 0, 0, 0]  min 0 · mean 0.000458 · max 0.0125
episode_lengths/step: [122, 564, 849, 892, 898, 894, 896, 889, 884, 783]  min 103 · mean 805 · max 899
reward/action_rate: [-0.0206, -0.0195, -0.0171, -0.0175, -0.0175, -0.0177, -0.0178, -0.0176, -0.0175, -0.0177]  min -0.0206 · mean -0.0179 · max -0.0165
reward/align: [0, 0.0914, 0.148, 0.248, 0.533, 1.12, 1.36, 1.4, 1.43, 1.43]  min 0 · mean 0.831 · max 1.48
reward/approach_rcv: [0.479, 0.576, 0.639, 0.644, 0.643, 0.645, 0.66, 0.674, 0.672, 0.671]  min 0.475 · mean 0.641 · max 0.677
reward/approach_src: [0.478, 0.599, 0.649, 0.647, 0.655, 0.668, 0.665, 0.661, 0.658, 0.663]  min 0.478 · mean 0.642 · max 0.67
reward/both_grasped: [0, 0.577, 0.829, 0.85, 0.863, 0.869, 0.877, 0.877, 0.883, 0.864]  min 0 · mean 0.789 · max 0.89
reward/both_lifted: [0, 0.407, 0.795, 0.837, 0.859, 0.863, 0.873, 0.873, 0.878, 0.855]  min 0 · mean 0.765 · max 0.883
reward/bring_together: [0, 0.164, 0.131, 0.189, 0.315, 0.506, 0.561, 0.575, 0.602, 0.595]  min 0 · mean 0.388 · max 0.62
reward/closing_speed: [-0.283, -0.111, -0.0897, -0.0677, -0.0925, -0.0798, -0.0567, -0.051, -0.05, -0.0483]  min -0.283 · mean -0.0703 · max -0.041
reward/cup_contact: [-0.207, -0.0139, -0.000625, -0.000594, -0.000579, -0.00489, -0.0132, -0.016, -0.012, -0.0162]  min -0.207 · mean -0.0114 · max 0
reward/drop: [-0.00977, -0.00525, -0.00061, -0.000122, 0, -0.000244, 0, -0.000122, -0.000122, -0.000122]  min -0.0188 · mean -0.00117 · max 0
reward/grasp_rcv: [0.0208, 0.733, 0.949, 0.954, 0.958, 0.953, 0.966, 0.978, 0.979, 0.964]  min 0.0208 · mean 0.896 · max 0.986
reward/grasp_src: [0.0207, 0.815, 0.974, 0.975, 0.971, 0.978, 0.984, 0.985, 0.983, 0.971]  min 0.0188 · mean 0.908 · max 0.997
reward/hand_foreign_rcv: [-0.0473, -0.0437, -0.00649, -0.00371, -0.00322, -0.0126, -0.0229, -0.0274, -0.016, -0.0249]  min -0.0547 · mean -0.0172 · max -0.00222
reward/hand_foreign_src: [-0.0439, -0.0258, -0.0116, -0.00347, -0.00535, -0.0272, -0.0376, -0.0308, -0.0141, -0.0242]  min -0.06 · mean -0.0198 · max -0.0022
reward/lift_rcv: [0, 1.03, 1.66, 1.69, 1.72, 1.73, 1.73, 1.76, 1.71, 1.69]  min 0 · mean 1.56 · max 1.77
reward/lift_src: [0, 1.23, 1.66, 1.72, 1.74, 1.76, 1.77, 1.78, 1.78, 1.75]  min 0 · mean 1.59 · max 1.79
reward/nested: [-0.0116, -0.00232, 0, 0, 0, -0.000244, -0.000244, -0.000244, 0, -0.000122]  min -0.0563 · mean -0.00101 · max 0
reward/pour_delta: [0.00061, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.26e-06 · max 0.00061
reward/pour_rate: [0, -0.000299, 0, 0, 0, -0.000142, -0.000363, 0, -0.000463, -0.000846]  min -0.00832 · mean -0.000466 · max 0
reward/spill_delta: [-0.0022, -0.0011, -0.00293, 0, 0, 0, 0, 0, 0, 0]  min -0.0033 · mean -0.000222 · max 0
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 0.00157, 0, 0, 0, 0.000763, 0.00117, 0, 0.00155, 0.0036]  min 0 · mean 0.00183 · max 0.0288
reward/tilt_premature: [-0.0184, -0.025, -0.0625, -0.0442, -0.017, -0.00319, -0.00044, -0.000646, -0.000322, -0.000233]  min -0.0877 · mean -0.0164 · max -6.17e-05
reward/total: [0.204, 5.84, 8.12, 8.52, 9.05, 9.87, 10.2, 10.3, 10.4, 10.2]  min 0.19 · mean 8.75 · max 10.4
reward/upright_rcv: [-0.14, -0.0866, -0.085, -0.0551, -0.0451, -0.0385, -0.0444, -0.0433, -0.0506, -0.0575]  min -0.14 · mean -0.0581 · max -0.0379
rewards/step: [26.8, 2.79e+03, 6.84e+03, 7.65e+03, 8.07e+03, 8.74e+03, 9.1e+03, 9.09e+03, 9e+03, 8.12e+03]  min 21.6 · mean 7.41e+03 · max 9.29e+03
task/aim_dist: [0.121, 0.397, 0.324, 0.302, 0.243, 0.16, 0.136, 0.13, 0.128, 0.129]  min 0.121 · mean 0.215 · max 0.614
task/cup_collision_rate: [0.261, 0.0164, 0.000732, 0.000732, 0.000732, 0.00635, 0.0181, 0.0266, 0.0205, 0.0239]  min 0 · mean 0.0161 · max 0.261
task/cups_center_dist: [0.178, 0.388, 0.309, 0.308, 0.255, 0.183, 0.176, 0.17, 0.179, 0.184]  min 0.169 · mean 0.238 · max 0.609
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0.0232, 0.00464, 0, 0, 0, 0.000488, 0.000488, 0.000488, 0, 0.000244]  min 0 · mean 0.00203 · max 0.113
task/rcv_cup_lift: [0.00398, 0.0928, 0.179, 0.146, 0.138, 0.114, 0.0984, 0.11, 0.0946, 0.0917]  min 0.0016 · mean 0.116 · max 0.191
task/rcv_grasped: [0, 0.654, 0.853, 0.861, 0.876, 0.88, 0.884, 0.889, 0.89, 0.874]  min 0 · mean 0.813 · max 0.897
task/rcv_hand_foreign_rate: [0.185, 0.139, 0.0374, 0.0239, 0.0205, 0.0706, 0.139, 0.18, 0.0977, 0.136]  min 0.0146 · mean 0.0885 · max 0.203
task/src_cup_lift: [0.00257, 0.123, 0.216, 0.197, 0.189, 0.156, 0.138, 0.149, 0.14, 0.134]  min 0.000908 · mean 0.151 · max 0.232
task/src_grasped: [0, 0.73, 0.865, 0.875, 0.88, 0.886, 0.892, 0.895, 0.897, 0.882]  min 0 · mean 0.82 · max 0.902
task/src_hand_foreign_rate: [0.169, 0.0979, 0.0588, 0.0212, 0.03, 0.151, 0.207, 0.189, 0.0869, 0.133]  min 0.0149 · mean 0.104 · max 0.243
task/src_tilt_deg: [26, 29.8, 42.6, 40.6, 34, 31.9, 29.5, 30, 29.3, 30]  min 10 · mean 31.6 · max 46.7
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Facts for this round (from the loop operator, no design advice):

1. Best result so far is NOT the previous function above but the one before it (iteration 1, trained on the older environment without collision signals / perception noise / physics randomisation). It reached episode_success 0.73 (max 0.75) at epoch 413 with: grasp S/R 0.88/0.87, src lift 0.30 m, rcv lift 0.15 m, src tilt 84°, bead_in_target 0.69, spill 0.07, nested_rate 0.01, aim_dist 0.10 m. Its components were: approach_src, approach_rcv, grasp_src, grasp_rcv, both_grasped, lift_src, lift_rcv, both_lifted, align, bring_together, tilt, tilt_premature, pour_delta, spill_delta, upright_rcv, drop, action_rate, success (weights: approach 1.0/arm exp(-4d); grasp 1.0 flag + 0.5 closure; both_grasped 1.0; lift toward 0.10 m; align/bring_together gated on lift ≥ 0.05 m; tilt saturating at 2.0 rad, free below 0.8 rad; pour_delta on d_in_target; spill_delta on d_spill; upright_rcv 0.3; drop 0.5). In the rollout video of that policy the two cups bumped into each other early and the source cup was lifted ~30 cm and poured from high above.

2. The environment changes that motivated the previous function (iteration 2) are permanent: the policy sees delayed/noisy cup poses, cup mass / joint gains / cup friction are randomised as training progresses (curriculum keyed on success — `adr/progress` stays 0 until the policy succeeds), and `ctx.cup_cup_force`, `ctx.src_hand_foreign_force`, `ctx.rcv_hand_foreign_force` are available. `task/cup_collision_rate` and `task/*_hand_foreign_rate` are now logged (see the table). This policy will run on the real robot, so cup–cup and hand–foreign collisions remain a safety hazard.

3. The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
