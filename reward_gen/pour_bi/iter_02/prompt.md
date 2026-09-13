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
    DZ_MIN, DZ_MAX = 0.03, 0.12            # [m] source rim above receiver rim window
    TILT_TARGET = 2.0         # [rad] beads leave past ~1.9 rad, so saturate slightly above
    TILT_FREE = 0.8           # [rad] grasp posture alone tilts the cup 20-35 deg; only charge beyond 46 deg
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped / knocked off the table
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side

    # ------------------------------------------------------------------ stage 1: approach
    # weight 1.0 per arm, exp(-4d): wider basin than exp(-5d) so a retreated hand (20 cm) is
    # still pulled back; saturates ~0.68 in the grasp posture (palm ~9 cm from cup origin).
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-4.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-4.0 * d_rcv)

    # ------------------------------------------------------------------ stage 2: grasp
    # 1.0 for the contact-established flag + 0.5 * closure weighted by a SMOOTH proximity
    # factor (the env already forbids closing far from the cup, so no hard gate is needed;
    # the old hard gate at 10 cm left a flat zone in which the receiver hand retreated for free).
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    prox_src = torch.exp(-8.0 * d_src)
    prox_rcv = torch.exp(-8.0 * d_rcv)
    grasp_src = 1.0 * g_src + 0.5 * prox_src * torch.clamp(ctx.src_hand_closure, 0.0, 1.0)
    grasp_rcv = 1.0 * g_rcv + 0.5 * prox_rcv * torch.clamp(ctx.rcv_hand_closure, 0.0, 1.0)
    # weight 1.0 bonus for holding BOTH cups: makes the second grasp worth more than the first,
    # so finishing the receiver branch beats idling with the source cup (3.7/step in the last run).
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
    # weight 1.0 bonus for both cups in the air (same combinatorial logic as both_grasped)
    both_lifted = 1.0 * both_lifted_f

    # ------------------------------------------------------------------ stage 4: bring cups together / align rims
    mouth_delta = ctx.src_cup_mouth_pos - ctx.rcv_cup_mouth_pos          # (N,3)
    d_xy = torch.norm(mouth_delta[:, :2], dim=-1)                          # (N,)
    dz = mouth_delta[:, 2]                                                 # (N,)
    dz_err = torch.clamp(DZ_MIN - dz, min=0.0) + torch.clamp(dz - DZ_MAX, min=0.0)
    align_xy = torch.exp(-8.0 * d_xy)
    align_z = torch.exp(-10.0 * dz_err)

    # bring_together, weight 1.0: once the SOURCE cup is lifted, pull its rim horizontally over the
    # receiver rim while staying ABOVE it (dz > DZ_MIN). Independent of the receiver grasp so the
    # cups stop drifting apart (0.38 m last run) and the source arm keeps a gradient after lift
    # saturation. The height gate means pushing the source cup INTO the receiver cup earns nothing.
    above_rcv = (dz > DZ_MIN).to(dt)
    bring_together = 1.0 * src_lifted_f * above_rcv * torch.exp(-4.0 * d_xy)

    # align, weight 3.0 — full alignment only with both cups lifted; must beat lift (2+2 banked)
    # plus bring_together (1.0) so lifting the receiver remains the better path.
    align = 3.0 * both_lifted_f * align_xy * align_z

    aligned_b = both_lifted_b & (d_xy < ALIGN_XY_GATE) & (dz > 0.0)
    aligned_f = aligned_b.to(dt)

    # ------------------------------------------------------------------ stage 5: tilt
    # weight 3.0, only while aligned; saturates at TILT_TARGET so full pour is the goal.
    tilt_frac = torch.clamp(ctx.src_cup_tilt / TILT_TARGET, 0.0, 1.0)
    tilt = 3.0 * aligned_f * tilt_frac
    # premature tilt (weight 0.5): tilting well past grasp posture when the rim is NOT over the
    # receiver. Real spills are priced by spill_delta, so this stays small.
    tilt_premature = -0.5 * (1.0 - aligned_f) * torch.clamp(ctx.src_cup_tilt - TILT_FREE, min=0.0)

    # ------------------------------------------------------------------ stage 6: bead transfer
    # 50 * Δfrac = 2.5 per bead (20 beads) — one-shot payments, must dominate the shaping stack.
    # 20 * Δfrac = -1.0 per spilled bead: one bead in the target outweighs two lost.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    spill_delta = -20.0 * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ constraints
    # Receiver-cup penalties are deliberately SMALL (0.3 + 0.5 = 0.8/step worst case) versus the
    # ~7/step the receiver branch can earn. Last run they were 1.0 + 2.0 = 3.0/step for the rest
    # of the episode after one accidental bump (~-2600 total), and the policy learned to keep the
    # left hand away from the receiver cup entirely (approach_rcv fell from 0.40 to 0.35).
    upright_rcv = -0.3 * torch.clamp(ctx.rcv_cup_tilt, 0.0, 1.0)
    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)

    # action-rate regularisation (weight 0.02 — tiny, must never compete with any stage term)
    action_rate = -0.02 * torch.mean((ctx.actions - ctx.prev_actions) ** 2, dim=-1)

    # ------------------------------------------------------------------ success bonus
    success = 10.0 * ctx.success.to(dt)

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
        "pour_delta": pour_delta,
        "spill_delta": spill_delta,
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

bead/in_target: [1.22e-05, 7.32e-05, 0.0196, 0.595, 0.719, 0.748, 0.769, 0.77, 0.772, 0.734]  min 1.22e-05 · mean 0.549 · max 0.798
bead/spill: [7.32e-05, 0.00283, 0.015, 0.142, 0.0779, 0.0672, 0.0482, 0.07, 0.0599, 0.0712]  min 7.32e-05 · mean 0.0616 · max 0.293
done/drop: [0.0105, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000217 · max 0.0132
episode_lengths/step: [106, 884, 830, 877, 891, 894, 856, 881, 890, 857]  min 96.1 · mean 850 · max 898
reward/action_rate: [-0.0205, -0.0183, -0.0174, -0.0163, -0.0165, -0.0163, -0.0165, -0.0161, -0.0164, -0.017]  min -0.0206 · mean -0.017 · max -0.0159
reward/align: [0, 0.637, 1.56, 1.9, 2, 2.1, 2.2, 2.21, 2.23, 2.22]  min 0 · mean 1.81 · max 2.29
reward/approach_rcv: [0.482, 0.667, 0.63, 0.633, 0.636, 0.634, 0.631, 0.626, 0.627, 0.632]  min 0.482 · mean 0.631 · max 0.669
reward/approach_src: [0.485, 0.676, 0.64, 0.651, 0.656, 0.665, 0.667, 0.65, 0.653, 0.658]  min 0.485 · mean 0.653 · max 0.676
reward/both_grasped: [0, 0.716, 0.776, 0.85, 0.856, 0.87, 0.866, 0.872, 0.872, 0.861]  min 0 · mean 0.807 · max 0.888
reward/both_lifted: [0, 0.695, 0.761, 0.841, 0.851, 0.861, 0.862, 0.861, 0.862, 0.855]  min 0 · mean 0.789 · max 0.884
reward/bring_together: [0, 0.12, 0.557, 0.718, 0.699, 0.765, 0.762, 0.756, 0.735, 0.753]  min 0 · mean 0.627 · max 0.8
reward/drop: [-0.0242, -0.00171, -0.00195, -0.000854, -0.000488, -0.000488, -0.000366, -0.000488, -0.000488, -0.000488]  min -0.0242 · mean -0.00187 · max 0
reward/grasp_rcv: [0.0148, 0.868, 0.914, 0.947, 0.944, 0.96, 0.958, 0.973, 0.968, 0.953]  min 0.0148 · mean 0.917 · max 0.982
reward/grasp_src: [0.0146, 0.91, 0.906, 0.96, 0.953, 0.968, 0.966, 0.981, 0.979, 0.975]  min 0.0146 · mean 0.926 · max 0.997
reward/lift_rcv: [2.58e-05, 1.49, 1.6, 1.71, 1.72, 1.74, 1.74, 1.75, 1.74, 1.72]  min 2.58e-05 · mean 1.62 · max 1.78
reward/lift_src: [0, 1.57, 1.62, 1.73, 1.73, 1.75, 1.74, 1.75, 1.74, 1.73]  min 0 · mean 1.63 · max 1.79
reward/pour_delta: [0.00061, 0, 0.00305, 0.047, 0.0586, 0.0464, 0.0629, 0.0439, 0.0555, 0.0488]  min 0 · mean 0.0388 · max 0.0806
reward/spill_delta: [0, -0.000732, -0.00146, -0.00488, -0.00391, -0.00317, -0.0022, -0.00171, -0.00244, -0.000732]  min -0.0151 · mean -0.00257 · max 0
reward/success: [0, 0, 0.0977, 6.5, 7.64, 8.07, 8.1, 8.15, 8.19, 7.96]  min 0 · mean 5.83 · max 8.32
reward/tilt: [0, 0.0529, 1, 2.02, 2.18, 2.25, 2.27, 2.38, 2.43, 2.38]  min 0 · mean 1.79 · max 2.48
reward/tilt_premature: [-0.0302, -0.0117, -0.0338, -0.0247, -0.0174, -0.0093, -0.00693, -0.0121, -0.00912, -0.00879]  min -0.0455 · mean -0.0147 · max -0.00535
reward/total: [0.721, 8.26, 10.9, 19.3, 20.7, 21.5, 21.7, 21.8, 21.9, 21.6]  min 0.592 · mean 17.9 · max 22.3
reward/upright_rcv: [-0.152, -0.0815, -0.0745, -0.108, -0.122, -0.106, -0.105, -0.098, -0.0961, -0.091]  min -0.152 · mean -0.0984 · max -0.0699
rewards/step: [61.2, 7.3e+03, 9.1e+03, 1.68e+04, 1.87e+04, 1.92e+04, 1.86e+04, 1.91e+04, 1.95e+04, 1.88e+04]  min 56.4 · mean 1.55e+04 · max 1.99e+04
task/aim_dist: [0.128, 0.138, 0.107, 0.112, 0.096, 0.111, 0.0948, 0.0922, 0.0835, 0.0883]  min 0.08 · mean 0.111 · max 0.352
task/cups_center_dist: [0.19, 0.137, 0.18, 0.231, 0.22, 0.237, 0.224, 0.23, 0.225, 0.227]  min 0.126 · mean 0.217 · max 0.459
task/episode_success: [0, 0, 0.00952, 0.64, 0.753, 0.796, 0.8, 0.805, 0.806, 0.789]  min 0 · mean 0.575 · max 0.824
task/nested_rate: [0.0239, 0.24, 0.0525, 0.0249, 0.00684, 0.00146, 0.00244, 0.00317, 0.0022, 0.00732]  min 0.000488 · mean 0.0442 · max 0.315
task/rcv_cup_lift: [0.00185, 0.151, 0.134, 0.153, 0.145, 0.152, 0.145, 0.143, 0.139, 0.128]  min 0.00185 · mean 0.135 · max 0.217
task/rcv_grasped: [0.000244, 0.77, 0.826, 0.87, 0.87, 0.884, 0.884, 0.893, 0.89, 0.876]  min 0.000244 · mean 0.839 · max 0.904
task/src_cup_lift: [0.000598, 0.174, 0.233, 0.323, 0.303, 0.33, 0.309, 0.317, 0.305, 0.293]  min 0.000598 · mean 0.274 · max 0.344
task/src_grasped: [0, 0.815, 0.827, 0.879, 0.873, 0.886, 0.881, 0.896, 0.893, 0.888]  min 0 · mean 0.843 · max 0.91
task/src_tilt_deg: [29.2, 30.8, 55.5, 86.1, 90.1, 90.5, 90.2, 97.3, 98.9, 96.5]  min 16.2 · mean 79.1 · max 101
task/success_now: [0, 0, 0.00977, 0.65, 0.764, 0.807, 0.81, 0.815, 0.819, 0.796]  min 0 · mean 0.583 · max 0.832

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Observations from the operator after watching the trained iter_01 policy and from the sim-to-real requirements (this round's environment changes):

1. In the rollout video the two cups often BUMP INTO EACH OTHER early — right after both are lifted, the policy drags them together fast and they collide before the source cup is tilted. On the real robot this is a collision hazard. The environment now exposes `ctx.cup_cup_force`, `ctx.src_hand_foreign_force` and `ctx.rcv_hand_foreign_force` (see the class definition and knowledge item 11); these are new since the previous reward. Also the previous metrics table shows `task/cup_collision_rate` / `task/*_hand_foreign_rate` were not yet logged, so treat them as unknown-but-observed-high.

2. The receiver cup is lifted less than the source cup (~15 cm vs ~30 cm) and the source pours from high above; a gentler pour (source rim only a few cm above the receiver rim, slower tilt) spills less. Spill was 7% at the end of training.

3. The environment now adds perception delay/noise on the cup poses seen by the policy and randomizes cup mass, joint gains and cup friction as training progresses (curriculum keyed on success). Rewards that depend on very precise instantaneous cup geometry (sharp exp(-k·d) with large k) will be noisier; prefer tolerances of a few centimetres.

The policy must still complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
