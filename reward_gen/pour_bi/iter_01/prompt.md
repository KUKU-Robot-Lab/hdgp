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
    zeros = torch.zeros(N, device=dev, dtype=ctx.src_palm_pos.dtype)
    ones = torch.ones_like(zeros)

    # ------------------------------------------------------------------ constants
    LIFT_TARGET = 0.10        # [m] cup height above spawn that counts as fully lifted
    LIFT_GATE = 0.05          # [m] minimum lift before alignment is rewarded
    NEAR_PALM = 0.10          # [m] palm-cup distance below which closure is rewarded
    ALIGN_XY_GATE = 1.5 * ctx.cup_radius   # [m] rim-over-rim tolerance for the tilt gate
    DZ_MIN, DZ_MAX = 0.03, 0.12            # [m] source rim above receiver rim window
    TILT_TARGET = 2.0         # [rad] beads leave past ~1.9 rad, so saturate slightly above
    TILT_FREE = 0.5           # [rad] tilt allowed before "premature tilt" is charged
    DROP_DEPTH = 0.03         # [m] cup below spawn height => dropped / knocked off
    RCV_TOPPLED = 1.2         # [rad] receiver cup lying on its side

    # ------------------------------------------------------------------ stage 1: approach
    # weight 1.0 per arm — smallest stage, only has to pull the palms toward the cups
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 1.0 * torch.exp(-5.0 * d_src)
    approach_rcv = 1.0 * torch.exp(-5.0 * d_rcv)

    # ------------------------------------------------------------------ stage 2: grasp
    # 1.0 for the contact-established flag + 0.5 for measured closure while the palm is near.
    # Closure is gated on proximity so closing in free air earns nothing.
    near_src = (d_src < NEAR_PALM).to(zeros.dtype)
    near_rcv = (d_rcv < NEAR_PALM).to(zeros.dtype)
    g_src = ctx.src_grasped.to(zeros.dtype)
    g_rcv = ctx.rcv_grasped.to(zeros.dtype)
    grasp_src = 1.0 * g_src + 0.5 * near_src * torch.clamp(ctx.src_hand_closure, 0.0, 1.0)
    grasp_rcv = 1.0 * g_rcv + 0.5 * near_rcv * torch.clamp(ctx.rcv_hand_closure, 0.0, 1.0)

    # ------------------------------------------------------------------ stage 3: lift
    # weight 2.0 per arm — larger than grasp so holding on and lifting beats holding on the table.
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET, 0.0, 1.0)
    lift_src = 2.0 * g_src * lift_frac_src
    lift_rcv = 2.0 * g_rcv * lift_frac_rcv

    both_lifted = (
        ctx.src_grasped & ctx.rcv_grasped & (h_src > LIFT_GATE) & (h_rcv > LIFT_GATE)
    )
    both_lifted_f = both_lifted.to(zeros.dtype)

    # ------------------------------------------------------------------ stage 4: align rims
    # weight 3.0 — must beat lift (2+2 already banked) so the arms keep moving after lifting.
    mouth_delta = ctx.src_cup_mouth_pos - ctx.rcv_cup_mouth_pos          # (N,3)
    d_xy = torch.norm(mouth_delta[:, :2], dim=-1)                          # (N,)
    dz = mouth_delta[:, 2]                                                 # (N,)
    dz_err = torch.clamp(DZ_MIN - dz, min=0.0) + torch.clamp(dz - DZ_MAX, min=0.0)
    align_xy = torch.exp(-8.0 * d_xy)
    align_z = torch.exp(-10.0 * dz_err)
    align = 3.0 * both_lifted_f * align_xy * align_z

    aligned = both_lifted & (d_xy < ALIGN_XY_GATE) & (dz > 0.0)
    aligned_f = aligned.to(zeros.dtype)

    # ------------------------------------------------------------------ stage 5: tilt
    # weight 3.0, only while aligned; saturates at TILT_TARGET so full pour is the goal.
    tilt_frac = torch.clamp(ctx.src_cup_tilt / TILT_TARGET, 0.0, 1.0)
    tilt = 3.0 * aligned_f * tilt_frac
    # premature tilt: tilting when the rim is NOT over the receiver spills beads on the table
    tilt_premature = -1.0 * (1.0 - aligned_f) * torch.clamp(ctx.src_cup_tilt - TILT_FREE, min=0.0)

    # ------------------------------------------------------------------ stage 6: bead transfer
    # 50 * Δfrac = 2.5 per bead (20 beads) — one-shot payments, must dominate the shaping stack.
    # 20 * Δfrac = -1.0 per spilled bead: one bead in the target outweighs two lost.
    pour_delta = 50.0 * torch.clamp(ctx.d_in_target, min=0.0)
    spill_delta = -20.0 * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ constraints
    # receiver cup upright (weight 1.0, saturates at 1 rad)
    upright_rcv = -1.0 * torch.clamp(ctx.rcv_cup_tilt, 0.0, 1.0)
    # dropped / knocked-off cup or toppled receiver cup (flat -2.0 each while it lasts)
    src_dropped = (h_src < -DROP_DEPTH).to(zeros.dtype)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(zeros.dtype)
    drop = -2.0 * (src_dropped + rcv_dropped)

    # action-rate regularisation (weight 0.02 — tiny, must never compete with any stage term)
    action_rate = -0.02 * torch.mean((ctx.actions - ctx.prev_actions) ** 2, dim=-1)

    # ------------------------------------------------------------------ success bonus
    success = 10.0 * ctx.success.to(zeros.dtype)

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
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

bead/in_target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.49e-06 · max 0.000293
bead/spill: [9.77e-05, 0.000244, 0.000342, 0.00215, 0.00132, 0.00083, 0.00083, 0.000635, 0.000781, 0.000391]  min 0 · mean 0.000975 · max 0.00459
done/drop: [0.00879, 0, 0.00195, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000411 · max 0.0107
episode_lengths/step: [105, 867, 752, 826, 806, 878, 885, 888, 859, 886]  min 94.4 · mean 803 · max 898
reward/action_rate: [-0.0204, -0.0193, -0.0185, -0.0168, -0.0176, -0.0168, -0.0172, -0.0177, -0.0176, -0.0175]  min -0.0207 · mean -0.0178 · max -0.0163
reward/align: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 5.43e-06 · max 0.000852
reward/approach_rcv: [0.402, 0.318, 0.355, 0.351, 0.35, 0.333, 0.333, 0.34, 0.347, 0.356]  min 0.318 · mean 0.348 · max 0.425
reward/approach_src: [0.407, 0.336, 0.472, 0.569, 0.605, 0.621, 0.628, 0.614, 0.615, 0.619]  min 0.326 · mean 0.559 · max 0.631
reward/drop: [-0.0957, -0.00195, -0.00586, 0, -0.00195, 0, 0, 0, 0, 0]  min -0.0957 · mean -0.00338 · max 0
reward/grasp_rcv: [0, 0.00488, 0.00488, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.0037 · max 0.0708
reward/grasp_src: [0, 0.00586, 0.639, 0.909, 1.01, 1.03, 1.06, 1.03, 1.02, 1.04]  min 0 · mean 0.82 · max 1.07
reward/lift_rcv: [0, 0.000364, 0.000577, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.00149 · max 0.0337
reward/lift_src: [0, 0.000404, 0.214, 1.41, 1.58, 1.64, 1.71, 1.68, 1.71, 1.73]  min 0 · mean 1.26 · max 1.8
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.89e-06 · max 0.00244
reward/spill_delta: [-0.000977, 0, 0, 0, -0.000977, 0, 0, 0, 0, 0]  min -0.00781 · mean -0.000129 · max 0
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt_premature: [-0.172, -0.00269, -0.0433, -0.188, -0.0444, -0.019, -0.00697, -0.00827, -0.0115, -0.0126]  min -0.215 · mean -0.035 · max -4.5e-06
reward/total: [-0.0121, 0.613, 1.56, 3.01, 3.44, 3.56, 3.68, 3.6, 3.63, 3.69]  min -0.0121 · mean 2.89 · max 3.78
reward/upright_rcv: [-0.489, -0.00624, -0.0234, -0.00654, -0.00359, -0.00323, -0.000574, -0.000405, -0.0018, -0.0016]  min -0.489 · mean -0.0165 · max -0.000145
rewards/step: [8.95, 527, 1.13e+03, 2.36e+03, 2.71e+03, 3.14e+03, 3.21e+03, 3.22e+03, 3.15e+03, 3.27e+03]  min 8.95 · mean 2.42e+03 · max 3.32e+03
task/aim_dist: [0.129, 0.319, 0.345, 0.457, 0.431, 0.405, 0.389, 0.377, 0.398, 0.371]  min 0.129 · mean 0.383 · max 0.756
task/cups_center_dist: [0.189, 0.32, 0.329, 0.451, 0.42, 0.393, 0.387, 0.38, 0.392, 0.377]  min 0.189 · mean 0.38 · max 0.759
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0.0293, 0, 0.00391, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.00227 · max 0.0732
task/rcv_cup_lift: [0.00244, 7.96e-05, 0.000631, 0.000119, 2.65e-05, 8.89e-06, 1.17e-05, 7.98e-06, 2.98e-05, 4.17e-05]  min -1.57e-05 · mean 0.000631 · max 0.0495
task/rcv_grasped: [0, 0.00488, 0.00488, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.00359 · max 0.0674
task/src_cup_lift: [0.000693, 3.49e-05, 0.0125, 0.09, 0.0974, 0.108, 0.109, 0.103, 0.112, 0.114]  min 2.5e-05 · mean 0.0814 · max 0.183
task/src_grasped: [0, 0.00586, 0.639, 0.854, 0.87, 0.874, 0.89, 0.881, 0.878, 0.891]  min 0 · mean 0.716 · max 0.919
task/src_tilt_deg: [29, 0.64, 16.7, 34.5, 25.7, 22.3, 19.1, 20.6, 23.1, 22.6]  min 0.233 · mean 19.9 · max 35
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.
