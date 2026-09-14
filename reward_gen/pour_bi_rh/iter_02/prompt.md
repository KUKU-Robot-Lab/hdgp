You are an expert in robotics, reinforcement learning and code generation.

We control a bimanual robot: two 7-DOF OpenArm arms, each carrying an Inspire RH56F1 five-finger \
hand, standing at a table. The RH56F1 is an UNDER-ACTUATED hand: only 6 joints per hand are driven \
(thumb abduction, thumb flexion, and one flexion drive for each of the index, middle, ring and pinky \
fingers); the remaining finger joints follow the driven ones through fixed mechanical couplings, so a \
finger always curls as a whole. The RIGHT arm is the "source" arm (src_*), the LEFT arm is the \
"receiver" arm (rcv_*). Two identical cups stand on the table: the source cup (in front of the right \
hand, filled with 20 small beads) and the receiver cup (in front of the left hand, empty). \
Positions are in metres in a frame whose origin is at the robot base; +z is up, +x is forward (away \
from the robot), +y is to the robot's left. The table top is at z = table_z.

The action space is a normalized `Box(-1, 1, (24,), float32)`:
  actions[0:6]   = source palm 6-DoF target offset (xyz + yaw/pitch/roll) — the arm is moved \
by a geometric-fabrics controller toward this target
  actions[6:12]  = source hand closure commands, one per driven joint \
(thumb abduction, thumb flexion, index, middle, ring, pinky; 0 = open, 1 = closed)
  actions[12:18] = receiver palm 6-DoF target offset
  actions[18:24] = receiver hand closure commands
The hand controller stops a finger automatically once one of its links touches its own cup \
(contact freeze); opening is always allowed. Fingers can only close when the palm is near its cup. \
The "cups" are slim shaker bodies about 6 cm in diameter and 11 cm tall, and the hand is small: a \
FINGERTIP (precision) grasp — thumb tip on one side, the tips of some fingers on the other — is the \
expected way to hold the cup; a full wrap-around power grasp is NOT required and usually not \
possible. The cups are light, so a finger brushing the cup easily knocks it over: approach slowly \
and touch it only with the fingertips.

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

I want it to fulfil the following task: Using both arms, grasp each cup (a fingertip grasp is fine, no wrap-around needed): the right hand grasps the source cup (which contains the beads) and the left hand grasps the empty receiver cup. Lift both cups off the table, bring the mouth of the source cup over the mouth of the receiver cup, and tilt the source cup so that the beads pour into the receiver cup. Keep the receiver cup upright, do not drop either cup, and spill as few beads as possible. The task is finished when at least half of the beads are inside the receiver cup with little spill.
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


def _near(d: torch.Tensor, k: float) -> torch.Tensor:
    """Bounded closeness in (0, 1]: 1 at d = 0, ~0.37 at d = 1/k."""
    return torch.exp(-k * d)


def _band_err(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """0 inside [lo, hi], linear distance outside."""
    return torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)


def _cyl_coords(points: torch.Tensor, cup_pos: torch.Tensor, cup_up: torch.Tensor):
    """Cup-cylinder coordinates of (N,F,3) points: axial height, radial distance, radial unit dir."""
    rel = points - cup_pos[:, None, :]
    up = cup_up[:, None, :]
    axial = (rel * up).sum(dim=-1)                                   # (N,F) height along cup axis
    radial_vec = rel - axial[..., None] * up                          # (N,F,3)
    radial = torch.norm(radial_vec, dim=-1)                           # (N,F)
    radial_dir = radial_vec / (radial[..., None] + 1e-6)
    return axial, radial, radial_dir


def _hand_terms(ctx, palm_pos, tips, cup_pos, cup_up, grasped):
    """Per-arm reach / fingertip-grasp geometry. Everything (N,)."""
    zero = torch.zeros_like(cup_pos[:, 0])
    r_wall = ctx.cup_radius + 0.004          # outer wall ~ inner radius + wall thickness
    lo = ctx.cup_bottom_z + 0.03             # keep tips >= 3 cm above the cup bottom (table clearance)
    hi_f = ctx.cup_mouth_z - 0.02            # finger tips >= 2 cm below the rim
    hi_t = ctx.cup_mouth_z - 0.025           # thumb tip >= 2.5 cm below the rim (anti rim-hook)

    d_palm = torch.norm(palm_pos - cup_pos, dim=-1)
    # two-scale approach: k=4 keeps gradient at 20-30 cm, k=15 sharpens the last few cm
    approach = 0.5 * _near(d_palm, 4.0) + 0.5 * _near(d_palm, 15.0)

    axial, radial, rdir = _cyl_coords(tips, cup_pos, cup_up)
    rad_err = torch.clamp(torch.abs(radial - r_wall) - 0.01, min=0.0)  # 1 cm tolerance for tip-pad offset
    thumb_err = rad_err[:, 0] + _band_err(axial[:, 0], lo, hi_t)
    finger_err = (rad_err[:, 1:3] + _band_err(axial[:, 1:3], lo, hi_f)).mean(dim=-1)  # index + middle

    # opposition: thumb radial direction vs mean index/middle radial direction (cos -1 = opposite sides)
    fdir = rdir[:, 1:3, :].mean(dim=1)
    fdir = fdir / (torch.norm(fdir, dim=-1, keepdim=True) + 1e-6)
    cos_opp = (rdir[:, 0, :] * fdir).sum(dim=-1)
    opp = torch.clamp(-cos_opp, 0.0, 1.0)

    # soft palm gate: 1 within 8 cm, 0.37 at 18 cm -> the receiver (at ~19 cm) still feels it
    near_gate = _near(torch.clamp(d_palm - 0.08, min=0.0), 10.0)
    tip_place = near_gate * (0.5 * _near(thumb_err, 25.0) + 0.5 * _near(finger_err, 25.0)) * (0.5 + 0.5 * opp)

    thumb_below_rim = axial[:, 0] < (ctx.cup_mouth_z - 0.015)
    good_grasp = (grasped & thumb_below_rim & (cos_opp < -0.3)).to(zero.dtype)
    held = (grasped & thumb_below_rim).to(zero.dtype)                 # used for lift (opposition may flicker)

    # rim hook: thumb tip at/above rim height AND over the cup footprint, while the palm is near
    over_rim = torch.clamp((axial[:, 0] - (ctx.cup_mouth_z - 0.02)) / 0.02, 0.0, 1.0)
    over_footprint = (radial[:, 0] < r_wall + 0.025).to(zero.dtype)
    rim_hook = near_gate * over_rim * over_footprint

    return approach, tip_place, good_grasp, held, rim_hook


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = torch.zeros_like(ctx.src_cup_tilt)

    # ------------------------------------------------------------------ stages 0-1: reach + fingertip grasp
    a_s, tp_s, gg_s, held_s, hook_s = _hand_terms(ctx, ctx.src_palm_pos, ctx.src_tips_pos,
                                                  ctx.src_cup_pos, ctx.src_cup_up, ctx.src_grasped)
    a_r, tp_r, gg_r, held_r, hook_r = _hand_terms(ctx, ctx.rcv_palm_pos, ctx.rcv_tips_pos,
                                                  ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_grasped)
    approach_src = 1.0 * a_s                  # weight 1.0 per arm (<= 1.0/step)
    approach_rcv = 1.0 * a_r
    tips_src = 1.0 * tp_s                     # weight 1.0: fingertips onto the wall below the rim, opposed
    tips_rcv = 1.0 * tp_r
    grasp_src = 1.0 * gg_s                    # weight 1.0: only a geometrically correct grasp is paid
    grasp_rcv = 1.0 * gg_r
    rim_hook_pen = -1.0 * (hook_s + hook_r)   # weight 1.0 = the grasp bonus it used to steal

    # ------------------------------------------------------------------ stage 2: lift (independent per arm)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # tanh(h/3cm): 1 cm -> 0.32, 3 cm -> 0.76; weight 3.0 so lifting 1 cm beats any constant pre-grasp term
    lift_src = 3.0 * held_s * torch.tanh(torch.clamp(h_src, min=0.0) / 0.03)
    lift_rcv = 3.0 * held_r * torch.tanh(torch.clamp(h_rcv, min=0.0) / 0.03)
    lifted_src = held_s * (h_src > 0.03).to(zero.dtype)
    lifted_rcv = held_r * (h_rcv > 0.03).to(zero.dtype)
    lifted = lifted_src * lifted_rcv

    # ------------------------------------------------------------------ stage 3: align rims (both lifted)
    mouth_dxy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    dz_err = torch.clamp(0.04 - mouth_dz, min=0.0) + torch.clamp(mouth_dz - 0.12, min=0.0)  # 4..12 cm window
    align_q = (0.5 * _near(mouth_dxy, 5.0) + 0.5 * _near(mouth_dxy, 20.0)) * _near(dz_err, 15.0)
    align = 4.0 * lifted * align_q            # weight 4.0 > lift 3.0: carrying toward the receiver must pay
    aligned = lifted * (mouth_dxy < 0.05).to(zero.dtype) * (dz_err < 0.015).to(zero.dtype)

    # ------------------------------------------------------------------ stage 4: tilt and pour
    tilt = 3.0 * aligned * torch.clamp(ctx.src_cup_tilt / 2.0, 0.0, 1.0)   # doorway to pouring only
    # 20 beads: +10 per transferred bead (full transfer = 200 ~ 50 steps of align) -> pouring dominates
    pour_delta = 200.0 * ctx.d_in_target
    spill_delta = -100.0 * ctx.d_spill        # -5 per spilled bead: costly but never worth refusing to tilt

    # ------------------------------------------------------------------ stage 5: success
    success = 10.0 * ctx.success.to(zero.dtype)   # 10/step > align + tilt (7): holding the goal is best

    # ------------------------------------------------------------------ constraints
    # tilting a bead-filled source cup before alignment spills; < 0.8 rad is harmless while carrying
    has_beads = (ctx.bead_in_source_frac > 0.05).to(zero.dtype)
    pre_tilt_pen = -1.0 * (1.0 - aligned) * has_beads * torch.tanh(2.0 * torch.clamp(ctx.src_cup_tilt - 0.8, min=0.0))
    # receiver upright: strict ONLY while held (the always-on version taught the left arm to stay away)
    rcv_upright_pen = -1.0 * ctx.rcv_grasped.to(zero.dtype) * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.1, min=0.0))
    # knocked-over unheld cup, symmetric for both arms, small (0.3) so it does not deter approaching
    src_free = 1.0 - ctx.src_grasped.to(zero.dtype)
    rcv_free = 1.0 - ctx.rcv_grasped.to(zero.dtype)
    knock_pen = -0.3 * (src_free * torch.tanh(3.0 * torch.clamp(ctx.src_cup_tilt - 0.2, min=0.0))
                        + rcv_free * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.2, min=0.0)))
    # drop: an unheld cup above the table falling down
    drop_pen = -1.0 * (src_free * (h_src > 0.015).to(zero.dtype) * torch.tanh(torch.clamp(-ctx.src_cup_lin_vel[:, 2], min=0.0) / 0.5)
                       + rcv_free * (h_rcv > 0.015).to(zero.dtype) * torch.tanh(torch.clamp(-ctx.rcv_cup_lin_vel[:, 2], min=0.0) / 0.5))
    # fast cup motion (thrown / swung): only speeds above 0.4 m/s are penalised, weight 0.2 per cup
    cup_speed_pen = -0.2 * (torch.tanh(torch.clamp(torch.norm(ctx.src_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4)
                            + torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4))
    nested_pen = -2.0 * ctx.cups_nested.to(zero.dtype)
    # real-robot safety
    cup_collision_pen = -1.0 * torch.tanh(ctx.cup_cup_force / 5.0)
    hand_foreign_pen = -1.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0) + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    palm_push_pen = -0.5 * (torch.tanh(ctx.src_palm_force / 5.0) + torch.tanh(ctx.rcv_palm_force / 5.0))

    # ------------------------------------------------------------------ regularisation
    action_rate_pen = -0.05 * ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    arm_speed_pen = -0.05 * (torch.tanh((ctx.src_arm_qd ** 2).mean(dim=-1))
                             + torch.tanh((ctx.rcv_arm_qd ** 2).mean(dim=-1)))

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "tips_src": tips_src,
        "tips_rcv": tips_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "rim_hook_pen": rim_hook_pen,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "align": align,
        "tilt": tilt,
        "pour_delta": pour_delta,
        "spill_delta": spill_delta,
        "success": success,
        "pre_tilt_pen": pre_tilt_pen,
        "rcv_upright_pen": rcv_upright_pen,
        "knock_pen": knock_pen,
        "drop_pen": drop_pen,
        "cup_speed_pen": cup_speed_pen,
        "nested_pen": nested_pen,
        "cup_collision_pen": cup_collision_pen,
        "hand_foreign_pen": hand_foreign_pen,
        "palm_push_pen": palm_push_pen,
        "action_rate_pen": action_rate_pen,
        "arm_speed_pen": arm_speed_pen,
    }
    reward = zero
    for v in components.values():
        reward = reward + v
    return reward, components
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.35e-07 · max 4.88e-05
bead/spill: [0.000293, 0, 0, 0, 0.000195, 0, 0, 0.000195, 0.000439, 0.000732]  min 0 · mean 0.00017 · max 0.00249
done/drop: [0.000977, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000215 · max 0.00879
done/mimic_runaway: [0.000977, 0, 0, 0, 0, 0, 0, 0.000977, 0, 0]  min 0 · mean 0.000108 · max 0.00293
episode_lengths/step: [115, 891, 746, 782, 831, 855, 877, 836, 848, 813]  min 115 · mean 812 · max 899
reward/action_rate_pen: [-0.0515, -0.0427, -0.0355, -0.0338, -0.0325, -0.0335, -0.0327, -0.0354, -0.0359, -0.0354]  min -0.0521 · mean -0.036 · max -0.03
reward/align: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/approach_rcv: [0.22, 0.248, 0.349, 0.384, 0.393, 0.401, 0.408, 0.4, 0.403, 0.399]  min 0.149 · mean 0.367 · max 0.425
reward/approach_src: [0.231, 0.251, 0.338, 0.372, 0.397, 0.405, 0.401, 0.404, 0.408, 0.406]  min 0.152 · mean 0.368 · max 0.429
reward/arm_speed_pen: [-0.0571, -0.0279, -0.0166, -0.0122, -0.0114, -0.0105, -0.00904, -0.00869, -0.0085, -0.00744]  min -0.0571 · mean -0.0141 · max -0.00727
reward/cup_collision_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00106 · mean -9.14e-06 · max 0
reward/cup_speed_pen: [-0.00646, 0, -4.04e-05, 0, -0.000391, 0, -0.000195, -0.000202, -0.000391, -7.73e-06]  min -0.00934 · mean -0.00021 · max 0
reward/drop_pen: [-0.00109, 0, 0, 0, 0, 0, 0, 0, -0.000976, 0]  min -0.0104 · mean -0.000251 · max 0
reward/grasp_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grasp_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/hand_foreign_pen: [-0.00388, -0.00179, -0.0084, -0.0129, -0.006, -0.0104, -0.0142, -0.0159, -0.0159, -0.018]  min -0.0301 · mean -0.00988 · max 0
reward/knock_pen: [-0.0398, 0, -0.00116, -0.000565, -0.0016, -0.00172, -0.000706, -0.00176, -0.00177, -0.00167]  min -0.0468 · mean -0.00215 · max 0
reward/lift_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.7e-06 · max 0.00293
reward/lift_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 6.11e-07 · max 0.000184
reward/nested_pen: [-0.00195, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00977 · mean -7.21e-05 · max 0
reward/palm_push_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.57e-05 · max 0.00977
reward/pre_tilt_pen: [-0.00695, 0, 0, 0, -0.00041, 0, 0, 0, -0.000957, -0.000753]  min -0.0183 · mean -0.00052 · max 0
reward/rcv_upright_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.000975 · mean -2.19e-06 · max 0
reward/rim_hook_pen: [-0.178, -0.0133, -0.0544, -0.0128, -0.0118, -0.00869, -0.00775, -0.0172, -0.0159, -0.0113]  min -0.192 · mean -0.0172 · max -0.00187
reward/spill_delta: [0, 0, 0, 0, 0, 0, 0, -0.0195, 0, 0]  min -0.0439 · mean -0.000361 · max 0.00488
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tips_rcv: [0.00858, 0.00791, 0.0834, 0.155, 0.134, 0.182, 0.176, 0.192, 0.181, 0.205]  min 0.000281 · mean 0.138 · max 0.229
reward/tips_src: [0.00803, 0.00591, 0.0717, 0.139, 0.189, 0.172, 0.204, 0.185, 0.19, 0.227]  min 0.00026 · mean 0.153 · max 0.231
reward/total: [0.109, 0.41, 0.718, 0.972, 1.04, 1.08, 1.11, 1.07, 1.08, 1.15]  min 0.00431 · mean 0.935 · max 1.2
rewards/step: [6.02, 346, 526, 731, 848, 921, 981, 899, 939, 923]  min 5.85 · mean 760 · max 993
task/aim_dist: [0.325, 0.319, 0.321, 0.32, 0.321, 0.32, 0.32, 0.321, 0.385, 0.321]  min 0.318 · mean 0.369 · max 1.04
task/cup_collision_rate: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.1e-05 · max 0.00195
task/cups_center_dist: [0.328, 0.319, 0.321, 0.32, 0.321, 0.32, 0.32, 0.321, 0.385, 0.32]  min 0.318 · mean 0.369 · max 1.04
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0.000977, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.61e-05 · max 0.00488
task/rcv_cup_lift: [0.000306, 1.97e-05, 6.39e-05, 0.000127, 0.00135, 0.000232, 0.00016, 0.000264, 0.000326, 0.000228]  min -2.08e-05 · mean 0.00481 · max 0.0975
task/rcv_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.14e-06 · max 0.000977
task/rcv_hand_foreign_rate: [0.00391, 0, 0.00586, 0.00781, 0.000977, 0.00977, 0.00684, 0.00977, 0.0127, 0.00879]  min 0 · mean 0.00588 · max 0.0234
task/src_cup_lift: [0.00675, 1.31e-05, 4.75e-05, 5.85e-05, 0.000732, 0.000234, 0.00126, 0.000381, 0.0252, 0.000459]  min -3.04e-05 · mean 0.00325 · max 0.0804
task/src_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 6.27e-06 · max 0.000977
task/src_hand_foreign_rate: [0, 0.00195, 0.00391, 0.00977, 0.00586, 0.00293, 0.00879, 0.00977, 0.00879, 0.0156]  min 0 · mean 0.00626 · max 0.0195
task/src_tilt_deg: [3.64, 0.0177, 0.173, 0.195, 1.32, 0.843, 1.29, 1.37, 1.96, 1.88]  min 0.0163 · mean 1.16 · max 4.67
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Observations from watching the trained policy (rollout video at epoch 600, camera on the robot's right side, 4 envs) plus the training metrics. These are facts about what the policy does, not design instructions.

1. The thumb-on-rim hook from the previous round is gone, and the LEFT (receiver) arm now comes down to its own cup. Both hands reach the side of their cup at mid height and then STOP there for the whole episode. Neither hand ever holds its cup: `task/src_grasped` and `task/rcv_grasped` stayed at 0 for all 620 epochs (a single environment out of 1024 at most). Neither arm moves upward and neither cup moves.

2. Neither hand wraps its cup. The RIGHT (source) hand sits beside the cup with its four fingers curled back toward its own palm (a loose half fist), so the finger pads face the palm, not the cup; only the thumb tip hangs close to the cup wall. The LEFT hand hangs next to its cup with the palm facing down and the half-curled fingers pointing down along the cup side. Measured over the last 30 epochs: palm-to-cup distance 0.129 m (source) and 0.130 m (receiver); hand closure 0.69 and 0.60 (the fingers are already mostly closed before any contact); maximum finger contact force on the own cup 0.03 N and 0.07 N, far below the 1 N that the grasp flag needs.

3. The run plateaued from about epoch 280 to the end: `reward/total` stayed between 1.08 and 1.16. At the end the per-step reward was approach_src 0.40, approach_rcv 0.41, tips_src 0.22, tips_rcv 0.22, rim_hook_pen −0.01, hand_foreign_pen −0.014. `reward/grasp_src`, `reward/grasp_rcv`, `reward/lift_src`, `reward/lift_rcv`, `reward/tilt` and `reward/success` were 0 for the whole run. The fingertip-placement term of the previous function is computed only from fingertip positions relative to the cup wall and from thumb/finger opposition; it requires no contact, and with the palm about 13 cm from the cup its palm gate was still about 0.6, so the hovering pose already earned most of it.

4. Hands touch things other than their own cup more often than before: `task/src_hand_foreign_rate` 0.010 and `task/rcv_hand_foreign_rate` 0.007 at the end (previous round ≤ 0.0003), consistent with the hands being held low beside the cups near the table.

5. Environment changes since the previous round (not caused by the reward): the policy now also observes five fingertip contact-force readings per hand (like the real RH56F1 fingertip tactile sensors), and an environment is also terminated when a hand's mimic coupling error exceeds 3 rad (`done/mimic_err_runaway`, triggered in a handful of epochs). The reward context fields are unchanged. Other facts unchanged: RH56F1 hands have 6 driven joints each, fingertip grasps are acceptable, the cup is a slim shaker (~0.06 m diameter, 0.134 kg).

The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
