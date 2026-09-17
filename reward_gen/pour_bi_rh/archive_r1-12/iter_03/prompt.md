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


def _hand_terms(ctx, palm_pos, tips, finger_force, closure, cup_pos, cup_up, grasped):
    """Per-arm reach / fingertip placement / contact / grasp. Everything (N,)."""
    dtype = cup_pos.dtype
    r_wall = ctx.cup_radius + 0.004          # outer wall ~ inner radius + wall thickness
    lo = ctx.cup_bottom_z + 0.03             # tips >= 3 cm above the cup bottom (table clearance)
    hi_f = ctx.cup_mouth_z - 0.02            # finger tips >= 2 cm below the rim
    hi_t = ctx.cup_mouth_z - 0.025           # thumb tip >= 2.5 cm below the rim (anti rim-hook)

    # ---- reach: flat inside 8 cm so the tips (not the palm) set the final pose; 13 cm hover -> 0.61
    d_palm = torch.norm(palm_pos - cup_pos, dim=-1)
    reach = _near(torch.clamp(d_palm - 0.08, min=0.0), 10.0)

    # ---- fingertip placement on the wall (no palm gate: the gate is what paid the hover last round)
    axial, radial, rdir = _cyl_coords(tips, cup_pos, cup_up)
    rad_err = torch.clamp(torch.abs(radial - r_wall) - 0.005, min=0.0)      # 5 mm tolerance
    thumb_err = rad_err[:, 0] + _band_err(axial[:, 0], lo, hi_t)
    f_err = rad_err[:, 1:3] + _band_err(axial[:, 1:3], lo, hi_f)           # index, middle
    finger_err = f_err.min(dim=-1).values                                   # best one: thumb+index pinch is fine

    fdir = rdir[:, 1:3, :].mean(dim=1)
    fdir = fdir / (torch.norm(fdir, dim=-1, keepdim=True) + 1e-6)
    cos_opp = (rdir[:, 0, :] * fdir).sum(dim=-1)
    opp = torch.clamp(-cos_opp, 0.0, 1.0)                                    # 1 = thumb opposite the fingers

    thumb_q = 0.5 * _near(thumb_err, 30.0) + 0.5 * _near(thumb_err, 8.0)     # k=8 far gradient, k=30 last cm
    finger_q = 0.5 * _near(finger_err, 30.0) + 0.5 * _near(finger_err, 8.0)
    tip_q = 0.5 * (thumb_q + finger_q)
    tip_place = tip_q * (0.5 + 0.5 * opp)

    # ---- contact ramp 0 N -> ~1 N (the grasp flag threshold); product term pays opposing contacts most
    f_thumb = finger_force[:, 0]
    f_other = finger_force[:, 1:].max(dim=-1).values
    c_t = torch.tanh(f_thumb / 0.5)
    c_o = torch.tanh(f_other / 0.5)
    below = (axial[:, 0] < (ctx.cup_mouth_z - 0.015)).to(dtype)             # thumb below rim (anti hook)
    contact = below * (0.25 * (c_t + c_o) + c_t * c_o) * (0.5 + 0.5 * tip_q)  # knuckle/fist pushes earn half

    # ---- grasp flag and soft hold (flag flickers around 1 N; lift must not flicker with it)
    grasped_f = grasped.to(dtype)
    good_grasp = below * grasped_f * (0.5 + 0.5 * opp)
    hold = below * torch.maximum(grasped_f, c_t * c_o)                        # (N,) in [0,1]

    # ---- pre-curled fist without any contact (observed: closure 0.6-0.7, 0.03 N)
    any_contact = torch.tanh((f_thumb + f_other) / 0.3)
    fist = torch.clamp((closure - 0.45) / 0.35, 0.0, 1.0) * (1.0 - any_contact)

    # ---- rim hook: thumb tip at/above rim height over the cup footprint while the palm is near
    near_gate = _near(torch.clamp(d_palm - 0.08, min=0.0), 10.0)
    over_rim = torch.clamp((axial[:, 0] - (ctx.cup_mouth_z - 0.02)) / 0.02, 0.0, 1.0)
    over_footprint = (radial[:, 0] < r_wall + 0.025).to(dtype)
    rim_hook = near_gate * over_rim * over_footprint

    return reach, tip_place, contact, good_grasp, hold, fist, rim_hook


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = torch.zeros_like(ctx.src_cup_tilt)

    # ------------------------------------------------------------------ stages 0-2: reach, tips, contact, grasp
    re_s, tp_s, ct_s, gg_s, hold_s, fist_s, hook_s = _hand_terms(
        ctx, ctx.src_palm_pos, ctx.src_tips_pos, ctx.src_finger_force, ctx.src_hand_closure,
        ctx.src_cup_pos, ctx.src_cup_up, ctx.src_grasped)
    re_r, tp_r, ct_r, gg_r, hold_r, fist_r, hook_r = _hand_terms(
        ctx, ctx.rcv_palm_pos, ctx.rcv_tips_pos, ctx.rcv_finger_force, ctx.rcv_hand_closure,
        ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_grasped)

    reach_src = 0.3 * re_s                   # weight 0.3: hovering at 13 cm now earns only 0.18
    reach_rcv = 0.3 * re_r
    tips_src = 0.5 * tp_s                    # weight 0.5: geometry must stay below any real contact
    tips_rcv = 0.5 * tp_r
    contact_src = 1.0 * ct_s                 # weight 1.0 (max 1.5): the ramp from 0 N to the 1 N grasp flag
    contact_rcv = 1.0 * ct_r
    grasp_src = 2.0 * gg_s                   # weight 2.0 > reach+tips+single-finger contact: pinching must pay
    grasp_rcv = 2.0 * gg_r
    grasp_both = 1.0 * gg_s * gg_r           # weight 1.0: both hands holding is the gateway to everything else
    fist_pen = -0.3 * (fist_s + fist_r)      # weight 0.3 < tips 0.5: open the hand while approaching
    rim_hook_pen = -1.0 * (hook_s + hook_r)

    # ------------------------------------------------------------------ stage 3: lift (independent per arm)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # tanh(h/3cm): 1 cm -> 0.32, 3 cm -> 0.76; weight 3.0 so lifting beats holding on the table
    lift_src = 3.0 * hold_s * torch.tanh(torch.clamp(h_src, min=0.0) / 0.03)
    lift_rcv = 3.0 * hold_r * torch.tanh(torch.clamp(h_rcv, min=0.0) / 0.03)
    lifted_src = hold_s * (h_src > 0.03).to(zero.dtype)
    lifted_rcv = hold_r * (h_rcv > 0.03).to(zero.dtype)
    lifted = lifted_src * lifted_rcv

    # ------------------------------------------------------------------ stage 4: align rims (both lifted)
    mouth_dxy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    dz_err = torch.clamp(0.04 - mouth_dz, min=0.0) + torch.clamp(mouth_dz - 0.12, min=0.0)  # 4..12 cm window
    align_q = (0.5 * _near(mouth_dxy, 5.0) + 0.5 * _near(mouth_dxy, 20.0)) * _near(dz_err, 15.0)
    align = 4.0 * lifted * align_q           # weight 4.0 > lift 3.0: carrying toward the receiver must pay
    aligned = lifted * (mouth_dxy < 0.05).to(zero.dtype) * (dz_err < 0.015).to(zero.dtype)

    # ------------------------------------------------------------------ stage 5: tilt and pour
    tilt = 3.0 * aligned * torch.clamp(ctx.src_cup_tilt / 2.0, 0.0, 1.0)   # doorway to pouring only
    pour_delta = 200.0 * ctx.d_in_target     # +10 per transferred bead (20 beads)
    spill_delta = -100.0 * ctx.d_spill       # -5 per spilled bead: costly but never worth refusing to pour

    # ------------------------------------------------------------------ stage 6: success
    success = 15.0 * ctx.success.to(zero.dtype)   # 15/step > held+lifted+aligned+tilt income: hold the goal

    # ------------------------------------------------------------------ constraints
    has_beads = (ctx.bead_in_source_frac > 0.05).to(zero.dtype)
    pre_tilt_pen = -1.0 * (1.0 - aligned) * has_beads * torch.tanh(2.0 * torch.clamp(ctx.src_cup_tilt - 0.8, min=0.0))
    # receiver upright only while held (always-on version taught the left arm to stay away)
    rcv_upright_pen = -1.0 * hold_r * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.1, min=0.0))
    src_free = 1.0 - hold_s
    rcv_free = 1.0 - hold_r
    knock_pen = -0.3 * (src_free * torch.tanh(3.0 * torch.clamp(ctx.src_cup_tilt - 0.2, min=0.0))
                        + rcv_free * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.2, min=0.0)))
    # unheld cup pushed across the table: contact is now rewarded, so shoving the cup away must cost
    disp_src = torch.norm(ctx.src_cup_pos[:, :2] - ctx.src_cup_spawn_pos[:, :2], dim=-1)
    disp_rcv = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    cup_push_pen = -0.3 * (src_free * torch.tanh(torch.clamp(disp_src - 0.02, min=0.0) / 0.03)
                           + rcv_free * torch.tanh(torch.clamp(disp_rcv - 0.02, min=0.0) / 0.03))
    drop_pen = -1.0 * (src_free * (h_src > 0.015).to(zero.dtype) * torch.tanh(torch.clamp(-ctx.src_cup_lin_vel[:, 2], min=0.0) / 0.5)
                       + rcv_free * (h_rcv > 0.015).to(zero.dtype) * torch.tanh(torch.clamp(-ctx.rcv_cup_lin_vel[:, 2], min=0.0) / 0.5))
    cup_speed_pen = -0.2 * (torch.tanh(torch.clamp(torch.norm(ctx.src_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4)
                            + torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4))
    nested_pen = -2.0 * ctx.cups_nested.to(zero.dtype)
    # real-robot safety
    cup_collision_pen = -1.0 * torch.tanh(ctx.cup_cup_force / 5.0)
    # weight 2.0 (was 1.0): foreign-contact rate rose 30x with hands held low near the table
    hand_foreign_pen = -2.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0) + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    palm_push_pen = -0.5 * (torch.tanh(ctx.src_palm_force / 5.0) + torch.tanh(ctx.rcv_palm_force / 5.0))

    # ------------------------------------------------------------------ regularisation
    action_rate_pen = -0.05 * ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    arm_speed_pen = -0.05 * (torch.tanh((ctx.src_arm_qd ** 2).mean(dim=-1))
                             + torch.tanh((ctx.rcv_arm_qd ** 2).mean(dim=-1)))

    components = {
        "reach_src": reach_src,
        "reach_rcv": reach_rcv,
        "tips_src": tips_src,
        "tips_rcv": tips_rcv,
        "contact_src": contact_src,
        "contact_rcv": contact_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "grasp_both": grasp_both,
        "fist_pen": fist_pen,
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
        "cup_push_pen": cup_push_pen,
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
bead/in_target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.17e-07 · max 9.77e-05
bead/spill: [0.000244, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 6.09e-05 · max 0.00283
done/drop: [0.00195, 0.000977, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000142 · max 0.00684
done/mimic_runaway: [0.00488, 0, 0, 0, 0, 0, 0, 0.000977, 0, 0]  min 0 · mean 0.000129 · max 0.00488
episode_lengths/step: [113, 799, 878, 874, 814, 850, 846, 821, 804, 831]  min 113 · mean 830 · max 893
reward/action_rate_pen: [-0.0514, -0.0438, -0.0414, -0.0406, -0.0381, -0.0381, -0.036, -0.0356, -0.0344, -0.0351]  min -0.0518 · mean -0.0388 · max -0.0341
reward/align: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/arm_speed_pen: [-0.0553, -0.0212, -0.013, -0.0105, -0.0104, -0.00899, -0.00929, -0.00871, -0.00767, -0.00775]  min -0.0598 · mean -0.0125 · max -0.00691
reward/contact_rcv: [0, 0, 0, 0.000284, 0.002, 0.00374, 0.00183, 0.00619, 0.00828, 0.0119]  min 0 · mean 0.00382 · max 0.0173
reward/contact_src: [0, 0, 0, 0.000204, 0.000501, 0.000731, 0.00555, 0.00476, 0.00489, 0.00348]  min 0 · mean 0.00242 · max 0.0108
reward/cup_collision_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00248 · mean -1.47e-05 · max 0
reward/cup_push_pen: [-0.0344, -0.000825, -0.000629, -0.000388, -0.00151, -0.000514, -0.00182, -0.0011, -0.00113, -0.000834]  min -0.0493 · mean -0.00167 · max 0
reward/cup_speed_pen: [-0.00863, 0, 0, -0.000121, -0.000216, -1.57e-05, -0.000423, -0.000195, -0.000283, 0]  min -0.00863 · mean -0.00022 · max 0
reward/drop_pen: [-0.00244, 0, 0, 0, -0.000532, 0, -0.000763, 0, 0, 0]  min -0.00602 · mean -0.000256 · max 0
reward/fist_pen: [0, -0.00182, -0.00133, -0.00239, -0.00408, -0.00368, -0.00233, -0.00105, -0.00137, -0.00206]  min -0.0373 · mean -0.0035 · max 0
reward/grasp_both: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grasp_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grasp_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/hand_foreign_pen: [-0.00195, -0.0129, -0.00027, -0.0125, -0.00641, -0.00326, -0.0046, -0.00528, -0.00638, -0.00685]  min -0.0191 · mean -0.00569 · max 0
reward/knock_pen: [-0.0351, -0.000915, -0.000443, -0.000414, -0.0011, -0.000447, -0.00124, -0.00131, -0.00125, -0.00116]  min -0.0478 · mean -0.00147 · max 0
reward/lift_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/lift_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/nested_pen: [-0.00391, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0117 · mean -0.000168 · max 0
reward/palm_push_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.000488 · mean -1.45e-06 · max 0
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/pre_tilt_pen: [-0.00259, 0, -0.000694, 0, 0, 0, -0.00067, 0, 0, 0]  min -0.0168 · mean -0.000351 · max 0
reward/rcv_upright_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/reach_rcv: [0.0759, 0.115, 0.13, 0.147, 0.158, 0.159, 0.158, 0.159, 0.161, 0.16]  min 0.0281 · mean 0.144 · max 0.165
reward/reach_src: [0.0799, 0.102, 0.14, 0.158, 0.17, 0.179, 0.186, 0.196, 0.212, 0.201]  min 0.0304 · mean 0.167 · max 0.214
reward/rim_hook_pen: [-0.189, -0.00477, -0.0148, -0.00737, -0.0238, -0.0147, -0.00993, -0.0123, -0.0099, -0.00922]  min -0.189 · mean -0.0127 · max -0.000391
reward/spill_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0977 · mean -0.000348 · max 0.00488
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tips_rcv: [0.0393, 0.0874, 0.117, 0.14, 0.16, 0.158, 0.157, 0.161, 0.163, 0.158]  min 0.0139 · mean 0.138 · max 0.169
reward/tips_src: [0.0376, 0.0629, 0.0917, 0.132, 0.138, 0.131, 0.141, 0.145, 0.139, 0.145]  min 0.0138 · mean 0.121 · max 0.153
reward/total: [-0.154, 0.279, 0.405, 0.502, 0.541, 0.56, 0.582, 0.605, 0.622, 0.614]  min -0.241 · mean 0.497 · max 0.634
rewards/step: [-22.3, 198, 360, 423, 430, 480, 500, 491, 495, 511]  min -76.4 · mean 415 · max 543
task/aim_dist: [0.317, 0.319, 0.32, 0.319, 0.319, 0.32, 0.549, 0.322, 0.323, 0.32]  min 0.317 · mean 0.477 · max 57.6
task/cup_collision_rate: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.32e-05 · max 0.00293
task/cups_center_dist: [0.32, 0.319, 0.32, 0.319, 0.319, 0.32, 0.549, 0.322, 0.323, 0.32]  min 0.318 · mean 0.477 · max 57.6
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0.00195, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 8.4e-05 · max 0.00586
task/rcv_cup_lift: [0.000795, 2.59e-05, 9.71e-05, 8.13e-05, 0.000213, 0.00021, 0.0898, 0.000372, 0.00109, 0.000368]  min -0.00014 · mean 0.0575 · max 27.1
task/rcv_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/rcv_hand_foreign_rate: [0.000977, 0.00586, 0, 0.00195, 0.00195, 0.000977, 0, 0.000977, 0.00195, 0.000977]  min 0 · mean 0.00154 · max 0.00879
task/src_cup_lift: [0.00258, 9.91e-06, 3.51e-05, 0.000113, 0.000173, 0.000153, 0.000254, 0.000303, 0.000298, 0.000214]  min -1.85e-05 · mean 0.000954 · max 0.083
task/src_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/src_hand_foreign_rate: [0, 0.00293, 0, 0.00488, 0.00293, 0.000977, 0.00391, 0.00293, 0.00195, 0.00391]  min 0 · mean 0.00222 · max 0.0107
task/src_tilt_deg: [3.17, 0.0647, 0.221, 0.383, 0.681, 0.51, 1.04, 1.08, 1.05, 0.751]  min 0.0139 · mean 0.696 · max 4.41
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Observations from watching the trained policy (rollout video at epoch 600, camera on the robot's right side, 4 envs), the training metrics, and operator notes. Items 1-5 are facts about what the policy does. Item 6 is an operator-supplied reference from another track of this project and item 7 is the operator's judgement.

1. Operator evaluation of the video: both hands approach their cups with the hand ROTATED (the wrist is turned, so the palm and finger pads do not face the cup body), and the hand is NOT open — the fingers stay partly curled during the approach. As a result each hand simply stops in front of its cup, at about the height of the cup mouth, and stays there for the whole episode; the fingers never close around the cup body, neither arm moves up and neither cup moves. In the video the right hand's curled fingers end up hooked over the top of the cup from the robot side, and the left hand hangs above its cup with the fingertips at the near rim.

2. Neither hand ever holds its cup: `task/src_grasped` and `task/rcv_grasped` were 0 for all 674 epochs. Hand closure (median of the last 30 epochs, 0 = fully open) is 0.40 (source) and 0.36 (receiver): lower than the 0.69 / 0.60 of the previous round, but the hand is still partly closed, not open, when it reaches the cup. The hands also stop short of the cup: palm-to-cup distance is 0.122 m (source) and 0.159 m (receiver), and the maximum finger contact force on the own cup is 0.06 N and 0.07 N, far below the 1 N the grasp flag needs.

3. The run crept up and then flattened: `reward/total` went from 0.56 (epoch 300) to 0.62 (epoch 674). Per-step reward at the end: reach_src 0.20, reach_rcv 0.16 (maximum 0.3 each), tips_src 0.15, tips_rcv 0.16 (maximum 0.5 each), contact_src 0.008, contact_rcv 0.011 (maximum 1.5 each), fist_pen −0.003, rim_hook_pen −0.007, hand_foreign_pen −0.006. `reward/grasp_src`, `reward/grasp_rcv`, `reward/grasp_both`, `reward/lift_src`, `reward/lift_rcv` and everything after them were 0 for the whole run.

4. In the previous function the reach term is flat once the palm is within 8 cm of the cup centre and is worth at most 0.3; the contact term is multiplied by a thumb-below-rim indicator (0 or 1) and by 0.5 + 0.5 · fingertip-placement quality, the grasp term is multiplied by the thumb-below-rim indicator, and the soft hold used by the lift term also requires it. So any contact made while the thumb tip is above that height earns nothing, and contact with the fingertips away from the ideal wall band earns half.

5. Physics blow-ups of the underactuated hand (mimic coupling error above 10 rad) happened in 18 epochs, each in a single environment that the environment then terminated. The two largest (epochs 627 and 659) threw the receiver cup far away, which is why the averaged `task/rcv_cup_lift` (max 27 m) and `task/aim_dist` (max 57 m) show absurd values; these are not lifts.

6. Operator reference — the bimanual pouring track of this project with a 20-DoF dexterous hand (same environment family, same RewardContext) solved approach and grasp with a much looser structure and reached grasp 0.88 on both hands and task success 0.77: approach = exp(−4 · palm-to-cup distance) with weight 1.0 and no flat zone; grasp = 1.0 · grasp flag + 0.5 · exp(−6 · palm-to-cup distance) · hand closure (dense reward for closing the hand near the cup before any contact; the flag itself has no position condition); grasp quality was never a gate on the grasp term — it only multiplied the later lift / aim / tilt income as a factor 0.3 + 0.7 · quality, so a poor grasp still earned 30 % and improved while lifting. Its quality measure used contact forces and palm orientation. For this robot two things differ: the hand is meant to hold the slim cup with the fingertips (no palm wrap, so palm force is not a quality signal), and in the first round of this track the average hand closure was about 0.65 while roughly a third of the environments held the cup, well above the 0.25-0.40 range used for the dexterous hand.

7. Operator judgement: once both hands reliably approach and grasp their cups, the later stages (lift, bring together, tilt, pour) are expected to follow, as they did on the other track.

The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
