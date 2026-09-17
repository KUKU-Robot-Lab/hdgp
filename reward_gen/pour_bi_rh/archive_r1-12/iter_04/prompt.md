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
7. `ctx.success` is computed by the environment (enough beads in the receiver cup, little spill, cups close together, the receiver cup held nearly upright — `ctx.rcv_cup_tilt` at most 20° — and the cups NOT nested). You may add a bonus on it but you cannot redefine it. Beads only count as "in the receiver" once they have LEFT the source cup — pushing the source cup into the receiver cup (`ctx.cups_nested`) transfers nothing and is never a success; the beads must fall out of the tilted source cup through the air.
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
    axial = (rel * up).sum(dim=-1)                                   # (N,F)
    radial_vec = rel - axial[..., None] * up                          # (N,F,3)
    radial = torch.norm(radial_vec, dim=-1)                           # (N,F)
    radial_dir = radial_vec / (radial[..., None] + 1e-6)
    return axial, radial, radial_dir


def _hand_terms(ctx, palm_pos, palm_axes, tips, finger_force, closure, cup_pos, cup_up, grasped):
    """Per-arm approach / orientation / tip placement / closing / contact / grasp. All (N,)."""
    dtype = cup_pos.dtype
    r_wall = ctx.cup_radius + 0.004          # outer wall ~ inner radius + wall thickness
    lo = ctx.cup_bottom_z + 0.03             # tips >= 3 cm above the cup bottom (table clearance)
    hi = ctx.cup_mouth_z - 0.015             # tips >= 1.5 cm below the rim

    # ---- approach: no flat zone; coarse k=4 from afar + fine k=15 over the last 6 cm (max 1.5)
    d_palm = torch.norm(palm_pos - cup_pos, dim=-1)
    reach = _near(d_palm, 4.0) + 0.5 * _near(torch.clamp(d_palm - 0.06, min=0.0), 15.0)

    # ---- orientation: pads face the cup body (sign-agnostic) and fingertips point toward the cup
    normal = palm_axes[:, 0:3]
    normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
    to_cup = cup_pos - palm_pos
    horiz_mask = torch.tensor([1.0, 1.0, 0.0], device=to_cup.device, dtype=dtype)
    to_cup_h = to_cup * horiz_mask
    to_cup_h = to_cup_h / (torch.norm(to_cup_h, dim=-1, keepdim=True) + 1e-6)
    face = torch.abs((normal * to_cup_h).sum(dim=-1))                          # 1 = palm normal points at cup
    d_palm_xy = torch.norm(to_cup[:, :2], dim=-1)
    tip_mid = tips[:, 0:3, :].mean(dim=1)                                      # thumb, index, middle
    d_tip_xy = torch.norm(tip_mid[:, :2] - cup_pos[:, :2], dim=-1)
    ahead = torch.clamp((d_palm_xy - d_tip_xy) / 0.03, 0.0, 1.0)              # tips nearer the cup axis than palm
    orient = _near(torch.clamp(d_palm - 0.08, min=0.0), 5.0) * (0.5 * face + 0.5 * ahead)

    # ---- fingertip placement: distance of each tip to the usable outer wall band
    axial, radial, rdir = _cyl_coords(tips, cup_pos, cup_up)
    rad_out = torch.clamp(radial - r_wall - 0.005, min=0.0)                    # 5 mm radial tolerance
    band = _band_err(axial, lo, hi)
    surf = torch.sqrt(rad_out ** 2 + band ** 2 + 1e-8)                         # (N,F)
    s_thumb = surf[:, 0]
    s_finger = surf[:, 1:3].min(dim=-1).values                                 # thumb+index pinch is fine
    thumb_q = 0.5 * _near(s_thumb, 30.0) + 0.5 * _near(s_thumb, 8.0)
    finger_q = 0.5 * _near(s_finger, 30.0) + 0.5 * _near(s_finger, 8.0)
    tip_q = 0.5 * (thumb_q + finger_q)

    fdir = rdir[:, 1:3, :].mean(dim=1)
    fdir = fdir / (torch.norm(fdir, dim=-1, keepdim=True) + 1e-6)
    opp = torch.clamp(-(rdir[:, 0, :] * fdir).sum(dim=-1), 0.0, 1.0)         # 1 = thumb opposite the fingers
    tip_place = tip_q * (0.5 + 0.5 * opp)

    # ---- dense closing reward, only when the tips are actually at the wall (not when hovering)
    close_near = closure * _near(0.5 * (s_thumb + s_finger), 15.0)

    # ---- hand must be OPEN while the palm is far (> 8 cm); contact does not cancel this
    far = 1.0 - _near(torch.clamp(d_palm - 0.08, min=0.0), 20.0)
    curl_far = torch.clamp((closure - 0.2) / 0.4, 0.0, 1.0) * far

    # ---- contact ramp 0 -> ~0.6 N, ungated; product pays thumb+finger opposition most (max 1.5)
    f_thumb = finger_force[:, 0]
    f_other = finger_force[:, 1:].max(dim=-1).values
    c_t = torch.tanh(f_thumb / 0.3)
    c_o = torch.tanh(f_other / 0.3)
    contact = 0.25 * (c_t + c_o) + c_t * c_o

    # ---- grasp flag (no position gate) and soft hold for lift (flag flickers near 1 N)
    grasped_f = grasped.to(dtype)
    hold = torch.maximum(grasped_f, c_t * c_o)

    # ---- grasp quality: a multiplier on later income, never a gate (0.3 .. 1.0)
    below_soft = torch.clamp(((ctx.cup_mouth_z - 0.015) - axial[:, 0]) / 0.02, 0.0, 1.0)
    quality = below_soft * (0.5 + 0.5 * opp)
    q = 0.3 + 0.7 * quality

    # ---- rim hook: thumb tip at/above rim height over the cup footprint while the palm is near
    near_gate = _near(torch.clamp(d_palm - 0.08, min=0.0), 10.0)
    over_rim = torch.clamp((axial[:, 0] - (ctx.cup_mouth_z - 0.02)) / 0.02, 0.0, 1.0)
    over_footprint = (radial[:, 0] < r_wall + 0.025).to(dtype)
    rim_hook = near_gate * over_rim * over_footprint

    return {
        "reach": reach, "orient": orient, "tip_place": tip_place, "close_near": close_near,
        "curl_far": curl_far, "contact": contact, "grasped": grasped_f, "hold": hold,
        "q": q, "rim_hook": rim_hook,
    }


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = torch.zeros_like(ctx.src_cup_tilt)
    dtype = zero.dtype

    s = _hand_terms(ctx, ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos, ctx.src_finger_force,
                    ctx.src_hand_closure, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_grasped)
    r = _hand_terms(ctx, ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                    ctx.rcv_hand_closure, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_grasped)

    # ------------------------------------------------------------------ stages 0-2: approach, orient, tips, close, grasp
    approach_src = 1.0 * s["reach"]          # weight 1.0 (max 1.5): 12 cm -> 0.78, 6 cm -> 1.29; closing in must pay
    approach_rcv = 1.0 * r["reach"]
    orient_src = 0.4 * s["orient"]           # weight 0.4: pads toward the cup; smaller than approach so it cannot stall it
    orient_rcv = 0.4 * r["orient"]
    tips_src = 0.5 * s["tip_place"]          # weight 0.5: geometry stays below real contact income
    tips_rcv = 0.5 * r["tip_place"]
    close_src = 0.5 * s["close_near"]        # weight 0.5: closing pays only with tips at the wall
    close_rcv = 0.5 * r["close_near"]
    contact_src = 1.0 * s["contact"]         # weight 1.0 (max 1.5): bridge from touching to the 1 N grasp flag
    contact_rcv = 1.0 * r["contact"]
    grasp_src = 2.0 * s["grasped"]           # weight 2.0 > any approach/orient gain: holding must dominate
    grasp_rcv = 2.0 * r["grasped"]
    grasp_both = 1.0 * s["grasped"] * r["grasped"]
    curl_far_pen = -0.4 * (s["curl_far"] + r["curl_far"])    # open hand while far; 0.4 < approach gain
    rim_hook_pen = -1.0 * (s["rim_hook"] + r["rim_hook"])    # separate cost, no longer zeroes contact/grasp

    hold_s, hold_r = s["hold"], r["hold"]
    q_s, q_r = s["q"], r["q"]

    # ------------------------------------------------------------------ stage 3: lift (independent per arm, x quality)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # tanh(h/3cm): 1 cm -> 0.32, 3 cm -> 0.76; weight 3.0 on top of grasp income, x (0.3..1.0) grasp quality
    lift_src = 3.0 * hold_s * q_s * torch.tanh(torch.clamp(h_src, min=0.0) / 0.03)
    lift_rcv = 3.0 * hold_r * q_r * torch.tanh(torch.clamp(h_rcv, min=0.0) / 0.03)
    lifted_src = hold_s * (h_src > 0.03).to(dtype)
    lifted_rcv = hold_r * (h_rcv > 0.03).to(dtype)
    lifted = lifted_src * lifted_rcv

    # ------------------------------------------------------------------ stage 4: align rims (both lifted)
    mouth_dxy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    dz_err = torch.clamp(0.04 - mouth_dz, min=0.0) + torch.clamp(mouth_dz - 0.12, min=0.0)  # 4..12 cm window
    align_q = (0.5 * _near(mouth_dxy, 5.0) + 0.5 * _near(mouth_dxy, 20.0)) * _near(dz_err, 15.0)
    align = 4.0 * lifted * q_s * align_q     # weight 4.0 > lift 3.0: carrying toward the receiver must pay
    aligned = lifted * (mouth_dxy < 0.05).to(dtype) * (dz_err < 0.015).to(dtype)

    # ------------------------------------------------------------------ stage 5: tilt and pour
    tilt = 3.0 * aligned * q_s * torch.clamp(ctx.src_cup_tilt / 2.0, 0.0, 1.0)   # doorway to pouring only
    pour_delta = 200.0 * ctx.d_in_target     # +10 per transferred bead (20 beads)
    spill_delta = -100.0 * ctx.d_spill       # -5 per spilled bead: costly but never worth refusing to pour

    # ------------------------------------------------------------------ stage 6: success
    success = 15.0 * ctx.success.to(dtype)   # 15/step > held+lifted+aligned+tilt income: hold the goal

    # ------------------------------------------------------------------ constraints
    has_beads = (ctx.bead_in_source_frac > 0.05).to(dtype)
    pre_tilt_pen = -1.0 * (1.0 - aligned) * has_beads * torch.tanh(2.0 * torch.clamp(ctx.src_cup_tilt - 0.8, min=0.0))
    # receiver upright only while held (always-on version taught the left arm to stay away)
    rcv_upright_pen = -1.0 * hold_r * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.1, min=0.0))
    src_free = 1.0 - hold_s
    rcv_free = 1.0 - hold_r
    knock_pen = -0.3 * (src_free * torch.tanh(3.0 * torch.clamp(ctx.src_cup_tilt - 0.2, min=0.0))
                        + rcv_free * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.2, min=0.0)))
    # unheld cup pushed across the table: stronger approach must not turn into shoving the cup
    disp_src = torch.norm(ctx.src_cup_pos[:, :2] - ctx.src_cup_spawn_pos[:, :2], dim=-1)
    disp_rcv = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    cup_push_pen = -0.3 * (src_free * torch.tanh(torch.clamp(disp_src - 0.02, min=0.0) / 0.03)
                           + rcv_free * torch.tanh(torch.clamp(disp_rcv - 0.02, min=0.0) / 0.03))
    drop_pen = -1.0 * (src_free * (h_src > 0.015).to(dtype) * torch.tanh(torch.clamp(-ctx.src_cup_lin_vel[:, 2], min=0.0) / 0.5)
                       + rcv_free * (h_rcv > 0.015).to(dtype) * torch.tanh(torch.clamp(-ctx.rcv_cup_lin_vel[:, 2], min=0.0) / 0.5))
    cup_speed_pen = -0.2 * (torch.tanh(torch.clamp(torch.norm(ctx.src_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4)
                            + torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4))
    nested_pen = -2.0 * ctx.cups_nested.to(dtype)
    # real-robot safety
    cup_collision_pen = -1.0 * torch.tanh(ctx.cup_cup_force / 5.0)
    hand_foreign_pen = -2.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0) + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    # palm is not part of a fingertip grasp: pushing the cup with it costs
    palm_push_pen = -0.5 * (torch.tanh(ctx.src_palm_force / 5.0) + torch.tanh(ctx.rcv_palm_force / 5.0))

    # ------------------------------------------------------------------ regularisation
    action_rate_pen = -0.05 * ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    arm_speed_pen = -0.05 * (torch.tanh((ctx.src_arm_qd ** 2).mean(dim=-1))
                             + torch.tanh((ctx.rcv_arm_qd ** 2).mean(dim=-1)))

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "orient_src": orient_src,
        "orient_rcv": orient_rcv,
        "tips_src": tips_src,
        "tips_rcv": tips_rcv,
        "close_src": close_src,
        "close_rcv": close_rcv,
        "contact_src": contact_src,
        "contact_rcv": contact_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "grasp_both": grasp_both,
        "curl_far_pen": curl_far_pen,
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
bead/in_target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.67e-06 · max 9.77e-05
bead/spill: [0.000244, 0.000293, 0, 0, 0, 0, 9.77e-05, 0, 0, 0]  min 0 · mean 5.82e-05 · max 0.00151
done/drop: [0.000977, 0, 0, 0, 0, 0, 0.000977, 0, 0, 0]  min 0 · mean 0.000311 · max 0.0107
done/mimic_runaway: [0.000977, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.00013 · max 0.00586
episode_lengths/step: [113, 375, 856, 869, 849, 867, 874, 892, 898, 875]  min 113 · mean 795 · max 898
reward/action_rate_pen: [-0.0515, -0.0498, -0.0484, -0.0489, -0.0448, -0.0447, -0.0457, -0.0432, -0.0446, -0.0427]  min -0.052 · mean -0.0461 · max -0.04
reward/align: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/approach_rcv: [0.452, 0.345, 0.357, 0.432, 0.462, 0.501, 0.549, 0.593, 0.596, 0.629]  min 0.338 · mean 0.513 · max 0.684
reward/approach_src: [0.479, 0.351, 0.352, 0.439, 0.445, 0.479, 0.551, 0.59, 0.584, 0.603]  min 0.337 · mean 0.507 · max 0.648
reward/arm_speed_pen: [-0.0573, -0.0522, -0.0428, -0.0412, -0.038, -0.0319, -0.0253, -0.0223, -0.0212, -0.0207]  min -0.0573 · mean -0.0316 · max -0.018
reward/close_rcv: [0.00737, 0.0136, 0.0151, 0.035, 0.0497, 0.0696, 0.0898, 0.0985, 0.0979, 0.109]  min 0.00737 · mean 0.0656 · max 0.141
reward/close_src: [0.00808, 0.0133, 0.0117, 0.0306, 0.0306, 0.0439, 0.0697, 0.0692, 0.0702, 0.0702]  min 0.00746 · mean 0.0468 · max 0.0932
reward/contact_rcv: [0.00536, 0, 0, 0, 0, 3.77e-05, 0.000488, 0.000456, 0, 0.000729]  min 0 · mean 0.000424 · max 0.00581
reward/contact_src: [0.00732, 0, 0, 0, 0, 0, 0, 0.000488, 0.000186, 0]  min 0 · mean 0.000437 · max 0.00808
reward/cup_collision_pen: [-0.000116, -9.77e-05, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00214 · mean -9.27e-05 · max 0
reward/cup_push_pen: [-0.0378, -0.00797, -0.000252, 0, -0.000379, -0.000712, -0.00141, -0.000258, -3.09e-05, -0.000425]  min -0.0481 · mean -0.00346 · max 0
reward/cup_speed_pen: [-0.00794, 0, 0, 0, 0, 0, -0.00035, -7.26e-05, 0, -5.46e-06]  min -0.00837 · mean -0.000377 · max 0
reward/curl_far_pen: [0, -0.206, -0.132, -0.345, -0.394, -0.449, -0.397, -0.307, -0.274, -0.241]  min -0.455 · mean -0.259 · max 0
reward/drop_pen: [-0.00572, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00906 · mean -0.000291 · max 0
reward/grasp_both: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grasp_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grasp_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/hand_foreign_pen: [-0.0142, -0.000307, -0.000424, -0.00177, -0.00143, 0, -0.00195, -0.00218, 0, -0.000224]  min -0.0185 · mean -0.00178 · max 0
reward/knock_pen: [-0.0351, -0.000786, 0, 0, -0.000304, -0.000435, -0.00147, -0.000359, 0, -0.00068]  min -0.0428 · mean -0.00258 · max 0
reward/lift_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/lift_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/nested_pen: [-0.00195, -0.00391, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0195 · mean -0.00115 · max 0
reward/orient_rcv: [0.139, 0.124, 0.122, 0.146, 0.153, 0.17, 0.186, 0.181, 0.19, 0.195]  min 0.121 · mean 0.166 · max 0.202
reward/orient_src: [0.135, 0.13, 0.129, 0.142, 0.134, 0.158, 0.191, 0.171, 0.177, 0.173]  min 0.122 · mean 0.16 · max 0.191
reward/palm_push_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 6.18e-05 · max 0.00977
reward/pre_tilt_pen: [-0.00984, 0, 0, 0, 0, 0, -0.000111, 0, 0, -0.000757]  min -0.0201 · mean -0.000956 · max 0
reward/rcv_upright_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/rim_hook_pen: [-0.181, -0.0139, -0.00239, -0.0125, -0.0109, -0.0261, -0.084, -0.0505, -0.056, -0.0714]  min -0.199 · mean -0.0497 · max -0.00239
reward/spill_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0146 · mean -0.000155 · max 0.00488
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tips_rcv: [0.0504, 0.0337, 0.038, 0.0518, 0.0619, 0.0731, 0.0903, 0.105, 0.11, 0.126]  min 0.0327 · mean 0.082 · max 0.149
reward/tips_src: [0.0519, 0.0339, 0.0369, 0.053, 0.0516, 0.0638, 0.093, 0.1, 0.105, 0.111]  min 0.0314 · mean 0.0774 · max 0.129
reward/total: [0.883, 0.703, 0.824, 0.867, 0.881, 0.971, 1.22, 1.42, 1.47, 1.57]  min 0.628 · mean 1.17 · max 1.8
rewards/step: [71.3, 221, 682, 734, 753, 839, 1.03e+03, 1.25e+03, 1.34e+03, 1.38e+03]  min 71.3 · mean 919 · max 1.49e+03
task/aim_dist: [0.317, 0.317, 0.32, 0.32, 0.32, 0.32, 0.321, 0.319, 0.32, 0.32]  min 0.315 · mean 0.326 · max 0.557
task/cup_collision_rate: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000149 · max 0.00391
task/cups_center_dist: [0.32, 0.317, 0.32, 0.32, 0.32, 0.32, 0.321, 0.319, 0.32, 0.321]  min 0.316 · mean 0.326 · max 0.558
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0.000977, 0.00195, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000578 · max 0.00977
task/rcv_cup_lift: [0.00185, 2.11e-06, 1.8e-05, 9.63e-06, 1.21e-05, 2.49e-05, -1.55e-05, 3.52e-05, 2.5e-05, 5.59e-05]  min -0.000162 · mean 0.00195 · max 0.0669
task/rcv_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/rcv_hand_foreign_rate: [0.00391, 0, 0, 0.000977, 0.000977, 0, 0.000977, 0.000977, 0, 0]  min 0 · mean 0.000765 · max 0.00586
task/src_cup_lift: [0.000745, 3.19e-05, 8.45e-06, 2.73e-06, 8.88e-06, 2.18e-05, 1.74e-05, 2.86e-05, 1.58e-05, 9.7e-06]  min -8.48e-05 · mean 0.000151 · max 0.0104
task/src_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/src_hand_foreign_rate: [0.00488, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000398 · max 0.00488
task/src_tilt_deg: [3.33, 0.173, 0.0253, 0.0153, 0.0326, 0.112, 0.103, 0.0613, 0.0453, 0.163]  min 0.0143 · mean 0.299 · max 4.7
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Observations for this round. Items 1-5 are facts (measured training metrics and the previous reward function's own formulas); item 6 is what the operator saw; item 7 records operator decisions.

1. Between the previous round and this one the robot model was corrected, and the SAME previous reward function was retrained from scratch:
   - The hand collision shapes were fixed. Duplicated overlapping fingertip shapes were removed, and the palm shells and the robot body no longer have inflated shapes. With self-collision ON, no hand link touches another hand link or the body during any finger sweep.
   - The finger drives were made stiffer. A commanded finger position is now followed within 0.01 rad; before, the finger lagged about 0.5 rad behind the command.
   - Finger contact force is still reported per finger, and the observations and actions keep their meaning.
   - The operator judged the model to be fine now. The failure below is therefore attributed to the reward function.

2. Both hands stop about 16-17 cm from their cups, never touch them, and never grasp. The pattern is the same as in the previous round with the old model at the same epochs (medians over the given epoch windows):

| metric | this round, ep 63-83 | this round, ep 116-146 | previous round, ep 116-146 | previous round, ep 280-310 |
|---|---|---|---|---|
| `task/src_palm_to_cup` [m] | 0.214 | 0.170 | 0.164 | 0.127 |
| `task/rcv_palm_to_cup` [m] | 0.210 | 0.164 | 0.190 | 0.166 |
| `contact/src_max` [N] | 0.000 | 0.000 | 0.001 | 0.138 |
| `task/src_grasped` / `task/rcv_grasped` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `task/src_closure` / `task/rcv_closure` | 0.33 / 0.39 | 0.32 / 0.42 | 0.44 / 0.33 | 0.51 / 0.22 |
| `reward/close_src` / `reward/close_rcv` | 0.040 / 0.054 | 0.078 / 0.118 | 0.098 / 0.061 | 0.177 / 0.060 |
| `reward/curl_far_pen` | −0.336 | −0.283 | −0.315 | −0.192 |
| `reward/rim_hook_pen` | −0.029 | −0.059 | −0.020 | −0.024 |
| `reward/approach_src` | 0.480 | 0.605 | 0.631 | 0.806 |

3. Approach and rim-hook terms of the previous function, per hand:
   - Approach, `exp(−4 d) + 0.5 exp(−15 max(d − 0.06, 0))`, pays 0.60 at d = 17 cm and 1.29 at d = 6 cm.
   - The rim-hook penalty is multiplied by `exp(−10 max(d − 0.08, 0))`. That factor is 0.41 at 17 cm and 1.0 at 8 cm or closer. So any thumb-over-rim posture costs 2.4 times more once the palm closes the last 9 cm, up to −1.0 per hand.
   - As the palms came from about 21 cm to about 17 cm, `reward/rim_hook_pen` grew from −0.029 to −0.059. The thumb tip is already at rim height over the cup opening while the palm is still outside 8 cm.

4. The closing term `close_near = closure · exp(−15 · mean tip-to-wall distance)` has no palm-distance condition. `reward/close_src` / `reward/close_rcv` rose from 0.015 / 0.011 (epochs 0-10) to 0.078 / 0.118. Over the same epochs the palm stayed about 16-17 cm away and finger contact stayed 0. The hands are 32-42 % closed while far, which also costs `reward/curl_far_pen` about −0.28 (both hands).

5. Physics blow-ups of the underactuated hand stayed small with the stiffer drives: `ctrl/mimic_err_max` median 0.14 rad over the last 30 epochs, versus about 0.77 rad median in the previous round.

6. Operator observations:
   - This round: the policy approaches so that the thumb gets caught on the rim, and without a proper approach no grasp is possible.
   - Previous round (video at epoch 800, old model): the right hand held its cup with the thumb over the cup mouth and the four fingers on the near side; the cup was never lifted. The left hand hovered above and behind its cup and never wrapped it.

7. Operator decisions:
   - The model problems are fixed, so the reward must change: generate a new reward and retrain from scratch.
   - The judgement from earlier rounds still stands: once both hands reliably approach their cups and grasp them with the thumb on the cup wall opposite the fingers (not over the rim), the later stages are expected to follow.

The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
