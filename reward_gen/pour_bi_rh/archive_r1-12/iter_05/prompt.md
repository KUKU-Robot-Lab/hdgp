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
    """Cup-cylinder coordinates of (N,K,3) points: axial height, radial distance, radial unit dir."""
    rel = points - cup_pos[:, None, :]
    up = cup_up[:, None, :]
    axial = (rel * up).sum(dim=-1)                                   # (N,K)
    radial_vec = rel - axial[..., None] * up                          # (N,K,3)
    radial = torch.norm(radial_vec, dim=-1)                           # (N,K)
    radial_dir = radial_vec / (radial[..., None] + 1e-6)
    return axial, radial, radial_dir


def _hand_terms(ctx, palm_pos, palm_axes, tips, finger_force, closure, cup_pos, cup_up, grasped):
    """Per-arm approach / orientation / tip placement / closing / contact / grasp. All (N,)."""
    dtype = cup_pos.dtype
    mouth = ctx.cup_mouth_z
    r_wall = ctx.cup_radius + 0.004          # outer wall ~ inner radius + wall thickness
    tip_lo = ctx.cup_bottom_z + 0.03         # tips >= 3 cm above the cup bottom (table clearance)
    tip_hi = mouth - 0.015                   # tips >= 1.5 cm below the rim

    # ---- pre-grasp region for the PALM: beside the cup (radial wall+1cm .. 7cm) at wall height
    #      (bottom+3cm .. rim+1cm). Error 0 inside. Over-the-mouth palms are OUTSIDE (radial < wall+1cm).
    p_ax, p_rad, _ = _cyl_coords(palm_pos[:, None, :], cup_pos, cup_up)
    rad_err = _band_err(p_rad[:, 0], r_wall + 0.01, 0.07)
    ht_err = _band_err(p_ax[:, 0], ctx.cup_bottom_z + 0.03, mouth + 0.01)
    d_pre = torch.sqrt(rad_err ** 2 + ht_err ** 2 + 1e-8)
    reach_lin = torch.clamp(1.0 - d_pre / 0.25, 0.0, 1.0)            # constant slope 4/m, no flat zone
    reach_fine = _near(d_pre, 25.0)                                    # sharp pull over the last few cm
    near_pre = _near(d_pre, 10.0)

    # ---- orientation: pad normal faces the cup horizontally (sign-agnostic), tips nearer the axis than palm
    normal = palm_axes[:, 0:3]
    normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
    to_cup_xy = cup_pos[:, :2] - palm_pos[:, :2]
    d_palm_xy = torch.norm(to_cup_xy, dim=-1)
    u_xy = to_cup_xy / (d_palm_xy[:, None] + 1e-6)
    face = torch.abs((normal[:, :2] * u_xy).sum(dim=-1))
    tip_mid = tips[:, 0:3, :].mean(dim=1)                                      # thumb, index, middle
    d_tip_xy = torch.norm(tip_mid[:, :2] - cup_pos[:, :2], dim=-1)
    ahead = torch.clamp((d_palm_xy - d_tip_xy) / 0.03, 0.0, 1.0)
    orient = near_pre * (0.5 * face + 0.5 * ahead)

    # ---- fingertip placement on the outer-wall band (below the rim)
    t_ax, t_rad, t_dir = _cyl_coords(tips, cup_pos, cup_up)
    rad_out = torch.clamp(t_rad - r_wall - 0.005, min=0.0)                     # 5 mm radial tolerance
    surf = torch.sqrt(rad_out ** 2 + _band_err(t_ax, tip_lo, tip_hi) ** 2 + 1e-8)   # (N,F)
    s_thumb = surf[:, 0]
    s_finger = surf[:, 1:3].min(dim=-1).values                                 # thumb+index pinch is fine
    thumb_q = 0.5 * _near(s_thumb, 30.0) + 0.5 * _near(s_thumb, 8.0)
    finger_q = 0.5 * _near(s_finger, 30.0) + 0.5 * _near(s_finger, 8.0)
    fdir = t_dir[:, 1:3, :].mean(dim=1)
    fdir = fdir / (torch.norm(fdir, dim=-1, keepdim=True) + 1e-6)
    opp = torch.clamp(-(t_dir[:, 0, :] * fdir).sum(dim=-1), 0.0, 1.0)        # 1 = thumb opposite the fingers
    tip_place = 0.5 * (thumb_q + finger_q) * (0.5 + 0.5 * opp)

    # ---- thumb height w.r.t. the rim: low = on the wall (income allowed), over = hooked on the mouth
    thumb_ax = t_ax[:, 0]
    thumb_low = torch.clamp((mouth - thumb_ax) / 0.015, 0.0, 1.0)             # 1 at >= 1.5 cm below rim, 0 at rim
    over_rim = torch.clamp((thumb_ax - (mouth - 0.005)) / 0.015, 0.0, 1.0)    # 0 below rim-0.5cm, 1 at rim+1cm
    over_footprint = (t_rad[:, 0] < r_wall + 0.025).to(dtype)
    rim_hook = over_rim * over_footprint                                       # flat: independent of palm distance

    # ---- closing: only with the tips AT the wall and the thumb low
    s_mean = 0.5 * (s_thumb + s_finger)
    close_near = closure * _near(s_mean, 40.0) * thumb_low                     # 1 cm -> 0.67, 5 cm -> 0.14
    # hand must be open while outside the pre-grasp region (action 0 = 50 % closed in this env)
    far = 1.0 - _near(torch.clamp(d_pre - 0.03, min=0.0), 20.0)
    curl_far = torch.clamp((closure - 0.3) / 0.4, 0.0, 1.0) * far

    # ---- contact ramp 0 -> ~0.6 N; thumb+finger opposition pays most (max 1.5); void with thumb over the rim
    c_t = torch.tanh(finger_force[:, 0] / 0.3)
    c_o = torch.tanh(finger_force[:, 1:].max(dim=-1).values / 0.3)
    contact = (0.25 * (c_t + c_o) + c_t * c_o) * thumb_low

    # ---- grasp: env flag x quality (thumb below rim, thumb opposite fingers)
    grasped_f = grasped.to(dtype)
    quality = thumb_low * (0.5 + 0.5 * opp)
    grasp = grasped_f * quality
    hold = torch.maximum(grasped_f, c_t * c_o) * thumb_low                    # soft hold (flag flickers near 1 N)
    q = 0.3 + 0.7 * quality                                                    # multiplier on lift / carry / tilt

    return {
        "reach_lin": reach_lin, "reach_fine": reach_fine, "orient": orient, "tip_place": tip_place,
        "close_near": close_near, "curl_far": curl_far, "contact": contact, "grasp": grasp,
        "grasped": grasped_f, "hold": hold, "q": q, "rim_hook": rim_hook,
    }


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = torch.zeros_like(ctx.src_cup_tilt)
    dtype = zero.dtype

    s = _hand_terms(ctx, ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos, ctx.src_finger_force,
                    ctx.src_hand_closure, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_grasped)
    r = _hand_terms(ctx, ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                    ctx.rcv_hand_closure, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_grasped)

    # ------------------------------------------------------------------ stages 0-2: approach, orient, tips, close, grasp
    # approach max 1.5 per hand: start pose (d_pre ~ 12 cm) -> 0.45, over-the-mouth hover -> 0.90, beside the cup -> 1.5
    approach_src = 0.8 * s["reach_lin"] + 0.7 * s["reach_fine"]
    approach_rcv = 0.8 * r["reach_lin"] + 0.7 * r["reach_fine"]
    orient_src = 0.3 * s["orient"]           # weight 0.3: helper only, must not compete with approach
    orient_rcv = 0.3 * r["orient"]
    tips_src = 0.6 * s["tip_place"]          # weight 0.6: geometry below real contact income (1.5)
    tips_rcv = 0.6 * r["tip_place"]
    close_src = 0.5 * s["close_near"]        # weight 0.5: closing pays only with tips at the wall
    close_rcv = 0.5 * r["close_near"]
    contact_src = 1.0 * s["contact"]         # weight 1.0 (max 1.5): bridge from touching to the 1 N grasp flag
    contact_rcv = 1.0 * r["contact"]
    grasp_src = 2.0 * s["grasp"]             # weight 2.0 > whole approach gain: a proper grasp must dominate
    grasp_rcv = 2.0 * r["grasp"]
    grasp_both = 1.0 * s["grasp"] * r["grasp"]
    curl_far_pen = -0.3 * (s["curl_far"] + r["curl_far"])    # open hand while outside the pre-grasp region
    rim_hook_pen = -0.5 * (s["rim_hook"] + r["rim_hook"])    # flat cost; the real lever is thumb_low on all income

    hold_s, hold_r = s["hold"], r["hold"]
    q_s, q_r = s["q"], r["q"]

    # ------------------------------------------------------------------ stage 3: lift (independent per arm, x quality)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # tanh(h/3cm): 1 cm -> 0.32, 3 cm -> 0.76; weight 3.0 on top of grasp income
    lift_src = 3.0 * hold_s * q_s * torch.tanh(torch.clamp(h_src, min=0.0) / 0.03)
    lift_rcv = 3.0 * hold_r * q_r * torch.tanh(torch.clamp(h_rcv, min=0.0) / 0.03)
    lifted = hold_s * (h_src > 0.03).to(dtype) * hold_r * (h_rcv > 0.03).to(dtype)

    # ------------------------------------------------------------------ stage 4: carry together (tilt-aware target)
    tilt = ctx.src_cup_tilt
    # upright: mouths 10 cm apart (cup bodies side by side, no nesting); the gap closes to 0 as tilt -> ~90 deg,
    # because in a real pour the source ORIGIN stays ~5 cm aside and the tilted mouth swings over the receiver.
    pour_prog = torch.clamp((tilt - 0.35) / 1.2, 0.0, 1.0)
    tgt_dxy = 0.10 * (1.0 - pour_prog)
    mouth_dxy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    dxy_err = torch.abs(mouth_dxy - tgt_dxy)
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    dz_err = _band_err(mouth_dz, 0.04, 0.12)                                   # source mouth 4..12 cm higher
    carry_q = (0.5 * torch.clamp(1.0 - dxy_err / 0.30, 0.0, 1.0) + 0.5 * _near(dxy_err, 30.0)) * _near(dz_err, 15.0)
    align = 4.0 * lifted * q_s * carry_q     # weight 4.0 > lift 3.0: carrying toward the receiver must pay

    # ------------------------------------------------------------------ stage 5: tilt and pour
    zone = _near(dxy_err, 40.0) * _near(dz_err, 40.0)                          # 1 cm error -> 0.67
    tilt_r = 3.0 * lifted * q_s * zone * torch.clamp(tilt / 2.0, 0.0, 1.0)     # tilt pays only on the schedule
    pour_delta = 200.0 * ctx.d_in_target     # +10 per transferred bead (20 beads)
    spill_delta = -100.0 * ctx.d_spill       # -5 per spilled bead: costly but never worth refusing to pour

    # ------------------------------------------------------------------ stage 6: success
    success = 15.0 * ctx.success.to(dtype)   # 15/step > held+lifted+carry+tilt income: hold the goal

    # ------------------------------------------------------------------ constraints
    has_beads = (ctx.bead_in_source_frac > 0.05).to(dtype)
    # tilting with beads outside the pour schedule spills them
    pre_tilt_pen = -1.0 * (1.0 - lifted * zone) * has_beads * torch.tanh(2.0 * torch.clamp(tilt - 0.8, min=0.0))
    # receiver upright only while held (always-on version taught the left arm to stay away)
    rcv_upright_pen = -1.0 * hold_r * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.1, min=0.0))
    src_free = 1.0 - hold_s
    rcv_free = 1.0 - hold_r
    # unheld cup knocked / shoved: milder than before (a toppled cup already loses all later income)
    knock_pen = -0.3 * (src_free * torch.tanh(3.0 * torch.clamp(ctx.src_cup_tilt - 0.3, min=0.0))
                        + rcv_free * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.3, min=0.0)))
    disp_src = torch.norm(ctx.src_cup_pos[:, :2] - ctx.src_cup_spawn_pos[:, :2], dim=-1)
    disp_rcv = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    cup_push_pen = -0.2 * (src_free * torch.tanh(torch.clamp(disp_src - 0.03, min=0.0) / 0.03)
                           + rcv_free * torch.tanh(torch.clamp(disp_rcv - 0.03, min=0.0) / 0.03))
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
        "tilt": tilt_r,
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
bead/in_target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.65e-06 · max 0.000195
bead/spill: [9.77e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.26e-05 · max 0.00244
done/drop: [0.00391, 0, 0, 0, 0.000977, 0, 0, 0, 0, 0]  min 0 · mean 0.000219 · max 0.00879
done/mimic_runaway: [0.00488, 0.000977, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 9.07e-05 · max 0.00488
episode_lengths/step: [114, 831, 853, 879, 865, 850, 835, 848, 865, 839]  min 112 · mean 833 · max 892
reward/action_rate_pen: [-0.0515, -0.0468, -0.0414, -0.0433, -0.043, -0.0417, -0.0403, -0.0389, -0.0401, -0.0394]  min -0.0517 · mean -0.042 · max -0.0375
reward/align: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/approach_rcv: [0.376, 0.828, 1.21, 1.34, 1.39, 1.35, 1.39, 1.38, 1.4, 1.4]  min 0.371 · mean 1.26 · max 1.43
reward/approach_src: [0.405, 0.592, 0.615, 0.782, 0.96, 1.04, 1.11, 1.15, 1.19, 1.23]  min 0.392 · mean 0.953 · max 1.25
reward/arm_speed_pen: [-0.0562, -0.0224, -0.0171, -0.0137, -0.0114, -0.0118, -0.0107, -0.0107, -0.011, -0.0101]  min -0.0562 · mean -0.0141 · max -0.00922
reward/close_rcv: [3.88e-05, 0.00546, 0.0168, 0.0325, 0.0574, 0.0536, 0.0965, 0.119, 0.107, 0.101]  min 3.88e-05 · mean 0.0655 · max 0.13
reward/close_src: [4.36e-05, 0.00267, 0.0061, 0.00721, 0.00864, 0.0159, 0.00525, 0.00206, 0.00128, 0.00486]  min 4.36e-05 · mean 0.0059 · max 0.0256
reward/contact_rcv: [0, 0.000839, 0.00397, 0.0135, 0.0451, 0.0267, 0.0408, 0.0833, 0.0917, 0.131]  min 0 · mean 0.0541 · max 0.161
reward/contact_src: [0.000159, 0, 0, 0.000212, 0.000164, 0.000587, 0.000869, 0, 0, 0.00023]  min 0 · mean 0.000251 · max 0.00192
reward/cup_collision_pen: [-0.000122, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00391 · mean -3.86e-05 · max 0
reward/cup_push_pen: [-0.0207, -0.000987, -0.000478, -0.00085, -0.000637, -0.00126, -0.000239, -0.00107, -0.000828, -0.00119]  min -0.0355 · mean -0.00134 · max 0
reward/cup_speed_pen: [-0.00891, -4.89e-05, 0, 0, -0.000154, 0, -0.000111, -0.00019, -0.000272, -0.000345]  min -0.00974 · mean -0.000238 · max 0
reward/curl_far_pen: [0, -0.163, -0.104, -0.0651, -0.0179, -0.00906, -0.00287, -0.00228, -0.00137, -0.000511]  min -0.215 · mean -0.043 · max 0
reward/drop_pen: [-0.0049, 0, 0, 0, 0, 0, 0, 0, 0, -0.000917]  min -0.00813 · mean -0.000195 · max 0
reward/grasp_both: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grasp_rcv: [0, 4.74e-05, 0, 0, 0.000977, 0, 0.000977, 0.000977, 0.00178, 0.00195]  min 0 · mean 0.00056 · max 0.00751
reward/grasp_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/hand_foreign_pen: [-0.00416, -0.00133, -0.00109, -0.00282, 0, -3.08e-06, -0.000131, 0, -0.00195, 0]  min -0.0187 · mean -0.00143 · max 0
reward/knock_pen: [-0.027, -0.00173, -0.000788, -0.00105, -0.00069, -0.000965, -0.000156, -0.000717, -0.000564, -0.000431]  min -0.0462 · mean -0.00148 · max 0
reward/lift_rcv: [0, 0, 0, 0, 0.00187, 0, 2.26e-05, 0.000227, 0.000298, 0.00033]  min 0 · mean 0.000184 · max 0.00255
reward/lift_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/nested_pen: [-0.00586, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0293 · mean -0.000508 · max 0
reward/orient_rcv: [0.0573, 0.132, 0.145, 0.161, 0.154, 0.175, 0.166, 0.199, 0.18, 0.177]  min 0.0573 · mean 0.163 · max 0.21
reward/orient_src: [0.0571, 0.0795, 0.0793, 0.0969, 0.0876, 0.0877, 0.0922, 0.092, 0.115, 0.11]  min 0.0571 · mean 0.0938 · max 0.118
reward/palm_push_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.000567 · mean -8.12e-06 · max 0
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.4e-05 · max 0.00977
reward/pre_tilt_pen: [-0.00682, -0.000166, -3.41e-05, -0.000784, 0, -0.000507, 0, 0, 0, 0]  min -0.0242 · mean -0.000515 · max 0
reward/rcv_upright_pen: [0, 0, 0, 0, -0.000486, 0, 0, 0, -7.11e-05, -0.00023]  min -0.00123 · mean -9.62e-05 · max 0
reward/rim_hook_pen: [-0.334, -0.0734, -0.053, -0.039, -0.033, -0.0236, -0.0167, -0.0154, -0.0183, -0.0134]  min -0.334 · mean -0.0386 · max -0.00684
reward/spill_delta: [-0.00977, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0293 · mean -0.000133 · max 0.00488
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tips_rcv: [0.0622, 0.114, 0.139, 0.165, 0.177, 0.176, 0.185, 0.202, 0.198, 0.197]  min 0.0561 · mean 0.167 · max 0.207
reward/tips_src: [0.0626, 0.109, 0.13, 0.137, 0.122, 0.123, 0.102, 0.102, 0.087, 0.0954]  min 0.0605 · mean 0.11 · max 0.148
reward/total: [0.449, 1.52, 2.09, 2.55, 2.87, 2.93, 3.1, 3.25, 3.27, 3.35]  min 0.341 · mean 2.7 · max 3.43
rewards/step: [36.5, 1.16e+03, 1.8e+03, 2.2e+03, 2.44e+03, 2.5e+03, 2.55e+03, 2.72e+03, 2.83e+03, 2.8e+03]  min 36.5 · mean 2.27e+03 · max 2.99e+03
task/aim_dist: [0.318, 0.32, 0.318, 0.319, 0.32, 0.321, 0.323, 0.322, 0.334, 0.325]  min 0.31 · mean 0.328 · max 0.776
task/cup_collision_rate: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.6e-05 · max 0.00781
task/cups_center_dist: [0.321, 0.32, 0.318, 0.319, 0.32, 0.321, 0.322, 0.321, 0.333, 0.323]  min 0.311 · mean 0.328 · max 0.777
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0.00293, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000254 · max 0.0146
task/rcv_cup_lift: [0.00381, 4.08e-05, 0.00014, 0.000224, 0.000484, 0.000285, 0.000397, 0.00061, 0.0117, 0.00134]  min -0.000304 · mean 0.00284 · max 0.129
task/rcv_grasped: [0, 0.000977, 0, 0, 0.000977, 0, 0.000977, 0.000977, 0.00195, 0.00195]  min 0 · mean 0.000633 · max 0.00781
task/rcv_hand_foreign_rate: [0.000977, 0, 0, 0, 0, 0, 0, 0.000977, 0, 0]  min 0 · mean 0.00044 · max 0.00879
task/src_cup_lift: [0.00105, 7.4e-06, 1.21e-05, 4.11e-05, 6.08e-05, 0.000158, 0.000107, 0.000131, 0.000275, 0.000286]  min -7.9e-05 · mean 0.000196 · max 0.0173
task/src_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.79e-06 · max 0.000977
task/src_hand_foreign_rate: [0.00195, 0.00195, 0.000977, 0, 0, 0, 0, 0, 0.000977, 0]  min 0 · mean 0.000551 · max 0.00684
task/src_tilt_deg: [3.09, 0.122, 0.104, 0.36, 0.216, 0.67, 0.398, 0.477, 0.944, 0.984]  min 0.0201 · mean 0.552 · max 5.64
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Observations for this round (previous reward function iter_04, corrected robot model, trained from scratch for 686 epochs). Items 1-6 are facts: training metrics, the environment's own grasp definition and a rollout video. Item 7 records operator decisions.

1. **The approach now works.** Both palms reach the pre-grasp region beside their cups with the thumb below the rim. The environment also logs an approach check that is independent of the reward: over envs whose palm is within 10 cm of the cup origin, how often the thumb tip is at or above rim height over the cup opening, and the mean thumb-tip height relative to the rim. Values are medians over epoch windows:

| metric | ep 0-10 | ep 225-235 | ep 400-410 | ep 590-600 | ep 676-686 |
|---|---|---|---|---|---|
| `task/src_palm_to_cup` [m] | 0.222 | 0.138 | 0.118 | 0.109 | 0.106 |
| `task/rcv_palm_to_cup` [m] | 0.241 | 0.086 | 0.083 | 0.088 | 0.082 |
| `task/src_near_rate` (palm < 10 cm) | 0.00 | 0.00 | 0.10 | 0.54 | 0.63 |
| `task/rcv_near_rate` | 0.00 | 0.88 | 0.91 | 0.92 | 0.93 |
| `task/src_thumb_over_rim_near` | – | – | 0.00 | 0.00 | 0.00 |
| `task/rcv_thumb_over_rim_near` | – | 0.10 | 0.02 | 0.14 | 0.03 |
| `task/src_thumb_above_rim_mm_near` [mm] | – | – | +0.2 | −5.3 | −11.8 |
| `task/rcv_thumb_above_rim_mm_near` [mm] | – | +3.0 | −11.1 | −11.3 | −15.0 |

2. **Contact grows on the receiver side only, and no grasp is ever registered.**

| metric | ep 0-10 | ep 225-235 | ep 400-410 | ep 590-600 | ep 676-686 |
|---|---|---|---|---|---|
| `contact/src_max` [N] | 0.32 | 0.011 | 0.023 | 0.002 | 0.001 |
| `contact/rcv_max` [N] | 0.45 | 0.38 | 0.96 | 1.64 | 2.25 |
| `task/src_grasped` / `task/rcv_grasped` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0.0015 | 0 / 0 |
| `task/src_closure` / `task/rcv_closure` | 0.13 / 0.13 | 0.55 / 0.57 | 0.62 / 0.66 | 0.59 / 0.57 | 0.59 / 0.55 |
| `task/src_cup_lift` / `task/rcv_cup_lift` [m] | 0.002 / 0.004 | 0.0001 / 0.0002 | 0.0001 / 0.0005 | 0.0002 / 0.0010 | 0.0002 / 0.0011 |

   - `contact/*_max` is the largest per-finger force on the hand's own cup, averaged over envs.
   - The environment's grasp flag (`task/*_grasped`, and `ctx.src_grasped` / `ctx.rcv_grasped`) is true only when the THUMB force is above 1 N AND at least one other finger's force is above 1 N at the same time.
   - So a receiver finger force of 2.25 N with the flag at 0 means that one side of the grasp (thumb, or all other fingers) stays at or below 1 N.

3. **Where the reward comes from.** Per-step medians over the last 30 epochs:
   - Approach, orient and tip terms: approach_src 1.23, approach_rcv 1.40 (max 1.5 each), orient_src 0.11, orient_rcv 0.18, tips_src 0.09, tips_rcv 0.20. Together 3.21 of `reward/total` 3.37.
   - Closing and contact terms: close_rcv 0.10, close_src 0.007, contact_rcv 0.148, contact_src 0.0002.
   - Terms at zero: grasp_src, grasp_rcv, grasp_both, lift_src (0.0000), lift_rcv (0.0002), align, tilt, pour_delta, success.
   - Penalties: rim_hook_pen −0.012, curl_far_pen −0.0004; all other penalties ≈ 0.
   - Most of the income is paid for being in the approach region, which the policy already holds.

4. **Rollout video** at the epoch-700 checkpoint (4 envs, default camera, env 0 shown, `our_source/pour_t2r_rh_i04_ep700_0916.mp4`, frame sheets in `pour_t2r_rh_i04_ep700_0916_frames/`):
   - From about step 100 until the reset at step 900, both hands stay low beside their cups. Their posture barely changes, no cup is lifted, and the arms do not move up.
   - The hand on the image left stands beside its cup with the four fingers curled downward and the thumb curled toward them. The fingers do not wrap the cup body, and the cup stands just outside the hand.
   - The hand at the image centre is closed, with fingers and thumb curled. Its cup is hidden behind or inside the hand, so thumb placement on the cup wall cannot be seen at this resolution.
   - Which image hand is source and which is receiver was not determined from the video. The contact metrics above suggest the centre hand is the one touching its cup.

5. **Physics stayed clean.** `ctrl/mimic_err_max` median was 0.12 rad over the last 10 epochs. `done/mimic_err_runaway`, `done/drop`, `task/cup_collision_rate` and `task/*_hand_foreign_rate` were all ≈ 0 after epoch 225.

6. **Previous rounds, for context.**
   - Round 5 (iter_03 on the corrected model): both palms stopped 16-17 cm away, with zero contact.
   - Round 4 (old model): the right hand held its cup with the thumb over the mouth, and the left hand never touched its cup.

7. **Operator decisions** (after reviewing items 1-6, the frame sheet and the zoomed hands):
   - This draft is approved as the feedback for this round, and a new reward is to be generated.
   - The approach stage is considered solved. Both palms reach the pre-grasp region beside the cup with the thumb below the rim, and that behaviour must be kept.
   - The next bottleneck is the grasp itself: the thumb and an opposing finger pressing the cup at the same time (both above 1 N), with the thumb on the wall opposite the fingers. After that comes lifting. Neither has ever happened for either hand.
   - The judgement from earlier rounds still stands: once both hands reliably grasp their cups, the later stages are expected to follow.

The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
