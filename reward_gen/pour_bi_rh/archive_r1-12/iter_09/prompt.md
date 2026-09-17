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
    bead_fill_level: torch.Tensor       # (N,) 에피소드 시작 시 소스 컵이 부피로 얼마나 찼는지 [0,1] (정착 후 실측, 에피소드 동안 고정; 비드 양은 에피소드마다 다르다)

    # ---- 충돌 (실기 안전 — 09.14) ---------------------------------------------------------
    cup_cup_force: torch.Tensor         # (N,) 두 컵이 서로 부딪히는 접촉력 [N] (0 = 안 닿음)
    src_hand_foreign_force: torch.Tensor  # (N,) 소스 손이 **자기 컵 외**(상대 손·상대 컵·테이블)에 닿는 힘 [N]
    rcv_hand_foreign_force: torch.Tensor  # (N,) 리시버 손이 자기 컵 외에 닿는 힘 [N]

    # ---- 과제 판정 / 시간 ------------------------------------------------------------
    cups_nested: torch.Tensor           # (N,) bool 두 컵 원점 거리 < 9 cm — 소스 컵이 리시버 컵에 끼워져 있음(붓기가 아니라 성공 무효)
    premature_tilt: torch.Tensor        # (N,) bool 에피소드 래치 — 소스 입구가 리시버 입구에서 xy 10 cm 보다 멀 때 소스가 30° 를 넘은 적이 있음(성공 무효, 리셋 전까지 유지)
    success: torch.Tensor               # (N,) bool 성공 조건 충족 (env 가 판정, 보상이 바꿀 수 없음; cups_nested·premature_tilt 면 항상 False)
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
5. Cup tilt: `ctx.src_cup_tilt` is the angle between the cup's up axis and world +z (0 = upright, π/2 = horizontal). The amount of beads in the source cup changes every episode and `ctx.bead_fill_level` (0 = empty, 1 = full to the rim, measured once the beads have settled, constant during the episode) tells the policy how full the cup is by volume. Beads start leaving the cup at a tilt that depends on that fill (measured): the first bead leaves at about 70° when the cup is full and at about 90° when it holds only a few beads, and a full cup has lost a fifth of its beads by 80° and half by 86°. `ctx.src_cup_mouth_pos` and `ctx.rcv_cup_mouth_pos` are the rim centres.
6. Bead bookkeeping: `bead_in_target_frac` rises as beads land in the receiver cup; `bead_spill_frac` counts beads lost outside both cups (permanent). `d_in_target` and `d_spill` are this step's increments — reward INCREMENTS of beads transferred rather than the level, otherwise the policy is paid for standing still with a filled cup.
7. `ctx.success` is computed by the environment (enough beads in the receiver cup, little spill, cups close together, the receiver cup held nearly upright — `ctx.rcv_cup_tilt` at most 20° — the cups NOT nested, and NO premature tilt: `ctx.premature_tilt` latches for the rest of the episode as soon as the source cup exceeds 30° while its mouth is still more than 10 cm (xy) from the receiver's mouth, and a latched episode can never succeed). You may add a bonus on it but you cannot redefine it. Beads only count as "in the receiver" once they have LEFT the source cup — pushing the source cup into the receiver cup (`ctx.cups_nested`) transfers nothing and is never a success; the beads must fall out of the tilted source cup through the air.
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


def _cyl(points: torch.Tensor, cup_pos: torch.Tensor, cup_up: torch.Tensor):
    """Cup-cylinder coords of (N,K,3) points: axial height (N,K), radial vector (N,K,3), radial distance (N,K)."""
    rel = points - cup_pos[:, None, :]
    up = cup_up[:, None, :]
    axial = (rel * up).sum(dim=-1)
    rvec = rel - axial[..., None] * up
    radial = torch.norm(rvec, dim=-1)
    return axial, rvec, radial


def _axis_to_segments(rv_a: torch.Tensor, rv_b: torch.Tensor) -> torch.Tensor:
    """Distance (N,K) from the cup AXIS to each segment joining rv_a (N,1,3) to rv_b (N,K,3),
    measured in the plane normal to the axis. ~0 only when the axis lies BETWEEN the two tips."""
    seg = rv_b - rv_a
    u = torch.clamp(-(rv_a * seg).sum(dim=-1) / ((seg * seg).sum(dim=-1) + 1e-8), 0.0, 1.0)
    closest = rv_a + u[..., None] * seg
    return torch.norm(closest, dim=-1)


def _hand_terms(ctx, palm_pos, palm_axes, tips, finger_force, palm_force, closure,
                cup_pos, cup_up, grasped, a_hand):
    """Per-arm approach / pad-direction / opening / opposition / contact signals, all (N,) in [0, 1]."""
    dtype = cup_pos.dtype
    mouth = ctx.cup_mouth_z
    r_in = ctx.cup_radius
    r_wall = r_in + 0.004                    # outer wall ~ inner radius + wall thickness
    d_out = 2.0 * r_wall                     # outer diameter
    w_cup = max(2.0 * r_in, 0.055)           # the cup is 57 mm across; floored so the target can never collapse
    gap_lo = w_cup + 0.003                   # the opening must CLEAR the cup before it can go around it
    gap_hi = w_cup + 0.018                   # comfortably wider than the cup -> full credit
    tip_lo = ctx.cup_bottom_z + 0.03         # tips >= 3 cm above the cup bottom (table clearance)
    tip_hi = mouth - 0.015                   # tips >= 1.5 cm below the rim

    # ---- palm pre-grasp region (this stage is solved; it stays deliberately cheap)
    p_ax, _, p_rad = _cyl(palm_pos[:, None, :], cup_pos, cup_up)
    rad_err = _band_err(p_rad[:, 0], r_wall + 0.01, 0.10)
    ht_err = _band_err(p_ax[:, 0], ctx.cup_bottom_z + 0.03, mouth + 0.01)
    d_pre = torch.sqrt(rad_err ** 2 + ht_err ** 2 + 1e-8)
    reach_lin = torch.clamp(1.0 - d_pre / 0.25, 0.0, 1.0)
    reach_fine = _near(d_pre, 25.0)
    near_pre = _near(d_pre, 10.0)

    # ---- PAD DIRECTION, SIGNED (round 7 used |.| here, so a palm turned away scored like a palm
    #      turned toward, and the policy laid the BACK of its fingers on the cup for free).
    #      palm_axes[:, 0:3] is the palm normal, i.e. the direction the pad faces.
    normal = palm_axes[:, 0:3]
    normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
    to_cup = cup_pos - palm_pos
    u3 = to_cup / (torch.norm(to_cup, dim=-1, keepdim=True) + 1e-6)
    facing = (normal * u3).sum(dim=-1)                       # +1 pad at the cup, -1 back of the hand
    face_lin = torch.clamp(facing / 0.5, 0.0, 1.0)           # income: saturates at 60 deg, a fingertip
    #                                                          grasp does not need the cup square on the pad
    pad = torch.clamp(facing / 0.35, 0.0, 1.0)               # HARD gate, no floor: 0 for any pad not
    #                                                          turned toward the cup. Multiplies every
    #                                                          orientation / posture / contact term below.
    back = torch.clamp(-facing, 0.0, 1.0)                    # 1 = the cup is behind the hand

    # ---- tips ahead of the palm (convention-free: the cup is in front of the fingers, not beside them)
    d_palm_xy = torch.norm(cup_pos[:, :2] - palm_pos[:, :2], dim=-1)
    tip_mid_all = tips[:, 0:3, :].mean(dim=1)
    d_tip_xy = torch.norm(tip_mid_all[:, :2] - cup_pos[:, :2], dim=-1)
    ahead = torch.clamp((d_palm_xy - d_tip_xy) / 0.03, 0.0, 1.0)
    orient_raw = near_pre * pad * (0.6 * face_lin + 0.4 * ahead)

    # ---- fingertips in cup-cylinder coordinates
    t_ax, t_rv, t_rad = _cyl(tips, cup_pos, cup_up)
    t_dir = t_rv / (t_rad[..., None] + 1e-6)

    # ---- OPPOSITION. +1 = this finger sits on the far wall from the thumb, -1 = same side as the thumb.
    u_th = t_dir[:, 0, :]
    opp_all = (t_dir[:, 1:, :] * u_th[:, None, :]).sum(dim=-1) * (-1.0)      # (N,F-1) in [-1, 1]
    opp_best = torch.clamp(opp_all.max(dim=-1).values, 0.0, 1.0)             # logged as oppose_*
    # HARD gate, no floor: 90 deg apart -> 0, 120 deg -> 1.  A thumb beside or inside the fingers scores 0.
    opp_gate_all = torch.clamp((opp_all - 0.3) / 0.5, 0.0, 1.0)
    opp_gate = opp_gate_all.max(dim=-1).values
    u_fm = t_dir[:, 1:3, :].mean(dim=1)
    u_fm = u_fm / (torch.norm(u_fm, dim=-1, keepdim=True) + 1e-6)
    opp_mid = -(u_th * u_fm).sum(dim=-1)                                     # -1 = thumb among the fingers
    opp_soft = torch.clamp((opp_mid + 0.1) / 0.6, 0.0, 1.0)

    # ---- tip admissibility: at wall height and outside the bore (not dipped in, not far outside)
    ht_bad = _band_err(t_ax, tip_lo, tip_hi)                                  # (N,F)
    h_ok = _near(ht_bad, 40.0)
    r_ok = (torch.clamp((t_rad - r_in) / 0.005, 0.0, 1.0)
            * torch.clamp((r_wall + 0.05 - t_rad) / 0.02, 0.0, 1.0))
    tip_ok = h_ok * r_ok                                                      # (N,F)
    thumb_ax = t_ax[:, 0]
    thumb_low = torch.clamp((mouth - thumb_ax) / 0.015, 0.0, 1.0)             # thumb below the rim = on the wall

    # ---- distance of each tip to the outer wall (radial gap + height-band error)
    rad_out = torch.clamp(t_rad - r_wall - 0.003, min=0.0)
    surf = torch.sqrt(rad_out ** 2 + ht_bad ** 2 + 1e-8)
    wall_all = 0.35 * _near(surf, 12.0) + 0.65 * _near(surf, 50.0)
    wall_t = wall_all[:, 0]
    wall_f = wall_all[:, 1:].max(dim=-1).values
    wall_pair = 0.5 * (wall_t + wall_f)
    at_cup = _near(surf.min(dim=-1).values, 15.0)                             # a tip is actually on the cup

    # ---- POSE: strict per-finger straddle. The cup axis lies between the thumb tip and THIS finger's
    #      tip, both tips admissible, the pair genuinely opposed, the pad turned toward the cup. No floor.
    seg_d = _axis_to_segments(t_rv[:, 0:1, :], t_rv[:, 1:, :])                # (N,F-1)
    pair_geo = _near(seg_d, 40.0) * opp_gate_all * tip_ok[:, 1:] * tip_ok[:, 0:1]
    pose = pair_geo.max(dim=-1).values * pad

    # ---- OPENING (requirement 2). gap = thumb tip to the NEARER of the index / middle tips, the same
    #      quantity as task/{src,rcv}_tip_gap_mm_near. The policy settled at 48-50 mm, i.e. narrower
    #      than the 57 mm cup, because nothing paid for opening.
    gap_pair = torch.norm(tips[:, 0:1, :] - tips[:, 1:3, :], dim=-1).min(dim=-1).values
    # spread: pays ONLY once the opening clears the cup. Not gated on opposition - it is the rung below it.
    gap_ok = (torch.clamp((gap_pair - gap_lo) / (gap_hi - gap_lo), 0.0, 1.0)
              * torch.clamp((0.155 - gap_pair) / 0.025, 0.0, 1.0))
    # ... but a hand laid ON the cup with the thumb tucked in among the fingers is not an opening,
    # however wide it measures: it is the round-7 posture with the object outside the jaw. `jaw_ok`
    # zeroes the whole pre-grasp ladder there (it is ~1 whenever the tips are not yet on the cup,
    # so it never blocks the approach).
    same_side = torch.clamp((-0.1 - opp_mid) / 0.5, 0.0, 1.0)
    jaw_ok = 1.0 - at_cup * same_side
    orient = orient_raw * jaw_ok
    spread = near_pre * pad * gap_ok * jaw_ok
    # open: the sub-threshold gradient. Squared, so a 48 mm hand collects ~0.05 of the 0.5 available
    # while d(reward)/d(opening) stays strictly positive all the way from a fist to 75 mm.
    open_gap = torch.clamp((gap_pair - 0.025) / max(gap_lo - 0.025, 1e-3), 0.0, 1.0)
    open_cmd = torch.clamp((0.45 - closure) / 0.40, 0.0, 1.0)
    open_hand = near_pre * pad * (0.5 * (open_gap + open_cmd)) ** 2 * (1.0 - pose) * jaw_ok

    # ---- POCKET: is the cup body actually inside the opening? midpoint of the thumb tip and the
    #      index/middle mean must sit ON the cup axis, the opening wider than the cup, both sides at
    #      wall height, thumb below the rim, and exactly 0 when the thumb is on the fingers' side.
    tip_f_mid = tips[:, 1:3, :].mean(dim=1)
    mid = 0.5 * (tips[:, 0, :] + tip_f_mid)
    _, _, d_pocket = _cyl(mid[:, None, :], cup_pos, cup_up)
    d_pocket = d_pocket[:, 0]
    h_pocket = h_ok[:, 0] * h_ok[:, 1:3].max(dim=-1).values
    pocket = ((0.35 * _near(d_pocket, 15.0) + 0.65 * _near(d_pocket, 45.0))
              * gap_ok * h_pocket * opp_soft * thumb_low * pad)

    # ---- close the last centimetres: ONLY inside a pose that can trap the cup
    close = pose * wall_pair

    # ---- squeeze intent from the COMMANDED closure, only inside a valid pose in contact range
    cmd = 0.5 * (torch.clamp(a_hand, -1.0, 1.0) + 1.0)
    cmd_close = (cmd[:, 0] + cmd[:, 1] + cmd[:, 2:6].mean(dim=-1)) / 3.0
    squeeze = pose * cmd_close * wall_pair

    # ---- closing on air / curling while far / hooking the rim
    gap_min = torch.norm(tips[:, 0:1, :] - tips[:, 1:, :], dim=-1).min(dim=-1).values
    air_pinch = torch.clamp((w_cup - 0.005 - gap_min) / 0.025, 0.0, 1.0)
    far = 1.0 - _near(torch.clamp(d_pre - 0.03, min=0.0), 20.0)
    curl_far = torch.clamp((closure - 0.3) / 0.4, 0.0, 1.0) * far
    over_rim = torch.clamp((thumb_ax - (mouth - 0.005)) / 0.015, 0.0, 1.0)
    over_footprint = (t_rad[:, 0] < r_wall + 0.025).to(dtype)
    rim_hook = over_rim * over_footprint

    # ---- FORCES. Only a finger that OPPOSES the thumb counts as the far side of the pinch, and only
    #      while the pad faces the cup: the force sensors report any contact in any direction, so
    #      without `pad` the back of the fingers reads as legitimate grasp progress (round 7's failure).
    f_t = finger_force[:, 0]
    f_o = (finger_force[:, 1:] * opp_gate_all).max(dim=-1).values
    f_min = torch.minimum(f_t, f_o)
    s_t = 1.0 - torch.exp(-f_t / 0.3)
    s_o = 1.0 - torch.exp(-f_o / 0.3)
    q_grasp = opp_gate * thumb_low * pad                                      # hard: no opposition or no pad, no pay
    touch = (0.6 * s_t * wall_f + 0.4 * s_o * wall_t) * q_grasp
    grip = (0.5 * (1.0 - torch.exp(-f_min / 0.25)) + 0.5 * torch.clamp(f_min / 1.0, 0.0, 1.0)) * q_grasp
    grasped_f = grasped.to(dtype)
    grasp = grasped_f * q_grasp                                               # the env flag alone is not enough
    hold_soft = torch.clamp((f_min - 0.3) / 0.7, 0.0, 1.0)
    hold = torch.maximum(grasped_f, hold_soft) * thumb_low * opp_gate * pad
    q = 0.3 + 0.7 * q_grasp

    # ---- pressing a cup with a palm turned away, and loitering near it back-first
    dorsal = back * (0.15 * near_pre
                     + 1.2 * torch.tanh((finger_force.sum(dim=-1) + palm_force) / 1.0))

    return {
        "reach_lin": reach_lin, "reach_fine": reach_fine, "orient": orient,
        "open": open_hand, "spread": spread, "oppose": opp_best * near_pre * pad * jaw_ok,
        "pocket": pocket, "pose": pose, "close": close, "squeeze": squeeze,
        "air_pinch": air_pinch, "curl_far": curl_far, "rim_hook": rim_hook, "dorsal": dorsal,
        "touch": touch, "grip": grip, "grasp": grasp, "hold": hold, "q": q,
    }


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = torch.zeros_like(ctx.src_cup_tilt)
    dtype = zero.dtype
    act = ctx.actions

    s = _hand_terms(ctx, ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos, ctx.src_finger_force,
                    ctx.src_palm_force, ctx.src_hand_closure, ctx.src_cup_pos, ctx.src_cup_up,
                    ctx.src_grasped, act[:, 6:12])
    r = _hand_terms(ctx, ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                    ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_cup_pos, ctx.rcv_cup_up,
                    ctx.rcv_grasped, act[:, 18:24])
    hold_s, hold_r = s["hold"], r["hold"]
    q_s, q_r = s["q"], r["q"]

    # ---------------------------------------------------- stage 0: approach (solved; kept cheap)
    approach_src = 0.2 * s["reach_lin"] + 0.1 * s["reach_fine"]      # max 0.3 per hand
    approach_rcv = 0.2 * r["reach_lin"] + 0.1 * r["reach_fine"]
    # weight 0.3: the SIGNED pad direction is now a prerequisite, but it must stay far below the
    # opening terms below, or facing the cup with a closed fist becomes the next plateau.
    orient_src = 0.3 * s["orient"]
    orient_rcv = 0.3 * r["orient"]

    # ---------------------------------------------------- stage 1: OPEN the hand wider than the cup
    open_src = 0.5 * s["open"]            # sub-threshold gradient only; ~0.05 for the 48 mm hand
    open_rcv = 0.5 * r["open"]
    spread_src = 1.5 * s["spread"]        # weight 1.5 > orient 0.3 + approach 0.3: opening beats standing
    spread_rcv = 1.5 * r["spread"]

    # ---------------------------------------------------- stage 2: get the cup BETWEEN thumb and fingers
    oppose_src = 0.3 * s["oppose"]        # logged diagnostic + gentle pull; 0 for a same-side thumb
    oppose_rcv = 0.3 * r["oppose"]
    pocket_src = 1.0 * s["pocket"]        # dense "cup inside the opening" signal
    pocket_rcv = 1.0 * r["pocket"]
    pose_src = 1.5 * s["pose"]            # strict straddle; the gate for everything below
    pose_rcv = 1.5 * r["pose"]
    # min(), not sum(): one hand improving while the other goes backwards must not pay.
    pose_both = 2.0 * torch.minimum(s["pose"], r["pose"])

    # ---------------------------------------------------- stage 3: onto the wall and squeeze
    close_src = 2.0 * s["close"]          # weight 2.0 but NO floor: 0 unless the pose can trap the cup
    close_rcv = 2.0 * r["close"]
    squeeze_src = 0.8 * s["squeeze"]
    squeeze_rcv = 0.8 * r["squeeze"]
    touch_src = 1.0 * s["touch"]          # one side pressing while the other is in place — a step, not a rest
    touch_rcv = 1.0 * r["touch"]
    grip_src = 3.0 * s["grip"]            # weight 3.0: both sides above 1 N is what the env calls a grasp
    grip_rcv = 3.0 * r["grip"]
    grasp_src = 4.0 * s["grasp"]          # weight 4.0 > grip 3.0: the flag itself is the target
    grasp_rcv = 4.0 * r["grasp"]
    grasp_both = 3.0 * s["grasp"] * r["grasp"]
    curl_far_pen = -0.3 * (s["curl_far"] + r["curl_far"])
    air_pinch_pen = -0.5 * (s["air_pinch"] * (1.0 - hold_s) + r["air_pinch"] * (1.0 - hold_r))
    rim_hook_pen = -0.5 * (s["rim_hook"] + r["rim_hook"])
    # the back of the hand on the cup is not contact progress — it costs
    dorsal_pen = -1.0 * (s["dorsal"] + r["dorsal"])

    # ---------------------------------------------------- stage 4: lift
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # weight 8.0 per hand > the whole grasp plateau's marginal value: a finished grasp on the table is not the end
    lift_src = 8.0 * hold_s * q_s * torch.tanh(torch.clamp(h_src, min=0.0) / 0.025)
    lift_rcv = 8.0 * hold_r * q_r * torch.tanh(torch.clamp(h_rcv, min=0.0) / 0.025)
    lifted = hold_s * (h_src > 0.03).to(dtype) * hold_r * (h_rcv > 0.03).to(dtype)

    # ---------------------------------------------------- stage 5: carry together (tilt-aware target)
    tilt = ctx.src_cup_tilt
    # upright: mouths 10 cm apart (bodies side by side, no nesting); the gap closes as tilt -> ~90 deg,
    # because in a real pour the source ORIGIN stays aside and the tilted mouth swings over the receiver.
    pour_prog = torch.clamp((tilt - 0.35) / 1.2, 0.0, 1.0)
    tgt_dxy = 0.10 * (1.0 - pour_prog)
    mouth_dxy = torch.norm(ctx.src_cup_mouth_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    dxy_err = torch.abs(mouth_dxy - tgt_dxy)
    mouth_dz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    dz_err = _band_err(mouth_dz, 0.04, 0.12)                     # source mouth 4..12 cm higher
    carry_q = (0.5 * torch.clamp(1.0 - dxy_err / 0.30, 0.0, 1.0) + 0.5 * _near(dxy_err, 30.0)) * _near(dz_err, 15.0)
    align = 8.0 * lifted * q_s * carry_q     # weight 8.0 = lift: carrying toward the receiver must keep paying

    # ---------------------------------------------------- stage 6: tilt and pour
    zone = _near(dxy_err, 40.0) * _near(dz_err, 40.0)            # 1 cm error -> 0.67
    tilt_r = 7.0 * lifted * q_s * zone * torch.clamp(tilt / 2.0, 0.0, 1.0)   # tilt pays only on the schedule
    pour_delta = 200.0 * ctx.d_in_target     # +10 per transferred bead (20 beads); increments, never the level
    spill_delta = -100.0 * ctx.d_spill       # -5 per spilled bead: costly but never worth refusing to pour

    # ---------------------------------------------------- stage 7: success
    success = 30.0 * ctx.success.to(dtype)   # 30/step on top of all held income: keep the goal state

    # ---------------------------------------------------- constraints (guards, not shaping)
    has_beads = (ctx.bead_in_source_frac > 0.05).to(dtype)
    pre_tilt_pen = -1.0 * (1.0 - lifted * zone) * has_beads * torch.tanh(2.0 * torch.clamp(tilt - 0.8, min=0.0))
    rcv_upright_pen = -0.8 * hold_r * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.15, min=0.0))
    src_free = 1.0 - hold_s
    rcv_free = 1.0 - hold_r
    disp_src = torch.norm(ctx.src_cup_pos[:, :2] - ctx.src_cup_spawn_pos[:, :2], dim=-1)
    disp_rcv = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    # knocking a cup over and shoving it across the table are the same failure (pressing one side only)
    disturb_pen = -0.3 * (src_free * torch.tanh(3.0 * torch.clamp(ctx.src_cup_tilt - 0.3, min=0.0))
                          + rcv_free * torch.tanh(3.0 * torch.clamp(ctx.rcv_cup_tilt - 0.3, min=0.0))) \
                  - 0.2 * (src_free * torch.tanh(torch.clamp(disp_src - 0.03, min=0.0) / 0.03)
                           + rcv_free * torch.tanh(torch.clamp(disp_rcv - 0.03, min=0.0) / 0.03))
    drop_pen = -1.0 * (src_free * (h_src > 0.015).to(dtype) * torch.tanh(torch.clamp(-ctx.src_cup_lin_vel[:, 2], min=0.0) / 0.5)
                       + rcv_free * (h_rcv > 0.015).to(dtype) * torch.tanh(torch.clamp(-ctx.rcv_cup_lin_vel[:, 2], min=0.0) / 0.5))
    cup_speed_pen = -0.2 * (torch.tanh(torch.clamp(torch.norm(ctx.src_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4)
                            + torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.4, min=0.0) / 0.4))
    nested_pen = -2.0 * ctx.cups_nested.to(dtype)
    # real-robot safety: the cups never touch each other and each hand touches only its own cup
    cup_collision_pen = -1.0 * torch.tanh(ctx.cup_cup_force / 5.0)
    hand_foreign_pen = -2.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0) + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    # the palm is not part of a fingertip grasp: pushing the cup with it costs
    palm_push_pen = -0.5 * (torch.tanh(ctx.src_palm_force / 5.0) + torch.tanh(ctx.rcv_palm_force / 5.0))

    # ---------------------------------------------------- regularisation
    action_rate_pen = -0.05 * ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    arm_speed_pen = -0.05 * (torch.tanh((ctx.src_arm_qd ** 2).mean(dim=-1))
                             + torch.tanh((ctx.rcv_arm_qd ** 2).mean(dim=-1)))

    components = {
        "approach_src": approach_src, "approach_rcv": approach_rcv,
        "orient_src": orient_src, "orient_rcv": orient_rcv,
        "open_src": open_src, "open_rcv": open_rcv,
        "spread_src": spread_src, "spread_rcv": spread_rcv,
        "oppose_src": oppose_src, "oppose_rcv": oppose_rcv,
        "pocket_src": pocket_src, "pocket_rcv": pocket_rcv,
        "pose_src": pose_src, "pose_rcv": pose_rcv, "pose_both": pose_both,
        "close_src": close_src, "close_rcv": close_rcv,
        "squeeze_src": squeeze_src, "squeeze_rcv": squeeze_rcv,
        "touch_src": touch_src, "touch_rcv": touch_rcv,
        "grip_src": grip_src, "grip_rcv": grip_rcv,
        "grasp_src": grasp_src, "grasp_rcv": grasp_rcv, "grasp_both": grasp_both,
        "curl_far_pen": curl_far_pen, "air_pinch_pen": air_pinch_pen,
        "rim_hook_pen": rim_hook_pen, "dorsal_pen": dorsal_pen,
        "lift_src": lift_src, "lift_rcv": lift_rcv,
        "align": align, "tilt": tilt_r,
        "pour_delta": pour_delta, "spill_delta": spill_delta, "success": success,
        "pre_tilt_pen": pre_tilt_pen, "rcv_upright_pen": rcv_upright_pen,
        "disturb_pen": disturb_pen, "drop_pen": drop_pen, "cup_speed_pen": cup_speed_pen,
        "nested_pen": nested_pen, "cup_collision_pen": cup_collision_pen,
        "hand_foreign_pen": hand_foreign_pen, "palm_push_pen": palm_push_pen,
        "action_rate_pen": action_rate_pen, "arm_speed_pen": arm_speed_pen,
    }
    reward = zero
    for v in components.values():
        reward = reward + v
    return reward, components
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 7.22e-08 · max 4.88e-05
bead/spill: [0.000342, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 9.1e-06 · max 0.0019
done/drop: [0.00195, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 6.93e-05 · max 0.00684
done/mimic_runaway: [0.00977, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 6.49e-05 · max 0.00977
episode_lengths/step: [115, 858, 899, 897, 881, 892, 897, 899, 894, 898]  min 115 · mean 875 · max 899
reward/action_rate_pen: [-0.0517, -0.0444, -0.0359, -0.0326, -0.0314, -0.03, -0.0282, -0.0251, -0.024, -0.0217]  min -0.0517 · mean -0.031 · max -0.0202
reward/air_pinch_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0463 · mean -0.000648 · max 0
reward/align: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/approach_rcv: [0.103, 0.0919, 0.154, 0.192, 0.238, 0.271, 0.269, 0.283, 0.289, 0.287]  min 0.0372 · mean 0.225 · max 0.292
reward/approach_src: [0.11, 0.155, 0.261, 0.282, 0.288, 0.282, 0.281, 0.282, 0.286, 0.289]  min 0.0699 · mean 0.26 · max 0.292
reward/arm_speed_pen: [-0.0573, -0.0276, -0.0159, -0.0111, -0.00948, -0.0101, -0.00962, -0.00751, -0.0068, -0.00728]  min -0.0573 · mean -0.0132 · max -0.00549
reward/close_rcv: [0.000202, 4.64e-08, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.35e-06 · max 0.00128
reward/close_src: [0.000157, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.01e-06 · max 0.000456
reward/cup_collision_pen: [-0.00213, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00213 · mean -1.84e-05 · max 0
reward/cup_speed_pen: [-0.00963, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00963 · mean -8.6e-05 · max 0
reward/curl_far_pen: [0, 0, -6.45e-06, 0, -1.46e-05, 0, 0, 0, 0, 0]  min -0.0221 · mean -0.000296 · max 0
reward/disturb_pen: [-0.0528, -0.00104, -0.000195, -4.48e-05, 0, -0.00011, -2.55e-05, 0, 0, 0]  min -0.0589 · mean -0.000835 · max 0
reward/dorsal_pen: [-0.0165, -1.15e-05, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0165 · mean -7.3e-05 · max 0
reward/drop_pen: [-0.00687, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00687 · mean -6.66e-05 · max 0
reward/grasp_both: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grasp_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.46e-06 · max 0.00099
reward/grasp_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.87e-06 · max 0.0033
reward/grip_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.1e-06 · max 0.000742
reward/grip_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 6.04e-07 · max 0.000409
reward/hand_foreign_pen: [-0.00378, 0, -0.00064, -0.000159, 0, -0.00359, 0, -0.00363, 0, 0]  min -0.0256 · mean -0.00132 · max 0
reward/lift_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.7e-07 · max 0.000183
reward/lift_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 8.58e-07 · max 0.000581
reward/nested_pen: [-0.0117, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0176 · mean -0.000185 · max 0
reward/open_rcv: [0.0698, 0.0741, 0.139, 0.18, 0.22, 0.251, 0.261, 0.282, 0.3, 0.305]  min 0.0426 · mean 0.221 · max 0.327
reward/open_src: [0.0592, 0.14, 0.194, 0.208, 0.23, 0.274, 0.289, 0.298, 0.309, 0.316]  min 0.0486 · mean 0.245 · max 0.333
reward/oppose_rcv: [0.00721, 7.01e-05, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 7.13e-05 · max 0.00721
reward/oppose_src: [0.00641, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 6.19e-05 · max 0.00641
reward/orient_rcv: [0.0422, 0.0592, 0.0916, 0.0829, 0.0989, 0.119, 0.121, 0.126, 0.126, 0.128]  min 0.036 · mean 0.103 · max 0.128
reward/orient_src: [0.0352, 0.0723, 0.124, 0.13, 0.131, 0.126, 0.123, 0.131, 0.133, 0.132]  min 0.0352 · mean 0.12 · max 0.135
reward/palm_push_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.000488 · mean -1.44e-06 · max 0
reward/pocket_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.7e-07 · max 0.000139
reward/pocket_src: [0.000123, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 5.43e-07 · max 0.000188
reward/pose_both: [3.92e-08, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 9.27e-10 · max 4.43e-07
reward/pose_rcv: [0.00027, 1.24e-07, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.74e-06 · max 0.00117
reward/pose_src: [0.00021, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.63e-06 · max 0.000556
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/pre_tilt_pen: [-0.0074, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0147 · mean -0.000172 · max 0
reward/rcv_upright_pen: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/rim_hook_pen: [-0.32, -0.0156, -0.0388, -0.0142, -0.00439, -0.00391, -0.0109, -0.0107, -0.00635, -0.00635]  min -0.332 · mean -0.0177 · max -0.00244
reward/spill_delta: [0.00488, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00977 · mean -1.44e-05 · max 0.00488
reward/spread_rcv: [0.244, 0.299, 0.516, 0.65, 0.819, 0.986, 1.01, 1.06, 1.1, 1.1]  min 0.118 · mean 0.821 · max 1.11
reward/spread_src: [0.208, 0.576, 1.05, 1.12, 1.13, 1.1, 1.08, 1.12, 1.14, 1.14]  min 0.0573 · mean 1.01 · max 1.16
reward/squeeze_rcv: [4.59e-05, 5.84e-09, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 6.96e-07 · max 0.000144
reward/squeeze_src: [3.38e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.01e-07 · max 0.000108
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/total: [0.299, 1.32, 2.38, 2.75, 3.09, 3.31, 3.31, 3.49, 3.62, 3.62]  min 0.183 · mean 2.89 · max 3.7
reward/touch_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 6.96e-07 · max 0.000215
reward/touch_src: [2.69e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 9.86e-07 · max 0.000558
rewards/step: [28.7, 1.02e+03, 2.13e+03, 2.4e+03, 2.61e+03, 2.93e+03, 3.06e+03, 3.15e+03, 3.17e+03, 3.22e+03]  min 26.9 · mean 2.55e+03 · max 3.25e+03
task/aim_dist: [0.308, 0.321, 0.32, 0.321, 0.321, 0.32, 0.32, 0.32, 0.32, 0.32]  min 0.308 · mean 0.321 · max 0.437
task/cup_collision_rate: [0.00293, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.03e-05 · max 0.00488
task/cups_center_dist: [0.311, 0.321, 0.32, 0.321, 0.321, 0.32, 0.32, 0.32, 0.32, 0.32]  min 0.311 · mean 0.321 · max 0.438
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0.00586, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 9.25e-05 · max 0.00879
task/rcv_cup_lift: [0.000759, 1.71e-05, 2.03e-05, 2.15e-05, 1.58e-05, 3.8e-05, 2.99e-05, 2.64e-05, 3.08e-05, 2.55e-05]  min -5.84e-05 · mean 6.51e-05 · max 0.00611
task/rcv_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.33e-06 · max 0.000977
task/rcv_hand_foreign_rate: [0.00195, 0, 0, 0, 0, 0.00195, 0, 0.00195, 0, 0]  min 0 · mean 0.000524 · max 0.00879
task/src_cup_lift: [0.00199, 1.92e-05, 2.44e-05, 3.2e-05, 1.81e-05, 9.37e-06, 2.91e-05, 4.57e-06, 5.57e-06, 6.1e-06]  min -6.7e-05 · mean 7.27e-05 · max 0.0273
task/src_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.44e-06 · max 0.000977
task/src_hand_foreign_rate: [0, 0, 0.000977, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000287 · max 0.0117
task/src_tilt_deg: [3.43, 0.15, 0.061, 0.087, 0.0538, 0.0267, 0.0847, 0.0159, 0.0166, 0.0168]  min 0.0139 · mean 0.089 · max 4.18
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Observations for this round (previous reward function iter_08, trained from scratch for 675 epochs, 1024 envs, 2.43 h). Items 1-6 are facts: training metrics, the previous reward function's own terms, the environment's own measurements and a rollout video. Item 7 records operator decisions.

1. **The hand now opens wider than the cup. This was the round's first question and it is answered.** The previous reward added a `spread` term paying only above 60 mm of thumb-to-finger gap, full at 75 mm, ungated on opposition. Medians over epoch windows (the cup is 57 mm across):

| metric | ep 0-10 | ep 150-160 | ep 300-310 | ep 450-460 | ep 665-675 |
|---|---|---|---|---|---|
| `task/src_tip_gap_mm_near` [mm] | 54.9 | 83.5 | 80.4 | 78.4 | **83.9** |
| `task/rcv_tip_gap_mm_near` [mm] | 64.0 | – | 84.9 | 90.7 | **92.7** |
| `task/src_thumb_above_rim_mm_near` [mm] | 0.0 | +3.7 | +5.5 | −10.1 | **−8.7** |
| `task/rcv_thumb_above_rim_mm_near` [mm] | +8.5 | – | +3.2 | −3.0 | **−10.8** |
| `task/{src,rcv}_thumb_over_rim_near` | 0 / 1.0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

   - The previous round ended at 48-50 mm and narrowing. One reward term reversed that inside 150 epochs, and it held for the rest of the run.
   - Both thumbs are also below the rim now, and the old rim-hook posture never returned.
   - So two preconditions that had never held together before — hand wider than the cup, thumb at wall height — now hold together on both hands.

2. **But the hands never touched their cups. Not once in the last 200 epochs, on either side.**

| metric | last 200 epochs |
|---|---|
| `contact/src_max` non-zero | **0 / 200** |
| `contact/rcv_max` non-zero | **0 / 200** |
| `reward/dorsal_pen` non-zero | 0 / 200 |
| `reward/oppose_src` / `oppose_rcv` non-zero | 1 / 200 · 1 / 200 (max 0.0000) |
| `reward/pocket_src`, `pose_src`, `close_*`, `grip_*` | 0 / 200 |
| `task/{src,rcv}_cup_in_pocket_near` | 0 / 200 |
| `task/{src,rcv}_grasped`, `task/*_cup_lift`, `task/episode_success` | 0 |

   - Question 2 of this round (does the dorsal-contact penalty fall as the palm turns toward the cup?) could not be answered at all: the penalty stayed at zero because there was no contact of any kind to penalise.
   - Question 3 (does opposition follow?) never started.

3. **Where the reward came from, and why the policy stopped.** Per-step medians over the last 10 epochs, `reward/total` 3.61:
   - `spread_src` 1.136 + `spread_rcv` 1.091 = 2.227
   - `open_src` 0.326 + `open_rcv` 0.313 = 0.639
   - `approach_src` 0.286 + `approach_rcv` 0.283 = 0.569
   - Together **3.43 of 3.61 — 95 %** — and every one of those terms is payable **without the hand ever touching the cup**.
   - Everything downstream (oppose, pocket, pose, close, touch, grip, grasp, lift, align, tilt, pour, success) is exactly 0.
   - `task/src_closure` fell 0.17 → 0.10 and `task/rcv_closure` 0.17 → 0.10: the hands ended the run held almost flat open.
   - `task/src_palm_to_cup` 0.234 → 0.119 m and `task/rcv_palm_to_cup` 0.252 → 0.119 m, pinned at about 12 cm from epoch 150 onward.
   - The near-rate moved the wrong way as the spread income grew: `task/src_near_rate` peaked at 0.22 around epoch 300 and ended at 0.018; `task/rcv_near_rate` peaked at 0.15 and ended at 0.063. Standing back and holding the hand open pays more than closing the last 12 cm.

4. **This is the same failure shape as the previous round, one rung further up the ladder.** In round 9 the policy settled on the approach term (88 % of income, palm pinned at 12 cm). In round 10 it settled on the opening terms (95 % of income, palm pinned at 12 cm). Each time, the rung that was added to unblock the ladder became a place to stand, because it pays a standing posture rather than a transition. The rungs themselves worked: approach was solved in round 6, opening in round 10. What is missing is that reaching a rung must stop paying once the next one is available.

5. **Rollout video** at the epoch-650 checkpoint (4 envs, 900 steps, `our_source/pour_t2r_rh_i08_ep650_0916.mp4`, frame sheet and 2.4x crops in `pour_t2r_rh_i08_ep650_0916_frames/`). The question was whether the hand holds still with the fingers spread or approaches and retreats. It holds still:
   - At step 60 the hand is already beside the cup with all five fingers spread wide and flat, laid out over the mat next to the cup rather than around it. The thumb is spread away from the cup as well.
   - At step 700 the posture is unchanged. Nothing approaches, nothing closes, and no part of the hand reaches the cup wall at any point in the episode.
   - **Operator observation (takes precedence over the frame reading above): what the arm brings close to the cup is the WRIST — the j7 motor end of the arm. The palm side never comes close at all.**
   - That observation and the logged numbers agree, and together they show how the plateau is built. `spread` is already gated on the signed pad direction (`spread = near_pre · pad · gap_ok · jaw_ok`), and it was paying 1.136 of its 1.5 weight, i.e. a raw 0.76 — so the pad was *pointing* at the cup. But pointing and approaching are separate conditions in the reward: `approach` is computed from palm distance alone and is worth at most 0.3 per hand, while `spread` is worth 1.5. The cheapest way to collect is therefore to park the arm with the wrist nearest the cup, aim the pad at it from 12 cm away, and spread the fingers. Everything that would require the palm itself to arrive — pocket, pose, close, touch, grip — needs the cup between the fingertips, and stayed at 0.
   - The fingers are spread in the plane of the mat, beside the cup, not on either side of it. So the wide `tip_gap` in item 1 is real but it is not a pocket around the cup — the opening happens in the wrong place, which is exactly what `cup_in_pocket` at 0 / 200 says.

6. **Physics was the cleanest of any round.** `ctrl/mimic_err_max` 1.52 at start → **0.025 rad** at the end. No mimic runaway, no drop, no cup collision, no foreign contact.

7. **Operator decisions** (after reviewing items 1-6 and the video):
   - This draft is approved as the feedback for this round, and a new reward is to be generated. Training starts from scratch.
   - **How to fix item 3 and item 4 is left to the reward designer.** The operator is not prescribing a mechanism this time. What is required is that the facts above be taken as binding: a hand standing 12 cm away with the fingers spread flat beside the cup currently collects 95 % of the available reward, and that has now happened twice in a row with a different term each time (approach in round 9, opening in round 10).
   - **Keep the three gates that are working**, all introduced in the last two rounds and none of them implicated in this failure: the opposition gate with no floors, the signed pad-facing condition, and the rule that force on the back of the fingers is not progress. Round 10 produced no ungraspable posture that was paid, and no dorsal contact.
   - The goal is unchanged: the thumb and at least one opposing finger both above 1 N at the same time (the environment's grasp flag), then lifting, then the rest of the task.
   - `task/{src,rcv}_cup_in_pocket_near` is the metric that separates a useful opening from this one. It was 0 for the entire round while `tip_gap` sat at 84-93 mm.

The policy must complete the full task (grasp both → lift → bring together without contact → tilt → beads in receiver, receiver upright, no drop, little spill).
