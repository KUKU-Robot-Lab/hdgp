"""RewardContext — 생성된 보상 함수가 읽는 **유일한** 입력 (grasp_fj_t2r: 단일 팔 · DG-5F full-joint).

★설계 계약(붓기 트랙 `modules/t2r/context.py` 와 같은 규약, 코드는 공유하지 않는다)
  · 모든 텐서는 배치 (N, …) 이고 같은 device 다. N 은 병렬 환경 수.
  · 위치는 전부 **env-local**(각 환경 원점 기준, m). 각도 rad, 힘 N.
  · 이 클래스의 필드 이름·주석이 곧 LLM 프롬프트의 환경 설명이다(`prompts.py` 가 소스를
    그대로 렌더링한다). 필드를 바꾸면 프롬프트도 같이 바뀐다 — 사본이 없다.
  · 주석은 영어다: 생성기(나중에 openai 백엔드 포함)가 그대로 읽는다.
  · frozen — 보상 함수는 읽기만 한다.
★주석의 수치·규약은 09.14 코드 대조로 확인했다(프로필 `tesollo_right_short_tl`·뱅크 `shaker_sweep`·
  `fj_core_env._get_dones`·`keypoint_goal.sample_first_goal`). 추정을 적지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass(frozen=True)
class RewardContext:
    # ---- constants (python numbers, fixed for the whole run) ---------------------------
    table_z: float                 # height of the table top [m]
    lift_latch_height: float       # the env sets `lifted` once the cup has risen this far above its starting height [m]
    success_hold_steps: int        # consecutive steps with goal_dist <= success_tol needed to count one success; a success also requires, on that step, that the fingers are wrapped around the cup (the env measures how far the finger links enclose the cup body and requires it above a threshold that starts low and rises as the policy succeeds more often)
    max_successes: int             # the episode ends after this many successes

    # ---- hand: right Tesollo DG-5F, finger index 0 thumb, 1 index, 2 middle, 3 ring, 4 pinky ----
    palm_pos: torch.Tensor         # (N,3) centre of the palm (a virtual point on the palm, not a collision surface)
    palm_normal: torch.Tensor      # (N,3) unit vector pointing out of the palm surface, towards an object held in the hand
    palm_side: torch.Tensor        # (N,3) unit vector lying in the palm plane (palm frame y axis)
    palm_finger_dir: torch.Tensor  # (N,3) unit vector lying in the palm plane, pointing from the palm towards the fingers (palm frame z axis); palm_normal, palm_side, palm_finger_dir form a right-handed frame. In the start pose palm_normal points along +y and palm_finger_dir along +x
    link_pos: torch.Tensor         # (N,5,3,3) finger link positions: [:, f, 0] link moved by joint _3, [:, f, 1] link moved by joint _4, [:, f, 2] fingertip
    link_cup_force: torch.Tensor   # (N,5,3) contact force magnitude between each of those links and the cup only [N] (0 = not touching the cup)
    palm_cup_force: torch.Tensor   # (N,) contact force magnitude between the palm and the cup only [N]
    hand_q: torch.Tensor           # (N,19) finger joint angles [rad], order = the hand joint table in the robot description
    hand_q_norm: torch.Tensor      # (N,19) joint angles normalised to each joint's commandable range: 0 = lower limit (straight), 1 = upper limit (most flexed)
    hand_target_norm: torch.Tensor  # (N,19) commanded finger joint targets (after filtering), same normalisation as hand_q_norm
    hand_default_q_norm: torch.Tensor  # (N,19) the hand's default pose: the finger joint angles every episode starts with, same normalisation as hand_q_norm (the same in every environment); in this pose the four fingers are straight, and the thumb, rotated into opposition, is straight and points along palm_normal, reaching about 0.12 m out from the palm surface at the wrist end of the palm (behind palm_pos along palm_finger_dir)
    hand_qd: torch.Tensor          # (N,19) finger joint velocities [rad/s]
    hand_z_min: torch.Tensor       # (N,) height of the lowest finger/thumb link [m]; the palm is not included in this one
    palm_clearance: torch.Tensor   # (N,) height of the lowest point of the palm itself above the table top [m]: 0 means the palm is resting on the table, negative means it is pressed into it. The palm is a large flat body and palm_pos is a virtual point up to 0.065 m away from its surface, so palm_pos alone cannot tell you whether the palm is on the table; this is the measured distance. Note that the episode-ending floor check looks only at the finger and thumb links, so the palm resting on the table does not end the episode by itself

    # ---- arm: 7-DOF ------------------------------------------------------------------------
    arm_q: torch.Tensor            # (N,7) arm joint angles [rad]
    arm_qd: torch.Tensor           # (N,7) arm joint velocities [rad/s]

    # ---- cup: a roughly cylindrical cup standing upright on the table at the start ----------
    cup_pos: torch.Tensor          # (N,3) cup reference point on its axis; the graspable band (cup_half_height above and below) is centred on it
    cup_quat: torch.Tensor         # (N,4) cup orientation quaternion (w,x,y,z)
    cup_axis: torch.Tensor         # (N,3) unit vector along the cylinder axis towards its top; (0,0,1) when upright
    cup_tilt: torch.Tensor         # (N,) angle between cup_axis and world +z [rad]
    cup_lin_vel: torch.Tensor      # (N,3) cup linear velocity [m/s]
    cup_ang_vel: torch.Tensor      # (N,3) cup angular velocity [rad/s]
    cup_spawn_pos: torch.Tensor    # (N,3) cup_pos at the start of the episode, resting on the table
    cup_radius: torch.Tensor       # (N,) radius of the cup's outer surface in the graspable band [m]; differs between environments
    cup_half_height: torch.Tensor  # (N,) half height of the graspable band, centred on cup_pos along cup_axis [m]

    # ---- goal and task status: computed by the environment --------------------------------
    goal_pos: torch.Tensor         # (N,3) position where cup_pos must be held (with the cup upright)
    goal_dist: torch.Tensor        # (N,) largest distance between keypoints fixed on the cup and the same keypoints at the goal pose [m]
    success_tol: torch.Tensor      # (N,) current success tolerance on goal_dist [m]; shrinks as training succeeds
    lifted: torch.Tensor           # (N,) bool, True once the cup has risen above lift_latch_height in this episode (stays True)
    success: torch.Tensor          # (N,) bool, True on the step a success is counted
    num_successes: torch.Tensor    # (N,) number of successes counted so far in this episode
    episode_progress: torch.Tensor  # (N,) elapsed fraction [0,1] of the step budget; the budget restarts after every success

    # ---- stage completion flags: kept by the environment, stay True until the episode ends ----
    approach_done: torch.Tensor    # (N,) bool, set on the first step on which all of these held at once, i.e. the cup sat between the extended thumb and the four fingers in front of the palm: the palm plane was within 2 cm of the cup's side with the cup in front of the palm (the distance from palm_pos to the cup axis along palm_normal, minus cup_radius, was between -0.01 m and 0.02 m); the cup axis was ahead of palm_pos along palm_finger_dir by between cup_radius - 0.005 m and cup_radius + 0.02 m; palm_pos was within the height of the graspable band (at most cup_half_height from cup_pos along cup_axis); the hand kept its start-pose orientation (palm_normal within about 45 degrees of +y and palm_finger_dir within about 45 degrees of +x); every movable finger joint was within 0.3 of hand_default_q_norm (joints with a range of 0.05 rad or less are ignored); and no finger or thumb link touched the cup (contact force above 0.1 N). Episodes that start beside the cup (see the scene description) have approach_done already set on the first step
    envelope_done: torch.Tensor    # (N,) bool, set once, after approach_done, the palm, the thumb and at least 4 digits in total (the thumb included) touched the cup for 5 consecutive steps; the palm or a finger counts as touching when its contact force with the cup is above 0.1 N

    # ---- actions ---------------------------------------------------------------------------
    actions: torch.Tensor          # (N,26) policy action of this step, clipped to [-1,1]; each step the joints receive one of the last 3 policy actions picked at random (a 0-2 step delay)
    prev_actions: torch.Tensor     # (N,26) policy action of the previous step (zeros right after a reset)

    # ------------------------------------------------------------------
    @property
    def num_envs(self) -> int:
        return int(self.palm_pos.shape[0])

    @property
    def device(self) -> torch.device:
        return self.palm_pos.device


TENSOR_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(RewardContext) if f.type == "torch.Tensor")
SCALAR_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(RewardContext) if f.type in ("float", "int"))


#: ★09.20 좌팔판 — 스텁 주석 중 손 방향에 묶인 문장만 좌손용으로 바꾼다(env 가 같은 규약으로 ctx 를 만든다:
#:   법선 −y · 손 정규화는 모든 관절 "0 = 곧음 · 1 = 가장 굽힘", `grasp_fj_t2r_env._hand_norm`). 원문이 바뀌면 치환이 조용히
#:   빠지지 않게 `context_stub_source` 가 각 원문이 정확히 한 번 있는지 확인한다.
LEFT_STUB_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("# ---- hand: right Tesollo DG-5F", "# ---- hand: left Tesollo DG-5F"),
    ("In the start pose palm_normal points along +y and palm_finger_dir along +x",
     "In the start pose palm_normal points along -y and palm_finger_dir along +x"),
    ("0 = lower limit (straight), 1 = upper limit (most flexed)",
     "0 = straight, 1 = most flexed, for every joint (on this left hand thumb_3 and thumb_4 flex towards negative "
     "angles, so for them 0 is the upper limit and 1 the lower limit)"),
    ("palm_normal within about 45 degrees of +y", "palm_normal within about 45 degrees of -y"),
)


def context_stub_source(side: str = "r") -> str:
    """`class RewardContext` 본문(필드+주석) — 프롬프트 재료. 헬퍼 속성 구간은 뺀다. side "l" = 좌손 문구."""
    import inspect
    src = inspect.getsource(RewardContext)
    cut = src.find("    # ----------")
    src = src[:cut].rstrip() + "\n"
    if side == "l":
        for old, new in LEFT_STUB_REPLACEMENTS:
            if src.count(old) != 1:
                raise ValueError(f"좌손 스텁 치환 원문이 {src.count(old)}번 있다(1번이어야): {old!r}")
            src = src.replace(old, new)
    elif side != "r":
        raise ValueError(f"side 는 'r' | 'l' — got {side!r}")
    return src
