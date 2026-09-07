#!/usr/bin/env python3
"""좌팔 49D 관측을 통합 씬 상태로 조립 — **배포 빌더를 그대로 쓴다**.

sim2real `left_obs_builder.assemble_actor_obs` 는 env 표본 대조(49/49 세그먼트)를
통과한 배포 코드다. 여기서 그걸 쓰는 것 자체가 배포 관측 경로의 리허설이다.
컵 pose 는 씬 참값 — 실기에서 FD++ 가 주는 그 자리다.
"""
from __future__ import annotations

import numpy as np

from left_obs_builder import assemble_actor_obs, quat_to_matrix
from robot_profile import load_hdgp_module, load_robot_profile

from bimanual_chain import LEFT9

_CACHE: dict = {}


def _preset():
    if "p" not in _CACHE:
        _CACHE["p"] = load_hdgp_module(load_robot_profile("gripper_left"), "preset")
    return _CACHE["p"]


def left_actor_obs(env, chain, *, goal7, last_action, q_default) -> np.ndarray:
    """씬 상태 + goal → v2B25 actor obs (49,). 프레임 규약은 빌더 독스트링 참조."""
    P = _preset()
    robot = env.robot
    e = 0
    if "il" not in _CACHE:
        _CACHE["il"] = [robot.joint_names.index(n) for n in LEFT9]
        _CACHE["gi"] = robot.body_names.index(P.GRIPPER_BASE_BODY)
    il, gi = _CACHE["il"], _CACHE["gi"]
    org = env.scene.env_origins[e]
    q = robot.data.joint_pos[e, il].cpu().numpy()
    qd = robot.data.joint_vel[e, il].cpu().numpy()
    gp = (robot.data.body_pos_w[e, gi] - org).cpu().numpy()
    gq = robot.data.body_quat_w[e, gi].cpu().numpy()
    tcp = gp + quat_to_matrix(gq) @ np.array([0.0, 0.0, P.TCP_OFFSET_IN_BASE_Z])
    cup = env.left_target_cup
    cpos = (cup.data.root_pos_w[e] - org).cpu().numpy()
    cq = cup.data.root_quat_w[e].cpu().numpy()
    root_p = (robot.data.root_pos_w[e] - org).cpu().numpy()
    root_q = robot.data.root_quat_w[e].cpu().numpy()
    goal7 = np.asarray(goal7, dtype=float).reshape(7)
    return assemble_actor_obs(
        joint_pos=q, joint_vel=qd,
        joint_pos_default=np.asarray(q_default, dtype=float),
        joint_vel_default=np.zeros(9),
        root_pos=root_p, root_quat=root_q,
        cup_pos=cpos, cup_quat=cq,
        goal_pos=goal7[:3], goal_quat=goal7[3:],
        tcp_pos=tcp, gripper_base_pos=gp, gripper_base_quat=gq,
        last_action=np.asarray(last_action, dtype=float),
        gripper_gate=float(chain.gate_open[e].item()),
        palm_box=(P.PALM_BOX_X, P.PALM_BOX_Y, P.PALM_BOX_Z),
    ).astype(np.float32)
