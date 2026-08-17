# Copyright 2025 Enactic, Inc. (Apache-2.0)
"""[Ablation #1 — joint-space action 대조군] 별도 cfg (base 미변경).

Path B novelty #1(task-space policy + Fabric/IK)을 뒷받침: 우팔을 palm-pose/Fabric 대신
action[:7](palm6+null1 슬롯)을 **관절 delta로 직접 구동**(Fabric 우회)하는 대조군.
task-space+Fabric(=D_full/M4) 대비 raw joint-space RL이 얼마나 나쁜지로 #1을 정량화한다.

구현: base env에 `right_arm_jointspace` flag + _pre_physics_step 분기(fabric_q[:7] 덮어쓰기).
실행 방식(둘 다 가능):
  - hydra override:  ./train.sh open-tesol_b_pour_sensor-lstm JS_s42 ... env.right_arm_jointspace=true
  - 이 cfg 클래스:    아래 PourRightEnvCfg_JointSpace (gym 등록은 선택)

⚠️ demo nullspace prior는 Fabric-bound라 jointspace에선 무효 → 이 ablation은
   "task-space+Fabric+demo 패키지 vs raw joint-space"를 비교(action-space 구조 기여).
   반드시 검증 probe 통과 후 학습(GPU 전 확인).
"""

from isaaclab.utils import configclass

from .pour_right_env_cfg import PourRightEnvCfg


@configclass
class PourRightEnvCfg_JointSpace(PourRightEnvCfg):
    """joint-space action 대조군 (#1). 우팔 Fabric 우회 + 관절 delta 직접 구동."""

    right_arm_jointspace: bool = True
    jointspace_action_scale: float = 0.03   # 관절 delta [rad/step] — probe로 튜닝
