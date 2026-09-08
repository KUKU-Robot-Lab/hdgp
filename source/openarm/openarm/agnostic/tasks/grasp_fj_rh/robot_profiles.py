"""grasp_fj_rh 로봇 프로필 — `grasp_s2r.robot_profiles` 의 **얇은 재수출**.

왜 사본이 아닌가: 부모 cfg 체인(`GraspS2REnvCfg.finalize_after_overrides`)이
`grasp_s2r.robot_profiles.PROFILES` 를 직접 읽는다. 여기에 사본을 두면 이 트랙에서만
보이는 프로필이 되어 부모가 `KeyError` 를 낸다 — 로봇 레지스트리는 한 곳뿐이다
(그 파일 상단의 설계 목표: "새 로봇 추가 = 이 파일에 프로필 1개 추가").
`modules/tests/test_vendor_gains.py` 가 이 네임스페이스의 `actuator_specs` 보유 객체를
훑으므로 프로필 객체가 여기 보여야 한다. isaaclab 을 import 하지 않는다.
"""

from ..grasp_s2r.robot_profiles import *  # noqa: F401,F403
from ..grasp_s2r.robot_profiles import PROFILES, RobotProfile  # noqa: F401  (명시 재수출)
