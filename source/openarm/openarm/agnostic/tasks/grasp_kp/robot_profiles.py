"""grasp_kp 로봇 프로필 — `grasp_s2r.robot_profiles` 의 **얇은 재수출**.

왜 사본이 아닌가: 두 트랙은 같은 로봇·같은 벤더 게인·같은 palm 박스를 쓴다. 사본을
두면 grasp_s2r 쪽 캘리브 갱신이 여기에 조용히 안 실린다(단일 출처 유지).
`modules/tests/test_vendor_gains.py` 는 이 모듈의 `vars()` 에서 `actuator_specs` 를
가진 객체를 훑으므로 프로필 객체가 이 네임스페이스에 있어야 한다 — `import *` 가
공개 이름(TESOLLO_RIGHT·GRIPPER_LEFT·PROFILES·RobotProfile)을 전부 가져온다.
isaaclab 을 import 하지 않는다(시스템 python3 pytest 로 돈다).
"""

from ..grasp_s2r.robot_profiles import *  # noqa: F401,F403
from ..grasp_s2r.robot_profiles import PROFILES, RobotProfile  # noqa: F401  (명시 재수출)
