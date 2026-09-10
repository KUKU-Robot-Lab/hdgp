"""grasp_s2r 로봇 프로필 — `agnostic/modules/robot_profiles` 의 **얇은 재수출**.

★★09.10 이동. 이 파일은 원래 실체(890줄)였고 5개 트랙 20곳이 여기를 읽었다 —
자리만 `grasp_s2r/` 밑이었을 뿐 사실상 공용 모듈이었다. 그 탓에 `grasp_fj` 가
"s2r 를 import 하는" 형태가 되어, 사용자 확정 "s2r 하고 fj 는 공유 금지"와
충돌했다. 실체를 중립 위치(`modules/`)로 옮기고 여기는 재수출만 남긴다.

**복제가 아니라 이동이다.** 로봇 레지스트리는 여전히 한 곳뿐이고, 벤더 게인·
palm 박스·리셋 자세도 한 벌이다("모든 값은 벤더 기준, 단일 출처").

`modules/tests/test_vendor_gains.py` 는 각 트랙 모듈의 `vars()` 에서
`actuator_specs` 보유 객체를 훑으므로 프로필 객체가 이 네임스페이스에 보여야
한다 — `import *` 가 공개 이름을 전부 가져온다.
isaaclab 을 import 하지 않는다(시스템 python3 pytest 로 돈다).
"""

from ...modules.robot_profiles import *  # noqa: F401,F403
from ...modules.robot_profiles import PROFILES, RobotProfile  # noqa: F401  (명시 재수출)
