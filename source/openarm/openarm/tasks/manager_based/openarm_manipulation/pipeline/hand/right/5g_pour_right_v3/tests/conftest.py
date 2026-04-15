"""pytest conftest: isaaclab 없이 독립 실행 가능하도록 sys.path 격리."""
import sys
from pathlib import Path

# 테스트 파일들은 cfg/env .py를 pathlib.Path.read_text()로 읽으므로
# 패키지 import가 불필요하다. 부모 __init__.py(isaaclab 의존)가
# 수집 경로에 들어오지 않도록 task 디렉토리만 sys.path에 추가한다.
_TASK_DIR = Path(__file__).parent.parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))
