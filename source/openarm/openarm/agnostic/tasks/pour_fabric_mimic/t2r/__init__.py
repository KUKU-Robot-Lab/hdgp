"""t2r — text2reward 방식 보상 자동생성의 **환경 쪽** 계약.

Isaac 을 import 하지 않는다(순수 torch). 세 조각:
  context.py  RewardContext — env 가 매 스텝 채우는 관측 묶음. 프롬프트가 이 클래스를
              그대로 설명하므로 **필드 이름 = LLM 이 쓰는 이름**(문자열 치환 단계 없음).
  loader.py   생성된 reward 파일을 읽어 `compute_reward(ctx) -> (reward, terms)` 를 돌려준다.
  validator.py 정적 검사 + 드라이런(Phase 2).
"""
