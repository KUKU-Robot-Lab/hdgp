"""agnostic 트랙 공용 모듈.

태스크(`agnostic/tasks/*`)가 공유하는 재사용 부품만 둔다. 어떤 모듈도 특정 태스크를
import 하지 않는다(의존은 tasks → modules 단방향).

★`agnostic/tasks/grasp_lift` 는 이 패키지를 쓰지 않는다(학습 중이라 손대지 않는다).
"""
