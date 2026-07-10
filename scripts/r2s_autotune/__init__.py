"""Real2Sim autotune: actuator system identification for OpenArm assets.

RL 학습 이전 단계. real 로봇과 Isaac Lab 로봇의 관절 응답 차이를 줄인다.
후보 파라미터는 USD 복사본이 아니라 calibration JSON + 런타임 gain override로 표현한다.
"""
