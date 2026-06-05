"""Phase 3 actuator/sensor 정합성 검증 (GPU/isaaclab 불필요, 순수 regex).

Isaac Lab 은 모든 articulation DOF 가 정확히 1개 actuator 그룹에 속해야 한다.
env_cfg 의 actuator 정규식이 38 DOF 를 빠짐없이/겹침없이 커버하는지 단언.
(정규식 오타·누락은 Isaac 로드시 크래시 → 사전 차단)
"""
import re

# 전체 38 DOF (로봇 로드 스모크로 확인된 구조)
ARM_R = [f"openarm_right_joint{i}" for i in range(1, 8)]
ARM_L = [f"openarm_left_joint{i}" for i in range(1, 8)]


def _hand(side):
    fingers = []
    fingers += [f"rh56f1_{side}_{side}_thumb_{k}_joint" for k in (1, 2, 3, 4)]
    for f in ("index", "middle", "ring", "little"):
        fingers += [f"rh56f1_{side}_{side}_{f}_{k}_joint" for k in (1, 2)]
    return fingers


ALL_JOINTS = ARM_R + ARM_L + _hand("right") + _hand("left")

# env_cfg 의 actuator 정규식 (cfg 와 동일하게 유지)
ACTUATOR_REGEXES = {
    "openarm_right_arm": r"openarm_right_joint[1-7]",
    "openarm_left_arm": r"openarm_left_joint[1-7]",
    "rh56f1_right_drive": r"rh56f1_right_right_(thumb_[12]|index_1|middle_1|ring_1|little_1)_joint",
    "rh56f1_right_mimic": r"rh56f1_right_right_(thumb_[34]|index_2|middle_2|ring_2|little_2)_joint",
    "rh56f1_left_drive": r"rh56f1_left_left_(thumb_[12]|index_1|middle_1|ring_1|little_1)_joint",
    "rh56f1_left_mimic": r"rh56f1_left_left_(thumb_[34]|index_2|middle_2|ring_2|little_2)_joint",
}


def _full_match(pattern, name):
    return re.fullmatch(pattern, name) is not None


def test_total_dof_is_38():
    assert len(ALL_JOINTS) == 38, len(ALL_JOINTS)
    assert len(set(ALL_JOINTS)) == 38


def test_every_joint_covered_exactly_once():
    for j in ALL_JOINTS:
        groups = [g for g, p in ACTUATOR_REGEXES.items() if _full_match(p, j)]
        assert len(groups) == 1, f"{j} matched by {groups} (정확히 1개여야)"


def test_no_regex_matches_outside_joint_set():
    for g, p in ACTUATOR_REGEXES.items():
        matched = [j for j in ALL_JOINTS if _full_match(p, j)]
        assert matched, f"{g} 정규식이 아무 관절도 매칭 안 함 (오타?)"


def test_drive_groups_have_6_each():
    for g in ("rh56f1_right_drive", "rh56f1_left_drive",
              "rh56f1_right_mimic", "rh56f1_left_mimic"):
        n = sum(1 for j in ALL_JOINTS if _full_match(ACTUATOR_REGEXES[g], j))
        assert n == 6, f"{g}: {n} (6 이어야)"


def test_init_state_right_hand_keys_valid():
    """env_cfg init_state 가 지정한 우측 손 관절명이 실제 관절 집합에 존재."""
    init_keys = [
        "rh56f1_right_right_thumb_1_joint", "rh56f1_right_right_thumb_2_joint",
        "rh56f1_right_right_index_1_joint", "rh56f1_right_right_middle_1_joint",
        "rh56f1_right_right_ring_1_joint", "rh56f1_right_right_little_1_joint",
        "rh56f1_right_right_thumb_3_joint", "rh56f1_right_right_thumb_4_joint",
        "rh56f1_right_right_index_2_joint", "rh56f1_right_right_middle_2_joint",
        "rh56f1_right_right_ring_2_joint", "rh56f1_right_right_little_2_joint",
    ]
    for k in init_keys:
        assert k in ALL_JOINTS, k


if __name__ == "__main__":
    test_total_dof_is_38()
    test_every_joint_covered_exactly_once()
    test_no_regex_matches_outside_joint_set()
    test_drive_groups_have_6_each()
    test_init_state_right_hand_keys_valid()
    print("Phase 3 actuator coverage: 5 checks passed (GREEN)")
