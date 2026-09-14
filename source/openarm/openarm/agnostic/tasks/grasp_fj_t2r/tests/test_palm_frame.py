"""손바닥 중심(palm_ee) 오프셋 — 자산 URDF 대조 (Isaac 불요).

★09.14 사용자 "손바닥의 중심쪽은 palm_ee xform": ctx.palm_pos 는 손바닥 링크 원점(손목 쪽)이 아니라 palm_ee 여야 한다.

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_fj_t2r/tests -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openarm.agnostic.tasks.grasp_fj_t2r.palm_frame import palm_center_offset

_HDGP = Path(__file__).resolve().parents[7]
_URDF = _HDGP / "assets" / "robot" / "openarm_dg5f-m-short-tl_bi_rl" / "openarm_dg5f-m-short-tl_bi_rl.urdf"


def test_short_tl_palm_center_is_28mm_along_the_normal_and_40mm_toward_the_fingers():
    assert palm_center_offset(_URDF, "r_hl_palm") == pytest.approx((0.028, 0.0, 0.040))


def _urdf(tmp_path, rpy="0 0 0", jtype="fixed", child="r_hl_palm_ee"):
    p = tmp_path / "r.urdf"
    p.write_text(f"""<robot name="r"><link name="r_hl_palm"/><link name="{child}"/>
<joint name="r_hj_palm_ee" type="{jtype}"><parent link="r_hl_palm"/><child link="{child}"/>
<origin rpy="{rpy}" xyz="0.01 0.02 0.03"/></joint></robot>""", encoding="utf-8")
    return p


def test_rotated_or_missing_palm_ee_joint_refuses_to_boot(tmp_path):
    assert palm_center_offset(_urdf(tmp_path), "r_hl_palm") == pytest.approx((0.01, 0.02, 0.03))
    with pytest.raises(ValueError, match="rpy"):
        palm_center_offset(_urdf(tmp_path, rpy="0 0.1 0"), "r_hl_palm")
    with pytest.raises(ValueError, match="fixed"):
        palm_center_offset(_urdf(tmp_path, jtype="revolute"), "r_hl_palm")
    with pytest.raises(ValueError, match="없다"):
        palm_center_offset(_urdf(tmp_path, child="r_hl_other"), "r_hl_palm")
