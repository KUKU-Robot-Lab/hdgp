# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""증류 카메라 공용 유틸 (태스크·좌우 무관)."""

from __future__ import annotations

import math

import warp as wp


def make_cam_matrix(width: int, height: int, focal: float, aperture: float):
    """depth normal_noise 커널이 픽셀→광선 역투영에 쓰는 정규화 투영행렬.

    DEXTRAH 규약: a = focal_px/(W/2) = 1/tan(hfov/2), b = focal_px/(H/2).
    주점(cx, cy) 오프셋은 원본과 동일하게 넣지 않는다 — 표면 법선 추정용이라
    전 픽셀에 동일한 shear 가 걸릴 뿐 노이즈 특성에 영향이 없다.
    """
    fov = 2.0 * math.atan(aperture / (2.0 * focal))
    focal_px = width * 0.5 / math.tan(fov / 2.0)

    mat = wp.mat44f()
    mat[0, 0] = focal_px / (width * 0.5)
    mat[1, 1] = focal_px / (height * 0.5)
    mat[2, 3] = -1.0
    mat[3, 2] = 1.0e-3
    return mat


def depth_randomization_cfg(cam_matrix, d_min: float, d_max: float) -> dict:
    """D435i 스테레오 IR 깊이 노이즈 모사 파라미터 (DEXTRAH 원본 값)."""
    return {
        "pixel_dropout_and_randu": {
            "p_dropout": 0.0125 / 4,
            "p_randu": 0.0125 / 4,
            "d_min": d_min,
            "d_max": d_max,
        },
        "sticks": {
            "p_stick": 0.001 / 4,
            "max_stick_len": 18.0,
            "max_stick_width": 3.0,
            "d_min": d_min,
            "d_max": d_max,
        },
        "correlated_noise": {
            "sigma_s": 1.0 / 2,
            "sigma_d": 1.0 / 6,
            "d_min": d_min,
            "d_max": d_max,
        },
        "normal_noise": {
            "sigma_theta": 0.01,
            "cam_matrix": cam_matrix,
            "d_min": d_min,
            "d_max": d_max,
        },
    }
