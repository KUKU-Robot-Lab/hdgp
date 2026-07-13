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

"""증류용 시각 도메인 랜덤화 (DEXTRAH env 의 distillation 분기 이식).

student 는 RGB 를 입력으로 본다(a2c_* 전부 use_depth=False). 따라서 외형이
고정되면 단 하나의 장면에만 맞는 정책이 나온다. 매 reset 마다 조명·테이블·물체·
로봇 재질을 흔들어 실기 외형 분포를 덮는다.

DEXTRAH 원본과의 차이: shader prim 경로를 하드코딩하지 않는다.
원본은 Kuka-Allegro 전용 이름(Looks/arm_gray, Looks/allegro_black …)을 박아뒀는데
OpenArm+Tesollo 는 Looks 이름이 다르다. 여기선 stage 를 걸어 UsdShade.Shader 를
찾는다 — 로봇이 바뀌어도 동작한다.

텍스처 에셋은 DEXTRAH textures.zip (HuggingFace nvidia/dextrah_textures, 8.4GB).
git 에 넣지 않는다 — assets/dextrah_textures/ 는 .gitignore 대상이고 server 에는
따로 내려받아야 한다.
"""

from __future__ import annotations

import glob
import os
import random

import numpy as np

# 조명 랜덤화 확률 (DEXTRAH 원본: reset 당 0.3)
DOME_LIGHT_RAND_PROB = 0.3
DOME_LIGHT_INTENSITY_RANGE = (1000.0, 4000.0)

# 물체 재질 랜덤화 범위 (DEXTRAH 원본)
OBJECT_TEXTURE_SCALE_RANGE = (0.7, 5.0)

# 로봇 재질 랜덤화 범위 (DEXTRAH 원본)
ARM_ROUGHNESS_RANGE = (0.2, 1.0)
ARM_METALLIC_RANGE = (0.0, 0.8)

# 테이블 재질 랜덤화 범위 (DEXTRAH 원본)
TABLE_TINT_R_RANGE = (0.3, 0.6)
TABLE_TINT_G_RANGE = (0.2, 0.4)
TABLE_TINT_B_RANGE = (0.1, 0.2)
TABLE_ROUGHNESS_RANGE = (0.3, 0.9)


class TextureBank:
    """텍스처 파일 목록. 디렉토리가 비어 있으면 즉시 실패한다.

    조용히 빈 리스트로 넘어가면 랜덤화가 통째로 no-op 이 되는데, 학습은 멀쩡히
    돌아 보이고 sim2real 에서만 무너진다 — 그래서 여기서 터뜨린다.
    """

    def __init__(self, texture_root: str, require: bool = True):
        self.table = sorted(glob.glob(
            os.path.join(texture_root, "curated_table_textures", "*.png")
        ))
        self.dome = sorted(glob.glob(
            os.path.join(texture_root, "dome_light_textures", "*.exr")
        ))
        self.object = sorted(glob.glob(
            os.path.join(texture_root, "object_textures", "**", "*.png"),
            recursive=True,
        ))

        if not require:
            return

        missing = [
            name for name, files in
            (("curated_table_textures/*.png", self.table),
             ("dome_light_textures/*.exr", self.dome),
             ("object_textures/**/*.png", self.object))
            if not files
        ]
        if missing:
            raise FileNotFoundError(
                f"시각 도메인 랜덤화용 텍스처가 없다 (root={texture_root}): "
                f"{', '.join(missing)}\n"
                "DEXTRAH textures.zip 을 받아 풀 것:\n"
                "  huggingface.co/datasets/nvidia/dextrah_textures → textures.zip\n"
                f"  unzip textures.zip -d {texture_root}"
            )


def find_shaders(stage, root_path: str) -> list:
    """root_path 하위의 UsdShade.Shader prim 을 전부 찾는다 (로봇 비종속)."""
    from pxr import UsdShade

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return []
    return [prim for prim in _walk(root) if UsdShade.Shader(prim)]


def _walk(prim):
    yield prim
    for child in prim.GetChildren():
        yield from _walk(child)


class VisualDomainRandomizer:
    """env 별 shader prim 을 캐시해두고 reset 마다 재질·조명을 흔든다."""

    def __init__(
        self,
        num_envs: int,
        texture_root: str,
        randomize_dome_light: bool = True,
        randomize_robot: bool = True,
    ):
        import omni.usd

        self.num_envs = num_envs
        self.textures = TextureBank(texture_root)
        self.randomize_dome_light = randomize_dome_light
        self.randomize_robot = randomize_robot
        self.stage = omni.usd.get_context().get_stage()

        # env 별 shader prim 캐시 (매 reset 마다 stage 를 걷지 않기 위해)
        self.object_shaders: list[list] = []
        self.table_shaders: list[list] = []
        self.robot_shaders: list[list] = []
        for i in range(num_envs):
            env_root = f"/World/envs/env_{i}"
            self.object_shaders.append(find_shaders(self.stage, f"{env_root}/Cup"))
            self.table_shaders.append(find_shaders(self.stage, f"{env_root}/Table"))
            self.robot_shaders.append(
                find_shaders(self.stage, f"{env_root}/Robot/Looks")
                if randomize_robot else []
            )

        if not any(self.object_shaders):
            raise RuntimeError(
                "물체 shader prim 을 하나도 못 찾았다 (/World/envs/env_*/Cup). "
                "USD 구조가 바뀌었는지 확인할 것 — 랜덤화가 no-op 이 되면 "
                "student 가 단일 외형에 과적합된다."
            )

        # 초기 조명 1회 설정
        if self.randomize_dome_light and self.textures.dome:
            self._set_dome_light(random.choice(self.textures.dome))

    # ------------------------------------------------------------------
    def randomize(self, env_ids) -> None:
        """reset 된 env 들의 외형을 재샘플. env_ids 는 int 시퀀스."""
        from pxr import Sdf

        if self.randomize_dome_light and random.random() < DOME_LIGHT_RAND_PROB:
            self._set_dome_light(random.choice(self.textures.dome), randomize_pose=True)

        with Sdf.ChangeBlock():
            for env_id in env_ids:
                self._randomize_object(int(env_id))
                self._randomize_table(int(env_id))
                if self.randomize_robot:
                    self._randomize_robot(int(env_id))

    # ------------------------------------------------------------------
    def _set_dome_light(self, texture: str, randomize_pose: bool = False) -> None:
        from pxr import Gf
        from scipy.spatial.transform import Rotation as R

        light = self.stage.GetPrimAtPath("/World/Light")
        if not light or not light.IsValid():
            return

        _set_attr(light, "inputs:texture:file", texture)
        if randomize_pose:
            x, y, z, w = R.random().as_quat()
            _set_attr(light, "xformOp:orient", Gf.Quatd(w, Gf.Vec3d(x, y, z)))
            _set_attr(
                light, "inputs:intensity",
                float(np.random.uniform(*DOME_LIGHT_INTENSITY_RANGE)),
            )

    def _randomize_object(self, env_id: int) -> None:
        from pxr import Gf, Sdf

        for shader in self.object_shaders[env_id]:
            _set_input(shader, "diffuse_texture", Sdf.ValueTypeNames.Asset,
                       random.choice(self.textures.object))
            _set_input(shader, "project_uvw", Sdf.ValueTypeNames.Bool, True)
            _set_input(
                shader, "texture_scale", Sdf.ValueTypeNames.Float2,
                Gf.Vec2f(*np.random.uniform(*OBJECT_TEXTURE_SCALE_RANGE, size=2)),
            )
            _set_input(
                shader, "diffuse_tint", Sdf.ValueTypeNames.Color3f,
                Gf.Vec3f(*np.random.rand(3)),
            )
            _set_input(shader, "reflection_roughness_constant",
                       Sdf.ValueTypeNames.Float, float(np.random.uniform(0.0, 1.0)))
            _set_input(shader, "metallic_constant",
                       Sdf.ValueTypeNames.Float, float(np.random.uniform(0.0, 1.0)))
            _set_input(shader, "specular_level",
                       Sdf.ValueTypeNames.Float, float(np.random.uniform(0.0, 1.0)))

    def _randomize_table(self, env_id: int) -> None:
        from pxr import Gf, Sdf

        for shader in self.table_shaders[env_id]:
            _set_input(shader, "diffuse_texture", Sdf.ValueTypeNames.Asset,
                       random.choice(self.textures.table))
            _set_input(
                shader, "diffuse_tint", Sdf.ValueTypeNames.Color3f,
                Gf.Vec3f(
                    float(np.random.uniform(*TABLE_TINT_R_RANGE)),
                    float(np.random.uniform(*TABLE_TINT_G_RANGE)),
                    float(np.random.uniform(*TABLE_TINT_B_RANGE)),
                ),
            )
            _set_input(shader, "specular_level",
                       Sdf.ValueTypeNames.Float, float(np.random.uniform(0.0, 1.0)))
            _set_input(shader, "reflection_roughness_constant",
                       Sdf.ValueTypeNames.Float,
                       float(np.random.uniform(*TABLE_ROUGHNESS_RANGE)))
            _set_input(shader, "texture_rotate",
                       Sdf.ValueTypeNames.Float,
                       float(np.random.uniform(0.0, 2.0 * np.pi)))

    def _randomize_robot(self, env_id: int) -> None:
        from pxr import Sdf

        for shader in self.robot_shaders[env_id]:
            _set_input(shader, "reflection_roughness_constant",
                       Sdf.ValueTypeNames.Float,
                       float(np.random.uniform(*ARM_ROUGHNESS_RANGE)))
            _set_input(shader, "metallic_constant",
                       Sdf.ValueTypeNames.Float,
                       float(np.random.uniform(*ARM_METALLIC_RANGE)))
            _set_input(shader, "specular_level",
                       Sdf.ValueTypeNames.Float, float(np.random.uniform(0.0, 1.0)))


def _set_attr(prim, name: str, value) -> None:
    attr = prim.GetAttribute(name)
    if attr:
        attr.Set(value)


def _set_input(shader_prim, name: str, value_type, value) -> None:
    """shader input 을 세팅. 없으면 만든다 (DEXTRAH 원본 규약)."""
    from pxr import UsdShade

    attr_name = "inputs:" + name
    if attr_name not in shader_prim.GetPropertyNames():
        UsdShade.Shader(shader_prim).CreateInput(name, value_type)
    attr = shader_prim.GetAttribute(attr_name)
    if attr:
        attr.Set(value)
