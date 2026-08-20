"""모듈 스위치 조합 검증 probe.

`resolve_cfg` 가 스위치를 파생값(자산 cfg · 차원 · replicate_physics)까지 반영하는지
확인한다. ★hydra 는 `env_cfg.from_dict(...)` 로 **필드만** 덮어쓰고 `__post_init__` 을
다시 돌리지 않으므로, 이 재해석이 없으면 CLI 오버라이드가 조용히 무시된다.

pytest 로 못 돌리는 이유: cfg 가 isaaclab.sim → pxr 을 타는데 서버 conda 환경엔 pxr 이
없다. 앱을 띄워야 한다.

    isaaclab.sh -p scripts/reinforcement_learning/probes/probe_cfg_switches.py
"""
from __future__ import annotations

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import traceback   # noqa: E402

from openarm.agnostic.tasks.grasp_lift_fabric import (   # noqa: E402
    grasp_lift_fabric_env_cfg as C,
)

FAIL = []


def check(label, cond, detail=""):
    print(f"  {'OK  ' if cond else '★FAIL'} {label}   {detail}", flush=True)
    if not cond:
        FAIL.append(label)


print("\n=== 1. 기본값 (Phase A) ===", flush=True)
cfg = C.GraspLiftFabricEnvCfg()
base_obs, base_act = cfg.observation_space, cfg.action_space
print(f"  profile={cfg.profile_name} bank={cfg.object_bank} "
      f"action={base_act} obs={base_obs} critic={cfg.state_space} "
      f"replicate={cfg.scene.replicate_physics}")
check("bis_right 차원 26/122/128", (base_act, base_obs, cfg.state_space) == (26, 122, 128))
check("single_cup 은 physics 복제 가능", cfg.scene.replicate_physics is True)

print("\n=== 2. 멱등성 (두 번 풀어도 같아야) ===", flush=True)
C.resolve_cfg(cfg)
check("재해석 후 차원 불변", (cfg.action_space, cfg.observation_space) == (base_act, base_obs))

print("\n=== 3. 물체 뱅크 스위치 (hydra 처럼 필드만 바꾼 뒤 재해석) ===", flush=True)
cfg.object_bank = "cup_family"
cfg.enable_object_onehot = True
C.resolve_cfg(cfg)
check("MultiAsset → replicate_physics=False", cfg.scene.replicate_physics is False)
check("onehot 8 반영", cfg.observation_space == base_obs + 8,
      f"{base_obs} → {cfg.observation_space}")
check("critic = policy + 6", cfg.state_space == cfg.observation_space + 6)

print("\n=== 4. 로봇 프로필 스위치 (손 자유도가 차원까지 따라오는가) ===", flush=True)
for name, hand in (("bis_right", 20), ("rh56_right", 12), ("sens_left", 1)):
    c = C.GraspLiftFabricEnvCfg()
    c.profile_name = name
    C.resolve_cfg(c)
    check(f"{name}: action = 6 + {hand}", c.action_space == 6 + hand,
          f"action={c.action_space} obs={c.observation_space}")

print("\n=== 5. Fabrics 자산 없는 프로필은 fail-loud ===", flush=True)
c = C.GraspLiftFabricEnvCfg()
c.profile_name = "rh56_left"
try:
    C.resolve_cfg(c)
    check("rh56_left 은 예외를 던져야", False, "예외가 안 났다")
except RuntimeError as e:
    check("rh56_left fail-loud", "Fabrics" in str(e), str(e)[:70])

print("\n=== 6. DR / ADR 스위치 ===", flush=True)
c = C.GraspLiftFabricEnvCfg()
c.enable_physics_dr = True
c.enable_adr = True
C.resolve_cfg(c)
check("DR/ADR 은 차원에 영향 없음(체크포인트 호환)",
      (c.action_space, c.observation_space) == (base_act, 122))

print("\n" + ("=" * 60))
print("전부 통과" if not FAIL else f"★실패 {len(FAIL)}건: {FAIL}")
app.close()
