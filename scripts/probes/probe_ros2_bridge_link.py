#!/usr/bin/env python3
"""Isaac Sim ↔ ROS 2 링크 확인 — **IsaacLab 실행 안에서** 브리지가 도는가.

왜 ActionGraph+ScriptNode 가 아니라 여기인가:
  `isaacsim_bridge/ISAACSIM_POLICY_WIRING.md` 의 설계는 Isaac Sim GUI 앱의 ActionGraph
  안 ScriptNode 에서 정책을 굴린다. 하지만 `grip/left/grasp_sensor` 는 IsaacLab
  `ManagerBasedRLEnv` 전체(ObservationManager · fabric 액션항 · 접촉센서 · 컵)가 있어야
  성립한다. ScriptNode 로 그걸 재현하면 **학습 env 와 갈라진다** — 이 트랙에서 반복해서
  대가를 치른 함정이다. 그래서 이미 검증된 IsaacLab 경로 안에서 브리지를 켠다.

★Isaac 번들 python 3.11 에는 rclpy 가 없다. `isaacsim.ros2.bridge` 는 rclpy 가 아니라
 자체 구현(OmniGraph 노드)이라 그 제약을 받지 않는다. 이 프로브가 확인하는 것이 정확히
 그 지점이다 — **rclpy 없이 ROS 2 토픽이 나가는가.**

확인 순서(각 단계에서 실패하면 그 자리에서 말한다):
  ① 확장이 켜지는가            ② OmniGraph 가 만들어지는가
  ③ 틱이 도는가                ④ 토픽이 실제로 나가는가(외부에서 `ros2 topic hz`)

★★**실행법 — `--kit_args` 없이는 안 된다.**

    source /opt/ros/humble/setup.bash
    ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_ros2_bridge_link.py \
        --ticks 60000 \
        --kit_args "--enable isaacsim.ros2.bridge --enable omni.graph.action --enable omni.graph.nodes"

  런타임에 `enable_extension()` 으로 켜면 노드 타입은 등록되지만 **그래프 evaluator 가
  등록되지 않아** `og.Controller.edit` 이 "Failed to wrap graph in node" 로 죽는다.
  IsaacLab 의 headless 경험파일(isaaclab.python.headless.kit)이 OmniGraph 파이프라인을
  안 싣기 때문이다. **기동 시점에 실어야 한다.** 확장 활성 여부만 보면 True 라서
  이 차이가 안 보인다 — 실제로 여기서 여러 판을 태웠다.

실측(2026-08-27, vision-3090 · Isaac Sim 5.1.0 · ROS 2 Humble, 같은 머신):
    /clock                  482~507 Hz
    /isaacsim/left_arm_cmd  601~604 Hz
  둘 다 `ros2 topic list` 에 뜨고 `ros2 topic hz` 로 수신된다. 주기는 물리에 묶여 있지
  않은 자유 틱이라 이 숫자 자체는 의미가 없다 — **경로가 산다는 것**이 결과다.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--topic", default="/isaacsim/left_arm_cmd",
                    help="발행할 토픽. 기본은 배포 구성이 쓰는 좌팔 명령 채널")
parser.add_argument("--dof", type=int, default=7, help="발행할 배열 길이(좌팔 7)")
parser.add_argument("--ticks", type=int, default=600, help="틱 수 (60 Hz 기준 10 s)")
parser.add_argument("--domain_id", type=int, default=None,
                    help="ROS_DOMAIN_ID. 안 주면 환경변수 그대로 둔다")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.domain_id is not None:
    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaacsim.core.utils.extensions import enable_extension    # noqa: E402

# ★import 순서가 규약이다. `omni.graph.*` 는 해당 확장이 켜진 **뒤에만** import 된다 —
#   위에서 미리 import 하면 ModuleNotFoundError 로 죽는다. 실제로 당했다.
_GRAPH_EXTS = ("omni.graph.core", "omni.graph.action", "omni.graph.nodes",
               "omni.graph.scriptnode", "isaacsim.ros2.bridge")
for _e in _GRAPH_EXTS:
    enable_extension(_e)

# ★함수 안에서 `import omni.usd` 를 하면 그 함수 안의 `omni` 가 **지역변수**가 되어
#   같은 함수의 다른 `omni.kit.app...` 참조가 UnboundLocalError 로 죽는다. 파이썬 스코프
#   규칙이고 실제로 당했다. omni 계열은 전부 모듈 수준에서만 import 한다.
import omni.kit.app                                            # noqa: E402
import omni.timeline                                           # noqa: E402
import omni.usd                                                # noqa: E402

_app = omni.kit.app.get_app()
_app.update()

import omni.graph.core as og                                   # noqa: E402


#: 그래프 프림 경로. 루트 직하가 아니라 2단계여야 한다.
GRAPH_PATH = "/ActionGraph/ROS2LinkProbe"


def main() -> int:
    print(f"[ROS2] ROS_DISTRO={os.environ.get('ROS_DISTRO', '(없음)')} "
          f"· ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '(기본 0)')} "
          f"· RMW={os.environ.get('RMW_IMPLEMENTATION', '(기본)')}")

    # ── ① 확장 ────────────────────────────────────────────────────────
    mgr = omni.kit.app.get_app().get_extension_manager()
    for _e in _GRAPH_EXTS:
        print(f"[ROS2] 확장 {_e:28s} 활성={mgr.is_extension_enabled(_e)}")
    enabled = mgr.is_extension_enabled("isaacsim.ros2.bridge")
    if not enabled:
        print("[ROS2] ❌ 확장이 안 켜졌다. ROS 2 를 source 한 뒤 실행했는지 확인하라 "
              "(브리지는 시스템 rclcpp 라이브러리를 찾는다).")
        return 1
    simulation_app.update()

    # ── ①-b 스테이지 ─────────────────────────────────────────────────
    # OmniGraph 는 **USD 프림**이다. 스테이지가 없으면 붙일 데가 없어
    # "Failed to wrap graph in node" 로 죽는다. 실제 배포에서는 IsaacLab env 가
    # 이미 스테이지를 갖고 있으므로 이 단계가 필요 없다 — 여기서는 단독 확인이라 만든다.
    ctx = omni.usd.get_context()
    if ctx.get_stage() is None:
        ctx.new_stage()
        omni.kit.app.get_app().update()
    print(f"[ROS2] 스테이지 {'있음' if ctx.get_stage() is not None else '없음'}")

    # ── ② 그래프 ──────────────────────────────────────────────────────
    # ── ② 그래프: 구성을 **여러 개 시험**한다 ────────────────────────
    # Isaac 부팅이 한 번에 3 분이라 추측을 한 번씩 태울 수 없다. 후보를 한 실행에서
    # 전부 돌리고 어느 것이 서는지 본다. 각 시도는 서로를 오염시키지 않게 격리한다.
    keys = og.Controller.Keys
    stage = ctx.get_stage()

    def attempt(name: str, path: str, spec: dict) -> bool:
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
        try:
            og.Controller.edit({"graph_path": path, "evaluator_name": "execution"}, spec)
        except Exception as exc:                                # noqa: BLE001
            print(f"[ROS2] ✗ {name:22s} {type(exc).__name__}: {str(exc)[:110]}")
            return False
        print(f"[ROS2] ✓ {name:22s} {path}")
        return True

    # ⓐ 가장 얇은 확인 — 시계만 낸다. 대상 프림도, 메시지 introspection 도 필요 없다.
    #    이것이 서면 "브리지가 실제로 ROS 2 로 뱉는다"가 증명된다.
    ok_clock = attempt("PublishClock", "/ActionGraph/ClockProbe", {
        keys.CREATE_NODES: [
            ("tick", "omni.graph.action.OnPlaybackTick"),
            ("time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("pub", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ],
        keys.CONNECT: [
            ("tick.outputs:tick", "pub.inputs:execIn"),
            ("time.outputs:simulationTime", "pub.inputs:timeStamp"),
        ],
    })

    # ⓑ 범용 퍼블리셔 — 배포에서 쓰려는 `/isaacsim/left_arm_cmd` 모양.
    #    메시지 타입을 런타임에 해석하므로 여기서 걸릴 수 있다.
    ok_generic = attempt("ROS2Publisher(std_msgs)", "/ActionGraph/CmdProbe", {
        keys.CREATE_NODES: [
            ("tick", "omni.graph.action.OnPlaybackTick"),
            ("pub", "isaacsim.ros2.bridge.ROS2Publisher"),
        ],
        keys.SET_VALUES: [
            ("pub.inputs:topicName", args.topic),
            ("pub.inputs:messagePackage", "std_msgs"),
            ("pub.inputs:messageSubfolder", "msg"),
            ("pub.inputs:messageName", "Float64MultiArray"),
        ],
        keys.CONNECT: [("tick.outputs:tick", "pub.inputs:execIn")],
    })

    if not (ok_clock or ok_generic):
        print("[ROS2] ❌ 어떤 구성도 서지 않았다.")
        return 1

    omni.timeline.get_timeline_interface().play()
    print("[ROS2] 타임라인 재생 시작 (OnPlaybackTick 은 재생 중에만 온다)")
    print(f"[ROS2] 확인할 토픽: /clock={ok_clock} · {args.topic}={ok_generic}")

    # ── ③ 틱 ──────────────────────────────────────────────────────────
    print(f"[ROS2] {args.ticks} 틱 발행 시작 — 지금 다른 터미널에서:")
    print(f"[ROS2]   ros2 topic hz {args.topic}")
    for i in range(args.ticks):
        simulation_app.update()
        if (i + 1) % 120 == 0:
            print(f"[ROS2]   {i+1}/{args.ticks} 틱", flush=True)
    print("[ROS2] ✅ 틱 완료 — 발행 쪽은 죽지 않았다. "
          "실제 수신 여부는 **밖에서** 확인해야 한다(이 프로세스는 자기 발행을 못 센다).")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
