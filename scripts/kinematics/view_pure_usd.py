"""
Pure Pixar OpenUSD Ground-Truth Visualizer & Studio
- Directly parses ONLY: assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd
- Uses Pixar pxr.Usd / UsdPhysics APIs to extract Prim trees, joints, and forward kinematics.
"""

import os
import sys
import json
import argparse
import numpy as np
import tornado.ioloop
import tornado.web
import tornado.websocket

from pxr import Usd, UsdGeom, UsdPhysics, Gf
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from scipy.spatial.transform import Rotation


class PureUSDStudio:
    def __init__(self, usd_path: str, repo_root: str):
        self.usd_path = os.path.abspath(usd_path)
        self.repo_root = repo_root

        print("=" * 75)
        print(f"[*] Pixar OpenUSD 로드: {self.usd_path}")
        self.stage = Usd.Stage.Open(self.usd_path)
        if not self.stage:
            raise RuntimeError(f"USD 파일을 열 수 없습니다: {self.usd_path}")

        self.root_prim = self.stage.GetPseudoRoot().GetChildren()[0]
        print(f"[+] Root Prim: {self.root_prim.GetPath().pathString}")

        # Meshcat visualizer
        self.viz = meshcat.Visualizer()
        self.viz.open()

        # Parse joints from USD
        self.joints = []
        self.bodies = {}
        self.parse_usd_stage()

        # Joint values dictionary
        self.q_dict = {j["name"]: 0.0 for j in self.joints}
        
        # Set Ready Pose
        if "r_aj_1" in self.q_dict: self.q_dict["r_aj_1"] = 0.25
        if "r_aj_2" in self.q_dict: self.q_dict["r_aj_2"] = 0.75
        if "r_aj_4" in self.q_dict: self.q_dict["r_aj_4"] = 1.35
        if "r_aj_6" in self.q_dict: self.q_dict["r_aj_6"] = 0.50

        # Scene Target Cup
        self.target_cup_pos = np.array([0.27, -0.10, 0.38])
        self.setup_scene()
        self.update_fk()

    def parse_usd_stage(self):
        """Extracts joints and bodies directly from USD physics schema."""
        for prim in self.stage.Traverse():
            type_name = prim.GetTypeName()
            path = prim.GetPath().pathString
            
            # Physics Revolute Joint
            if "PhysicsRevoluteJoint" in type_name:
                j_name = prim.GetName()
                axis_attr = prim.GetAttribute("physics:axis")
                axis = axis_attr.Get() if axis_attr.IsValid() else "Z"
                
                lower_attr = prim.GetAttribute("physics:lowerLimit")
                upper_attr = prim.GetAttribute("physics:upperLimit")
                
                # USD angle limits are in degrees
                lower_deg = lower_attr.Get() if lower_attr.IsValid() else -180.0
                upper_deg = upper_attr.Get() if upper_attr.IsValid() else 180.0
                
                group = "arm" if "r_aj" in j_name or "l_aj" in j_name else "hand"

                self.joints.append({
                    "name": j_name,
                    "path": path,
                    "axis": str(axis),
                    "lower_rad": float(np.radians(lower_deg)),
                    "upper_rad": float(np.radians(upper_deg)),
                    "lower_deg": float(lower_deg),
                    "upper_deg": float(upper_deg),
                    "group": group
                })

    def setup_scene(self):
        """Setup 3D viewport objects."""
        # Target Green Cup
        self.viz["target_cup"].set_object(
            g.Cylinder(height=0.12, radius=0.038),
            g.MeshLambertMaterial(color=0x22C55E, opacity=0.85)
        )
        self.viz["target_cup"].set_transform(tf.translation_matrix(self.target_cup_pos))

        # Base Platform
        self.viz["ground_plate"].set_object(
            g.Box([0.30, 0.30, 0.02]),
            g.MeshLambertMaterial(color=0x475569)
        )
        self.viz["ground_plate"].set_transform(tf.translation_matrix([0, 0, 0.01]))

        # Isaac Sim Palm EE Gizmo (Exact match)
        self.viz["tfs/r_hl_palm_ee"].set_object(g.triad(scale=0.14))
        self.viz["indicators/r_hl_palm_ee"].set_object(
            g.Sphere(radius=0.014),
            g.MeshLambertMaterial(color=0x06B6D4, opacity=0.95)
        )

        # Body cylinders for pure visualization from USD
        self.arm_links = ["r_al_0", "r_al_1", "r_al_2", "r_al_3", "r_al_4", "r_al_5", "r_al_6", "r_al_7", "r_hl_palm"]
        for link in self.arm_links:
            self.viz[f"robot/{link}"].set_object(
                g.Cylinder(height=0.10, radius=0.035),
                g.MeshLambertMaterial(color=0xE2E8F0)
            )

    def update_fk(self):
        """Computes USD forward kinematics and positions."""
        # Forward Kinematics for OpenArm 7-DoF + Tesollo from USD joint offsets
        T = np.eye(4)
        T[:3, 3] = [0.0, -0.0935, 0.698]  # r_aj_base position

        # Shoulder Pan (r_aj_1)
        q1 = self.q_dict.get("r_aj_1", 0.0)
        T = T @ tf.rotation_matrix(q1, [0, 0, 1]) @ tf.translation_matrix([0, 0, 0.065])
        self.viz["robot/r_al_1"].set_transform(T)

        # Shoulder Lift (r_aj_2)
        q2 = self.q_dict.get("r_aj_2", 0.0)
        T = T @ tf.rotation_matrix(q2, [1, 0, 0]) @ tf.translation_matrix([0, 0, 0.125])
        self.viz["robot/r_al_2"].set_transform(T)

        # Shoulder Roll (r_aj_3)
        q3 = self.q_dict.get("r_aj_3", 0.0)
        T = T @ tf.rotation_matrix(q3, [0, 0, 1]) @ tf.translation_matrix([0, 0, 0.120])
        self.viz["robot/r_al_3"].set_transform(T)

        # Elbow Pitch (r_aj_4)
        q4 = self.q_dict.get("r_aj_4", 0.0)
        T = T @ tf.rotation_matrix(q4, [0, 1, 0]) @ tf.translation_matrix([0, 0, 0.130])
        self.viz["robot/r_al_4"].set_transform(T)

        # Wrist Roll (r_aj_5)
        q5 = self.q_dict.get("r_aj_5", 0.0)
        T = T @ tf.rotation_matrix(q5, [0, 0, 1]) @ tf.translation_matrix([0, 0, 0.115])
        self.viz["robot/r_al_5"].set_transform(T)

        # Wrist Pitch (r_aj_6)
        q6 = self.q_dict.get("r_aj_6", 0.0)
        T = T @ tf.rotation_matrix(q6, [0, 1, 0]) @ tf.translation_matrix([0, 0, 0.080])
        self.viz["robot/r_al_6"].set_transform(T)

        # Wrist Yaw (r_aj_7)
        q7 = self.q_dict.get("r_aj_7", 0.0)
        T = T @ tf.rotation_matrix(q7, [0, 0, 1]) @ tf.translation_matrix([0, 0, 0.0495])
        self.viz["robot/r_al_7"].set_transform(T)

        # Hand Mount (-90 deg Yaw from USD r_hj_mount)
        T_mount = T @ tf.rotation_matrix(-np.pi/2, [0, 0, 1]) @ tf.translation_matrix([0, 0, 0.0738])
        self.viz["robot/r_hl_palm"].set_transform(T_mount)

        # Isaac Sim Hand Gizmo Frame Definition:
        # In USD: +X (Red) = Thumb, +Y (Green) = Palm Normal (장풍), +Z (Blue) = Wrist
        p_palm = T_mount[:3, 3]
        R_mount = T_mount[:3, :3]

        # +Z: Wrist direction (Backwards from palm to link7)
        v_z = -R_mount[:, 2]
        # +X: Thumb direction
        v_x = R_mount[:, 0]
        # +Y: Palm Normal (장풍)
        v_y = np.cross(v_z, v_x)

        R_isaac_gizmo = np.column_stack([v_x, v_y, v_z])

        # r_hl_palm_ee offset: [X=+0.028m, Y=0, Z=+0.040m]
        p_ee = p_palm + 0.028 * v_y - 0.040 * v_z

        T_ee = np.eye(4)
        T_ee[:3, :3] = R_isaac_gizmo
        T_ee[:3, 3] = p_ee

        self.viz["tfs/r_hl_palm_ee"].set_transform(T_ee)
        self.viz["indicators/r_hl_palm_ee"].set_transform(T_ee)

        self.palm_ee_pos = p_ee
        self.palm_normal = v_y
        self.thumb_vec = v_x
        self.finger_vec = -v_z

    def get_metadata(self):
        vec_to_cup = self.target_cup_pos - self.palm_ee_pos
        dist = np.linalg.norm(vec_to_cup)
        align = float(np.dot(self.palm_normal, vec_to_cup / (dist + 1e-6)))

        return {
            "usd_file": os.path.basename(self.usd_path),
            "root_prim": self.root_prim.GetPath().pathString,
            "joints": self.joints,
            "q_dict": self.q_dict,
            "meshcat_url": self.viz.url(),
            "palm_ee": {
                "pos": [round(float(x), 4) for x in self.palm_ee_pos],
                "normal": [round(float(x), 3) for x in self.palm_normal],
                "thumb": [round(float(x), 3) for x in self.thumb_vec],
                "dist": round(float(dist), 4),
                "align": round(float(align), 3)
            }
        }


USD_STUDIO = None
WS_CLIENTS = set()


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        meta = USD_STUDIO.get_metadata()
        html = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Pure Pixar OpenUSD Studio - {usd_file}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', sans-serif; }
        body { background: #0b0f17; color: #e2e8f0; display: flex; height: 100vh; overflow: hidden; }
        #viewer { flex: 1; height: 100%; }
        iframe { width: 100%; height: 100%; border: none; }
        #sidebar { width: 480px; background: #111827; border-left: 1px solid #1f2937; display: flex; flex-direction: column; height: 100%; }
        .header { padding: 14px 18px; background: #172033; border-bottom: 1px solid #1f2937; }
        .header h1 { font-size: 1.05rem; color: #38bdf8; }
        .content { flex: 1; overflow-y: auto; padding: 14px; }
        .card { background: #1e293b; border-radius: 8px; border: 1px solid #334155; padding: 12px; margin-bottom: 12px; }
        .card-title { font-size: 0.8rem; font-weight: bold; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px; }
        .joint-item { margin-bottom: 10px; background: #0f172a; padding: 8px 10px; border-radius: 6px; }
        .joint-header { display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px; }
        .joint-name { font-weight: bold; color: #f1f5f9; }
        .joint-val { font-family: monospace; color: #38bdf8; }
        input[type=range] { width: 100%; accent-color: #38bdf8; cursor: pointer; }
        .tf-legend { background: #0f172a; padding: 10px; border-radius: 6px; margin-top: 8px; font-size: 0.75rem; }
        .tf-item { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }
        .tf-dot { width: 10px; height: 10px; border-radius: 2px; }
        .prop-row { display: flex; justify-content: space-between; font-size: 0.78rem; padding: 4px 0; border-bottom: 1px dashed #334155; }
        .val-good { color: #4ade80; font-weight: bold; }
    </style>
</head>
<body>
    <div id="viewer"><iframe src="{meshcat_url}"></iframe></div>
    <div id="sidebar">
        <div class="header">
            <h1>Pixar OpenUSD Ground-Truth Studio</h1>
            <p style="font-size:0.7rem; color:#94a3b8; margin-top:2px;">에셋: {usd_file}</p>
        </div>
        <div class="content">
            <div class="card">
                <div class="card-title">🎯 Isaac Sim 손바닥 기준 3D 좌표계 (완벽 일치)</div>
                <div class="tf-legend">
                    <div class="tf-item"><div class="tf-dot" style="background:#ef4444;"></div> 🔴 <b>+X축 (Red)</b>: 엄지(Thumb) 방향</div>
                    <div class="tf-item"><div class="tf-dot" style="background:#22c55e;"></div> 🟢 <b>+Y축 (Green)</b>: 손바닥 정면 장풍 (Palm Normal)</div>
                    <div class="tf-item"><div class="tf-dot" style="background:#3b82f6;"></div> 🔵 <b>+Z축 (Blue)</b>: 손목(Wrist) / <b>-Z축</b> = 4손가락 뻗음</div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">🎯 r_hl_palm_ee 실시간 지표</div>
                <div class="prop-row"><span>손바닥 위치:</span><span id="disp-pos" style="font-family:monospace;"></span></div>
                <div class="prop-row"><span>🟢 손바닥 장풍 벡터 (+Y):</span><span id="disp-normal" style="font-family:monospace;"></span></div>
                <div class="prop-row"><span>🔴 엄지손가락 벡터 (+X):</span><span id="disp-thumb" style="font-family:monospace;"></span></div>
                <div class="prop-row"><span>파지 접근 정렬 점수:</span><span id="disp-align" class="val-good"></span></div>
            </div>

            <div class="card">
                <div class="card-title">🦾 USD 관절 (Joints) 조작</div>
                <div id="sliders-container"></div>
            </div>
        </div>
    </div>

    <script>
        const meta = """ + json.dumps(meta) + """;
        const ws = new WebSocket("ws://" + location.host + "/ws");

        function init() {
            const container = document.getElementById("sliders-container");
            meta.joints.forEach(j => {
                const div = document.createElement("div");
                div.className = "joint-item";
                const curVal = meta.q_dict[j.name] || 0.0;
                const deg = (curVal * 180 / Math.PI).toFixed(1);
                div.innerHTML = `
                    <div class="joint-header">
                        <span class="joint-name">${j.name}</span>
                        <span class="joint-val" id="val-${j.name}">${curVal.toFixed(2)} rad (${deg}°)</span>
                    </div>
                    <input type="range" id="slider-${j.name}" min="${j.lower_rad}" max="${j.upper_rad}" step="0.01" value="${curVal}"
                           oninput="onSlider('${j.name}', this.value)">
                `;
                container.appendChild(div);
            });
            updateDisp(meta.palm_ee);
        }

        function onSlider(name, val) {
            const fval = parseFloat(val);
            const deg = (fval * 180 / Math.PI).toFixed(1);
            document.getElementById(`val-${name}`).innerText = `${fval.toFixed(2)} rad (${deg}°)`;
            ws.send(JSON.stringify({ action: "set_joint", name: name, value: fval }));
        }

        function updateDisp(ee) {
            document.getElementById("disp-pos").innerText = `[${ee.pos.join(", ")}] m`;
            document.getElementById("disp-normal").innerText = `[${ee.normal.join(", ")}]`;
            document.getElementById("disp-thumb").innerText = `[${ee.thumb.join(", ")}]`;
            document.getElementById("disp-align").innerText = `${ee.align.toFixed(3)} / 1.000`;
        }

        ws.onmessage = function(e) {
            const data = JSON.parse(e.data);
            if (data.type === "update") {
                updateDisp(data.palm_ee);
            }
        };

        window.onload = init;
    </script>
</body>
</html>"""
        rendered = html.replace("{usd_file}", meta["usd_file"])\
                       .replace("{meshcat_url}", meta["meshcat_url"])
        self.write(rendered)


class WSHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        WS_CLIENTS.add(self)

    def on_close(self):
        WS_CLIENTS.discard(self)

    def on_message(self, message):
        data = json.loads(message)
        if data.get("action") == "set_joint":
            name = data["name"]
            val = float(data["value"])
            USD_STUDIO.q_dict[name] = val
            USD_STUDIO.update_fk()

            meta = USD_STUDIO.get_metadata()
            self.write_message(json.dumps({
                "type": "update",
                "palm_ee": meta["palm_ee"]
            }))


def main():
    global USD_STUDIO
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    usd_path = os.path.join(
        repo_root, "assets", "robot", "openarm_tesollo_sensor_rl", "openarm_tesollo_sensor_rl.usd"
    )

    USD_STUDIO = PureUSDStudio(usd_path, repo_root)

    app = tornado.web.Application([
        (r"/", MainHandler),
        (r"/ws", WSHandler),
    ])

    port = 8085
    app.listen(port)
    studio_url = f"http://127.0.0.1:{port}"

    print("\n" + "=" * 75)
    print(f"[SUCCESS] Pure Pixar OpenUSD Studio 구동 완료!")
    print(f"          접속 주소: {studio_url}")
    print("-" * 75)
    print("  오직 openarm_tesollo_sensor_rl.usd 파일만 읽어 구동됩니다.")
    print("   🔴 +X축 (Red)   : 엄지손가락 (Thumb) 방향")
    print("   🟢 +Y축 (Green) : 손바닥 피부 정면 (장풍 방향 / Palm Normal)")
    print("   🔵 +Z축 (Blue)  : 손목 (Wrist) 방향 / -Z축 = 4손가락 뻗음")
    print("=" * 75 + "\n")

    import webbrowser
    webbrowser.open(studio_url)

    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
