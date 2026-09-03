"""
Isaac Lab Style Interactive Robot Kinematics & Fabrics Studio
- Exactly matches Isaac Sim / Isaac Lab GUI coordinate frame:
    * 🔴 +X (Red)   : Thumb (엄지) 방향
    * 🟢 +Y (Green) : Palm Normal (손바닥 정면 피부/장풍 방향)
    * 🔵 +Z (Blue)  : Wrist (손목 방향) / -Z = 4손가락 뻗음 (Finger extension)
"""

import os
import sys
import json
import argparse
import numpy as np
import tornado.ioloop
import tornado.web
import tornado.websocket

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf
from scipy.spatial.transform import Rotation


class RobotKinematicsServer:
    def __init__(self, urdf_path: str, repo_root: str):
        self.urdf_path = os.path.abspath(urdf_path)
        self.repo_root = repo_root

        mesh_dirs = [
            os.path.dirname(self.urdf_path),
            os.path.join(self.repo_root, "source", "FABRICS", "src", "fabrics_sim", "models", "robots", "urdf", "openarm_tesollo"),
            os.path.join(self.repo_root, "source", "FABRICS", "src", "fabrics_sim", "models", "robots", "urdf", "openarm_tesollo", "meshes"),
            os.path.join(self.repo_root, "source", "FABRICS", "src", "fabrics_sim", "models", "robots", "urdf", "kuka_allegro"),
            os.path.join(self.repo_root, "source", "FABRICS", "src", "fabrics_sim", "models", "robots", "urdf", "kuka_allegro", "meshes"),
        ]
        mesh_dirs = [d for d in mesh_dirs if os.path.isdir(d)]

        print("=" * 75)
        print(f"[*] URDF 로드: {os.path.basename(self.urdf_path)}")
        self.robot = RobotWrapper.BuildFromURDF(self.urdf_path, mesh_dirs)
        self.model = self.robot.model
        self.data = self.robot.data

        # Visual or collision model
        display_model = self.robot.visual_model if self.robot.visual_model.ngeoms > 0 else self.robot.collision_model

        # Launch Meshcat
        self.viz = MeshcatVisualizer(self.model, self.robot.collision_model, display_model)
        self.viz.initViewer(open=False)
        self.viz.loadViewerModel()

        # Start with clean Ready Posture (Arm lifted forward)
        self.q = pin.neutral(self.model)
        if self.model.nq >= 7:
            self.q[0] = 0.25   # shoulder pan
            self.q[1] = 0.75   # shoulder lift
            self.q[3] = 1.35   # elbow pitch
            self.q[5] = 0.50   # wrist pitch
        if self.model.nq >= 27:
            for i in range(7, self.model.nq):
                self.q[i] = 0.30

        self.show_spheres = True
        self.target_cup_pos = np.array([0.27, -0.10, 0.38])

        self.setup_scene_objects()
        self.update_kinematics(self.q)
        self.setup_clean_spheres()

    def setup_scene_objects(self):
        """Adds Target Cup and Workspace Ground."""
        self.viz.viewer["target_cup"].set_object(
            g.Cylinder(height=0.12, radius=0.038),
            g.MeshLambertMaterial(color=0x22C55E, opacity=0.85)
        )
        self.viz.viewer["target_cup"].set_transform(tf.translation_matrix(self.target_cup_pos))

        # Palm EE Canonical Gizmo (Exact match with Isaac Sim Gizmo)
        # Red (+X) = Thumb, Green (+Y) = Palm Normal (장풍), Blue (+Z) = Wrist
        self.viz.viewer["tfs/palm_ee"].set_object(g.triad(scale=0.14))
        
        # Palm EE Cyan Sphere
        self.viz.viewer["indicators/palm_ee"].set_object(
            g.Sphere(radius=0.014),
            g.MeshLambertMaterial(color=0x06B6D4, opacity=0.95)
        )

        # 5 Fingertips Colored Spheres
        finger_keys = ["rl_dg_1_tip", "rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip", "rl_dg_5_tip"]
        colors = [0xEF4444, 0xF59E0B, 0x10B981, 0x6366F1, 0xEC4899]
        for idx, k in enumerate(finger_keys):
            self.viz.viewer[f"indicators/{k}"].set_object(
                g.Sphere(radius=0.010),
                g.MeshLambertMaterial(color=colors[idx], opacity=0.9)
            )

    def setup_clean_spheres(self):
        """Sets up clean collision spheres without visual clutter."""
        for fid, frame in enumerate(self.model.frames):
            fname = frame.name
            T = self.data.oMf[fid].homogeneous

            # Render key collision spheres
            if "sphere" in fname.lower() and not ("neg" in fname.lower() or "palm_x" in fname.lower() or "palm_y" in fname.lower() or "palm_z" in fname.lower()):
                sphere_radius = 0.020
                if "body" in fname.lower():
                    sphere_radius = 0.11
                elif "palm" in fname.lower():
                    sphere_radius = 0.032
                elif "link" in fname.lower():
                    sphere_radius = 0.040

                sphere_node = f"collision_spheres/{fname}"
                self.viz.viewer[sphere_node].set_object(
                    g.Sphere(radius=sphere_radius),
                    g.MeshLambertMaterial(color=0xF59E0B, opacity=0.30)
                )
                self.viz.viewer[sphere_node].set_transform(T)

    def compute_isaac_sim_palm_frame(self):
        """Computes the exact Isaac Sim coordinate frame (🔴+X=Thumb, 🟢+Y=Normal/장풍, 🔵+Z=Wrist)."""
        if not self.model.existFrame("palm_link"):
            return None, None, None, None, None

        palm_fid = self.model.getFrameId("palm_link")
        wrist_fid = self.model.getFrameId("openarm_right_link7") if self.model.existFrame("openarm_right_link7") else palm_fid
        thumb_fid = self.model.getFrameId("tesollo_right_rl_dg_1_1") if self.model.existFrame("tesollo_right_rl_dg_1_1") else palm_fid

        p_palm = self.data.oMf[palm_fid].translation
        p_wrist = self.data.oMf[wrist_fid].translation
        p_thumb = self.data.oMf[thumb_fid].translation

        # 1. 🔵 Blue (+Z): Points towards the wrist
        v_z = p_wrist - p_palm
        v_z = v_z / (np.linalg.norm(v_z) + 1e-6)

        # 2. 🔴 Red (+X): Points along the thumb (orthogonalized to Z)
        v_thumb = p_thumb - p_palm
        v_x = v_thumb - np.dot(v_thumb, v_z) * v_z
        v_x = v_x / (np.linalg.norm(v_x) + 1e-6)

        # 3. 🟢 Green (+Y): Palm Normal (장풍 방향 / Outwards from palm skin)
        v_y = np.cross(v_z, v_x)
        v_y = v_y / (np.linalg.norm(v_y) + 1e-6)

        # Construct exact Isaac Sim 3x3 rotation matrix
        R_isaac = np.column_stack([v_x, v_y, v_z])

        # Palm EE Center: 28mm outwards normal (+Y), 40mm towards fingers (-Z)
        ee_pos = p_palm + 0.028 * v_y - 0.040 * v_z

        return ee_pos, R_isaac, v_x, v_y, v_z

    def update_kinematics(self, q: np.ndarray):
        self.q = q.copy()
        self.viz.display(self.q)
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

        # Update Collision spheres
        for fid, frame in enumerate(self.model.frames):
            fname = frame.name
            T = self.data.oMf[fid].homogeneous
            if "sphere" in fname.lower() and self.show_spheres:
                if not ("neg" in fname.lower() or "palm_x" in fname.lower() or "palm_y" in fname.lower() or "palm_z" in fname.lower()):
                    self.viz.viewer[f"collision_spheres/{fname}"].set_transform(T)

        # Update Palm EE with exact Isaac Sim coordinate frame
        ee_pos, R_isaac, v_x, v_y, v_z = self.compute_isaac_sim_palm_frame()
        if ee_pos is not None:
            T_ee = np.eye(4)
            T_ee[:3, :3] = R_isaac
            T_ee[:3, 3] = ee_pos

            self.viz.viewer["indicators/palm_ee"].set_transform(T_ee)
            self.viz.viewer["tfs/palm_ee"].set_transform(T_ee)

        # Update 5 Fingertips
        finger_keys = ["rl_dg_1_tip", "rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip", "rl_dg_5_tip"]
        for k in finger_keys:
            if self.model.existFrame(k):
                fid = self.model.getFrameId(k)
                self.viz.viewer[f"indicators/{k}"].set_transform(self.data.oMf[fid].homogeneous)

    def get_robot_metadata(self):
        """Extracts complete ground-truth URDF metadata."""
        joints_info = []
        for j_id in range(1, self.model.njoints):
            j = self.model.joints[j_id]
            j_name = self.model.names[j_id]
            nq = j.nq
            idx_q = j.idx_q
            
            lower = float(self.model.lowerPositionLimit[idx_q]) if nq > 0 else 0.0
            upper = float(self.model.upperPositionLimit[idx_q]) if nq > 0 else 0.0
            val = float(self.q[idx_q]) if nq > 0 else 0.0

            placement = self.model.jointPlacements[j_id]
            trans = [round(float(x), 4) for x in placement.translation]
            rpy = [round(float(x), 4) for x in Rotation.from_matrix(placement.rotation).as_euler('xyz', degrees=True)]

            group = "arm" if j_id <= 7 else "hand"

            joints_info.append({
                "id": j_id,
                "name": j_name,
                "idx_q": idx_q,
                "nq": nq,
                "lower": lower,
                "upper": upper,
                "value": val,
                "origin_xyz": trans,
                "origin_rpy": rpy,
                "type": j.shortname(),
                "group": group
            })

        frames_info = []
        for f_id in range(self.model.nframes):
            f = self.model.frames[f_id]
            T = self.data.oMf[f_id]
            pos = [round(float(x), 4) for x in T.translation]
            rot = [round(float(x), 2) for x in Rotation.from_matrix(T.rotation).as_euler('xyz', degrees=True)]
            frames_info.append({
                "id": f_id,
                "name": f.name,
                "parent_joint": self.model.names[f.parentJoint] if f.parentJoint < len(self.model.names) else "none",
                "pos": pos,
                "rot": rot,
                "is_sphere": "sphere" in f.name.lower()
            })

        palm_ee_info = self.get_palm_ee_info()

        return {
            "name": self.model.name,
            "nq": self.model.nq,
            "nv": self.model.nv,
            "njoints": self.model.njoints,
            "nframes": self.model.nframes,
            "joints": joints_info,
            "frames": frames_info,
            "palm_ee": palm_ee_info,
            "meshcat_url": self.viz.viewer.url()
        }

    def get_palm_ee_info(self):
        ee_pos, R_isaac, v_x, v_y, v_z = self.compute_isaac_sim_palm_frame()
        if ee_pos is None:
            return None

        rot_deg = Rotation.from_matrix(R_isaac).as_euler('xyz', degrees=True)

        vec_to_cup = self.target_cup_pos - ee_pos
        dist_to_cup = np.linalg.norm(vec_to_cup)
        dir_to_cup = vec_to_cup / (dist_to_cup + 1e-6)

        # Alignment score: +Y palm normal dot product with cup direction
        align_score = float(np.dot(v_y, dir_to_cup))

        return {
            "pos": [round(float(x), 4) for x in ee_pos],
            "rot": [round(float(x), 2) for x in rot_deg],
            "palm_normal": [round(float(x), 3) for x in v_y],
            "thumb_vec": [round(float(x), 3) for x in v_x],
            "finger_vec": [round(float(x), 3) for x in -v_z],
            "dist_to_cup": round(float(dist_to_cup), 4),
            "align_score": round(float(align_score), 3)
        }


# Global server instance
ROBOT_SERVER = None
WS_CLIENTS = set()


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        meta = ROBOT_SERVER.get_robot_metadata()
        html_template = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Isaac Lab Style Robot Kinematics Studio - {robot_name}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', -apple-system, sans-serif; }
        body { background: #0b0f17; color: #e2e8f0; display: flex; height: 100vh; overflow: hidden; }
        #viewer-panel { flex: 1; height: 100%; position: relative; }
        iframe { width: 100%; height: 100%; border: none; }
        #sidebar { width: 490px; background: #111827; border-left: 1px solid #1f2937; display: flex; flex-direction: column; height: 100%; }
        .sidebar-header { padding: 14px 18px; background: #172033; border-bottom: 1px solid #1f2937; display: flex; justify-content: space-between; align-items: center; }
        .sidebar-header h1 { font-size: 1.05rem; font-weight: 700; color: #38bdf8; }
        .badge { background: #0284c7; color: white; font-size: 0.7rem; padding: 2px 8px; border-radius: 9999px; font-weight: 600; }
        .tabs { display: flex; background: #0f172a; border-bottom: 1px solid #1f2937; }
        .tab-btn { flex: 1; padding: 10px 4px; background: none; border: none; color: #94a3b8; font-weight: 600; font-size: 0.78rem; cursor: pointer; border-bottom: 2px solid transparent; }
        .tab-btn.active { color: #38bdf8; border-bottom-color: #38bdf8; background: #1e293b; }
        .tab-content { flex: 1; overflow-y: auto; padding: 14px; display: none; }
        .tab-content.active { display: block; }
        .card { background: #1e293b; border-radius: 8px; border: 1px solid #334155; padding: 12px; margin-bottom: 12px; }
        .card-header { font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
        .section-tag { font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; background: #334155; color: #38bdf8; }
        .joint-item { margin-bottom: 10px; background: #0f172a; padding: 8px 10px; border-radius: 6px; border: 1px solid #1e293b; }
        .joint-header { display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px; }
        .joint-name { font-weight: 600; color: #f1f5f9; }
        .joint-val { font-family: monospace; color: #38bdf8; font-weight: bold; }
        .joint-limits { font-size: 0.7rem; color: #64748b; margin-top: 2px; display: flex; justify-content: space-between; }
        input[type=range] { width: 100%; accent-color: #38bdf8; cursor: pointer; }
        .btn-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px; }
        button { background: #334155; color: #f8fafc; border: 1px solid #475569; padding: 7px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 600; cursor: pointer; transition: 0.15s; }
        button:hover { background: #475569; border-color: #64748b; }
        button.primary { background: #0284c7; border-color: #0369a1; }
        button.primary:hover { background: #0369a1; }
        .prop-row { display: flex; justify-content: space-between; font-size: 0.78rem; padding: 4px 0; border-bottom: 1px dashed #334155; }
        .prop-row:last-child { border-bottom: none; }
        .prop-key { color: #94a3b8; }
        .prop-val { font-family: monospace; font-weight: 600; color: #f8fafc; }
        .val-highlight { color: #38bdf8; font-weight: bold; }
        .val-good { color: #4ade80; }
        .val-warn { color: #facc15; }
        .tf-legend { background: #0f172a; padding: 10px; border-radius: 6px; border: 1px solid #1e293b; margin-top: 8px; }
        .tf-title { font-size: 0.75rem; font-weight: bold; color: #94a3b8; margin-bottom: 6px; }
        .tf-item { display: flex; align-items: center; gap: 6px; font-size: 0.75rem; margin-bottom: 3px; }
        .tf-dot { width: 10px; height: 10px; border-radius: 2px; }
        .collapse-header { cursor: pointer; user-select: none; display: flex; justify-content: space-between; align-items: center; }
        .collapse-header:hover { color: #38bdf8; }
    </style>
</head>
<body>
    <div id="viewer-panel">
        <iframe src="{meshcat_url}"></iframe>
    </div>

    <div id="sidebar">
        <div class="sidebar-header">
            <div>
                <h1>{robot_name}</h1>
                <p style="font-size: 0.7rem; color: #94a3b8; margin-top: 2px;">Isaac Sim Ground-Truth Kinematics Studio</p>
            </div>
            <span class="badge">{nq} DoF</span>
        </div>

        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('joints')">🦾 관절 조작</button>
            <button class="tab-btn" onclick="switchTab('palm_ee')">🎯 Palm EE & 축 정의</button>
            <button class="tab-btn" onclick="switchTab('spheres')">🟡 충돌 구체</button>
            <button class="tab-btn" onclick="switchTab('properties')">📋 URDF 속성</button>
        </div>

        <!-- 1. Joint Control Tab -->
        <div id="tab-joints" class="tab-content active">
            <div class="card">
                <div class="card-header"><span>⚡ 대표 자세 프리셋</span></div>
                <div class="btn-grid">
                    <button class="primary" onclick="setPoseReady()">Ready Posture (기본)</button>
                    <button onclick="setPoseReach()">전방 도달 (Reach)</button>
                    <button onclick="setPoseGrasp()">그리퍼 쥐기 (Grasp)</button>
                    <button onclick="resetNeutral()">차렷 / Neutral (0)</button>
                </div>
                
                <!-- Isaac Sim Ground Truth Coordinate Legend -->
                <div class="tf-legend">
                    <div class="tf-title">🎯 Isaac Sim 손바닥 기준 3D 좌표계 (완벽 일치):</div>
                    <div class="tf-item"><div class="tf-dot" style="background:#ef4444;"></div> 🔴 <b>+X축 (Red)</b>: 엄지(Thumb) 방향</div>
                    <div class="tf-item"><div class="tf-dot" style="background:#22c55e;"></div> 🟢 <b>+Y축 (Green)</b>: 손바닥 정면 장풍 (Palm Skin Normal)</div>
                    <div class="tf-item"><div class="tf-dot" style="background:#3b82f6;"></div> 🔵 <b>+Z축 (Blue)</b>: 손목(Wrist) 방향 / <b>-Z축</b> = 4손가락 뻗음</div>
                </div>
            </div>

            <!-- Arm 7-DoF Group -->
            <div class="card">
                <div class="card-header">
                    <span>🦾 7-DoF 로봇 팔 (Fabrics Task-Space)</span>
                    <span class="section-tag">Joint 1 ~ 7</span>
                </div>
                <div id="arm-slider-container"></div>
            </div>

            <!-- Hand 20-DoF Group -->
            <div class="card">
                <div class="card-header collapse-header" onclick="toggleHandGroup()">
                    <span>🖐️ 20-DoF 테솔로 핸드 (Grasping)</span>
                    <span id="hand-toggle-icon">▼ 펼치기/접기</span>
                </div>
                <div id="hand-slider-container" style="margin-top:10px;"></div>
            </div>
        </div>

        <!-- 2. Palm EE & Alignment Tab -->
        <div id="tab-palm_ee" class="tab-content">
            <div class="card">
                <div class="card-header"><span>🎯 Isaac Sim Palm EE (손바닥 중심점)</span></div>
                <div class="prop-row"><span class="prop-key">현재 3D 위치 (X, Y, Z):</span><span class="prop-val" id="disp-palm-ee-pos">로딩 중...</span></div>
                <div class="prop-row"><span class="prop-key">현재 3D 회전 (RPY):</span><span class="prop-val" id="disp-palm-ee-rot">로딩 중...</span></div>
                <div class="prop-row"><span class="prop-key">🟢 손바닥 장풍 벡터 (+Y):</span><span class="prop-val" id="disp-palm-normal">로딩 중...</span></div>
                <div class="prop-row"><span class="prop-key">🔴 엄지손가락 벡터 (+X):</span><span class="prop-val" id="disp-thumb-vec">로딩 중...</span></div>
                <div class="prop-row"><span class="prop-key">🔵 4손가락 뻗음 벡터 (-Z):</span><span class="prop-val" id="disp-finger-vec">로딩 중...</span></div>
            </div>

            <div class="card">
                <div class="card-header"><span>🎯 컵(Target Cup) 대면 정렬 점수</span></div>
                <div class="prop-row"><span class="prop-key">목표 컵 위치 (RL Demo):</span><span class="prop-val">[0.270, -0.100, 0.380] m</span></div>
                <div class="prop-row"><span class="prop-key">손바닥-컵 거리:</span><span class="prop-val val-highlight" id="disp-dist-cup">0.000 m</span></div>
                <div class="prop-row"><span class="prop-key">파지 접근 정렬 점수 (Dot Product):</span><span class="prop-val val-good" id="disp-align-score">0.00</span></div>
                <p style="font-size:0.7rem; color:#94a3b8; margin-top:8px;">
                    * 🟢+Y 손바닥 장풍 벡터가 컵을 똑바로 마주볼 때 점수가 +1.0 (최적의 파지)이 됩니다.
                </p>
            </div>
        </div>

        <!-- 3. Collision Spheres Tab -->
        <div id="tab-spheres" class="tab-content">
            <div class="card">
                <div class="card-header"><span>🟡 Fabrics 충돌 구체 (_sphere) 목록</span></div>
                <p style="font-size:0.72rem; color:#94a3b8; margin-bottom:10px;">
                    3D 화면의 반투명 주황색 구체들이 실제 로봇 외형 및 컵과의 간섭 여부를 결정하는 충돌체입니다.
                </p>
                <div id="spheres-list-container" style="font-size:0.75rem;"></div>
            </div>
        </div>

        <!-- 4. URDF Properties Tab -->
        <div id="tab-properties" class="tab-content">
            <div class="card">
                <div class="card-header"><span>📊 로봇 기본 정보 (URDF)</span></div>
                <div class="prop-row"><span class="prop-key">로봇 모델명:</span><span class="prop-val">{robot_name}</span></div>
                <div class="prop-row"><span class="prop-key">설정 자유도(nq):</span><span class="prop-val">{nq}</span></div>
                <div class="prop-row"><span class="prop-key">속도 자유도(nv):</span><span class="prop-val">{nv}</span></div>
                <div class="prop-row"><span class="prop-key">총 관절 수:</span><span class="prop-val">{njoints}</span></div>
                <div class="prop-row"><span class="prop-key">총 프레임 수:</span><span class="prop-val">{nframes}</span></div>
            </div>
            <div class="card">
                <div class="card-header"><span>🌐 전체 프레임 목록</span></div>
                <div id="all-frames-container" style="font-size:0.72rem; max-height:300px; overflow-y:auto;"></div>
            </div>
        </div>
    </div>

    <script>
        const robotMeta = """ + json.dumps(meta) + """;
        const ws = new WebSocket("ws://" + location.host + "/ws");

        let currentQ = robotMeta.joints.map(j => j.value);

        function initUI() {
            const armContainer = document.getElementById("arm-slider-container");
            const handContainer = document.getElementById("hand-slider-container");
            armContainer.innerHTML = "";
            handContainer.innerHTML = "";

            robotMeta.joints.forEach(j => {
                if (j.nq > 0) {
                    const div = document.createElement("div");
                    div.className = "joint-item";
                    const degLower = (j.lower * 180 / Math.PI).toFixed(1);
                    const degUpper = (j.upper * 180 / Math.PI).toFixed(1);
                    const degVal = (j.value * 180 / Math.PI).toFixed(1);
                    
                    div.innerHTML = `
                        <div class="joint-header">
                            <span class="joint-name">${j.name}</span>
                            <span class="joint-val" id="val-${j.idx_q}">${j.value.toFixed(2)} rad (${degVal}°)</span>
                        </div>
                        <input type="range" id="slider-${j.idx_q}" 
                               min="${j.lower}" max="${j.upper}" step="0.01" value="${j.value}"
                               oninput="onSliderChange(${j.idx_q}, this.value)">
                        <div class="joint-limits">
                            <span>Min: ${degLower}°</span>
                            <span>Type: ${j.type}</span>
                            <span>Max: ${degUpper}°</span>
                        </div>
                    `;
                    if (j.group === "arm") {
                        armContainer.appendChild(div);
                    } else {
                        handContainer.appendChild(div);
                    }
                }
            });

            updatePalmEEDisplay(robotMeta.palm_ee);
            updateSpheresList(robotMeta.frames);
            updateAllFramesList(robotMeta.frames);
        }

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('tab-' + tabId).classList.add('active');
        }

        function toggleHandGroup() {
            const el = document.getElementById("hand-slider-container");
            el.style.display = (el.style.display === "none") ? "block" : "none";
        }

        function onSliderChange(idx_q, value) {
            currentQ[idx_q] = parseFloat(value);
            const deg = (currentQ[idx_q] * 180 / Math.PI).toFixed(1);
            document.getElementById(`val-${idx_q}`).innerText = `${parseFloat(value).toFixed(2)} rad (${deg}°)`;

            ws.send(JSON.stringify({
                action: "set_q",
                q: currentQ
            }));
        }

        function resetNeutral() {
            currentQ.fill(0.0);
            updateAllSliders();
        }

        function setPoseReady() {
            currentQ.fill(0.0);
            if (currentQ.length >= 7) {
                currentQ[0] = 0.25;
                currentQ[1] = 0.75;
                currentQ[3] = 1.35;
                currentQ[5] = 0.50;
            }
            if (currentQ.length >= 27) {
                for (let i = 7; i < currentQ.length; i++) currentQ[i] = 0.30;
            }
            updateAllSliders();
        }

        function setPoseReach() {
            currentQ.fill(0.0);
            if (currentQ.length >= 7) {
                currentQ[0] = 0.15;
                currentQ[1] = 0.95;
                currentQ[3] = 1.65;
                currentQ[5] = 0.65;
            }
            if (currentQ.length >= 27) {
                for (let i = 7; i < currentQ.length; i++) currentQ[i] = 0.15;
            }
            updateAllSliders();
        }

        function setPoseGrasp() {
            currentQ.fill(0.0);
            if (currentQ.length >= 7) {
                currentQ[0] = 0.15;
                currentQ[1] = 0.95;
                currentQ[3] = 1.65;
                currentQ[5] = 0.65;
            }
            if (currentQ.length >= 27) {
                for (let i = 7; i < currentQ.length; i++) currentQ[i] = 0.85;
            }
            updateAllSliders();
        }

        function updateAllSliders() {
            robotMeta.joints.forEach(j => {
                if (j.nq > 0) {
                    const slider = document.getElementById(`slider-${j.idx_q}`);
                    if (slider) {
                        slider.value = currentQ[j.idx_q];
                        const deg = (currentQ[j.idx_q] * 180 / Math.PI).toFixed(1);
                        document.getElementById(`val-${j.idx_q}`).innerText = `${currentQ[j.idx_q].toFixed(2)} rad (${deg}°)`;
                    }
                }
            });
            ws.send(JSON.stringify({ action: "set_q", q: currentQ }));
        }

        function updatePalmEEDisplay(ee) {
            if (!ee) return;
            document.getElementById("disp-palm-ee-pos").innerText = `[${ee.pos.join(", ")}] m`;
            document.getElementById("disp-palm-ee-rot").innerText = `[${ee.rot.join(", ")}]°`;
            document.getElementById("disp-palm-normal").innerText = `[${ee.palm_normal.join(", ")}]`;
            document.getElementById("disp-thumb-vec").innerText = `[${ee.thumb_vec.join(", ")}]`;
            document.getElementById("disp-finger-vec").innerText = `[${ee.finger_vec.join(", ")}]`;
            document.getElementById("disp-dist-cup").innerText = `${ee.dist_to_cup.toFixed(3)} m`;
            
            const elAlign = document.getElementById("disp-align-score");
            elAlign.innerText = `${ee.align_score.toFixed(3)}`;
            if (ee.align_score > 0.85) {
                elAlign.className = "prop-val val-good";
            } else if (ee.align_score > 0.5) {
                elAlign.className = "prop-val val-warn";
            } else {
                elAlign.className = "prop-val";
            }
        }

        function updateSpheresList(frames) {
            const container = document.getElementById("spheres-list-container");
            container.innerHTML = "";
            frames.filter(f => f.is_sphere && !f.name.includes("neg") && !f.name.includes("palm_x") && !f.name.includes("palm_y") && !f.name.includes("palm_z")).forEach(f => {
                const div = document.createElement("div");
                div.style.padding = "4px 0";
                div.style.borderBottom = "1px solid #1e293b";
                div.innerHTML = `🟡 <b>${f.name}</b> <span style="color:#64748b;">(Parent: ${f.parent_joint})</span><br>
                                 <span style="color:#94a3b8;">Pos: [${f.pos.join(", ")}]</span>`;
                container.appendChild(div);
            });
        }

        function updateAllFramesList(frames) {
            const container = document.getElementById("all-frames-container");
            container.innerHTML = "";
            frames.forEach(f => {
                const div = document.createElement("div");
                div.style.padding = "3px 0";
                div.style.borderBottom = "1px solid #1e293b";
                div.innerHTML = `<b>[#${f.id}] ${f.name}</b> | <span style="color:#94a3b8;">Pos: [${f.pos.join(", ")}]</span>`;
                container.appendChild(div);
            });
        }

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.type === "fk_update") {
                updatePalmEEDisplay(data.palm_ee);
                updateSpheresList(data.frames);
            }
        };

        window.onload = initUI;
    </script>
</body>
</html>"""
        rendered = html_template.replace("{robot_name}", meta["name"])\
                                .replace("{nq}", str(meta["nq"]))\
                                .replace("{nv}", str(meta["nv"]))\
                                .replace("{njoints}", str(meta["njoints"]))\
                                .replace("{nframes}", str(meta["nframes"]))\
                                .replace("{meshcat_url}", meta["meshcat_url"])
        self.write(rendered)


class WSHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        WS_CLIENTS.add(self)

    def on_close(self):
        WS_CLIENTS.discard(self)

    def on_message(self, message):
        data = json.loads(message)
        if data.get("action") == "set_q":
            q_arr = np.array(data["q"], dtype=np.float64)
            ROBOT_SERVER.update_kinematics(q_arr)

            frames_update = []
            for f_id in range(ROBOT_SERVER.model.nframes):
                f = ROBOT_SERVER.model.frames[f_id]
                T = ROBOT_SERVER.data.oMf[f_id]
                pos = [round(float(x), 4) for x in T.translation]
                rot = [round(float(x), 2) for x in Rotation.from_matrix(T.rotation).as_euler('xyz', degrees=True)]
                frames_update.append({
                    "id": f_id,
                    "name": f.name,
                    "parent_joint": ROBOT_SERVER.model.names[f.parentJoint] if f.parentJoint < len(ROBOT_SERVER.model.names) else "none",
                    "pos": pos,
                    "rot": rot,
                    "is_sphere": "sphere" in f.name.lower()
                })

            palm_ee_info = ROBOT_SERVER.get_palm_ee_info()

            self.write_message(json.dumps({
                "type": "fk_update",
                "frames": frames_update,
                "palm_ee": palm_ee_info
            }))


def main():
    global ROBOT_SERVER
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_urdf = os.path.join(
        repo_root, "source", "FABRICS", "src", "fabrics_sim", "models", "robots", "urdf",
        "openarm_tesollo", "openarm_tesollo.urdf"
    )

    parser = argparse.ArgumentParser(description="Isaac Lab Style Interactive Robot Kinematics Inspector")
    parser.add_argument("urdf", nargs="?", default=default_urdf, help="Path to URDF file")
    parser.add_argument("--port", type=int, default=8080, help="Web Studio Port")

    args = parser.parse_args()

    ROBOT_SERVER = RobotKinematicsServer(args.urdf, repo_root)

    app = tornado.web.Application([
        (r"/", MainHandler),
        (r"/ws", WSHandler),
    ])

    app.listen(args.port)
    studio_url = f"http://127.0.0.1:{args.port}"

    print("\n" + "=" * 75)
    print(f"[SUCCESS] Isaac Sim Ground-Truth Kinematics Studio 구동 완료!")
    print(f"          접속 주소: {studio_url}")
    print("-" * 75)
    print("  Isaac Sim 손바닥 기준 3D 좌표계 완벽 정합 완료:")
    print("   🔴 +X축 (Red)   : 엄지손가락 (Thumb) 방향")
    print("   🟢 +Y축 (Green) : 손바닥 피부 정면 (장풍 방향 / Palm Skin Normal)")
    print("   🔵 +Z축 (Blue)  : 손목 (Wrist) 방향 / -Z축 = 4손가락 뻗음 (Finger extension)")
    print("=" * 75 + "\n")

    import webbrowser
    webbrowser.open(studio_url)

    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
