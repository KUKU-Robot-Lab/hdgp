#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""grasp_v2용 RH56F1 손 PCA5 basis 계산.

inspire grasp 시연(grasp_ref_inspire.pkl, CrossDex retargeting, hand_qpos 22x6)을
RH56F1 6-drive 관절 순서·한계로 remap → sklearn PCA5 → fabric pca_matrix(5x6) 저장.

CrossDex inspire hand_qpos 순서(inspire_hand_right.yml target_joint_names):
  [0]pinky [1]ring [2]middle [3]index [4]thumb_pitch(flex) [5]thumb_yaw(abduction)
RH56F1 r_hj drive 순서:
  [0]thumb_1(abd) [1]thumb_2(flex) [2]index_1 [3]middle_1 [4]ring_1 [5]pinky_1
"""
import numpy as np
import pickle
import torch
from pathlib import Path
from sklearn.decomposition import PCA

REPO = Path("/home/user/rl_ws/hdgp")
SRC = REPO / "assets/multi_obj/demograsp_references/grasp_ref_inspire.pkl"
OUT = REPO / "assets/multi_obj/demograsp_references/rh56f1_grasp_pca5.pt"

# RH56F1 6-drive 관절 한계 (grasp_v1 HAND_JOINT_LIMITS_MAX)
RH_MAX = np.array([2.0943951, 0.4745550, 1.5285594, 1.5285594, 1.5285594, 1.5285594])
RH_MIN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# inspire[j] → RH56F1[i] 인덱스 매핑
INSPIRE_TO_RH = [5, 4, 3, 2, 1, 0]  # RH[thumb1,thumb2,index,middle,ring,pinky] = inspire[yaw,pitch,index,middle,ring,pinky]


def main() -> None:
    d = pickle.load(open(SRC, "rb"))
    q_in = np.asarray(d["hand_qpos"], dtype=np.float64)  # (22,6) inspire order
    q = q_in[:, INSPIRE_TO_RH]                            # (22,6) RH56F1 order
    # inspire 각 관절을 [관측 min,max] → [0,1] 정규화 후 RH56F1 [0,limit]로 스케일.
    # (inspire 손 kinematics ≠ RH56F1이라 절대값 직접전이 불가 → 시너지(상대변화)를 RH 범위에 매핑.)
    col_min, col_max = q.min(axis=0), q.max(axis=0)
    span = np.clip(col_max - col_min, 1e-6, None)
    q_norm = (q - col_min) / span                        # [0,1] per joint
    q_rh = RH_MIN + q_norm * (RH_MAX - RH_MIN)            # RH56F1 관절값(rad)
    q_rh = np.clip(q_rh, RH_MIN, RH_MAX)

    pca = PCA(n_components=5)
    z = pca.fit_transform(q_rh)                           # (22,5) centered PCA 좌표
    recon = pca.inverse_transform(z)
    rms = float(np.sqrt(((recon - q_rh) ** 2).mean()))
    # fabric LinearMap 은 x = C @ q (mean 미차감, uncentered) 로 투영한다.
    # → env 의 PCA action 범위(HAND_PCA_MINS/MAXS)는 이 uncentered 투영 범위여야
    #   attractor target 공간과 정합한다(centered z 범위 쓰면 C@mean 만큼 어긋남).
    proj = q_rh @ pca.components_.T                       # (22,5) uncentered = fabric taskmap 공간
    action_mins = proj.min(axis=0)                        # (5,)
    action_maxs = proj.max(axis=0)                        # (5,)

    print("=== RH56F1 grasp PCA5 ===")
    print("설명분산비:", np.round(pca.explained_variance_ratio_, 4),
          "누적:", round(float(pca.explained_variance_ratio_.sum()), 4))
    print("재구성 RMS(rad):", round(rms, 5))
    print("RH56F1 관절값 범위(remap 후, rad):")
    nm = ["thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "pinky_1"]
    for i, n in enumerate(nm):
        print(f"  {n:9s} [{q_rh[:,i].min():.3f}, {q_rh[:,i].max():.3f}] (한계 {RH_MAX[i]:.3f})")
    # PC1 방향(주 시너지) — 각 관절이 함께 닫히는지
    print("PC1 방향(주 닫힘 시너지):", np.round(pca.components_[0], 3))
    print("  → 부호 정렬(닫힘=+): 4손가락+엄지 함께 닫히면 firm envelope 시너지")

    # fabric용 저장: pca_matrix(5x6)=components, mean(6), PCA 좌표 범위(mins/maxs)
    payload = {
        "pca_matrix": torch.tensor(pca.components_, dtype=torch.float32),   # (5,6) PCA→? (basis rows)
        "mean": torch.tensor(pca.mean_, dtype=torch.float32),               # (6,)
        "pca_mins": torch.tensor(z.min(axis=0), dtype=torch.float32),       # (5,) centered(참고용)
        "pca_maxs": torch.tensor(z.max(axis=0), dtype=torch.float32),       # (5,) centered(참고용)
        "pca_action_mins": torch.tensor(action_mins, dtype=torch.float32),  # (5,) uncentered=env action 범위
        "pca_action_maxs": torch.tensor(action_maxs, dtype=torch.float32),  # (5,) uncentered=env action 범위
        "explained_variance_ratio": torch.tensor(pca.explained_variance_ratio_, dtype=torch.float32),
        "grasps_rh_qpos": torch.tensor(q_rh, dtype=torch.float32),          # (22,6) remap된 grasp
        "meta": {
            "source": str(SRC), "inspire_to_rh": INSPIRE_TO_RH,
            "rh_order": nm, "scaling": "per-joint [obs_min,max]->[0,limit]",
        },
    }
    torch.save(payload, OUT)
    print("저장:", OUT)
    # 가장 닫힌 grasp (firm 후보)
    closure = q_rh.sum(axis=1)
    mi = int(closure.argmax())
    print(f"가장 닫힌 grasp #{mi} (RH rad):", np.round(q_rh[mi], 3))


if __name__ == "__main__":
    main()
