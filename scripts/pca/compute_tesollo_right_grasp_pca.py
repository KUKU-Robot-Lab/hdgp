#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""right grasp_v2 성공 파지 데이터 → tesollo 고유 PCA(eigengrasp) + 물체별 시그니처 추출.

compute_tesollo_left_grasp_pca.py 의 right 판. 차이:
  1. 입력이 이미 right 좌표(q_hand_right) — 부호맵 변환 불필요.
  2. 현행 basis(Allegro retarget) 비교에 더해 left 추출 basis
     (tesollo_grasp_pca5_from_left.pt) 와의 좌우 일관성 비교 추가.
출력:
  - assets/demograsp_references/tesollo_grasp_pca5_from_right.pt
  - docs/pca/right_grasp_pca_lstm_test1.md

실행: python3 scripts/pca/compute_tesollo_right_grasp_pca.py [--npz <path>]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[2]
LEFT_PKG = REPO / "source/openarm/openarm/tesollo/left/grasp_v2"
RIGHT_PKG = REPO / "source/openarm/openarm/tesollo/right/grasp_v2"
DEFAULT_NPZ = REPO / "data/right_grasp_poses_lstm_test1.npz"
LEFT_PT = REPO / "assets/demograsp_references/tesollo_grasp_pca5_from_left.pt"
OUT_PT = REPO / "assets/demograsp_references/tesollo_grasp_pca5_from_right.pt"
OUT_MD = REPO / "docs/pca/right_grasp_pca_lstm_test1.md"
N_PC = 5


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"module load 실패: {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _principal_angles_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """두 행공간(row-space) 간 주각. a,b: (k, d) 정규직교 행."""
    qa, _ = np.linalg.qr(a.T)
    qb, _ = np.linalg.qr(b.T)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def _recon_rms(q: np.ndarray, basis: np.ndarray, anchor: np.ndarray) -> float:
    """anchor 기준 basis 부분공간 재구성 RMS (rad)."""
    d = q - anchor
    z = d @ basis.T
    recon = anchor + z @ basis
    return float(np.sqrt(((recon - q) ** 2).mean()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default=str(DEFAULT_NPZ))
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=False)
    q_right = data["q_hand_right"].astype(np.float64)         # (M, 20) — right 좌표 그대로
    obj_idx = data["object_idx"]                              # (M,)
    names = [str(s) for s in data["object_names"]]
    if q_right.shape[0] < 100:
        raise SystemExit(f"샘플 부족({q_right.shape[0]}) — 수집을 먼저 확인하세요")

    # B_L = B_R·S 파생용 부호맵 (left env 적용 시 필요 — left 스크립트와 동일 규약)
    preset = _load_module(LEFT_PKG / "grasp_left_preset.py", "_left_preset_for_pca")
    sign = np.asarray(preset._HAND_SIGN, dtype=np.float64)
    assert sign.shape == (20,), "부호맵 차원 불일치"

    # --- 전역 PCA (centered) — 전체 20축을 뽑고 상위 N_PC 를 기본 뷰로 사용 -----
    pca = PCA(n_components=q_right.shape[1])
    pca.fit(q_right)
    basis_full = pca.components_                               # (20, 20) 정규직교 행
    evr_full = pca.explained_variance_ratio_                   # (20,)
    basis = basis_full[:N_PC]                                  # (5, 20) 기본 뷰
    mean = pca.mean_                                           # (20,)
    evr = evr_full[:N_PC]
    proj = q_right @ basis.T                                   # (M, 5) uncentered
    coeff_mins, coeff_maxs = proj.min(axis=0), proj.max(axis=0)
    recon_new = _recon_rms(q_right, basis, mean)

    # --- 현행 basis(Allegro retarget) 대비 -------------------------------------
    cur = _load_module(RIGHT_PKG / "tesollo_hand_synergy.py", "_right_synergy_for_pca")
    cur_basis = np.asarray(cur.HAND_SYNERGY_BASIS, dtype=np.float64)     # (5, 20)
    cur_anchor = np.asarray(cur.HAND_SYNERGY_ANCHOR, dtype=np.float64)   # (20,)
    recon_cur = _recon_rms(q_right, cur_basis, cur_anchor)
    angles_cur = _principal_angles_deg(basis, cur_basis)

    # --- left 추출 basis 와 좌우 일관성 비교 -------------------------------------
    angles_left = None
    recon_left_basis = None
    if LEFT_PT.exists():
        left_pt = torch.load(LEFT_PT, weights_only=False)
        left_basis = left_pt["basis_right"].numpy()            # right-canonical (5,20)
        left_mean = left_pt["mean_right"].numpy()
        angles_left = _principal_angles_deg(basis, left_basis)
        recon_left_basis = _recon_rms(q_right, left_basis, left_mean)

    # --- 물체별 시그니처 (신규 basis uncentered 투영) ---------------------------
    sig: dict[str, dict[str, list[float]]] = {}
    counts = np.bincount(obj_idx, minlength=len(names))
    for i, nm in enumerate(names):
        m = obj_idx == i
        if int(m.sum()) < 3:
            continue
        p = proj[m]
        sig[nm] = {"mean": p.mean(axis=0).tolist(), "std": p.std(axis=0).tolist(),
                   "n": [int(m.sum())]}

    # --- 저장 (.pt) -------------------------------------------------------------
    OUT_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "basis_right": torch.tensor(basis, dtype=torch.float64),        # (5,20) B_R
        "mean_right": torch.tensor(mean, dtype=torch.float64),
        "hand_sign_left": torch.tensor(sign, dtype=torch.float64),      # B_L = B_R·diag(S)
        "coeff_mins_uncentered": torch.tensor(coeff_mins, dtype=torch.float64),
        "coeff_maxs_uncentered": torch.tensor(coeff_maxs, dtype=torch.float64),
        "explained_variance_ratio": torch.tensor(evr, dtype=torch.float64),
        "basis_full_right": torch.tensor(basis_full, dtype=torch.float64),   # (20,20) 전체 축
        "explained_variance_ratio_full": torch.tensor(evr_full, dtype=torch.float64),
        "per_object_signature": sig,
        "meta": {
            "source_npz": str(args.npz),
            "checkpoint": str(data["checkpoint"]),
            "n_samples": int(q_right.shape[0]),
            "n_objects_covered": int((counts > 0).sum()),
            "adr_level": int(data["adr_level"]),
            "coord": "right-native",
            "projection": "uncentered x = B q (fabric LinearMap 정합)",
            "date": _dt.date.today().isoformat(),
        },
    }, OUT_PT)

    # --- 보고서 (md) -------------------------------------------------------------
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    top = sorted(sig.items(), key=lambda kv: kv[1]["mean"][0])
    lines = [
        f"# tesollo eigengrasp — right lstm_test1(multiobj) 성공 파지 역추론 ({_dt.date.today().isoformat()})",
        "",
        f"- 원료: `{Path(args.npz).name}` — 샘플 {q_right.shape[0]}개, "
        f"물체 {int((counts > 0).sum())}/{len(names)}종, ADR {int(data['adr_level'])}, "
        f"checkpoint `{Path(str(data['checkpoint'])).name}`",
        "- 좌표 규약: right-native. left 용은 BASIS_L = B_R·S "
        "(tesollo_hand_synergy.py 규약과 동일, 계수 공간 불변)",
        "",
        "## 전역 PCA5 (신규, 성공 파지 기반)",
        f"- 설명분산비: {[round(float(v), 4) for v in evr]}  (누적 {float(evr.sum()):.4f})",
        f"- 전체 스펙트럼 누적: 5축 {float(evr_full[:5].sum()):.3f} / 8축 "
        f"{float(evr_full[:8].sum()):.3f} / 10축 {float(evr_full[:10].sum()):.3f}",
        f"- 재구성 RMS: **신규 {recon_new:.4f} rad** vs 현행(Allegro retarget) {recon_cur:.4f} rad",
        f"- uncentered 계수 범위 mins: {[round(float(v), 3) for v in coeff_mins]}",
        f"- uncentered 계수 범위 maxs: {[round(float(v), 3) for v in coeff_maxs]}",
        "",
        "## 현행 basis 와의 부분공간 비교",
        f"- 주각(deg): {[round(float(a), 1) for a in angles_cur]}",
    ]
    if angles_left is not None:
        lines += [
            "",
            "## left 추출 basis(tesollo_grasp_pca5_from_left) 와의 좌우 일관성",
            f"- 주각(deg): {[round(float(a), 1) for a in angles_left]}",
            f"- left basis 로 right 데이터 재구성 RMS: {recon_left_basis:.4f} rad",
            "  (주각이 작을수록 좌우 정책이 같은 eigengrasp 구조에 수렴했다는 뜻)",
        ]
    lines += [
        "",
        "## 물체별 시그니처 (신규 basis, PC1 mean 오름차순 상·하위)",
        "| 물체 | n | PC1 | PC2 | PC3 | PC4 | PC5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for nm, s in top[:10] + top[-10:]:
        m = s["mean"]
        lines.append(f"| {nm} | {s['n'][0]} | " + " | ".join(f"{v:+.3f}" for v in m) + " |")
    lines += [
        "",
        f"- 전체 물체별 mean/std 는 `{OUT_PT.relative_to(REPO)}` 의 per_object_signature 에 저장.",
        "",
        "## 사용법",
        "- 신규 basis 로 교체 시 계수 공간이 바뀌므로 **정책 재학습 필요** (분석 용도는 무해).",
        "- left env 적용은 tesollo_hand_synergy.py 규약대로 부호맵 파생만 하면 됨.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] basis 저장: {OUT_PT}")
    print(f"[OK] 보고서: {OUT_MD}")
    print(f"  설명분산 {[round(float(v),3) for v in evr]} | 재구성 RMS 신규 {recon_new:.4f} vs 현행 {recon_cur:.4f}")
    print(f"  주각 vs 현행(deg): {[round(float(a),1) for a in angles_cur]}")
    if angles_left is not None:
        print(f"  주각 vs left추출(deg): {[round(float(a),1) for a in angles_left]} | left basis 재구성 {recon_left_basis:.4f}")


if __name__ == "__main__":
    main()
