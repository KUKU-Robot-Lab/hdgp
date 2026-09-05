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

"""pour 체크포인트의 **액션 헤드를 12D → 15D 로 0-패딩**한다 (좌팔 TCP 3채널 추가).

왜 되는가. `left_arm_action_enable` 은 **관측을 안 바꾼다**(55/144 동일). 액션만
뒤에 3채널이 붙는다. 그래서 출력 헤드 세 파라미터만 늘리면 앞 12채널은 학습된 그대로
동작한다. 새 3채널의 mu 를 0 으로 두면 좌팔 TCP delta 가 0 → **좌팔이 rest 에 머문다**
= 좌팔 고정으로 학습한 현재 정책과 **완전히 같은 거동**에서 출발한다.

늘려야 하는 것 (rl_games a2c_continuous + LSTM):
    model  a2c_network.sigma      (12,)      → (15,)
           a2c_network.mu.weight  (12, 512)  → (15, 512)   새 행 = 0
           a2c_network.mu.bias    (12,)      → (15,)       새 값 = 0
    optimizer state 의 exp_avg / exp_avg_sq 도 **같은 모양**으로 늘린다
      (0 = Adam 히스토리 없음 = 새 파라미터에 맞는 값). 안 늘리면 resume 이
      shape mismatch 로 죽는다.

★새 채널의 sigma 를 얼마로 둘지가 실질적 선택이다. 0(=σ 1.0)이면 첫 스텝부터 좌팔이
  크게 흔들려 물려받은 정책을 망친다. 기본값 −1.0(σ≈0.37)은 palm 채널(−1.23~−0.78)과
  같은 대역이라 탐색은 하되 파괴적이지 않다.

실행:
    python3 scripts/tools/pad_pour_ckpt_left_action.py \\
        --in  <12D ckpt> --out <15D ckpt> [--new_logstd -1.0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

MODEL_KEYS = ("a2c_network.sigma", "a2c_network.mu.weight", "a2c_network.mu.bias")


def _pad_rows(t: torch.Tensor, n_new: int, fill: float) -> torch.Tensor:
    """1축(행)을 n_new 만큼 늘리고 fill 로 채운다."""
    shape = list(t.shape)
    shape[0] = n_new
    return torch.cat([t, torch.full(shape, fill, dtype=t.dtype, device=t.device)], dim=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--n_new", type=int, default=3, help="추가 액션 채널 수")
    ap.add_argument(
        "--new_logstd", type=float, default=-1.0,
        help="새 채널의 log-std. 0=σ1.0(과격) · -1.0=σ0.37(palm 대역, 기본)",
    )
    args = ap.parse_args()

    src, dst = Path(args.src).expanduser(), Path(args.dst).expanduser()
    if not src.is_file():
        raise SystemExit(f"입력 체크포인트가 없다: {src}")
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck["model"]

    old_n = int(sd["a2c_network.mu.bias"].shape[0])
    new_n = old_n + args.n_new
    print(f"액션 {old_n}D → {new_n}D  (추가 {args.n_new}채널, log-std {args.new_logstd})")

    # ---- 모델 ----
    for k in MODEL_KEYS:
        if k not in sd:
            raise SystemExit(f"체크포인트에 '{k}' 가 없다 — 네트워크 구조가 다르다")
        fill = args.new_logstd if k.endswith("sigma") else 0.0
        before = tuple(sd[k].shape)
        sd[k] = _pad_rows(sd[k], args.n_new, fill)
        print(f"  model  {k:32s} {before} → {tuple(sd[k].shape)}  fill={fill}")

    # ---- 옵티마이저 (shape 이 액션 차원인 항목만) ----
    opt = ck.get("optimizer")
    if opt is None:
        print("  ⚠optimizer 상태 없음 — resume 시 rl_games 가 KeyError 를 낼 수 있다")
    else:
        n_pad = 0
        for idx, entry in opt.get("state", {}).items():
            for field in ("exp_avg", "exp_avg_sq"):
                t = entry.get(field)
                if t is None or not hasattr(t, "shape") or t.ndim == 0:
                    continue
                if int(t.shape[0]) != old_n:
                    continue
                entry[field] = _pad_rows(t, args.n_new, 0.0)
                n_pad += 1
        print(f"  optimizer  액션차원 텐서 {n_pad}개 패딩(0 = Adam 히스토리 없음)")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(ck, tmp)
    tmp.replace(dst)

    # ---- 검증: 다시 읽어 모양 확인 ----
    chk = torch.load(dst, map_location="cpu", weights_only=False)["model"]
    for k in MODEL_KEYS:
        assert int(chk[k].shape[0]) == new_n, f"{k} 패딩 실패"
    tail = chk["a2c_network.mu.bias"][-args.n_new:]
    w_tail = chk["a2c_network.mu.weight"][-args.n_new:]
    print(f"\n검증 — 새 채널 mu.bias {[round(float(v),4) for v in tail]} · "
          f"mu.weight 절대합 {float(w_tail.abs().sum()):.6f} (0 이어야 좌팔이 rest 에 머문다)")
    print(f"       새 채널 sigma {[round(float(v),4) for v in chk['a2c_network.sigma'][-args.n_new:]]}")
    print(f"저장 {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
