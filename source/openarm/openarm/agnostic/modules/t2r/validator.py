"""생성된 보상 코드 검증 — 정적 검사 + 가짜 RewardContext 드라이런 (Isaac 불요).

정적 검사(fail-loud):
  · `def compute_reward(ctx)` 가 최상위에 정확히 하나
  · import 는 torch / math / typing 만
  · exec / eval / open / __import__ / getattr(문자열 우회) 호출 금지
  · `ctx.<field> = …` 대입·`ctx` 속성 변경 금지(frozen 이라 런타임에도 죽지만 먼저 잡는다)
  · 존재하지 않는 ctx 필드 접근 금지(오타·환각 필드 — 가장 흔한 실패)
드라이런:
  · 무작위 텐서로 채운 RewardContext(N,F,A,Dact 는 인자)로 호출 → (N,) 유한값·terms (N,) 확인
  · 항별 크기 표(mean / |max|)를 돌려준다 — audit 의 첫 재료
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field

import torch

from .context import RewardContext, SCALAR_FIELDS, TENSOR_FIELDS
from .loader import ENTRY_NAME, call_reward_fn, load_reward_fn

ALLOWED_IMPORTS = {"torch", "math", "typing"}
FORBIDDEN_CALLS = {"exec", "eval", "open", "__import__", "compile", "getattr", "setattr",
                   "delattr", "globals", "locals", "vars"}


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    term_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    total_stats: dict[str, float] = field(default_factory=dict)
    fields_used: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings,
                "term_stats": self.term_stats, "total_stats": self.total_stats,
                "fields_used": self.fields_used}


def static_check(src: str) -> ValidationReport:
    rep = ValidationReport(ok=True)
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        rep.ok = False
        rep.errors.append(f"SyntaxError: {e}")
        return rep
    entries = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == ENTRY_NAME]
    if len(entries) != 1:
        rep.ok = False
        rep.errors.append(f"최상위 `def {ENTRY_NAME}(ctx)` 가 정확히 하나여야 한다 (지금 {len(entries)})")
    else:
        args = entries[0].args
        if len(args.args) != 1 or args.vararg or args.kwarg:
            rep.ok = False
            rep.errors.append(f"`{ENTRY_NAME}` 인자는 (ctx) 하나여야 한다")
    ctx_name = entries[0].args.args[0].arg if entries and entries[0].args.args else "ctx"
    valid_fields = set(TENSOR_FIELDS) | set(SCALAR_FIELDS) | {"num_envs", "device"}
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = ([a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import)
                    else [(node.module or "").split(".")[0]])
            for m in mods:
                if m not in ALLOWED_IMPORTS:
                    rep.ok = False
                    rep.errors.append(f"허용되지 않은 import: {m}")
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
            if name in FORBIDDEN_CALLS:
                rep.ok = False
                rep.errors.append(f"금지 호출: {name}()")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == ctx_name:
            used.add(node.attr)
            if node.attr not in valid_fields:
                rep.ok = False
                rep.errors.append(f"RewardContext 에 없는 필드: ctx.{node.attr}")
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                rep.ok = False
                rep.errors.append(f"ctx 는 읽기 전용이다: ctx.{node.attr} 대입")
    rep.fields_used = sorted(used)
    if not used:
        rep.warnings.append("ctx 필드를 하나도 읽지 않는다 — 상수 보상인가?")
    return rep


def make_fake_context(n: int = 16, *, num_fingers: int = 5, num_arm: int = 7,
                      num_actions: int = 42, device: str = "cpu", seed: int = 0) -> RewardContext:
    """형태만 맞춘 무작위 ctx. 값의 물리적 정합은 보장하지 않는다(드라이런 전용)."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape, lo=-1.0, hi=1.0):
        return (torch.rand(*shape, generator=g) * (hi - lo) + lo).to(device)

    def unit(*shape):
        v = r(*shape)
        return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    kw: dict = dict(table_z=0.205, cup_radius=0.041, cup_mouth_z=0.10, cup_bottom_z=-0.0773,
                    num_beads=20)
    for side in ("src", "rcv"):
        kw[f"{side}_palm_pos"] = r(n, 3, lo=0.0, hi=0.6)
        kw[f"{side}_palm_axes"] = torch.cat([unit(n, 3), unit(n, 3)], dim=1)
        kw[f"{side}_tips_pos"] = r(n, num_fingers, 3, lo=0.0, hi=0.6)
        kw[f"{side}_hand_closure"] = r(n, lo=0.0, hi=1.0)
        kw[f"{side}_finger_force"] = r(n, num_fingers, lo=0.0, hi=5.0)
        kw[f"{side}_palm_force"] = r(n, lo=0.0, hi=5.0)
        kw[f"{side}_grasped"] = r(n) > 0.0
        kw[f"{side}_arm_qd"] = r(n, num_arm)
        kw[f"{side}_cup_pos"] = r(n, 3, lo=0.0, hi=0.6)
        q = unit(n, 4)
        kw[f"{side}_cup_quat"] = q
        up = unit(n, 3)
        kw[f"{side}_cup_up"] = up
        kw[f"{side}_cup_tilt"] = torch.acos(up[:, 2].clamp(-1, 1))
        kw[f"{side}_cup_mouth_pos"] = r(n, 3, lo=0.0, hi=0.6)
        kw[f"{side}_cup_lin_vel"] = r(n, 3)
        kw[f"{side}_cup_ang_vel"] = r(n, 3)
        kw[f"{side}_cup_spawn_pos"] = r(n, 3, lo=0.0, hi=0.6)
    fr = r(n, 3, lo=0.0, hi=1.0)
    fr = fr / fr.sum(dim=1, keepdim=True)
    kw.update(bead_in_source_frac=fr[:, 0], bead_in_target_frac=fr[:, 1], bead_spill_frac=fr[:, 2],
              bead_centroid=r(n, 3, lo=0.0, hi=0.6), d_in_target=r(n, lo=-0.1, hi=0.1),
              d_spill=r(n, lo=-0.1, hi=0.1), success=r(n) > 0.8,
              episode_progress=r(n, lo=0.0, hi=1.0),
              actions=r(n, num_actions), prev_actions=r(n, num_actions))
    return RewardContext(**kw)


def dry_run(path: str, *, n: int = 64, device: str = "cpu", **ctx_kw) -> ValidationReport:
    rep = ValidationReport(ok=True)
    try:
        fn, _ = load_reward_fn(path)
    except Exception as e:  # noqa: BLE001 — 로더 실패 자체가 보고 대상
        rep.ok = False
        rep.errors.append(f"로드 실패: {type(e).__name__}: {e}")
        return rep
    for seed in range(3):
        ctx = make_fake_context(n, device=device, seed=seed, **ctx_kw)
        try:
            total, terms = call_reward_fn(fn, ctx)
        except Exception as e:  # noqa: BLE001
            rep.ok = False
            rep.errors.append(f"호출 실패(seed {seed}): {type(e).__name__}: {e}")
            return rep
        if not torch.isfinite(total).all():
            rep.ok = False
            rep.errors.append(f"reward 에 NaN/Inf (seed {seed})")
        for k, v in terms.items():
            if not torch.isfinite(v).all():
                rep.ok = False
                rep.errors.append(f"terms['{k}'] 에 NaN/Inf (seed {seed})")
            st = rep.term_stats.setdefault(k, {"mean": 0.0, "abs_max": 0.0})
            st["mean"] += float(v.mean()) / 3.0
            st["abs_max"] = max(st["abs_max"], float(v.abs().max()))
        rep.total_stats["mean"] = rep.total_stats.get("mean", 0.0) + float(total.mean()) / 3.0
        rep.total_stats["abs_max"] = max(rep.total_stats.get("abs_max", 0.0), float(total.abs().max()))
    if not rep.term_stats:
        rep.warnings.append("terms 가 비어 있다 — 항별 로깅·피드백이 불가능하다")
    if rep.total_stats.get("abs_max", 0.0) > 1e3:
        rep.warnings.append(f"보상 크기가 크다(|max| {rep.total_stats['abs_max']:.1f}) — 스케일 확인")
    return rep


def validate(path: str, **ctx_kw) -> ValidationReport:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    rep = static_check(src)
    if not rep.ok:
        return rep
    dr = dry_run(path, **ctx_kw)
    rep.ok = dr.ok
    rep.errors += dr.errors
    rep.warnings += dr.warnings
    rep.term_stats, rep.total_stats = dr.term_stats, dr.total_stats
    return rep


__all__ = ["ValidationReport", "static_check", "dry_run", "validate", "make_fake_context", "math"]
