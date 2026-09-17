"""Validation of generated stage-2 reward code (IKER t2r fork) — static check + fake RewardContext dry run, no Isaac.

정적 검사(fail-loud):
  · 최상위 `def compute_reward(ctx)` 가 정확히 하나, 인자는 ctx 하나
  · import 는 torch / math / typing 만
  · exec / eval / open / __import__ / getattr 류 호출 금지
  · `ctx.<field>` 대입 금지 · 존재하지 않는 ctx 필드 접근 금지(환각 필드 — 가장 흔한 실패)
드라이런:
  · 무작위 텐서로 채운 RewardContext 로 3회 호출 → (N,) 유한값·terms (N,) 확인, 항별 크기 표
"""

from __future__ import annotations

import ast
import functools
from dataclasses import dataclass, field

import torch

from ..place_stage import PlaceRewardCfg
from .context import SCALAR_FIELDS, TENSOR_FIELDS, RewardContext
from .loader import ENTRY_NAME, call_reward_fn, load_reward_fn
from .. import grasp_stage as gs
from .. import layout

ALLOWED_IMPORTS = {"torch", "math", "typing"}
FORBIDDEN_CALLS = {"exec", "eval", "open", "__import__", "compile", "getattr", "setattr",
                   "delattr", "globals", "locals", "vars"}
NUM_ARM, NUM_ACTIONS, NUM_KEYPOINTS = 7, 7, 4
NUM_SURFACE = gs.SURFACE_POINT_COUNT


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


def _ctx_field(expr: ast.AST, ctx_name: str) -> str | None:
    """`ctx.<field>` 에서 시작하는 식이면 그 field 이름 — Attribute·Subscript 를 안쪽으로 따라간다."""
    while isinstance(expr, (ast.Attribute, ast.Subscript)):
        if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) and expr.value.id == ctx_name:
            return expr.attr
        expr = expr.value
    return None


def static_check(src: str) -> ValidationReport:
    rep = ValidationReport(ok=True)
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return ValidationReport(ok=False, errors=[f"SyntaxError: {e}"])
    entries = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == ENTRY_NAME]
    if len(entries) != 1:
        rep.ok = False
        rep.errors.append(f"최상위 `def {ENTRY_NAME}(ctx)` 가 정확히 하나여야 한다 (지금 {len(entries)})")
    elif len(entries[0].args.args) != 1 or entries[0].args.vararg or entries[0].args.kwarg:
        rep.ok = False
        rep.errors.append(f"`{ENTRY_NAME}` 인자는 (ctx) 하나여야 한다")
    ctx_name = entries[0].args.args[0].arg if entries and entries[0].args.args else "ctx"
    valid = set(TENSOR_FIELDS) | set(SCALAR_FIELDS) | {"num_envs", "device"}
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
            # ★제자리 연산(`ctx.x.clamp_()` · `ctx.x[:, 0].add_()`)은 AST Store 가 아니라 대입 검사에 안 걸린다(09.14 리뷰).
            if isinstance(fn, ast.Attribute) and fn.attr.endswith("_") and not fn.attr.startswith("_"):
                _f = _ctx_field(fn.value, ctx_name)
                if _f:
                    rep.ok = False
                    rep.errors.append(f"ctx 는 읽기 전용이다: ctx.{_f} 에 제자리 연산 .{fn.attr}()")
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)) \
                and _ctx_field(node.value, ctx_name):
            rep.ok = False
            rep.errors.append(f"ctx 는 읽기 전용이다: ctx.{_ctx_field(node.value, ctx_name)}[...] 대입")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == ctx_name:
            used.add(node.attr)
            if node.attr not in valid:
                rep.ok = False
                rep.errors.append(f"RewardContext 에 없는 필드: ctx.{node.attr}")
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                rep.ok = False
                rep.errors.append(f"ctx 는 읽기 전용이다: ctx.{node.attr} 대입")
    rep.fields_used = sorted(used)
    if not used:
        rep.warnings.append("ctx 필드를 하나도 읽지 않는다 — 상수 보상인가?")
    return rep


def make_fake_context(n: int = 16, *, device: str = "cpu", seed: int = 0) -> RewardContext:
    """Shape-correct random ctx for the dry run; no physical consistency."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    cfg = PlaceRewardCfg()

    def r(*shape, lo=-1.0, hi=1.0):
        return (torch.rand(*shape, generator=g) * (hi - lo) + lo).to(device)

    def unit(*shape):
        v = r(*shape)
        return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    return RewardContext(
        table_top_z=layout.TABLE_TOP_Z, rack_x_min=layout.RACK_X_RANGE[0], rack_x_max=layout.RACK_X_RANGE[1],
        rack_y_min=layout.RACK_Y_RANGE[0], rack_y_max=layout.RACK_Y_RANGE[1], rack_top_z=cfg.rack_top_z,
        episode_steps=200, control_dt=0.1,
        place_tolerance=cfg.place_tolerance, release_radius=cfg.release_radius, resting_tol=cfg.resting_tol,
        still_speed=cfg.still_speed, stable_steps=cfg.stable_steps, window_steps=cfg.window_steps,
        home_radius=cfg.home_radius,
        palm_pos=r(n, 3, lo=0.0, hi=0.6), palm_quat=unit(n, 4), palm_normal=unit(n, 3),
        home_palm_pos=r(n, 3, lo=0.0, hi=0.6), palm_home_dist=r(n, lo=0.0, hi=0.5),
        arm_q=r(n, NUM_ARM), arm_qd=r(n, NUM_ARM),
        grip_norm=r(n, lo=-1.0, hi=1.0),
        shoe_pos=r(n, 3, lo=0.0, hi=0.6), shoe_quat=unit(n, 4), shoe_lin_vel=r(n, 3), shoe_ang_vel=r(n, 3),
        shoe_surface=r(n, NUM_SURFACE, 3, lo=0.0, hi=0.6), shoe_bottom_z=r(n, lo=0.1, hi=0.4),
        palm_gap=r(n, lo=0.0, hi=0.2), palm_shoe_dist=r(n, lo=0.0, hi=0.3),
        target_keypoints=r(n, NUM_KEYPOINTS, 3, lo=0.0, hi=0.6), keypoints=r(n, NUM_KEYPOINTS, 3, lo=0.0, hi=0.6),
        init_keypoints=r(n, NUM_KEYPOINTS, 3, lo=0.0, hi=0.6), keypoint_err=r(n, NUM_KEYPOINTS, lo=0.0, hi=0.3),
        keypoint_dist=r(n, lo=0.0, hi=0.3),
        placed=r(n) > 0.0, released=r(n) > 0.0, resting=r(n) > 0.0, still=r(n) > 0.0, home=r(n) > 0.0,
        stable_count=torch.floor(r(n, lo=0.0, hi=25.0)), success=r(n) > 0.8,
        episode_progress=r(n, lo=0.0, hi=1.0), actions=r(n, NUM_ACTIONS), prev_actions=r(n, NUM_ACTIONS),
    )


@functools.lru_cache(maxsize=1)
def cuda_usable() -> bool:
    """cuda 로 **실제로** 연산이 되는가.

    ★`torch.cuda.is_available()` 은 커널 없는 GPU 에서도 True 다 — 09.14 이 호스트의 시스템 python torch 는
      RTX 5090(sm_120) 커널이 없어 `no kernel image is available` 로 죽었다(Isaac python 의 torch 는 된다).
    """
    if not torch.cuda.is_available():
        return False
    try:
        return float((torch.ones(2, device="cuda") * 2.0).sum()) == 4.0
    except RuntimeError:
        return False


def dry_run(path: str, *, n: int = 37, device: str = "cpu") -> ValidationReport:
    # finding 5: 37, not the round smoke's env count (64) — a reward that hard-codes the batch size must fail both checks
    rep = ValidationReport(ok=True)
    try:
        fn, _ = load_reward_fn(path)
    except Exception as e:  # noqa: BLE001 — 로더 실패 자체가 보고 대상
        return ValidationReport(ok=False, errors=[f"로드 실패: {type(e).__name__}: {e}"])
    for seed in range(3):
        try:
            ctx = make_fake_context(n, device=device, seed=seed)
        except Exception as e:  # noqa: BLE001 — 장치 자체가 안 되는 경우도 예외가 아니라 보고로 낸다
            return ValidationReport(ok=False, errors=[f"가짜 ctx 생성 실패({device}): {type(e).__name__}: {e}"])
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


def validate(path: str, devices: tuple[str, ...] | None = None) -> ValidationReport:
    """정적 검사 → 드라이런. 기본 장치 = cpu + (있으면) cuda.

    ★cuda 드라이런이 필요한 이유(09.14 리뷰): device 없이 만든 상수(`torch.tensor([...])`)는 cpu 가짜 ctx 에서
      통과하고 학습 첫 스텝(cuda)에서 장치 불일치로 죽는다.
    """
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    rep = static_check(src)
    if not rep.ok:
        return rep
    if devices is None:
        devices = ("cpu", "cuda") if cuda_usable() else ("cpu",)
    if "cuda" not in devices:
        rep.warnings.append("cuda 드라이런 생략(연산 가능한 cuda 없음) — Isaac python 으로 ingest 하거나 Isaac 스모크에서 확인")
    for i, dev in enumerate(devices):
        dr = dry_run(path, device=dev)
        rep.ok = rep.ok and dr.ok
        rep.errors += [f"[{dev}] {e}" for e in dr.errors]
        rep.warnings += [w for w in dr.warnings if w not in rep.warnings]
        if i == 0:
            rep.term_stats, rep.total_stats = dr.term_stats, dr.total_stats
    return rep


__all__ = ["ValidationReport", "static_check", "dry_run", "validate", "make_fake_context", "cuda_usable"]
