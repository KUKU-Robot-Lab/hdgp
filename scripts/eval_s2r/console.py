"""인터랙티브 세션 명령 파서·상태 전이 (순수 함수, Isaac 무관).

명령: spawn X Y [Z] | repeat | reset | back2ini | obj N | sweep XMIN XMAX NX YMIN YMAX NY REPEATS | quit
설계: docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp1-design.md §4.6
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from scripts.eval_s2r.grid import GridSpec

NUM_OBJECTS = 8  # grasp_v1 MultiAsset 물체 수 (env_id % 8 배정)


@dataclass(frozen=True)
class Command:
    kind: str
    x: float | None = None
    y: float | None = None
    z: float | None = None
    obj: int | None = None
    grid: GridSpec | None = None


@dataclass(frozen=True)
class SessionState:
    last_spawn: tuple[float, float, float] | None
    obj_idx: int | None


def parse_command(line: str) -> Command:
    parts = line.strip().split()
    if not parts:
        raise ValueError("빈 입력 (spawn X Y [Z] | repeat | obj N | sweep ... | quit)")
    kind, args = parts[0], parts[1:]
    if kind in ("quit", "repeat", "reset", "back2ini"):
        return Command(kind=kind)
    if kind == "spawn":
        if len(args) not in (2, 3):
            raise ValueError("사용법: spawn X Y [Z]")
        try:
            x, y = float(args[0]), float(args[1])
            z = float(args[2]) if len(args) == 3 else float("nan")
        except ValueError as e:
            raise ValueError(f"spawn 좌표 파싱 실패: {e}") from e
        return Command(kind="spawn", x=x, y=y, z=z)
    if kind == "obj":
        if len(args) != 1 or not args[0].lstrip("-").isdigit():
            raise ValueError("사용법: obj N (0~7)")
        n = int(args[0])
        if not 0 <= n < NUM_OBJECTS:
            raise ValueError(f"obj 는 0~{NUM_OBJECTS - 1} (got {n})")
        return Command(kind="obj", obj=n)
    if kind == "sweep":
        if len(args) != 7:
            raise ValueError("사용법: sweep XMIN XMAX NX YMIN YMAX NY REPEATS")
        try:
            grid = GridSpec(
                x_min=float(args[0]), x_max=float(args[1]), nx=int(args[2]),
                y_min=float(args[3]), y_max=float(args[4]), ny=int(args[5]),
                repeats=int(args[6]),
            )
        except ValueError as e:
            raise ValueError(f"sweep 인자 오류: {e}") from e
        return Command(kind="sweep", grid=grid)
    raise ValueError(f"알 수 없는 명령 {kind!r}")


def apply_command(state: SessionState, cmd: Command) -> tuple[SessionState, dict]:
    """새 SessionState + 실행 지시 dict 반환 (원본 state 불변)."""
    if cmd.kind == "quit":
        return state, {"action": "quit"}
    if cmd.kind == "obj":
        return replace(state, obj_idx=cmd.obj), {"action": "noop"}
    if cmd.kind == "spawn":
        new = replace(state, last_spawn=(cmd.x, cmd.y, cmd.z))
        return new, {"action": "spawn", "x": cmd.x, "y": cmd.y, "z": cmd.z}
    if cmd.kind == "repeat":
        if state.last_spawn is None:
            raise ValueError("repeat 전에 spawn 이력이 없습니다")
        x, y, z = state.last_spawn
        return state, {"action": "spawn", "x": x, "y": y, "z": z}
    if cmd.kind == "sweep":
        return state, {"action": "sweep", "grid": cmd.grid}
    if cmd.kind == "reset":
        return state, {"action": "reset"}
    if cmd.kind == "back2ini":
        return state, {"action": "back2ini"}
    raise ValueError(f"unhandled command kind {cmd.kind!r}")
