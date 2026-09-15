"""Run a VLM ``get_interaction_data`` response in a restricted namespace (design spec §6).

The response must contain exactly one fenced Python block that defines ``get_interaction_data``.
Only ``import numpy`` is allowed. Attribute access is limited to an explicit list of names, so the
numpy module graph cannot be walked to other modules (``numpy.lib.npyio.os``) and arrays cannot write
files (``ndarray.tofile``). Dunder names, ``try``, classes and the builtins that reach the interpreter
(``open``, ``exec``, ``eval``, ``getattr`` ...) are rejected before anything runs, and module execution
and the call run in a forked child process with a 1 GiB address-space limit and a wall-clock limit.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import multiprocessing
import os
import re
import resource
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

FUNCTION_NAME = "get_interaction_data"
DEFAULT_TIMEOUT_S = 2.0
_MEMORY_LIMIT_BYTES = 1 << 30  # 1 GiB of address space above what the child already uses
_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_BANNED_NAMES = frozenset(
    {
        "open", "exec", "eval", "compile", "globals", "locals", "vars", "getattr", "setattr", "delattr",
        "input", "breakpoint", "help", "exit", "quit", "memoryview", "type", "object", "super",
    }
)
_BANNED_NODES = (
    ast.ClassDef, ast.AsyncFunctionDef, ast.Global, ast.Nonlocal, ast.Try, getattr(ast, "TryStar", ast.Try),
    ast.Yield, ast.YieldFrom, ast.Await,
)
# Every attribute name the code may use: numpy math, ndarray/list/dict/str helpers. Anything else is
# rejected at parse time, whatever object it is taken from.
_ALLOWED_ATTRIBUTES = frozenset(
    {
        # numpy functions and constants
        "abs", "absolute", "allclose", "arange", "arccos", "arcsin", "arctan", "arctan2", "argmax", "argmin",
        "argsort", "array", "asarray", "average", "ceil", "clip", "concatenate", "copy", "cos", "cross",
        "cumsum", "deg2rad", "degrees", "det", "diff", "dot", "e", "exp", "expand_dims", "eye", "flip",
        "float32", "float64", "floor", "full", "hstack", "hypot", "inf", "int32", "int64", "inv", "isclose",
        "isfinite", "isnan", "linalg", "linspace", "log", "matmul", "max", "maximum", "mean", "median", "min",
        "minimum", "nan", "ndarray", "norm", "ones", "outer", "pi", "pinv", "prod", "rad2deg", "radians",
        "reshape", "round", "sign", "sin", "solve", "sort", "sqrt", "square", "squeeze", "stack", "sum", "svd",
        "tan", "transpose", "vstack", "where", "zeros",
        # ndarray attributes and methods without I/O
        "T", "all", "any", "astype", "dtype", "flatten", "item", "ndim", "ravel", "shape", "size", "std",
        "tolist",
        # list, dict and str helpers
        "append", "endswith", "extend", "get", "index", "items", "join", "keys", "lower", "pop", "replace",
        "split", "startswith", "strip", "update", "upper", "values",
    }
)
_ALLOWED_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance", "len", "list",
        "max", "min", "range", "reversed", "round", "sorted", "str", "sum", "tuple", "zip",
    )
}


class InteractionError(ValueError):
    """The response cannot be turned into an interaction."""


@dataclass(frozen=True)
class Interaction:
    object_name: str
    keypoint_ids: tuple[int, ...]
    grasp_mode: bool
    coordinates: Mapping[int, tuple[float, float, float]]
    done: bool


def extract_code_block(response: str) -> str:
    blocks = _CODE_BLOCK.findall(response)
    if len(blocks) != 1:
        raise InteractionError(f"expected exactly one python code block, found {len(blocks)}")
    return blocks[0]


def check_code(code: str) -> ast.Module:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise InteractionError(f"syntax error: {exc.msg} (line {exc.lineno})") from None
    for node in ast.walk(tree):
        _check_node(node)
    for node in tree.body:
        is_docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if not (is_docstring or isinstance(node, (ast.Import, ast.FunctionDef))):
            raise InteractionError(f"top-level statement not allowed: {type(node).__name__} (line {node.lineno})")
    if not any(isinstance(node, ast.FunctionDef) and node.name == FUNCTION_NAME for node in tree.body):
        raise InteractionError(f"no top-level function named {FUNCTION_NAME}")
    return tree


def _check_node(node: ast.AST) -> None:
    line = getattr(node, "lineno", "?")
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name != "numpy":
                raise InteractionError(f"only numpy may be imported, got {alias.name!r} (line {line})")
    elif isinstance(node, ast.ImportFrom):
        raise InteractionError(f"from-imports are not allowed (line {line})")
    elif isinstance(node, _BANNED_NODES):
        raise InteractionError(f"{type(node).__name__} is not allowed (line {line})")
    elif isinstance(node, ast.Name) and (node.id in _BANNED_NAMES or node.id.startswith("__")):
        raise InteractionError(f"name {node.id!r} is not allowed (line {line})")
    elif isinstance(node, ast.Attribute) and node.attr not in _ALLOWED_ATTRIBUTES:
        raise InteractionError(f"attribute {node.attr!r} is not allowed (line {line})")


def run_interaction(
    code: str, keypoints: Mapping[int, Sequence[float]], timeout_s: float = DEFAULT_TIMEOUT_S
) -> Interaction:
    """Execute checked code on a copy of ``{"1": np.array([x, y, z]), ...}`` and parse its return value.

    Forks a child process to run the module and the call, so this must not be called from a process
    that has initialized CUDA or Isaac Sim.
    """
    tree = check_code(code)
    inputs = {}
    for key, xyz in keypoints.items():
        point = np.asarray(xyz, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"keypoint {key} must be a finite xyz triple")
        inputs[str(int(key))] = point.copy()

    ctx = multiprocessing.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_run_in_child, args=(tree, inputs, child_conn))
    process.start()
    child_conn.close()
    try:
        if not parent_conn.poll(timeout_s):
            process.kill()
            process.join()
            raise InteractionError(f"{FUNCTION_NAME} exceeded {timeout_s} s")
        try:
            status, payload = parent_conn.recv()
        except EOFError:
            process.join()
            raise InteractionError(f"interaction process exited with code {process.exitcode}") from None
    finally:
        parent_conn.close()
        process.join()

    if status == "error":
        raise InteractionError(payload)
    return _parse_result(payload)


def _import_numpy_only(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002 - builtin signature
    """numpy and numpy's own submodules, nothing else.

    An ndarray method (``pts.mean(axis=0)``, ``.min()``, ``.max()``) imports ``numpy.core._methods`` through the
    calling frame's ``__import__`` every call, even when the module is already loaded, so rejecting every dotted
    name failed ordinary array maths (VLM gate 2026-09-16, stage_01/attempt_00). The checked code may still only
    write ``import numpy`` (``check_code``) and cannot reach ``__import__`` itself, so numpy's internal imports
    widen nothing it can read.
    """
    if level != 0 or (name != "numpy" and not name.startswith("numpy.")):
        raise ImportError(f"import of {name!r} is not allowed")
    return importlib.import_module(name) if fromlist else np


def _run_in_child(tree: ast.Module, inputs: dict, conn) -> None:
    """Child process entry point: bound its address space, then exec the module and call the function.

    Runs entirely inside a forked child. Any failure here - including a checked-code error or a
    MemoryError from the address-space limit - is sent back as an ("error", text) message, never
    raised, and the child always exits via ``os._exit`` so no parent cleanup handlers run in it.
    """
    try:
        _limit_child_memory()
        namespace = {"__builtins__": dict(_ALLOWED_BUILTINS, __import__=_import_numpy_only), "np": np, "numpy": np}
        try:
            exec(compile(tree, "<interaction>", "exec"), namespace)
        except BaseException as exc:  # noqa: BLE001 - untrusted code; report, don't propagate
            _send(conn, "error", f"module execution failed: {type(exc).__name__}: {exc}")
            return
        try:
            result = namespace[FUNCTION_NAME](inputs)
        except BaseException as exc:  # noqa: BLE001 - untrusted code; report, don't propagate
            _send(conn, "error", f"{FUNCTION_NAME} raised {type(exc).__name__}: {exc}")
            return
        _send(conn, "ok", result)
    except BaseException as exc:  # noqa: BLE001 - safety net around setup itself (e.g. RLIMIT_AS)
        _send(conn, "error", f"module execution failed: {type(exc).__name__}: {exc}")
    finally:
        os._exit(0)


def _limit_child_memory() -> None:
    with open("/proc/self/statm") as f:
        current_pages = int(f.read().split()[0])
    current_bytes = current_pages * resource.getpagesize()
    limit_bytes = current_bytes + _MEMORY_LIMIT_BYTES
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def _send(conn, status: str, payload) -> None:
    try:
        conn.send((status, payload))
    except Exception as exc:  # noqa: BLE001 - unpicklable payload
        conn.send(("error", f"{FUNCTION_NAME} returned a value that cannot be transferred: {type(exc).__name__}"))


def _parse_result(result) -> Interaction:
    if result is None:
        return Interaction(object_name="", keypoint_ids=(), grasp_mode=False, coordinates={}, done=True)
    if not isinstance(result, (tuple, list)) or len(result) != 4:
        raise InteractionError(
            f"{FUNCTION_NAME} must return (object_to_interact, keypoint_indices_to_interact, grasp_mode, "
            "keypoint_coordinates)"
        )
    name, ids, grasp_mode, coordinates = result
    if not isinstance(name, str) or not name.strip():
        raise InteractionError("object_to_interact must be a non-empty string")
    if isinstance(ids, (str, bytes)) or not isinstance(ids, (list, tuple, np.ndarray)) or len(ids) == 0:
        raise InteractionError("keypoint_indices_to_interact must be a non-empty list")
    keypoint_ids = tuple(_as_keypoint_id(value) for value in ids)
    if len(set(keypoint_ids)) != len(keypoint_ids):
        raise InteractionError(f"keypoint_indices_to_interact has duplicates: {list(keypoint_ids)}")
    if not isinstance(grasp_mode, (bool, np.bool_)):
        raise InteractionError("grasp_mode must be a boolean")
    if not isinstance(coordinates, dict):
        raise InteractionError("keypoint_coordinates must be a dict")
    parsed = {}
    for key, value in coordinates.items():
        try:
            point = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise InteractionError(f"coordinate {key!r} is not numeric") from None
        if point.shape != (3,):
            raise InteractionError(f"coordinate {key!r} must have shape (3,), got {point.shape}")
        parsed[_as_keypoint_id(key)] = (float(point[0]), float(point[1]), float(point[2]))
    return Interaction(name.strip(), keypoint_ids, bool(grasp_mode), parsed, done=False)


def _as_keypoint_id(value) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise InteractionError(f"keypoint id {value!r} is not an integer")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise InteractionError(f"keypoint id {value!r} is not an integer")
