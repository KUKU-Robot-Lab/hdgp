"""autotune 산출물을 부위별 calibration 파일로 쪼개고, 다시 합친다.

autotune 결과 JSON은 tune 대상이 **아니었던** group까지 config 기본값 그대로 담는다.
그 파일을 학습에 그대로 물리면 측정한 적 없는 값이 env_cfg를 덮어쓴다. 실제로 07.29
우팔 결과에는 손 group이 30/5(config placeholder)로 들어 있어서, 그대로 적용하면
grasp_v1의 손 강성이 400에서 30으로 떨어진다 — 에러 없이, 조용히.

그래서 부위별로 **실제로 튜닝한 group만** 뽑아 자산 옆에 두고(extract),
학습에 쓸 때 필요한 부위만 합친다(merge). 합칠 때 같은 group이 두 부위에 겹치면
거부한다 — 나중 파일이 조용히 이기는 것이 바로 위에서 말한 사고 유형이다.

사용:
    python3 -m r2s_autotune.calibration_parts extract \\
        --result log/logs/r2s_autotune/results/right_arm_best_calibration.json \\
        --part right_arm \\
        --output assets/robot/openarm_tesollo_sensor_rl/calibration/right_arm.json

    python3 -m r2s_autotune.calibration_parts merge \\
        --output /tmp/train_calibration.json \\
        assets/robot/openarm_tesollo_sensor_rl/calibration/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
_REQUIRED_GROUP_KEYS = ("stiffness", "damping")


class CalibrationPartError(ValueError):
    """부위 파일 규약 위반. 조용히 넘어가면 학습에 엉뚱한 게인이 들어간다."""


def _check_schema(payload: Mapping[str, Any], source: str) -> None:
    version = int(payload.get("schema_version", 0))
    if version != SCHEMA_VERSION:
        raise CalibrationPartError(
            f"{source}: schema_version {version} != {SCHEMA_VERSION}"
        )
    if not isinstance(payload.get("groups"), dict) or not payload["groups"]:
        raise CalibrationPartError(f"{source}: 'groups'가 비었거나 객체가 아니다")


def _check_group_values(groups: Mapping[str, Any], source: str) -> None:
    for name, values in groups.items():
        if not isinstance(values, dict):
            raise CalibrationPartError(f"{source}: group '{name}'이 객체가 아니다")
        missing = [k for k in _REQUIRED_GROUP_KEYS if values.get(k) is None]
        if missing:
            raise CalibrationPartError(f"{source}: group '{name}'에 {missing}가 없다")


def extract_part(
    result: Mapping[str, Any],
    *,
    part: str,
    groups: Sequence[str] | None = None,
    rename: Mapping[str, str] | None = None,
    measured_on: str | None = None,
    source_path: str | None = None,
    asset: str | None = None,
) -> dict[str, Any]:
    """autotune 결과에서 실제로 튜닝한 group만 뽑아 부위 파일 payload를 만든다.

    groups를 주지 않으면 결과에 기록된 tune_groups를 쓴다 — 그게 "측정된 것"의 정의다.
    rename은 group 이름이 바뀐 경우(arm_proximal → right_arm_proximal)에만 쓴다.

    asset을 주면 그 자산의 파일로 표시한다. 같은 팔이 여러 자산에 들어 있을 때
    (openarm_tesollo_sensor_rl / _bi_rl의 오른팔은 같은 하드웨어) 쓰는 길인데,
    측정이 실제로 이뤄진 자산은 provenance에 남겨 이관 사실이 지워지지 않게 한다.
    """
    _check_schema(result, source_path or "result")

    tuned = list(groups) if groups is not None else list(
        result.get("autotune", {}).get("tune_groups", [])
    )
    if not tuned:
        raise CalibrationPartError(
            "튜닝된 group을 알 수 없다 (autotune.tune_groups 없음). --groups로 지정하라"
        )

    available = result["groups"]
    unknown = [name for name in tuned if name not in available]
    if unknown:
        raise CalibrationPartError(f"결과에 없는 group: {unknown}")

    rename = dict(rename or {})
    extracted = {rename.get(name, name): dict(available[name]) for name in tuned}
    _check_group_values(extracted, source_path or "result")

    provenance: dict[str, Any] = {
        "tuned_groups_in_result": tuned,
        "source_dataset": result.get("source_dataset"),
        "fit_error": result.get("fit_error"),
        "autotune": result.get("autotune"),
    }
    if source_path:
        provenance["autotune_result"] = source_path
    if measured_on:
        provenance["measured_on"] = measured_on
    if rename:
        provenance["renamed_groups"] = rename

    measured_asset = result.get("robot_asset")
    if asset and asset != measured_asset:
        provenance["measured_with_asset"] = measured_asset

    return {
        "schema_version": SCHEMA_VERSION,
        "robot_asset": asset or measured_asset,
        "part": part,
        "groups": extracted,
        "provenance": provenance,
    }


def merge_parts(parts: Sequence[Mapping[str, Any]], *, sources: Sequence[str] | None = None) -> dict[str, Any]:
    """부위 파일 여러 개를 학습이 읽을 수 있는 단일 calibration으로 합친다."""
    if not parts:
        raise CalibrationPartError("합칠 부위 파일이 없다")

    labels = list(sources) if sources else [f"part[{i}]" for i in range(len(parts))]

    assets = {payload.get("robot_asset") for payload in parts}
    if len(assets) != 1:
        raise CalibrationPartError(f"robot_asset이 섞였다: {sorted(map(str, assets))}")

    merged: dict[str, Any] = {}
    owner: dict[str, str] = {}
    for payload, label in zip(parts, labels):
        _check_schema(payload, label)
        _check_group_values(payload["groups"], label)
        for name, values in payload["groups"].items():
            if name in merged:
                raise CalibrationPartError(
                    f"group '{name}'이 {owner[name]}와 {label} 양쪽에 있다 — "
                    "어느 쪽이 맞는지 사람이 정해야 한다"
                )
            merged[name] = dict(values)
            owner[name] = label

    return {
        "schema_version": SCHEMA_VERSION,
        "robot_asset": assets.pop(),
        "groups": merged,
        "merged_from": [
            {"part": payload.get("part"), "source": label}
            for payload, label in zip(parts, labels)
        ],
    }


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CalibrationPartError(f"파일이 없다: {path}")
    return json.loads(path.read_text())


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_rename(items: Sequence[str]) -> dict[str, str]:
    rename: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise CalibrationPartError(f"--rename은 old=new 형식이다: {item}")
        old, new = item.split("=", 1)
        rename[old.strip()] = new.strip()
    return rename


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="autotune 산출물을 부위별 calibration 파일로 쪼개고, 다시 합친다."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="autotune 결과 → 부위 파일")
    extract.add_argument("--result", type=Path, required=True)
    extract.add_argument("--part", required=True, help="예: right_arm, right_hand")
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--groups", nargs="*", default=None, help="기본: 결과의 tune_groups")
    extract.add_argument("--rename", nargs="*", default=[], metavar="OLD=NEW")
    extract.add_argument("--measured-on", default=None, help="예: 2026-07-29")
    extract.add_argument(
        "--asset",
        default=None,
        help="다른 자산의 파일로 저장할 때. 같은 하드웨어일 때만 쓰고, "
        "측정 자산은 provenance에 남는다",
    )

    merge = sub.add_parser("merge", help="부위 파일들 → 학습용 단일 calibration")
    merge.add_argument("parts", nargs="+", type=Path)
    merge.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)

    try:
        if args.command == "extract":
            payload = extract_part(
                _read(args.result),
                part=args.part,
                groups=args.groups,
                rename=_parse_rename(args.rename),
                measured_on=args.measured_on,
                source_path=str(args.result),
                asset=args.asset,
            )
            _write(args.output, payload)
            print(f"[parts] {args.part}: {', '.join(sorted(payload['groups']))} -> {args.output}")
        else:
            payloads = [_read(p) for p in args.parts]
            payload = merge_parts(payloads, sources=[str(p) for p in args.parts])
            _write(args.output, payload)
            print(f"[parts] merged {len(payloads)} parts, {len(payload['groups'])} groups -> {args.output}")
    except CalibrationPartError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
