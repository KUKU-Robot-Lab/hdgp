#!/usr/bin/env python3
"""범용 TFEvents 바이너리 파서 (tensorflow/protobuf 라이브러리 불필요).

Usage:
    python parse_tfevents.py <events_file> [--tags TAG1 TAG2 ...] [--epochs N] [--out /tmp/out.json]

출력: JSON {tag: [[step, value], ...]}
     --epochs N 지정 시 N개 구간으로 리샘플링해 평균값 반환
"""

import argparse
import collections
import json
import os
import struct
import sys
from pathlib import Path


# ─── TFRecord / protobuf struct 파서 ───────────────────────────

_MAX_RESYNC_BYTES = 1 << 20   # 재동기 예산 1 MB — 넘으면 진짜 손상으로 보고 중단

def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result, shift = 0, 0
    while pos < len(buf):
        b = buf[pos]; pos += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            break
    return result, pos


def _parse_summary_value(vbuf: bytes) -> tuple[str, float] | None:
    """Summary.Value bytes → (tag, float) or None."""
    p, nb = 0, len(vbuf)
    tag_str: str | None = None
    fval: float | None = None
    while p < nb:
        tw, p = _read_varint(vbuf, p)
        fnum, wire = tw >> 3, tw & 0x7
        if wire == 0:
            _, p = _read_varint(vbuf, p)
        elif wire == 1:
            p += 8
        elif wire == 2:
            ln, p = _read_varint(vbuf, p)
            chunk = vbuf[p:p + ln]; p += ln
            if fnum == 1:
                try:
                    tag_str = chunk.decode("utf-8")
                except Exception:
                    pass
        elif wire == 5:
            if p + 4 <= nb:
                fval = struct.unpack_from("<f", vbuf, p)[0]
            p += 4
        else:
            break
    if tag_str is not None and fval is not None:
        return tag_str, fval
    return None


def _parse_summary(sbuf: bytes) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    p, nb = 0, len(sbuf)
    while p < nb:
        tw, p = _read_varint(sbuf, p)
        fnum, wire = tw >> 3, tw & 0x7
        if wire == 2:
            ln, p = _read_varint(sbuf, p)
            chunk = sbuf[p:p + ln]; p += ln
            if fnum == 1:
                r = _parse_summary_value(chunk)
                if r:
                    out.append(r)
        elif wire == 0:
            _, p = _read_varint(sbuf, p)
        elif wire == 1:
            p += 8
        elif wire == 5:
            p += 4
        else:
            break
    return out


def _parse_event(buf: bytes) -> tuple[int | None, list[tuple[str, float]]]:
    """Event bytes → (step, [(tag, value), ...])."""
    step: int | None = None
    summary_bytes: bytes | None = None
    pos, n = 0, len(buf)
    while pos < n:
        tw, pos = _read_varint(buf, pos)
        fnum, wire = tw >> 3, tw & 0x7
        if wire == 0:
            val, pos = _read_varint(buf, pos)
            if fnum == 2:
                step = val
        elif wire == 1:
            pos += 8
        elif wire == 2:
            ln, pos = _read_varint(buf, pos)
            chunk = buf[pos:pos + ln]; pos += ln
            if fnum == 5:
                summary_bytes = chunk
        elif wire == 5:
            pos += 4
        else:
            break
    if step is not None and summary_bytes is not None:
        return step, _parse_summary(summary_bytes)
    return step, []


def load_tfevents(path: str) -> dict[str, list[tuple[int, float]]]:
    """TFEvents 파일 → {tag: [(step, value), ...]} (step 오름차순)."""
    tag_rows: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)
    # ★★fab_test40 사고: 학습 중인 파일을 rsync 하면 마지막 레코드가 토막나고, 그러면
    #   길이 필드가 쓰레기가 되어 **그 뒤 전체를 버렸다.** 실제로 ep174 이후 3,000 epoch
    #   분량이 통째로 안 읽혀 "학습이 ep173 에서 멈췄다"로 오판할 뻔했다(로그는 ep661).
    #   길이가 파일 크기와 안 맞으면 1 바이트씩 전진해 **재동기**한다(TFRecord 표준 복구).
    size = os.path.getsize(path)
    resync = 0
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            hdr = f.read(12)
            if len(hdr) < 12:
                break
            data_len = struct.unpack_from("<Q", hdr, 0)[0]
            if data_len == 0 or data_len > size - pos - 16:
                f.seek(pos + 1)
                resync += 1
                if resync > _MAX_RESYNC_BYTES:
                    break
                continue
            data = f.read(data_len)
            f.read(4)  # masked crc (skip)
            if len(data) < data_len:
                break
            step, pairs = _parse_event(data)
            if step is None:
                continue
            for tag, val in pairs:
                tag_rows[tag].append((step, val))

    return {tag: sorted(rows) for tag, rows in tag_rows.items()}


# ─── 에포크 단위 리샘플링 ───────────────────────────────────────

def resample_to_epochs(
    rows: list[tuple[int, float]],
    n_epochs: int,
) -> list[tuple[int, float]]:
    """step 기반 데이터를 n_epochs 구간으로 나눠 구간 평균 반환."""
    if not rows:
        return []
    max_step = rows[-1][0]
    if max_step == 0:
        return []
    bucket_size = max(1, max_step // n_epochs)
    buckets: dict[int, list[float]] = collections.defaultdict(list)
    for step, val in rows:
        bucket = (step // bucket_size) * bucket_size
        buckets[bucket].append(val)
    result = [(b, sum(vs) / len(vs)) for b, vs in sorted(buckets.items())]
    return result


# ─── 에포크 번호 추정 (step → epoch) ───────────────────────────
# rl_games 기본: 1 epoch = horizon(32) * num_envs(4096) steps → ~131k steps/epoch
# 실제 값은 체크포인트 파일명으로 보정

def estimate_epoch(step: int, steps_per_epoch: int = 131072) -> int:
    return max(1, step // steps_per_epoch)


# ─── 메인 ──────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Parse TFEvents to JSON")
    p.add_argument("events_file", help="Path to events.out.tfevents.* file")
    p.add_argument("--tags", nargs="*", help="Filter specific tags (prefix match)")
    p.add_argument("--epochs", type=int, default=0,
                   help="Resample to N epoch buckets (0=raw steps)")
    p.add_argument("--steps-per-epoch", type=int, default=131072,
                   help="Steps per epoch for epoch estimation")
    p.add_argument("--out", default="-", help="Output JSON file (- = stdout)")
    args = p.parse_args()

    data = load_tfevents(args.events_file)

    # 태그 필터링
    if args.tags:
        prefixes = tuple(args.tags)
        data = {t: v for t, v in data.items() if any(t.startswith(pf) for pf in prefixes)}

    # 리샘플링
    if args.epochs > 0:
        data = {t: resample_to_epochs(v, args.epochs) for t, v in data.items()}

    # epoch 번호 컬럼 추가 (step → epoch 추정)
    output: dict[str, list] = {}
    for tag, rows in data.items():
        if args.epochs > 0:
            # 이미 에포크 단위 리샘플
            output[tag] = [[estimate_epoch(s, args.steps_per_epoch), round(v, 6)]
                           for s, v in rows]
        else:
            output[tag] = [[s, round(v, 6)] for s, v in rows]

    json_str = json.dumps(output, ensure_ascii=False, indent=None)
    if args.out == "-":
        sys.stdout.write(json_str + "\n")
    else:
        Path(args.out).write_text(json_str, encoding="utf-8")
        print(f"Saved to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
