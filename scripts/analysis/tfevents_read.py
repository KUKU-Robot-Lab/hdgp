"""TFEvents 스칼라 리더 — **의존성 없음**(protobuf·tensorboard 불필요).

왜 직접 파싱하나. 이 워크스테이션의 파이썬은 tensorboard 를 부르는 순간 protobuf
descriptor 중복 등록으로 **인터프리터가 통째로 죽는다**(SIGABRT/SIGSEGV). 환경변수
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 도 소용없다 — 사이트 패키지가 파이썬
시작 시점에 이미 C++ protobuf 를 끌어오기 때문이다. 그래서 로그를 볼 때마다 서버로
복사해 왔는데, 지표 판독은 앞으로 매일 하는 일이라 그 우회로를 없앤다.

읽는 것은 스칼라뿐이고, 필요한 wire format 은 아래 세 겹이 전부다:

    TFRecord : uint64 길이 · uint32 crc · payload · uint32 crc   (crc 는 검사하지 않는다)
    Event    : field 2 = step(varint) · field 5 = Summary(message)
    Value    : field 1 = tag(string)  · field 2 = simple_value(float32)

⚠스칼라 외(히스토그램·이미지·텐서)는 건너뛴다. 그 이상이 필요해지면 그때 서버로 간다.
"""

from __future__ import annotations

import struct
from pathlib import Path


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    val = shift = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7


def _fields(buf: bytes):
    """(field_number, wire_type, payload) 를 차례로 낸다."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            val, i = _varint(buf, i)
            yield fno, wt, val
        elif wt == 1:
            yield fno, wt, buf[i:i + 8]
            i += 8
        elif wt == 2:
            ln, i = _varint(buf, i)
            yield fno, wt, buf[i:i + ln]
            i += ln
        elif wt == 5:
            yield fno, wt, buf[i:i + 4]
            i += 4
        else:                                   # 6·7 은 없어진 group — 나오면 포기한다
            raise ValueError(f"지원하지 않는 wire type {wt}")


def _records(raw: bytes):
    i, n = 0, len(raw)
    while i + 12 <= n:
        (ln,) = struct.unpack_from("<Q", raw, i)
        beg = i + 12
        end = beg + ln
        if end + 4 > n:                          # 기록 중인 마지막 레코드는 잘려 있다
            return
        yield raw[beg:end]
        i = end + 4


def read_scalars(path: Path | str) -> dict[str, list[tuple[int, float]]]:
    """{tag: [(step, value), ...]}. 디렉터리를 주면 그 안의 events 파일을 전부 합친다."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("events.out.tfevents.*"))
    else:
        files = [p]
    out: dict[str, list[tuple[int, float]]] = {}
    for f in files:
        raw = f.read_bytes()
        for rec in _records(raw):
            step = 0
            summaries = []
            try:
                for fno, wt, val in _fields(rec):
                    if fno == 2 and wt == 0:
                        step = val
                    elif fno == 5 and wt == 2:
                        summaries.append(val)
            except (IndexError, ValueError):
                continue                          # 깨진 레코드 하나가 전체를 막지 않게
            for s in summaries:
                try:
                    for fno, wt, val in _fields(s):
                        if fno != 1 or wt != 2:
                            continue
                        tag = None
                        sval = None
                        for vf, vw, vv in _fields(val):
                            if vf == 1 and vw == 2:
                                tag = vv.decode("utf-8", "replace")
                            elif vf == 2 and vw == 5:
                                (sval,) = struct.unpack("<f", vv)
                        if tag is not None and sval is not None:
                            out.setdefault(tag, []).append((step, sval))
                except (IndexError, ValueError, struct.error):
                    continue
    for v in out.values():
        v.sort(key=lambda t: t[0])
    return out
