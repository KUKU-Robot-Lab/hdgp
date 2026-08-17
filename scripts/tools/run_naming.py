#!/usr/bin/env python3
"""런 이름 규약 — `<논문>-<실험>-<조건>-a<자산>-s<시드>`

왜 필요한가 (2026-08-17)
------------------------
기존 네이밍에 네 가지 문제가 있었고, 마지막 하나가 실제 사고를 냈다.

1. **이원화** — train.py 가 폴더를 `lstm_test<N>` 자동증가로 만들고 의미 라벨은
   `test_history.md` 에만 넣었다. 그래서 폴더 `NS_demo_s42` 의 기록 헤더가
   `# lstm_test1` 인 상태(사후 개명)가 남았다.
2. **상태가 이름에 붙음** — `_COLLAPSED` / `_PARTIAL` / `_BROKEN` / `_ABANDONED`.
   이름이 가변이라 문서·스크립트의 참조가 깨진다. 상태는 이름이 아니라 메타로 둔다.
3. **실험 묶음이 없음** — 어떤 Table/Figure 를 위한 런인지 이름으로 알 수 없었다.
4. ★**자산 버전이 없음** — 로봇 USD 가 DG-5F → DG-5FS 로 교체됐는데 런 이름·기록
   헤더 어디에도 자산이 없어 구/신 런을 구분할 수 없었다. warm state 텐서 차원이
   같아(arm7+hand20) 구 캐시가 에러 없이 로드되기까지 해서, **어떤 자산에서 나온
   수치인지 사후에 확인할 수단이 없었다.**

규약
----
    <paper>-<exp>-<cond>-a<asset>-s<seed>
    예) A-E1-NSdemo-a2-s42     B-E1-frozen-a2-s42

    paper : 논문 구분   A=RA-L(both/pour_sensor)  B=양손 dexterous(both/pour_v1)
    exp   : 실험 묶음   E1, E2, ... (Table/Figure 단위)
    cond  : 조건        NSdemo, NSnaive, JS, Full, Rnoaim, frozen, learned, ...
    asset : 자산 버전   a1=openarm_tesollo_sensor_rl, a2=openarm_tesollo_bi_s_rl
    seed  : 시드        s42, s43, ...

상태는 이름에 넣지 않는다. 런 폴더의 `STATUS` 파일(한 줄)로 표시한다.

사용
----
    from run_naming import parse_label, assert_asset_matches
    info = parse_label("A-E1-NSdemo-a2-s42")
    assert_asset_matches("A-E1-NSdemo-a2-s42", "/.../openarm_tesollo_bi_s_rl/....usd")
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# 자산 태그 → 로봇 USD 디렉터리 이름(부분 일치로 검사).
# 새 자산을 추가할 때 여기에만 넣으면 게이트가 자동으로 따라온다.
ASSET_TAGS: dict[str, str] = {
    "a1": "openarm_tesollo_sensor_rl",   # DG-5F, 오른손만 Tesollo + 왼손 2-DOF 그리퍼
    "a2": "openarm_tesollo_bi_s_rl",     # DG-5FS, 좌우 20관절
    "a0": "openarm_bi_rh56f1_rl",        # RH56F1 (참고용)
}

_LABEL_RE = re.compile(
    r"^(?P<paper>[A-Z])-(?P<exp>E\d+)-(?P<cond>[A-Za-z0-9]+)-(?P<asset>a\d+)-s(?P<seed>\d+)$"
)

# 상태 파일 이름 + 허용 값 (이름에 붙이지 않는다)
STATUS_FILE = "STATUS"
STATUS_VALUES = ("running", "done", "collapsed", "partial", "aborted")


@dataclass(frozen=True)
class RunLabel:
    paper: str
    exp: str
    cond: str
    asset: str      # "a2"
    seed: int
    raw: str

    @property
    def asset_dir(self) -> str:
        return ASSET_TAGS[self.asset]


def parse_label(label: str) -> RunLabel | None:
    """규약에 맞으면 RunLabel, 아니면 None (구 라벨 하위호환 — 호출부가 판단)."""
    m = _LABEL_RE.match(label.strip())
    if not m:
        return None
    asset = m.group("asset")
    if asset not in ASSET_TAGS:
        raise ValueError(
            f"알 수 없는 자산 태그 '{asset}' (라벨 '{label}'). "
            f"허용: {sorted(ASSET_TAGS)} — 새 자산이면 run_naming.ASSET_TAGS 에 먼저 추가할 것."
        )
    return RunLabel(
        paper=m.group("paper"),
        exp=m.group("exp"),
        cond=m.group("cond"),
        asset=asset,
        seed=int(m.group("seed")),
        raw=label.strip(),
    )


def assert_asset_matches(label: str, robot_usd_path: str) -> RunLabel | None:
    """라벨의 자산 태그와 실제 로봇 USD 가 다르면 예외.

    규약 형식이 아닌 라벨은 검사하지 않고 None 을 돌려준다(구 라벨 하위호환).
    **이 게이트가 2026-08-17 사고(구 USD 캐시가 조용히 로드됨)의 재발 방지 장치다.**
    """
    info = parse_label(label)
    if info is None:
        return None
    if info.asset_dir not in str(robot_usd_path):
        raise RuntimeError(
            "런 라벨의 자산 태그와 실제 로봇 USD 가 다르다 — 학습을 중단한다.\n"
            f"  라벨      : {label}  (태그 {info.asset} = {info.asset_dir})\n"
            f"  실제 USD  : {robot_usd_path}\n"
            "  라벨의 자산 태그를 고치거나, env cfg 의 usd_path 를 고칠 것.\n"
            "  (이 게이트가 없던 시절 구 자산 런과 신 자산 런이 뒤섞여 논문 수치가 무효가 됐다)"
        )
    return info


def sanitize_for_dir(label: str) -> str:
    """폴더명으로 안전한 형태로 (공백·슬래시 제거). 규약 라벨은 그대로 통과한다."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", label.strip())


def describe_convention() -> str:
    return __doc__ or ""


if __name__ == "__main__":  # 간단 자기검사
    ok = parse_label("A-E1-NSdemo-a2-s42")
    assert ok and ok.paper == "A" and ok.exp == "E1" and ok.cond == "NSdemo"
    assert ok.asset == "a2" and ok.seed == 42 and ok.asset_dir == "openarm_tesollo_bi_s_rl"
    assert parse_label("lstm_test1") is None, "구 라벨은 None 이어야(하위호환)"
    assert parse_label("NS_demo_s42") is None
    assert assert_asset_matches("lstm_test1", "/any/path") is None
    assert_asset_matches("A-E1-NSdemo-a2-s42", "/x/openarm_tesollo_bi_s_rl/y.usd")
    try:
        assert_asset_matches("A-E1-NSdemo-a2-s42", "/x/openarm_tesollo_sensor_rl/y.usd")
    except RuntimeError:
        pass
    else:
        raise AssertionError("자산 불일치를 잡지 못했다")
    assert sanitize_for_dir("A-E1-NSdemo-a2-s42") == "A-E1-NSdemo-a2-s42"
    print("run_naming 자기검사 통과")
