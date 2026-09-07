"""SAPG 블록 산술 단위테스트 — Isaac 없이 돈다."""

import pytest
import torch

from openarm.agnostic.modules.sapg_blocks import (
    SAPGBlocks,
    SAPGCfg,
    filter_leader,
    leader_block_after_roll,
    roll_blocks,
    sample_repeat_idxs,
)

_SCALE = 0.002


def _blocks(n=4096, bs=1024, scale=_SCALE):
    return SAPGBlocks(n, SAPGCfg(block_size=bs, coef_scale=scale), "cpu")


# ------------------------------------------------------------------ 블록 분할
def test_original_simtoolreal_coefficients_are_reproduced():
    """원본 24,576 env ÷ 4,096 = 6블록. yaml 의 entropy_coef 0.0 은 리더 값일 뿐이다."""
    c = SAPGBlocks(24576, SAPGCfg(block_size=4096, coef_scale=_SCALE), "cpu").block_coefs()
    assert torch.allclose(c, torch.tensor([0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0]), atol=1e-9)


def test_our_four_block_layout():
    b = _blocks()
    assert (b.num_blocks, b.block_size, b.embd_dim) == (4, 1024, 1)
    assert torch.allclose(b.block_coefs(),
                          torch.tensor([0.001, 0.002 / 3.0, 0.001 / 3.0, 0.0]), atol=1e-9)


def test_leader_is_the_last_block_and_has_zero_bonus():
    b = _blocks()
    assert float(b.entropy_coef[-1]) == 0.0
    assert float(b.entropy_coef[0]) == pytest.approx(_SCALE * 0.5)
    # 리더 블록 전체가 0 이어야 한다 — 한 칸만 0 이면 슬라이싱이 어긋난 것이다.
    assert torch.count_nonzero(b.entropy_coef) == b.num_actors - b.block_size


def test_ids_and_embedding_are_constant_within_a_block():
    b = _blocks()
    for k in range(b.num_blocks):
        sl = slice(k * b.block_size, (k + 1) * b.block_size)
        assert int(b.ids[sl].unique().numel()) == 1
        assert int(b.embedding[sl].unique().numel()) == 1
    # 식별자는 블록마다 달라야 정책이 구분할 수 있다.
    assert int(b.embedding.unique().numel()) == b.num_blocks


def test_single_block_is_rejected():
    """★4,096 env 에 원본 block_size 4,096 을 그대로 쓰면 계수가 균일해져 그냥 PPO 가 된다."""
    with pytest.raises(ValueError, match="블록이 1개"):
        SAPGBlocks(4096, SAPGCfg(block_size=4096, coef_scale=_SCALE), "cpu")


def test_non_divisible_block_size_is_rejected():
    with pytest.raises(ValueError, match="나누어떨어져"):
        SAPGBlocks(4096, SAPGCfg(block_size=1000, coef_scale=_SCALE), "cpu")


# ------------------------------------------------------------------ 재라벨링
@pytest.mark.parametrize("k", [1, 2, 3])
def test_leader_block_after_roll_matches_the_actual_roll(k):
    """★핵심 항등식 — `filter_leader` 의 (idx-1) 슬라이스가 여기에 걸려 있다.

    롤한 계수 배열에서 **리더 값(0)** 을 달게 된 블록이 `leader_block_after_roll(k)` 와
    같아야 한다. 어긋나면 팔로워 경험이 엉뚱한 라벨로 리더 학습에 섞인다 — 조용히 틀린다.
    """
    b = _blocks()
    rolled = roll_blocks(b.entropy_coef, b.block_size, k)
    zero_blocks = {int(i) // b.block_size for i in torch.nonzero(rolled == 0.0).flatten()}
    assert zero_blocks == {leader_block_after_roll(k, b.num_blocks)}


def test_roll_zero_is_identity():
    b = _blocks()
    assert torch.equal(roll_blocks(b.entropy_coef, b.block_size, 0), b.entropy_coef)


def test_filter_leader_keeps_full_self_copy_and_one_block_per_other():
    b = _blocks()
    orig = b.num_actors
    val = torch.arange(orig * 2, dtype=torch.float32)      # 복사본 2개
    out = filter_leader(val, orig, [0, 2], b.num_blocks)
    # roll 0 → 통째(4096) · roll 2 → 한 블록(1024)
    assert out.numel() == orig + b.block_size
    assert torch.equal(out[:orig], val[:orig])
    lb = leader_block_after_roll(2, b.num_blocks)
    assert torch.equal(out[orig:], val[orig + lb * b.block_size: orig + (lb + 1) * b.block_size])


def test_filter_leader_selects_exactly_the_leader_labelled_rows():
    """복사본의 계수 라벨을 함께 걸러보면, 남는 행은 전부 리더 계수(0)여야 한다."""
    b = _blocks()
    idxs = [0, 1, 3]
    labels = torch.cat([roll_blocks(b.entropy_coef, b.block_size, k) for k in idxs])
    kept = filter_leader(labels, b.num_actors, idxs, b.num_blocks)
    # 자기 복사본(roll 0)은 전 블록이 남으므로 리더 블록 하나만 0 이다.
    self_part, other_part = kept[: b.num_actors], kept[b.num_actors:]
    assert torch.count_nonzero(self_part) == b.num_actors - b.block_size
    assert torch.count_nonzero(other_part) == 0          # ★나머지는 전부 리더 라벨
    assert other_part.numel() == b.block_size * (len(idxs) - 1)


def test_filter_leader_dim1_matches_dim0_semantics():
    """rnn_states 는 배치축이 1 이다 — 축만 다르고 자르는 위치는 같아야 한다."""
    b = _blocks()
    orig, idxs = b.num_actors, [0, 2]
    row = torch.arange(orig * len(idxs), dtype=torch.float32)
    got0 = filter_leader(row, orig, idxs, b.num_blocks)
    got1 = filter_leader(row.reshape(1, -1).repeat(3, 1), orig, idxs, b.num_blocks, dim=1)
    assert torch.equal(got1[0], got0) and torch.equal(got1[2], got0)


# ------------------------------------------------------------------ 롤 표본
def test_sample_repeat_idxs_always_includes_self_and_no_duplicates():
    for ratio in (0, 1, 2, 3, 10):
        got = sample_repeat_idxs(4, ratio)
        assert got[0] == 0
        assert len(set(got)) == len(got)
        assert all(0 <= v < 4 for v in got)
        assert len(got) == min(4, int(ratio) + 1)


def test_off_policy_ratio_one_mixes_exactly_one_other_block():
    """SimToolReal 값 1.0 → 자기 롤아웃 + 다른 블록 하나."""
    got = sample_repeat_idxs(4, 1.0)
    assert len(got) == 2 and got[0] == 0 and got[1] != 0


# ============================================================ 벤더 포크와의 등가성
# `hdgp/vendor/rl_games_sapg` 는 SimToolReal 포크를 그대로 들여온 것이고, 학습은 그쪽 코드로
# 돈다. 위 모듈은 같은 산술의 **독립 구현**이다. 둘이 갈라지면 우리가 문서·테스트로 믿고 있는
# 동작과 실제 학습이 달라지므로, 여기서 직접 대조한다.
import pathlib
import sys

_VENDOR = pathlib.Path(__file__).resolve().parents[6] / "vendor" / "rl_games_sapg"


def _vendor_custom_utils():
    if not (_VENDOR / "rl_games" / "common" / "custom_utils.py").is_file():
        pytest.skip("벤더 rl_games_sapg 가 없다")
    if str(_VENDOR) not in sys.path:
        sys.path.insert(0, str(_VENDOR))
    from rl_games.common import custom_utils
    if not str(pathlib.Path(custom_utils.__file__)).startswith(str(_VENDOR)):
        pytest.skip(f"설치본 rl_games 가 먼저 잡혔다: {custom_utils.__file__}")
    return custom_utils


@pytest.mark.parametrize("idxs", [[0, 1], [0, 2], [0, 3], [0, 1, 3]])
def test_filter_leader_matches_vendor_fork(idxs):
    """우리 `filter_leader` 가 포크 원본과 **같은 행**을 남기는지."""
    cu = _vendor_custom_utils()
    b = _blocks()
    val = torch.arange(b.num_actors * len(idxs), dtype=torch.float32)
    ours = filter_leader(val, b.num_actors, idxs, b.num_blocks)
    theirs = cu.filter_leader(val, b.num_actors, idxs, b.num_blocks)
    assert torch.equal(ours, theirs)


def test_vendor_axis_heuristic_requires_single_layer_rnn():
    """★포크의 축 선택은 `len(val) > 1` 휴리스틱이라 **층 수가 1일 때만** 맞는다.

    `filter_leader` 는 rnn_states 를 (layers, batch, hidden) 으로 받아 **배치축**을 잘라야 한다.
    그런데 원본은 첫 축 길이가 1 일 때만 축 1 로 간다 — 2층 이상이면 층 축을 자르며 조용히
    틀린다. 우리 SAPG 설정이 `layers: 1` 인 한 안전하므로, 여기서 그 전제를 직접 잠근다.
    """
    cu = _vendor_custom_utils()
    b = _blocks()
    idxs = [0, 2]
    one_layer = torch.arange(b.num_actors * len(idxs), dtype=torch.float32).reshape(1, -1)
    assert torch.equal(cu.filter_leader(one_layer, b.num_actors, idxs, b.num_blocks),
                       filter_leader(one_layer, b.num_actors, idxs, b.num_blocks, dim=1))

    cfg = (pathlib.Path(__file__).resolve().parents[3] / "tasks" / "grasp_fj" / "config" /
           "agents" / "rl_games_ppo_lstm_sapg_cfg.yaml")
    if cfg.is_file():
        rnn = cfg.read_text(encoding="utf-8").split("rnn:")[1].split("load_checkpoint")[0]
        assert "layers: 1" in rnn, "SAPG 설정의 LSTM 층이 1 이 아니다 — 포크 축 휴리스틱이 깨진다"


def test_block_coefficients_match_the_fork_formula():
    """포크 `A2CBase.__init__` 의 계수식을 그대로 재현하는지 — 식을 여기 박아 대조한다."""
    b = _blocks()
    nb, bs = b.num_blocks, b.block_size
    env_ids = torch.arange(nb).repeat_interleave(bs)
    fork_coef = torch.linspace(0.5, 0.0, nb)[env_ids] * _SCALE
    fork_embd = torch.linspace(50.0, 0.0, nb)[env_ids].reshape(-1, 1)
    assert torch.allclose(b.entropy_coef, fork_coef)
    assert torch.allclose(b.embedding, fork_embd)
