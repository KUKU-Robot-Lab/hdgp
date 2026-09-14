"""grasp_fj t2r 루프 도구 — 판정 · 피드백 표 · 다음 프롬프트 · 기동 명령 (Isaac·서버 불요).

잠그는 것:
  ① 판정(`t2r_fj_round.judge`)이 ROUND_POLICY 대로 crashed/dead/continue/advance/done_candidate 를 낸다.
     인벨롭 지표가 없으면(구 코드·성공 전) 판정 보류 — 영상이 가른다.
  ② 피드백 표에는 생성 항·접촉·과제 지표만 들어가고 B 의 설계 계측(grasp_q)은 안 들어간다.
  ③ reflect 가 이전 코드 + 표 + 꼬리말로 다음 iter 프롬프트를 만들고, 첫 라운드 프롬프트는 그대로다.
  ④ 기동 명령은 i00 과 같은 런처·GPU0·SERVER=1·SAPG 6블록이다.

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_fj_t2r/tests -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from openarm.agnostic.tasks.grasp_fj_t2r.t2r import prompts as P

_HDGP = Path(__file__).resolve().parents[7]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HDGP / "scripts" / "reward_gen" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("t2r_fj_round")
T = _load("t2r_fj")


def _summ(succ=0.0, fingers=None, palm=None, tol=0.1125, tol_ago=None):
    s = {R.SUCCESS_TAG: {"last": succ, "now": succ},
         "task/tol": {"now": tol, "ago": tol if tol_ago is None else tol_ago}}
    if fingers is not None:
        s["contact/fingers_touching_at_success"] = {"now": fingers}
    if palm is not None:
        s["contact/palm_touching_at_success"] = {"now": palm}
    return s


ALIVE = {"alive": True, "crashed": False}
RE = R.ROUND_POLICY["ROUND_EPOCHS"]


@pytest.mark.parametrize("summary, st, hours, want", [
    (_summ(), {"alive": True, "crashed": True, "epoch": 300}, 1.0, "crashed"),
    (_summ(), {"alive": False, "crashed": False, "epoch": 3}, 0.1, "crashed"),
    (_summ(), {"alive": False, "crashed": False, "epoch": 400}, 1.0, "dead"),
    (_summ(0.3), {**ALIVE, "epoch": 200}, 0.5, "continue"),
    (_summ(0.3), {**ALIVE, "epoch": RE}, 3.0, "advance"),
    (_summ(0.3), {**ALIVE, "epoch": 300}, R.ROUND_POLICY["ROUND_HOURS"], "advance"),
    (_summ(2.5), {**ALIVE, "epoch": RE + 200}, 3.5, "continue(success)"),
    # ★09.14 i00 실측: 공차가 조여질 때마다 성공이 게이트 2.0 쪽으로 떨어진다 — 최근에 조여졌으면 유지, 멈췄으면 advance
    (_summ(1.7, tol=0.082, tol_ago=0.0911), {**ALIVE, "epoch": RE + 100}, 4.2, "continue(curriculum)"),
    (_summ(1.7, tol=0.082, tol_ago=0.082), {**ALIVE, "epoch": RE + 100}, 4.2, "advance"),
    (_summ(2.5, fingers=2.0, palm=0.1), {**ALIVE, "epoch": 2 * RE}, 6.0, "advance(envelope)"),
    (_summ(4.3, fingers=4.6, palm=0.8, tol=0.02), {**ALIVE, "epoch": 3000}, 9.0, "done_candidate"),
    (_summ(4.3, fingers=4.6, palm=0.8, tol=0.08), {**ALIVE, "epoch": 1500}, 4.5, "continue(success)"),  # 느슨한 공차
    (_summ(4.3, fingers=2.0, palm=0.0, tol=0.02), {**ALIVE, "epoch": 1500}, 4.5, "continue(success)"),
    (_summ(4.3, tol=0.02), {**ALIVE, "epoch": 3000}, 9.0, "done_candidate"),         # 인벨롭 지표 없음 → 영상
    (_summ(4.3, fingers=-1.0, palm=-1.0, tol=0.02), {**ALIVE, "epoch": 3000}, 9.0, "done_candidate"),
])
def test_judge_follows_the_round_policy(summary, st, hours, want):
    verdict, info = R.judge(summary, st, hours)
    assert verdict == want, info


def test_parse_tail_reads_console_epoch_and_label_checked_pids():
    txt = ("  epoch                       : 1,635 / 20,000\n"
           "PID=2993763 LABEL=fj_t2r_i00 CUDA_VISIBLE_DEVICES=0\n")
    st = R.parse_tail(txt)
    assert st["epoch"] == 1635 and st["alive"] and not st["crashed"]
    assert st["procs"] == [(2993763, "fj_t2r_i00", "0")]
    assert R.parse_tail("Traceback (most recent call last):\n")["crashed"]


def test_procs_cmd_does_not_match_its_own_shell():
    cmd = R.procs_cmd("fj_t2r_i00")
    assert "train[.]py" in cmd and "RUN_LABEL=" in cmd and "'fj_t2r_i00'" in cmd


def test_launch_command_matches_the_i00_launcher_on_gpu0():
    cmd = R.launch_command("fj_t2r_i01", "reward_gen/grasp_fj_envelope/iter_01", 12288, 42)
    for tok in ("run_fj.sh", "SERVER=1", "GPU=0", "RUN=fj_t2r_i01", "ENVS=12288", "BLK=2048",
                "expl_coef_block_size=2048", "open-short_r_grasp_fj_t2r-lstm-sapg",
                "env.reward_code_path=/home/oem/rl_ws/hdgp/reward_gen/grasp_fj_envelope/iter_01/compute_reward.py",
                "nohup bash", "< /dev/null &"):
        assert tok in cmd, tok
    assert "GPU=1" not in cmd and "pour" not in cmd
    with pytest.raises(ValueError):
        R.launch_command("x", "reward_gen/grasp_fj_envelope/iter_01", 12289, 42)


def test_summarize_keeps_judge_tags_the_last_point_and_the_window_value():
    data = {"ctrl/prev_ep_successes_mean/iter": [(i, float(i)) for i in range(10)],
            "task/tol/iter": [(i, 0.1125) for i in range(5)] + [(i, 0.1013) for i in range(5, 10)],
            "contact/palm_touching_at_success/iter": [(0, -1.0), (1, 0.6)],
            "reward/envelope/iter": [(0, 0.1)], "losses/a_loss": [(0, 1.0)]}
    s = R.summarize(data, last_n=4, window=7)
    assert s[R.SUCCESS_TAG]["last"] == 7.5 and s[R.SUCCESS_TAG]["now"] == 9.0
    assert s["task/tol"]["ago"] == 0.1125 and R.curriculum_moving(s)
    assert not R.curriculum_moving(R.summarize(data, last_n=4, window=3))     # 3 epoch 전에도 이미 0.1013
    assert s["contact/palm_touching_at_success"]["now"] == 0.6
    assert "reward/envelope" in s and "losses/a_loss" not in s


def test_feedback_series_has_generated_terms_contact_and_task_but_not_design_metrics():
    data = {"reward/approach/iter": [(0, 1.0), (1, 2.0)],
            "contact/palm_touching_at_success/iter": [(0, 0.5)],
            "ctrl/prev_ep_successes_mean/iter": [(0, 0.0)],
            "task/grasp_q_at_success/iter": [(0, 0.2)],
            "rewards/step": [(0, 3.0)], "rewards/iter": [(0, 3.0)], "contact/x/step": [(0, 1.0)]}
    s = T.collect_feedback_series(data)
    assert set(s) == {"reward/approach", "contact/palm_touching_at_success",
                      "ctrl/prev_ep_successes_mean", "rewards/step"}


def test_first_round_prompt_has_no_feedback_tail():
    txt = P.render_prompt(P.PromptSpec(task="T"))
    assert "Please carefully analyse the policy feedback" not in txt
    assert "_at_success" not in txt


def _iter00(tmp_path: Path) -> Path:
    d = tmp_path / "grasp_fj_envelope" / "iter_00"
    d.mkdir(parents=True)
    (d / "compute_reward.py").write_text("import torch\n\ndef compute_reward(ctx):\n    return 0\n")
    (d / "meta.json").write_text(json.dumps({"track": "grasp_fj_envelope", "iter": 0, "task": "THE TASK"}))
    return d


def test_reflect_writes_feedback_and_the_next_prompt(tmp_path, monkeypatch):
    fake = types.ModuleType("parse_tfevents")
    fake.load_tfevents = lambda path: {"reward/approach/iter": [(i, 0.1 * i) for i in range(20)],
                                       "contact/fingers_touching/iter": [(i, 2.0) for i in range(20)],
                                       "task/grasp_q/iter": [(0, 0.1)]}
    monkeypatch.setitem(sys.modules, "parse_tfevents", fake)
    d = _iter00(tmp_path)
    assert T.main(["--root", str(tmp_path), "reflect", "--iter", str(d), "--events", "ev"]) == 0
    fb = (d / "feedback.md").read_text()
    assert "reward/approach:" in fb and "contact/fingers_touching:" in fb and "grasp_q" not in fb
    nxt = tmp_path / "grasp_fj_envelope" / "iter_01"
    prompt = (nxt / "prompt.md").read_text()
    assert "THE TASK" in prompt and "The previous reward function was" in prompt
    assert "Please carefully analyse the policy feedback" in prompt and "contact/<metric>_at_success" in prompt
    meta = json.loads((nxt / "meta.json").read_text())
    assert meta["iter"] == 1 and meta["prev_iter"] == str(d)
    with pytest.raises(SystemExit):     # 덮어쓰기는 --force 로만
        T.main(["--root", str(tmp_path), "reflect", "--iter", str(d), "--events", "ev"])


def test_reflect_refuses_events_without_generated_terms(tmp_path, monkeypatch):
    fake = types.ModuleType("parse_tfevents")
    fake.load_tfevents = lambda path: {"contact/fingers_touching/iter": [(0, 1.0)]}
    monkeypatch.setitem(sys.modules, "parse_tfevents", fake)
    with pytest.raises(SystemExit):
        T.main(["--root", str(tmp_path), "reflect", "--iter", str(_iter00(tmp_path)), "--events", "ev"])
