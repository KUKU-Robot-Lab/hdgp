"""grasp_fj t2r 루프 도구 — 판정 · 원본 t2r 피드백(관찰+개선) 프롬프트 · 기동/영상 명령 (Isaac·서버 불요).

잠그는 것:
  ① 판정(`t2r_fj_round.judge`)이 ROUND_POLICY 대로 crashed/dead/continue/advance/done_candidate 를 낸다.
     인벨롭 지표가 없으면(구 코드·성공 전) 판정 보류 — 영상이 가른다. 공차 커리큘럼이 움직이면 유지.
  ② reflect 는 원본 text2reward interactive 형식이다: 지난 (코드 · 로봇 관찰 · 개선 피드백) 전 이력 →
     "Re-imagine which steps is missed or wrong." — 관찰·피드백 없이는 다음 프롬프트를 만들지 않는다(09.14 사용자).
     지표 표는 참고로만 붙고, B 의 설계 계측(grasp_q)은 안 들어간다.
  ③ 첫 라운드 프롬프트는 그대로다.
  ④ 기동·영상 명령은 GPU0 · t2r task id · SERVER=1 · SAPG 6블록이다.

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
    # ★09.14 reach(사용자: 접근·파지·리프트가 잘 되는지 틱 확인): 성공·공차가 멈춰도 퍼널 단계가 오르는 중이면 2×ROUND 까지 유지
    ({**_summ(0.0), "stage/reach_ep": {"now": 0.62, "ago": 0.40}}, {**ALIVE, "epoch": RE + 100}, 4.2, "continue(stage)"),
    ({**_summ(0.0), "stage/lift_ep": {"now": 0.05, "ago": -1.0}}, {**ALIVE, "epoch": RE}, 3.0, "continue(stage)"),
    ({**_summ(0.0), "stage/reach_ep": {"now": 0.62, "ago": 0.61}}, {**ALIVE, "epoch": RE + 100}, 4.2, "advance"),
    ({**_summ(0.0), "stage/lift_ep": {"now": 0.30, "ago": 0.10}}, {**ALIVE, "epoch": 2 * RE}, 6.0, "advance"),
    ({**_summ(0.0), "stage/grasp_ep": {"now": -1.0, "ago": -1.0}}, {**ALIVE, "epoch": RE}, 3.0, "advance"),
])
def test_judge_follows_the_round_policy(summary, st, hours, want):
    verdict, info = R.judge(summary, st, hours)
    assert verdict == want, info


def test_stage_funnel_reads_episode_fractions_in_the_env_order():
    from openarm.agnostic.tasks.grasp_fj_t2r import stage_funnel as SF
    assert R.STAGE_NAMES == SF.STAGES
    data = {"stage/reach_ep/iter": [(0, -1.0), (1, 0.2), (2, 0.5)], "stage/palm_cup_gap/iter": [(0, 0.3)]}
    s = R.summarize(data, last_n=2, window=1)
    assert "stage/palm_cup_gap" in s
    fun = R.stage_funnel(s)
    assert fun["reach"] == (0.5, 0.2) and fun["success"] is None and list(fun) == list(SF.STAGES)
    assert R.stage_moving(s)


def test_reach_track_video_covers_a_full_episode():
    assert R.TRACKS["grasp_fj_reach"]["video_length"] >= 900
    assert R.TRACKS["grasp_fj_envelope"]["video_length"] == 700


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


def test_video_command_plays_the_t2r_task_on_gpu0_from_a_snapshot():
    cmd = R.video_command("/home/oem/rl_ws/hdgp/log/rl_games/open-short/right/grasp-fj-t2r/fj_t2r_i00",
                          "fj_t2r_i00", 0.082, "0914_1253")
    for tok in ("open-short_r_grasp_fj_t2r-play-lstm-sapg", "env.tol_eval=0.082", "--view_env_index 11",
                "--num_envs 12", "CUDA_VISIBLE_DEVICES=0", "snap_0914_1253.pth", "--video", "VIDEO "):
        assert tok in cmd, tok
    assert "open-sens" not in cmd


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


def test_first_round_prompt_has_no_feedback_blocks():
    txt = P.render_prompt(P.PromptSpec(task="T"))
    for tok in ("Please carefully analyse the policy feedback", "_at_success", "I can see from the robot that",
                "Re-imagine which steps is missed or wrong"):
        assert tok not in txt, tok


def _iter(tmp_path: Path, n: int, code: str = "import torch\n\ndef compute_reward(ctx):\n    return 0\n") -> Path:
    d = tmp_path / "grasp_fj_envelope" / f"iter_{n:02d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "compute_reward.py").write_text(code)
    if not (d / "meta.json").exists():
        (d / "meta.json").write_text(json.dumps({"track": "grasp_fj_envelope", "iter": n, "task": "THE TASK"}))
    return d


def _texts(tmp_path: Path, tag: str) -> tuple[str, str]:
    o = tmp_path / f"obs_{tag}.md"
    o.write_text(f"OBSERVED {tag}: the cup is lifted tilted.")
    f = tmp_path / f"fb_{tag}.md"
    f.write_text(f"FEEDBACK {tag}: keep the cup upright.")
    return str(o), str(f)


@pytest.fixture
def fake_events(monkeypatch):
    fake = types.ModuleType("parse_tfevents")
    fake.load_tfevents = lambda path: {"reward/approach/iter": [(i, 0.1 * i) for i in range(20)],
                                       "contact/fingers_touching/iter": [(i, 2.0) for i in range(20)],
                                       "task/grasp_q/iter": [(0, 0.1)]}
    monkeypatch.setitem(sys.modules, "parse_tfevents", fake)


def test_reflect_builds_the_original_t2r_feedback_prompt_with_full_history(tmp_path, fake_events):
    d0 = _iter(tmp_path, 0, code="import torch\n\ndef compute_reward(ctx):\n    return 'CODE0'\n")
    o0, f0 = _texts(tmp_path, "R0")
    assert T.main(["--root", str(tmp_path), "reflect", "--iter", str(d0), "--description", o0,
                   "--feedback", f0, "--events", "ev"]) == 0
    p1 = (tmp_path / "grasp_fj_envelope" / "iter_01" / "prompt.md").read_text()
    for tok in ("THE TASK", "Generated code shown as below:", "CODE0",
                "After training, I can see from the robot that:", "OBSERVED R0",
                "the feedback for improvement is:", "FEEDBACK R0",
                "Re-imagine which steps is missed or wrong.", "reward/approach:"):
        assert tok in p1, tok
    assert "Please carefully analyse the policy feedback" not in p1 and "grasp_q" not in p1
    assert "OBSERVED R0" in (d0 / "feedback.md").read_text()
    # 두 번째 라운드: 이력이 순서대로 전부 들어간다(원본 FewShot 예시 누적)
    d1 = _iter(tmp_path, 1, code="import torch\n\ndef compute_reward(ctx):\n    return 'CODE1'\n")
    o1, f1 = _texts(tmp_path, "R1")
    assert T.main(["--root", str(tmp_path), "reflect", "--iter", str(d1), "--description", o1,
                   "--feedback", f1]) == 0
    p2 = (tmp_path / "grasp_fj_envelope" / "iter_02" / "prompt.md").read_text()
    assert p2.index("CODE0") < p2.index("OBSERVED R0") < p2.index("CODE1") < p2.index("OBSERVED R1")
    assert "reward/approach:" not in p2          # --events 없으면 지표 표 없음
    hist = (tmp_path / "grasp_fj_envelope" / "history.jsonl").read_text().splitlines()
    assert [json.loads(h)["iter"] for h in hist] == [0, 1]
    with pytest.raises(SystemExit):     # 덮어쓰기는 --force 로만
        T.main(["--root", str(tmp_path), "reflect", "--iter", str(d1), "--description", o1, "--feedback", f1])


def test_reflect_requires_observation_and_feedback(tmp_path, fake_events):
    d0 = _iter(tmp_path, 0)
    empty = tmp_path / "empty.md"
    empty.write_text("  \n")
    _, f0 = _texts(tmp_path, "R0")
    with pytest.raises(SystemExit):
        T.main(["--root", str(tmp_path), "reflect", "--iter", str(d0), "--description", str(empty), "--feedback", f0])
    with pytest.raises(SystemExit):     # argparse: 관찰·피드백 둘 다 필수
        T.main(["--root", str(tmp_path), "reflect", "--iter", str(d0), "--events", "ev"])


def test_reflect_refuses_events_without_generated_terms(tmp_path, monkeypatch):
    fake = types.ModuleType("parse_tfevents")
    fake.load_tfevents = lambda path: {"contact/fingers_touching/iter": [(0, 1.0)]}
    monkeypatch.setitem(sys.modules, "parse_tfevents", fake)
    o0, f0 = _texts(tmp_path, "R0")
    with pytest.raises(SystemExit):
        T.main(["--root", str(tmp_path), "reflect", "--iter", str(_iter(tmp_path, 0)), "--description", o0,
                "--feedback", f0, "--events", "ev"])


def test_tracks_map_to_their_own_gym_ids_and_log_dirs():
    env, reach = R.track("grasp_fj_envelope"), R.track("grasp_fj_reach")
    assert env["task"] == "open-short_r_grasp_fj_t2r-lstm-sapg" and env["logdir"] == "grasp-fj-t2r"
    assert env["sapg"] and env["num_envs"] == 12288
    # ★09.14 사용자 "보상 구조가 확실하지 않은데 SAPG·env 수를 너무 늘린 게 아닌지" → reach 보상 루프는 PPO-LSTM 4096
    assert reach["task"] == "open-short_r_grasp_fj_t2r_reach-lstm" and reach["logdir"] == "grasp-fj-t2r-reach"
    assert reach["play"] == "open-short_r_grasp_fj_t2r_reach-play-lstm"
    assert not reach["sapg"] and reach["num_envs"] == 4096
    assert reach["server_logdir"].endswith("/open-short/right/grasp-fj-t2r-reach")
    cmd = R.launch_command("fj_reach_i01", "reward_gen/grasp_fj_reach/iter_01", reach["num_envs"], 42,
                           task=reach["task"], sapg=reach["sapg"])
    for tok in ("TASK=open-short_r_grasp_fj_t2r_reach-lstm ", "ENVS=4096", "GPU=0", "SERVER=1",
                "EXTRA='env.reward_code_path=/home/oem/rl_ws/hdgp/reward_gen/grasp_fj_reach/iter_01/compute_reward.py'",
                "scripts/experiments/run_fj.sh", "< /dev/null &"):
        assert tok in cmd, tok
    assert "expl_coef_block_size" not in cmd and "BLK=" not in cmd, "PPO-LSTM 설정에는 SAPG 블록 키가 없다"
    with pytest.raises(ValueError):
        R.launch_command("x", "reward_gen/grasp_fj_reach/iter_01", 4000, 42, task=reach["task"], sapg=False)
    vid = R.video_command("/x/fj_reach_i01", "fj_reach_i01", 0.1, "0914_1500", task=reach["task"], play_task=reach["play"])
    assert "--task open-short_r_grasp_fj_t2r_reach-play-lstm " in vid
    assert "open-short_r_grasp_fj_t2r_reach-lstm.pth" in vid
    with pytest.raises(SystemExit):
        R.track("pour_bi")          # 붓기 트랙은 이 도구가 다루지 않는다


def test_judge_ends_the_round_early_when_a_stage_is_stuck_behind_a_saturated_one():
    # ★09.14 사용자: 앞 단계 ≥ 0.9 인데 다음 단계가 0 으로 200 epoch 이상 안 오르면 e1000 전이라도 라운드 끝
    #   (reach i00 e571: 접근 0.98 · 파지 0 · 손바닥 접촉 0.61 · 손가락 0.03)
    stuck = {**_summ(0.0), "stage/reach_ep": {"now": 0.98, "ago": 0.95}, "stage/grasp_ep": {"now": 0.0, "ago": 0.01}}
    verdict, info = R.judge(stuck, {**ALIVE, "epoch": 700}, 2.2)
    assert verdict == "advance(stuck:grasp)" and info["stage_stuck"] == "grasp", info
    # 앞 단계가 창 안에서 막 포화됐으면(ago < 0.9) 아직 아니다 — 퍼널이 오르는 중이라 continue(stage)
    rising = {**stuck, "stage/reach_ep": {"now": 0.98, "ago": 0.73}}
    assert R.judge(rising, {**ALIVE, "epoch": 571}, 1.8)[0] == "continue(stage)"
    # 뒤 단계(리프트)가 오르는 중이면 인벨롭이 0 이어도 막힌 것이 아니다(퍼널은 포함 관계를 강제하지 않는다)
    lifting = {**_summ(0.0), "stage/grasp_ep": {"now": 0.95, "ago": 0.92}, "stage/envelope_ep": {"now": 0.0, "ago": 0.0},
               "stage/lift_ep": {"now": 0.30, "ago": 0.10}}
    assert R.judge(lifting, {**ALIVE, "epoch": 700}, 2.2)[0] == "continue(stage)"
    # 창이 차기 전(epoch < TOL_WINDOW)에는 끊지 않는다 · 성공이 유지 중이면 성공 규칙이 먼저다
    assert R.judge(stuck, {**ALIVE, "epoch": 150}, 0.6)[0] != "advance(stuck:grasp)"
    assert R.judge({**stuck, R.SUCCESS_TAG: {"last": 2.5, "now": 2.5}}, {**ALIVE, "epoch": 700}, 2.2)[0] == "continue(success)"


def test_reflect_carries_the_variant_into_the_next_prompt(tmp_path, fake_events):
    d0 = _iter(tmp_path, 0)
    meta = json.loads((d0 / "meta.json").read_text())
    (d0 / "meta.json").write_text(json.dumps({**meta, "variant": "reach"}))
    o0, f0 = _texts(tmp_path, "R0")
    assert T.main(["--root", str(tmp_path), "reflect", "--iter", str(d0), "--description", o0, "--feedback", f0]) == 0
    nxt = tmp_path / "grasp_fj_envelope" / "iter_01"
    assert "roughly 0.38 m" in (nxt / "prompt.md").read_text()
    assert json.loads((nxt / "meta.json").read_text())["variant"] == "reach"


def test_reach_ppo_agent_yaml_builds_under_the_vendored_rl_games_fork():
    # ★09.14 fj_reach_i01 크래시: run_fj.sh 는 PPO 여도 vendor/rl_games_sapg 를 PYTHONPATH 앞에 둔다. fork 는 fixed_sigma 가
    #   문자열('fixed' 공유 Parameter · 'coef_cond' 블록별 · 그 밖 = Linear)이라 상류 불리언 True 는 sigma 를 Linear 로 만들어
    #   초기화에서 죽는다. 'fixed' 는 상류(참 문자열)에서도 같은 뜻이다.
    import re
    agents = _HDGP / "source/openarm/openarm/agnostic/tasks/grasp_fj/config/agents"
    ppo = (agents / "rl_games_ppo_lstm_cfg.yaml").read_text(encoding="utf-8")
    sapg = (agents / "rl_games_ppo_lstm_sapg_cfg.yaml").read_text(encoding="utf-8")
    assert re.findall(r"^\s*fixed_sigma:\s*(\S+)", ppo, re.M) == ["fixed"]
    assert re.findall(r"^\s*fixed_sigma:\s*(\S+)", sapg, re.M) == ["coef_cond"]
    assert R.TRACKS["grasp_fj_reach"]["task"].endswith("-lstm"), "reach 루프가 이 PPO-LSTM yaml 을 쓴다"


def test_judge_windows_follow_frames_not_epochs():
    # ★09.14 사용자 "프레임 기준으로 맞춤": ROUND_POLICY 의 epoch 창은 SAPG 12,288 env 기준 — PPO 4096 은 epoch 당 프레임이 1/3
    env = R.track_policy(R.track("grasp_fj_envelope"))
    reach = R.track_policy(R.track("grasp_fj_reach"))
    for k in ("ROUND_EPOCHS", "TOL_WINDOW", "LAST_N"):
        assert env[k] == R.ROUND_POLICY[k], k
    assert reach["ROUND_EPOCHS"] == 3000 and reach["TOL_WINDOW"] == 600 and reach["LAST_N"] == 150
    for k in ("ROUND_EPOCHS", "TOL_WINDOW"):
        assert reach[k] * R.TRACKS["grasp_fj_reach"]["num_envs"] == env[k] * R.TRACKS["grasp_fj_envelope"]["num_envs"]
    assert reach["STAGE_EPS"] == env["STAGE_EPS"] and reach["ROUND_HOURS"] == env["ROUND_HOURS"]
    # 같은 요약이라도 reach 에서는 아직 라운드 끝이 아니다
    s = {**_summ(0.0), "stage/reach_ep": {"now": 0.5, "ago": 0.5}}
    assert R.judge(s, {**ALIVE, "epoch": 1500}, 1.7)[0] == "advance"
    assert R.judge(s, {**ALIVE, "epoch": 1500}, 1.7, reach)[0] == "continue"
