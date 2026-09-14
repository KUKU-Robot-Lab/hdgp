"""loop_probe — reading runs, logs, checkpoints and loop files into a Probe (design auto-loop §3, §8; no Isaac)."""

import os

from openarm.agnostic.modules.iker import loop_probe as lp
from openarm.agnostic.modules.iker import loop_state as ls
from openarm.agnostic.modules.iker import run_files

ISAAC = b"/home/u/IsaacLab/_isaac_sim/kit/python/bin/python3\0scripts/reinforcement_learning/rl_games/train.py\0"


def _proc(root, pid, label, cmdline):
    entry = root / str(pid)
    entry.mkdir(parents=True)
    (entry / "environ").write_bytes(b"HOME=/h\0RUN_LABEL=" + label.encode() + b"\0")
    (entry / "cmdline").write_bytes(cmdline)


def test_list_checkpoints_reads_saved_and_max_epoch_names(tmp_path):
    nn = tmp_path / "nn"
    nn.mkdir()
    for name in ("last_open-sens_l_iker_shoe_grasp_ep_50_rew_49.66499.pth", "last_open-sens_l_iker_shoe_ep_750_rew__582.5_.pth",
                 "open-sens_l_iker_shoe_grasp.pth", "last_x_frame_100_rew_1.pth",
                 "last_open-sens_l_iker_shoe_grasp_ep_500_rew__229.35422_.pth", "last_open-sens_l_iker_shoe_grasp_ep_500_rew_229.35422.pth"):
        (nn / name).write_bytes(b"")
    found = lp.list_checkpoints(nn)
    assert sorted(found) == [50, 500, 750] and found[50].endswith("ep_50_rew_49.66499.pth") and os.path.isabs(found[50])
    assert found[500].endswith("ep_500_rew_229.35422.pth")
    assert lp.list_checkpoints(tmp_path / "missing") == {}


def test_label_pids_match_the_exact_label_on_isaac_python_only(tmp_path):
    proc = tmp_path / "proc"
    _proc(proc, 11, "iker_grasp_c00_a", b"/bin/bash\0../IsaacLab/_isaac_sim/python.sh\0train.py\0")
    _proc(proc, 12, "iker_grasp_c00_a", ISAAC)
    _proc(proc, 13, "iker_grasp_c00_ab", ISAAC)
    (proc / "self").mkdir()
    assert lp.label_pids("iker_grasp_c00_a", proc) == [12]


def test_training_status_finishes_on_the_marker_or_max_epoch_and_crashes_when_it_dies_early():
    running = lp.training_status("epoch 3", True, {}, 1000, 5.0)
    assert running.alive and not running.finished and not running.crashed and running.idle_s == 5.0
    assert lp.training_status("MAX EPOCHS NUM!", False, {}, None, 0.0).finished
    assert lp.training_status("", False, {1000: "p"}, 1000, 0.0).finished
    died = lp.training_status("", False, {500: "p"}, 1000, 0.0)
    assert died.crashed and died.marker == ""
    trace = lp.training_status("Traceback (most recent call last):\nKeyError: 'x'\n", True, {}, 1000, 0.0)
    assert trace.crashed and trace.marker.startswith("Traceback")


def test_side_status_reads_the_last_result_marker_and_the_excepthook_marker():
    failed = lp.side_status("calibrate", "QUALITY config 00 {'envs': 512} passed False: 12 latch/success moments < 64\n", False, False, 1.0)
    assert failed.finished and failed.passed is False and not failed.crashed
    text = "HARVEST seed 0 envs 512 first-episode successes 40 verified 30\nHARVEST config 00 checkpoint x captured 90 verified 70 (min 64) passed True -> out\n"
    passed = lp.side_status("harvest", text, True, False, 130.0)
    assert passed.finished and passed.passed and passed.alive and passed.marker.startswith("HARVEST config")
    evaluated = lp.side_status("eval", 'EVAL {"success_5cm_end": 0.3}\n', False, False, 0.0)
    assert evaluated.finished and evaluated.passed
    broke = lp.side_status("observe_render", "Traceback\nOBSERVE FAILED\n", False, False, 0.0)
    assert broke.crashed and broke.marker == "OBSERVE FAILED"
    running = lp.side_status("env_smoke", "SMOKE obs (16, 38) finite True bank 80\n", True, False, 0.0)
    assert not running.finished and not running.crashed
    video = lp.side_status("video", "Traceback (most recent call last): harmless extension warning\n", False, True, 0.0)
    assert video.finished and not video.crashed


def test_boot_reward_parses_the_last_reward_line_and_gpu_memory_reading():
    line = "[iker_grasp] reward Stage1RewardCfg(palm_scale=50.0, latch_steps=3, g_min=0.5, q_lo=0.1234567890123, q_hi=0.4) · idle income 0\n"
    assert lp.boot_reward("noise\n" + line) == {"g_min": 0.5, "q_lo": 0.1234567890123, "q_hi": 0.4}
    assert lp.boot_reward("nothing") is None
    assert lp.parse_gpu_used_mib("13041\n") == 13041


def test_find_run_dir_takes_the_newest_folder_made_since_the_launch(tmp_path):
    task = tmp_path / "iker-shoe"
    old, new = task / "iker_vlm_c00_s1", task / "iker_vlm_c00_s1-r1"
    old.mkdir(parents=True)
    new.mkdir()
    os.utime(old, (1000.0, 1000.0))
    os.utime(new, (5000.0, 5000.0))
    assert lp.find_run_dir(task, "iker_vlm_c00_s1", 4000.0) == new
    assert lp.find_run_dir(task, "iker_vlm_c00_s1", 9000.0) is None
    assert lp.find_run_dir(tmp_path / "missing", "iker_vlm_c00_s1", 0.0) is None


def test_collect_reads_runs_events_checkpoints_and_the_calibration(tmp_path):
    paths = lp.LoopPaths.of(tmp_path / "hdgp", 0, "iker_shoe_c00")
    state = ls.new_state("t", phase="calibrate")
    label = state["policy"]["labels"]["stage1_a"]
    run_dir = paths.task_dir("stage1_a") / label
    (run_dir / "nn").mkdir(parents=True)
    (run_dir / "summaries").mkdir()
    (run_dir / "summaries" / "events.out.tfevents.1").write_bytes(b"")
    (run_dir / "nn" / "last_open-sens_l_iker_shoe_grasp_ep_250_rew_1.0.pth").write_bytes(b"")
    train_log = paths.train_log(label)
    train_log.write_text("epoch\n")
    side_log = paths.side_log("calibrate", "ep250")
    side_log.parent.mkdir(parents=True)
    side_log.write_text("QUALITY config 00 {} passed True -> x\n")
    state["runs"] = {
        "stage1_a": {"label": label, "log": str(train_log), "started_s": 0.0, "key": label},
        "calibrate": {"label": "iker_shoe_c00_calibrate", "log": str(side_log), "started_s": 0.0, "key": "k"},
    }
    proc = tmp_path / "proc"
    _proc(proc, 7, label, ISAAC)
    run_files.write_json(paths.calibration_file, {"schema": 1, "checkpoint": "k", "q_lo": 0.1, "q_hi": 0.3})
    events = {lp.LATCHED_TAG: [(1, 0.1), (2, 0.2)], lp.OVER_RACK_TAG: [(1, 0.0)]}
    probe = lp.collect(state, paths, now_s=train_log.stat().st_mtime + 3.0, gpu_used_mib=13000,
                       load_events=lambda path: events, proc_root=proc)
    assert probe.runs["stage1_a"].alive and not probe.runs["stage1_a"].crashed and probe.runs["stage1_a"].idle_s >= 3.0
    assert probe.runs["calibrate"].finished and probe.runs["calibrate"].passed and not probe.runs["calibrate"].alive
    assert probe.latched == ((1, 0.1), (2, 0.2)) and probe.over_rack == ((1, 0.0),) and list(probe.checkpoints) == [250]
    assert probe.calibration["q_hi"] == 0.3 and probe.gpu_used_mib == 13000 and probe.boot_reward is None
    assert probe.attempts == () and probe.final_rows is None and probe.files["stage_prompt"] is False


def test_attempts_evals_final_rows_and_files_come_from_the_stage_directory(tmp_path):
    paths = lp.LoopPaths.of(tmp_path, 0, "iker_shoe_c00")
    stage = paths.stage_dir
    for index, passed in ((0, False), (1, None)):
        attempt = stage / f"attempt_{index:02d}"
        attempt.mkdir(parents=True)
        (attempt / "response.md").write_text("```python\n```\n")
        if passed is not None:
            run_files.write_json(attempt / "gate.json", {"schema": 1, "passed": passed, "failures": []})
    (stage / "attempt_02").mkdir()
    run_files.write_json(paths.eval_file(250), {"schema": 1, "summary": {"success_5cm_end": 0.3}})
    run_files.write_json(paths.final_states_file, {"schema": 1, "rows": [{"env": 4, "success": True, "end_dist": 0.02, "joint_pos": []}]})
    for name in ("prompt.md", "snapshot.png", "keypoints.json"):
        (stage / name).write_text("x")
    assert lp.attempts(stage) == (ls.AttemptStatus(0, False), ls.AttemptStatus(1, None))
    assert lp.evals(stage) == {250: {"success_5cm_end": 0.3}}
    assert lp.final_rows(paths.final_states_file) == ({"env": 4, "success": True, "end_dist": 0.02},)
    probe = lp.collect(ls.new_state("t", phase="vlm_target"), paths, now_s=0.0, gpu_used_mib=0, load_events=lambda path: {},
                       proc_root=tmp_path / "no_proc")
    assert probe.files["stage_prompt"] and not probe.files["requery_prompt"] and not probe.files["video_raw"]
    assert probe.bank_meta is None and probe.checkpoints == {} and len(probe.attempts) == 2
