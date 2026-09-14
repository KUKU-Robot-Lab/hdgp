"""loop_vlm — prompts for the generator agent and the re-query parse (design auto-loop §6; no Isaac)."""

from openarm.agnostic.modules.iker import loop_vlm, prompts

TASK = "Place the shoe on the rack next to the other shoe."
KEYPOINTS = {1: (0.38, -0.06, 0.38), 2: (0.14, -0.06, 0.38), 3: (0.26, -0.11, 0.38), 4: (0.26, -0.02, 0.38)}
STAGE_RESPONSE = '''Plan: move the shoe onto the rack.

```python
import numpy as np
def get_interaction_data(keypoint_coordinates):
    """Stage 1: place shoe_move beside shoe_other."""
    done = False
    if done:
        return
    object_to_interact = "shoe_move"
    keypoint_indices_to_interact = [1, 2, 3, 4]
    grasp_mode = True
    for key in ["1", "2", "3", "4"]:
        keypoint_coordinates[key] = keypoint_coordinates[key] + np.array([0.0, -0.2, 0.12])
    return object_to_interact, keypoint_indices_to_interact, grasp_mode, keypoint_coordinates
```
'''
DONE_RESPONSE = '''```python
def get_interaction_data(keypoint_coordinates):
    """The shoe is on the rack next to the other shoe."""
    done = True
    if done:
        return
```
'''


def test_target_prompt_is_the_single_step_prompt():
    assert loop_vlm.target_prompt(TASK) == prompts.fill_single_step(TASK)


def test_requery_prompt_carries_the_stage_code_and_marks_the_stage_image():
    text = loop_vlm.requery_prompt(TASK, STAGE_RESPONSE)
    assert text.count(prompts.IMAGE_MARKER) == 1
    assert f"Stage 1 image: {loop_vlm.STAGE_IMAGE_MARKER}" in text
    assert 'object_to_interact = "shoe_move"' in text
    assert text.index("## History") < text.index(prompts.QUERY_HEADER)


def test_requery_result_is_done_only_when_the_function_returns_none():
    assert loop_vlm.requery_result(DONE_RESPONSE, KEYPOINTS) == {"done": True, "detail": "get_interaction_data returned None (done=True)"}
    new_stage = loop_vlm.requery_result(STAGE_RESPONSE, KEYPOINTS)
    assert new_stage["done"] is False and new_stage["detail"] == "new stage: move shoe_move keypoints [1, 2, 3, 4]"
    broken = loop_vlm.requery_result("no code block here", KEYPOINTS)
    assert broken["done"] is False and broken["detail"].startswith("G1: expected exactly one python code block")


def test_generator_brief_names_only_the_given_files():
    brief = loop_vlm.generator_brief("/s/prompt.md", "/s/attempt_00/response.md", {prompts.IMAGE_MARKER: "/s/snapshot.png"})
    assert brief.splitlines()[0] == "Read the prompt file /s/prompt.md and answer it exactly as it instructs."
    assert "The image for [IMAGE_WITH_KEYPOINTS] is /s/snapshot.png; open it with the Read tool." in brief
    assert "/s/attempt_00/response.md" in brief and "Do not open, list or search any other file" in brief
