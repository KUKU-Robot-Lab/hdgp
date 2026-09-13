"""interaction — code block extraction, restricted execution, result parsing (no Isaac)."""

import numpy as np
import pytest

from openarm.agnostic.modules.iker.interaction import InteractionError, check_code, extract_code_block, run_interaction

KEYPOINTS = {1: (0.39, 0.14, 0.25), 2: (0.15, 0.14, 0.25), 3: (0.27, 0.09, 0.25), 4: (0.27, 0.19, 0.25), 5: (0.40, -0.15, 0.38)}
GOOD_RESPONSE = '''The shoe goes on the rack.
```python
import numpy as np

def get_interaction_data(keypoint_coordinates):
    """Move the left shoe next to the other shoe."""
    object_to_interact = "left shoe"
    keypoint_indices_to_interact = [1, 2, 3, 4]
    grasp_mode = True
    for k in ["1", "2", "3", "4"]:
        keypoint_coordinates[k] = keypoint_coordinates[k] + np.array([0.0, -0.2, 0.12])
    return object_to_interact, keypoint_indices_to_interact, grasp_mode, keypoint_coordinates
```
'''


def _function(body: str) -> str:
    return "def get_interaction_data(k):\n" + "".join(f"    {line}\n" for line in body.splitlines())


def test_good_response_runs_and_parses():
    result = run_interaction(extract_code_block(GOOD_RESPONSE), KEYPOINTS)
    assert (result.object_name, result.keypoint_ids, result.grasp_mode, result.done) == ("left shoe", (1, 2, 3, 4), True, False)
    assert np.allclose(result.coordinates[1], (0.39, -0.06, 0.37))
    assert np.allclose(result.coordinates[5], KEYPOINTS[5])


def test_caller_keypoints_are_not_mutated():
    arrays = {k: np.array(v) for k, v in KEYPOINTS.items()}
    run_interaction(extract_code_block(GOOD_RESPONSE), arrays)
    assert all(np.array_equal(arrays[k], np.array(v)) for k, v in KEYPOINTS.items())


def test_string_ids_are_accepted():
    code = _function("return 'shoe', ['1', '2'], False, k")
    assert run_interaction(code, KEYPOINTS).keypoint_ids == (1, 2)


@pytest.mark.parametrize("response", ["no code at all", "```python\nx = 1\n```\n```python\ny = 2\n```"])
def test_exactly_one_python_block_is_required(response):
    with pytest.raises(InteractionError, match="exactly one"):
        extract_code_block(response)


@pytest.mark.parametrize(
    "code, fragment",
    [
        ("import os", "only numpy"),
        ("from numpy import array", "from-imports"),
        (_function("return open('x')"), "open"),
        (_function("return k.__class__"), "attribute"),
        (_function("return eval('1')"), "eval"),
        (_function("return __builtins__"), "__builtins__"),
        ("x = 1\n" + _function("return None"), "top-level"),
        (_function("try:\n    return None\nexcept Exception:\n    return None"), "Try"),
        ("def other(k):\n    return None\n", "no top-level function"),
        ("def get_interaction_data(k:\n", "syntax error"),
    ],
)
def test_forbidden_code_is_rejected_before_running(code, fragment):
    with pytest.raises(InteractionError, match=fragment):
        check_code(code)


def test_runtime_errors_and_timeouts_become_interaction_errors():
    with pytest.raises(InteractionError, match="ZeroDivisionError"):
        run_interaction(_function("return 1 / 0"), KEYPOINTS)
    with pytest.raises(InteractionError, match="exceeded"):
        run_interaction(_function("while True:\n    pass"), KEYPOINTS, timeout_s=0.2)


def test_multi_step_done_is_reported():
    result = run_interaction(_function("done = True\nif done:\n    return"), KEYPOINTS)
    assert result.done is True and result.keypoint_ids == ()


@pytest.mark.parametrize(
    "body, fragment",
    [
        ("return 1, 2", "must return"),
        ("return '', [1], True, k", "object_to_interact"),
        ("return 'shoe', [], True, k", "keypoint_indices"),
        ("return 'shoe', [1, 1], True, k", "duplicates"),
        ("return 'shoe', [1], 'yes', k", "grasp_mode"),
        ("return 'shoe', [True], True, k", "not an integer"),
        ("k['1'] = [1.0, 2.0]\nreturn 'shoe', [1], True, k", "shape"),
    ],
)
def test_malformed_return_values_are_rejected(body, fragment):
    with pytest.raises(InteractionError, match=fragment):
        run_interaction(_function(body), KEYPOINTS)
