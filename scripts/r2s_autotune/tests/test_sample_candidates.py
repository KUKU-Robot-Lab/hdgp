import numpy as np
import pytest

from r2s_autotune.config import load_config
from r2s_autotune.gain_matrix import build_gain_matrices
from r2s_autotune.sample_candidates import sample_candidates
from r2s_autotune.seed_from_config import seed_from_config


def test_candidate_zero_is_the_seed(config_path):
    config = load_config(config_path)
    seed = seed_from_config(config)

    candidates = sample_candidates(config, seed)

    assert candidates[0].groups == seed.groups


def test_population_size_is_respected(config_path):
    config = load_config(config_path)

    candidates = sample_candidates(config, seed_from_config(config))

    assert len(candidates) == config.population_size
    assert [c.index for c in candidates] == list(range(config.population_size))


def test_sampling_is_deterministic_for_a_given_seed(config_path):
    config = load_config(config_path)
    seed = seed_from_config(config)

    first = sample_candidates(config, seed)
    second = sample_candidates(config, seed)

    assert first == second


def test_sampled_gains_stay_within_configured_scale_range(config_path):
    config = load_config(config_path)
    seed = seed_from_config(config)

    candidates = sample_candidates(config, seed)

    for name in config.tune_groups:
        group = config.groups[name]
        base = seed.groups[name].stiffness
        if base == 0.0:
            continue
        scales = [c.groups[name].stiffness / base for c in candidates[1:]]
        assert min(scales) >= group.stiffness_scale.low - 1e-9
        assert max(scales) <= group.stiffness_scale.high + 1e-9


def test_untuned_groups_keep_seed_values(config_path):
    config = load_config(config_path)
    seed = seed_from_config(config)

    candidates = sample_candidates(config, seed)

    untuned = set(config.groups) - set(config.tune_groups)
    for name in untuned:
        assert all(c.groups[name] == seed.groups[name] for c in candidates)


def test_seed_missing_a_tune_group_is_rejected(config_path):
    config = load_config(config_path)
    seed = seed_from_config(config)
    stripped = seed.with_groups(
        {k: v for k, v in seed.groups.items() if k != config.tune_groups[0]}
    )

    with pytest.raises(ValueError, match="lacks tune groups"):
        sample_candidates(config, stripped)


def test_gain_matrix_fills_every_joint_so_no_actuator_dies(config_path):
    """group에 안 잡힌 관절이 0 gain으로 남으면 손이 흐물해진다."""
    config = load_config(config_path)
    seed = seed_from_config(config)
    candidates = sample_candidates(config, seed)

    num_joints = len(config.manifest.movable_joints)
    group_indices = {
        name: tuple(
            num_joints and config.manifest.movable_joints.index(j) for j in _resolve(config, name)
        )
        for name in config.groups
    }

    matrices = build_gain_matrices(config, candidates, group_indices, num_joints)

    covered = np.zeros(num_joints, dtype=bool)
    for indices in group_indices.values():
        covered[list(indices)] = True
    assert covered.all()
    assert matrices["stiffness"].shape == (len(candidates), num_joints)


def _resolve(config, group_name):
    from r2s_autotune.joint_contract import resolve_group_joints

    return resolve_group_joints(
        config.groups[group_name].joint_names_expr, config.manifest.movable_joints
    )
