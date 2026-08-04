from capo_jax.posthoc_paired_eval import ROLE_COMPARISONS, common_episode_seeds


def test_posthoc_paired_eval_uses_one_common_seed_set():
    seeds_a = common_episode_seeds(40, 10_000)
    seeds_b = common_episode_seeds(40, 10_000)
    assert seeds_a == seeds_b == list(range(10_000, 10_040))


def test_required_role_comparisons_are_available():
    assert ("student_before_refresh", "newly_generated_challenger") in ROLE_COMPARISONS
    assert ("active_incumbent_before_gate", "newly_generated_challenger") in ROLE_COMPARISONS
    assert ("student_before_refresh", "quarantined_incumbent_before_gate") in ROLE_COMPARISONS
    assert ("student_before_refresh", "active_teacher_after_gate") in ROLE_COMPARISONS
