import numpy as np
import pytest

from go2_autonomous_driving.real.control_transitions import JointTargetRamp, VelocityRamp


def test_space_decelerates_and_s_reverses_through_zero():
    ramp = VelocityRamp([.5, .4, 1], [1, .8, 2], .02)
    for _ in range(50):
        value = ramp.step([.5, 0, 0])
    np.testing.assert_allclose(value, [.5, 0, 0])
    for _ in range(25):
        value = ramp.step([-.5, 0, 0])
    np.testing.assert_allclose(value, [0, 0, 0], atol=1e-10)
    for _ in range(50):
        value = ramp.step([-.5, 0, 0])
    np.testing.assert_allclose(value, [-.5, 0, 0])
    for _ in range(25):
        value = ramp.step([0, 0, 0])
    np.testing.assert_allclose(value, [0, 0, 0], atol=1e-10)


def test_joint_target_jump_is_bounded_and_reset_uses_measured_joints():
    ramp = JointTargetRamp(12, .02)
    goal = np.full(12, 1., dtype=np.float32)
    first = ramp.step(goal, initial=np.zeros(12))
    np.testing.assert_allclose(first, .24)
    second = ramp.step(-goal)
    assert np.max(np.abs(second - first)) <= .240001
    reset = ramp.step(goal, initial=np.full(12, .6))
    np.testing.assert_allclose(reset, .84)


def test_invalid_limits_do_not_silently_disable_slew_guard():
    with pytest.raises(ValueError):
        JointTargetRamp(float('nan'), .02)
    with pytest.raises(ValueError):
        VelocityRamp([0, .4, 1], [1, .8, 2], .02)
    with pytest.raises(ValueError):
        JointTargetRamp(12, .02, transition_rate=13, transition_seconds=.4)
    with pytest.raises(ValueError):
        JointTargetRamp(12, .02, transition_rate=3, transition_seconds=float('nan'))


def test_gait_phase_boundary_eases_targets_without_hiding_stuck_joint():
    ramp = JointTargetRamp(12, .02, transition_rate=3, transition_seconds=.4)
    q = np.zeros(12, dtype=np.float32)
    ramp.step(q, initial=q, moving=True)
    goal = np.full(12, 1., dtype=np.float32)
    # Entering standing must not cause the previous 0.24 rad target jump.
    np.testing.assert_allclose(ramp.step(goal, moving=False), .06, atol=1e-6)
    for _ in range(10):
        value = ramp.step(goal, moving=False)
    assert np.max(value - q) > .65  # Stuck actual joints remain detectable.
    for _ in range(9):
        ramp.step(goal, moving=False)
    np.testing.assert_allclose(ramp.step(-goal, moving=False), .76, atol=1e-6)
    # A later walking start keeps the existing normal target rate.
    np.testing.assert_allclose(ramp.step(-goal, moving=True), .52, atol=1e-6)
    # Re-arming initializes from current joints, without a stale transition.
    np.testing.assert_allclose(ramp.step(goal, initial=q, moving=False), .24, atol=1e-6)


def test_high_speed_stop_and_reverse_use_full_deceleration_ramp():
    ramp = VelocityRamp([.5, .4, 1], [.5, .4, 1], .02)
    for _ in range(150):
        ramp.step([1.5, 0, 0])
    np.testing.assert_allclose(ramp.value, [1.5, 0, 0])
    for i in range(150):
        value = ramp.step([-1.5, 0, 0])
        np.testing.assert_allclose(value, [1.5 - (i + 1) * .01, 0, 0], atol=1e-9)
    for _ in range(150):
        ramp.step([-1.5, 0, 0])
    np.testing.assert_allclose(ramp.value, [-1.5, 0, 0])
    for _ in range(150):
        ramp.step([0, 0, 0])
    np.testing.assert_allclose(ramp.value, [0, 0, 0])
