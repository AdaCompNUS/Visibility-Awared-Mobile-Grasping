import vamp

def test_motion_validation():
    """
    Tests the motion validation function.
    """
    robot = vamp.fetch
    env = vamp.Environment()

    # 1. Test a valid motion in an empty environment
    start_config = [0.0] * robot.dimension()
    goal_config = [0.5] * robot.dimension()

    is_valid = robot.validate_motion(start_config, goal_config, env)
    assert is_valid, "Expected motion to be valid in an empty environment"


if __name__ == "__main__":
    test_motion_validation()
    print("Test passed!")
