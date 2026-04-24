import vamp


def main():
    """
    Tests the clear_spheres functionality in the Environment.
    """
    # 1. Create an environment
    env = vamp.Environment()
    print("Created vamp.Environment")

    # 2. Create a dummy sphere
    sphere = vamp.Sphere([1.0, 2.0, 3.0], 0.5)
    print("Created a dummy sphere.")

    # 3. Add sphere to the environment
    env.add_sphere(sphere)
    print("Added sphere to the environment.")

    # 4. Clear the spheres
    try:
        print("Attempting to call env.clear_spheres()...")
        env.clear_spheres()
        print("Successfully called env.clear_spheres().")
        print(
            "\nTest passed! The 'clear_spheres' method is available on the Environment object."
        )

    except AttributeError:
        print(
            "\nTest failed! The 'clear_spheres' method does not exist on the Environment object."
        )
        print(
            "This indicates that the C++ code changes have not been compiled and installed correctly."
        )
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
