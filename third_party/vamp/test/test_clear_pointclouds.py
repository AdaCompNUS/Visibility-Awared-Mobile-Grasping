import vamp


def main():
    """
    Tests the clear_pointclouds functionality in the Environment.
    """
    # 1. Create an environment
    env = vamp.Environment()
    print("Created vamp.Environment")

    # 2. Create a dummy pointcloud as a list of lists
    points = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
    print(f"Created a dummy pointcloud with {len(points)} points.")

    # 3. Add pointcloud to the environment
    env.add_pointcloud(points, 0.0, 1.0, 0.05)
    print("Added pointcloud to the environment.")

    # 4. Clear the pointclouds
    try:
        print("Attempting to call env.clear_pointclouds()...")
        env.clear_pointclouds()
        print("Successfully called env.clear_pointclouds().")
        print(
            "\nTest passed! The 'clear_pointclouds' method is available on the Environment object."
        )

    except AttributeError:
        print(
            "\nTest failed! The 'clear_pointclouds' method does not exist on the Environment object."
        )
        print(
            "This indicates that the C++ code changes have not been compiled and installed correctly."
        )
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
