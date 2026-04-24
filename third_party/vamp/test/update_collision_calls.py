import re


def process_file(filename):
    with open(filename, "r") as f:
        content = f.read()

    # Pattern for sphere_environment_in_collision with 4 parameters (not counting environment)
    # Capture each parameter separately
    pattern1 = r"sphere_environment_in_collision\s*\(\s*environment\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,)]+)\s*\)"

    # Apply the transformation to the coordinates
    def transform_coords(match):
        env = "environment"
        x = match.group(1)
        y = match.group(2)
        z = match.group(3)
        r = match.group(4)

        # Apply the transformation as in validity.hh:
        # const auto cos_theta = std::cos(base_theta);
        # const auto sin_theta = std::sin(base_theta);
        # const auto tx = sx * cos_theta - sy * sin_theta + base_x;
        # const auto ty = sx * sin_theta + sy * cos_theta + base_y;

        # Instead of directly adding parameters, we'll replace the coordinates with the transformed ones
        return (
            f"sphere_environment_in_collision({env}, "
            + f"({x}) * std::cos(base_theta) - ({y}) * std::sin(base_theta) + base_x, "
            + f"({x}) * std::sin(base_theta) + ({y}) * std::cos(base_theta) + base_y, "
            + f"{z}, {r})"
        )

    # First, process sphere_environment_in_collision calls
    modified_content = re.sub(pattern1, transform_coords, content)

    # Add the base parameter declarations at the beginning of each function
    # Find all functions that contain sphere_environment_in_collision calls
    function_pattern = (
        r"(template\s*<[^>]+>\s*inline\s+[^\n]+\s*\([^)]*\)\s*noexcept\s*\{)"
    )

    def add_base_declarations(match):
        function_header = match.group(1)
        return (
            function_header
            + "\n        double base_theta = -1.423f;\n        double base_x = -2.80515f;\n        double base_y = 0.03805f;\n"
        )

    # Add base parameter declarations to functions
    modified_content = re.sub(function_pattern, add_base_declarations, modified_content)

    # Write the modified content back
    with open(filename + ".modified", "w") as f:
        f.write(modified_content)

    print("Modified file has been written to", filename + ".modified")


# Use the script
if __name__ == "__main__":
    filename = "src/impl/vamp/robots/fetch/fk.hh"
    process_file(filename)
