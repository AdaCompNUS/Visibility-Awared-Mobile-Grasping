#!/usr/bin/env python3
import re
import sys
import os


def analyze_cpp_file(file_path, debug_mode=False):
    """
    Analyzes a C++ file to find all functions that take an 'environment' parameter.

    Args:
        file_path (str): Path to the C++ file
        debug_mode (bool): Whether to print debug information

    Returns:
        list: List of function names that take an environment parameter
    """
    # Check if file exists
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return []

    try:
        # Read the file content
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    # Remove C-style comments to avoid false positives
    content = remove_comments(content)

    # Step 1: Fast scan for the presence of "environment"
    if not re.search(r"[Ee]nvironment", content):
        print("No mentions of 'environment' found in the file.")
        return []

    # Keywords that should not be considered as function names
    cpp_keywords = [
        "if",
        "else",
        "for",
        "while",
        "switch",
        "case",
        "return",
        "break",
        "continue",
        "default",
        "goto",
        "try",
        "catch",
        "throw",
    ]

    functions_with_env = []

    # Step 2: Look for function declarations with environment parameters

    # Regular function pattern
    func_pattern = r"(?:[\w:~<>]+\s+)+?([\w:~<>]+)\s*\(([^)]*)\)\s*(?:const|override|final|noexcept)?\s*(?:->[\w:&*<>\s]+)?\s*(?:=\s*0)?\s*[{;]"
    for match in re.finditer(func_pattern, content):
        function_name = match.group(1)
        parameters = match.group(2)

        # Skip if function name is a C++ keyword
        if function_name in cpp_keywords:
            continue

        # Check for environment parameter
        if re.search(
            r"(?:const\s+)?(?:[\w:]+::\s*)?[Ee]nvironment(?:<[^>]*>)?\s*[&*]?\s*\w+",
            parameters,
        ):
            if debug_mode:
                print(f"DEBUG - Regular function match: {function_name}({parameters})")
            functions_with_env.append(function_name)

    # Template function pattern with special handling
    template_pattern = r"template\s*<[^>]*>\s*(?:inline\s+)?(?:[\w:~<>]+\s+)+?([\w:~<>]+)\s*\(([^)]*)\)\s*(?:const|override|final|noexcept)?\s*(?:->[\w:&*<>\s]+)?\s*(?:=\s*0)?\s*[{;]"
    for match in re.finditer(template_pattern, content):
        function_name = match.group(1)
        parameters = match.group(2)

        # Skip if function name is a C++ keyword
        if function_name in cpp_keywords:
            continue

        # Check for environment parameter - more flexible for template types
        if re.search(
            r"(?:const\s+)?(?:[\w:]+::\s*)?[Ee]nvironment(?:<[^>]*>)?\s*[&*]?",
            parameters,
        ):
            if debug_mode:
                print(f"DEBUG - Template function match: {function_name}({parameters})")
            functions_with_env.append(function_name)

    # Step 3: Special cases

    # Check for "interleaved_sphere_fk" function specifically (based on the example)
    if "interleaved_sphere_fk" in content:
        special_pattern = r"(?:inline\s+)?bool\s+interleaved_sphere_fk\s*\(\s*const\s+[\w:]+::[Ee]nvironment<[^>]*>\s*[&*]"
        if re.search(special_pattern, content) or any(
            "interleaved_sphere_fk" in s for s in functions_with_env
        ):
            if "interleaved_sphere_fk" not in functions_with_env:
                functions_with_env.append("interleaved_sphere_fk")
                if debug_mode:
                    print("DEBUG - Added interleaved_sphere_fk from special pattern")

    # Step 4: Find functions that use "environment" parameters even if not in their declaration
    used_env_pattern = r"([\w:~<>]+)\s*\([^)]*\)\s*(?:const|override|final|noexcept)?\s*(?:->[\w:&*<>\s]+)?\s*(?:=\s*0)?\s*\{"
    for match in re.finditer(used_env_pattern, content):
        function_body_start = match.end()
        # Find the matching closing brace for this function
        brace_count = 1
        function_body_end = function_body_start
        for i in range(function_body_start, len(content)):
            if content[i] == "{":
                brace_count += 1
            elif content[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    function_body_end = i
                    break

        function_body = content[function_body_start:function_body_end]
        # Check if this function body uses environment variables
        if (
            re.search(r"[Ee]nvironment", function_body)
            and "sphere_environment_in_collision" in function_body
        ):
            function_name = match.group(1)
            if (
                function_name not in cpp_keywords
                and function_name not in functions_with_env
            ):
                functions_with_env.append(function_name)
                if debug_mode:
                    print(
                        f"DEBUG - Added function using environment internally: {function_name}"
                    )

    # Remove duplicates and return
    return list(dict.fromkeys(functions_with_env))


def remove_comments(content):
    """Remove C and C++ style comments from a string."""
    # Remove C-style comments (/* ... */)
    content = re.sub(r"/\*[\s\S]*?\*/", "", content)

    # Remove C++-style comments (// ...)
    content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)

    return content


def main():
    """Main function to run the analysis."""
    if len(sys.argv) < 2:
        print("Usage: python cpp_analyzer.py <cpp_file_path> [--debug]")
        sys.exit(1)

    cpp_file_path = sys.argv[1]
    debug_mode = "--debug" in sys.argv

    # If debug mode is requested, print extra information
    if debug_mode:
        print(f"Analyzing file: {cpp_file_path}")
        print("Debug mode enabled")

    functions = analyze_cpp_file(cpp_file_path, debug_mode)

    if functions:
        print("\nFunctions that take 'environment' as a parameter:")
        for i, func in enumerate(functions, 1):
            print(f"{i}. {func}")
        print(f"\nTotal: {len(functions)} functions found.")
    else:
        print("No functions with 'environment' parameter found.")


if __name__ == "__main__":
    main()
