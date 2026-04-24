#!/usr/bin/env python

from __future__ import print_function

import os
import sys

from .compile import compile_ikfast

sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

# Build C++ extension by running: 'python setup.py'
# see: https://docs.python.org/3/extending/building.html

ARMS = ["main_arm"]


def main():
    sys.argv[:] = sys.argv[:1] + ["build"]
    robot_name = "fetch"
    compile_ikfast(
        module_name="ikfast_{}".format(robot_name),
        cpp_filename="ikfast_robot.cpp",
    )


if __name__ == "__main__":
    main()
