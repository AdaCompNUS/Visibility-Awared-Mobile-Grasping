from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "ikfast_fetch",
            sources=["grasp_anywhere/robot/ik/ikfast/fetch/ikfast_robot.cpp"],
            include_dirs=["grasp_anywhere/robot/ik/ikfast/fetch"],
            language="c++",
        )
    ]
)
