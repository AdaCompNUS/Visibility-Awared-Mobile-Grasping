# pytracik — ROS-free TRAC-IK Python bindings

A self-contained [TRAC-IK](https://traclabs.com/projects/trac-ik/) inverse-kinematics
solver exposed to Python via pybind11, with a vendored minimal urdfdom/tinyxml parser so
**no ROS is required**. This replaces the `trac_ik_python` ROS package that
`grasp_anywhere.robot.ik.trac_ik_solver` previously depended on.

## Build

The module is built with CMake driven by [scikit-build-core]. `cmake` and `ninja` are pulled
from PyPI by the build (the "pip cmake"); the C/C++ dependencies come from the active
conda/pixi environment:

- `orocos-kdl`, `nlopt`, `eigen`, `libboost-devel` (headers), `pybind11`, a C++ compiler.

With the pixi environment active it is built automatically on `pixi install` (declared as a
path dependency in the repo's `pixi.toml`). To build it by hand:

```bash
pip install ./third_party/pytracik      # uses the pip cmake + the env's orocos-kdl/nlopt
python -c "import pytracik; print(pytracik.SolveType.Distance)"
```

## API

`import pytracik` exposes `TRAC_IK`, the `SolveType` enum (`Speed`/`Distance`/`Manip1`/`Manip2`),
and helpers `ik`, `fk`, `get_num_joints`, `get_joint_lower_bounds`, `get_joint_upper_bounds`,
`set_joint_limits`. See `grasp_anywhere/robot/ik/trac_ik_solver.py` for the wrapper used by the
rest of the codebase.

## Provenance

The C++ sources under `trac_ik/` are vendored from
[AdaCompNUS/Autolife-Planning](https://github.com/AdaCompNUS/Autolife-Planning) (`ext/trac_ik`),
which in turn packages the TRAC-IK library (BSD) and a trimmed urdfdom/tinyxml parser.
