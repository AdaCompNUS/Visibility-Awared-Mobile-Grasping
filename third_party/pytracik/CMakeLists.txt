cmake_minimum_required(VERSION 3.20)
project(pytracik LANGUAGES C CXX)

# ----------------------------------------------------------------------------
# ROS-free TRAC-IK -> Python (pybind11) module `pytracik`.
#
# The C++ sources under trac_ik/ are a vendored copy of the TRAC-IK solver plus a
# minimal urdfdom/tinyxml parser (so no ROS is required). CMake is provided by
# the pip build (scikit-build-core pulls cmake+ninja from PyPI); the C/C++
# dependencies (orocos-kdl, NLopt, Eigen, Boost headers) come from the active
# conda/pixi environment via CMAKE_PREFIX_PATH.
# ----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Boost >= 1.70 prefers BoostConfig.cmake; force the legacy FindBoost interface
# so this builds against both old and modern Boost (matches upstream Autolife).
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()
set(Boost_NO_BOOST_CMAKE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)

# orocos-kdl ships orocos-kdl.pc; its Cflags carry the Eigen and Boost header
# search paths that the vendored kdl_parser/trac_ik sources need.
pkg_check_modules(orocos_kdl REQUIRED IMPORTED_TARGET orocos-kdl)

# NLopt: prefer its CMake config (NLopt::nlopt), fall back to pkg-config.
find_package(NLopt QUIET)
if(TARGET NLopt::nlopt)
    set(TRACIK_NLOPT NLopt::nlopt)
else()
    pkg_check_modules(nlopt REQUIRED IMPORTED_TARGET nlopt)
    set(TRACIK_NLOPT PkgConfig::nlopt)
endif()

# Defensive: add Eigen explicitly if a config is present (orocos-kdl already
# carries the include path, but this is harmless and more portable).
find_package(Eigen3 QUIET NO_MODULE)

file(GLOB_RECURSE TRAC_IK_SOURCES "trac_ik/*.cpp")

pybind11_add_module(pytracik ${TRAC_IK_SOURCES})

target_include_directories(pytracik PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/trac_ik
    ${CMAKE_CURRENT_SOURCE_DIR}/trac_ik/urdf
)

target_link_libraries(pytracik PRIVATE
    PkgConfig::orocos_kdl
    ${TRACIK_NLOPT}
)
if(TARGET Eigen3::Eigen)
    target_link_libraries(pytracik PRIVATE Eigen3::Eigen)
endif()

install(TARGETS pytracik LIBRARY DESTINATION .)
