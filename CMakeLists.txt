# Reference:
# https://github.com/pybind/scikit_build_example/blob/master/CMakeLists.txt

# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
python_add_library(_core MODULE python/src/zonoopt_py.cpp WITH_SOABI)

# link libraries
target_link_libraries(_core PRIVATE pybind11::headers)

# This is passing in the version as a define just as an example
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

# includes
include_directories(
    include
    extern/eigen
)

# compiler flags
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    target_compile_options(_core PRIVATE -march=native)
endif()

if(MSVC)
    target_compile_options(_core PRIVATE /bigobj /W3)
else()
    target_compile_options(_core PRIVATE -Wall -Wextra)
endif()

# set_target_properties(_core PROPERTIES COMPILE_WARNING_AS_ERROR ON)

# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION zonoopt)