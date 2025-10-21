# Reference:
# https://github.com/pybind/scikit_build_example/blob/master/CMakeLists.txt

# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
message(STATUS "Building Python Bindings: ${BUILD_PYTHON_BINDINGS}")

set(ZONO_FLOAT "double" CACHE STRING "The underlying floating-point type for ZonoOpt library.")
message(STATUS "ZONO_FLOAT set to: ${ZONO_FLOAT}")

set(ZONO_EPS "1e-12" CACHE STRING "Numerical precision for floating point comparisons in ZonoOpt.")
message(STATUS "ZONO_EPS set to: ${ZONO_EPS}")

# project name
if (BUILD_PYTHON_BINDINGS)
    # Scikit-build-core sets these values for you, or you can just hard-code the
    # name and version.
    project(
        ${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX
    )
    
    set(TARGET_VAR _core)
else()
    project(ZonoOpt LANGUAGES CXX)
    set(TARGET_VAR ZonoOpt)
endif()

# compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

if (BUILD_PYTHON_BINDINGS)
    # Find the module development requirements (requires FindPython from 3.17 or
    # scikit-build-core's built-in backport)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
endif()


include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# submodule
if (EXISTS "${CMAKE_SOURCE_DIR}/.git")
    message(STATUS "Updating Git submodules...")

    # Execute 'git submodule update --init --recursive'
    execute_process(
        COMMAND git submodule update --init --recursive
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE GIT_SUBMODULE_RESULT
        OUTPUT_VARIABLE GIT_SUBMODULE_OUTPUT
        ERROR_VARIABLE GIT_SUBMODULE_ERROR
    )

    if (NOT GIT_SUBMODULE_RESULT EQUAL "0")
        message(FATAL_ERROR "Failed to update Git submodules: ${GIT_SUBMODULE_ERROR}")
    else()
        message(STATUS "Git submodules updated successfully.")
    endif()
else()
    message(STATUS "Not a Git repository, skipping submodule update.")
endif()

# source files
set(SOURCE_FILES
    src/PolymorphicFunctions.cpp
    src/ADMM.cpp
    src/CholeskyUtilities.cpp
    src/HybZono.cpp
    src/ConZono.cpp
    src/Zono.cpp
    src/Point.cpp
    src/EmptySet.cpp
    src/Intervals.cpp
    src/MI_Solver.cpp
)

# build
if (BUILD_PYTHON_BINDINGS)
    # Add a library using FindPython's tooling (pybind11 also provides a helper like
    # this)
    python_add_library(${TARGET_VAR} MODULE
        python/src/zonoopt_py.cpp
        ${SOURCE_FILES}
        WITH_SOABI
    )

    # link libraries
    target_link_libraries(${TARGET_VAR} PRIVATE pybind11::headers)

    # This is passing in the version as a define just as an example
    target_compile_definitions(${TARGET_VAR} PRIVATE VERSION_INFO=${PROJECT_VERSION})

    # The install directory is the output (wheel) directory
    install(TARGETS ${TARGET_VAR} DESTINATION zonoopt)
else()
    add_library(${TARGET_VAR} STATIC
        ${SOURCE_FILES}
    )
endif()

target_include_directories(${TARGET_VAR} PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    SYSTEM $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen>
)

# preprocessor definitions
target_compile_definitions(${TARGET_VAR} PUBLIC
    zono_float=${ZONO_FLOAT}
    zono_eps=${ZONO_EPS}
)

# debug flags
if (MSVC)
    target_compile_options(${TARGET_VAR} PRIVATE /bigobj /W3)
else()
    target_compile_options(${TARGET_VAR} PRIVATE -Wall -Wextra -Wpedantic)
endif()

# C++ example
if (NOT BUILD_PYTHON_BINDINGS)
    # build test executable
    add_subdirectory(test)
endif()