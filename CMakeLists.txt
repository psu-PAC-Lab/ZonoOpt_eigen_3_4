cmake_minimum_required(VERSION 3.15...3.27)

option(ZONOOPT_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
message(STATUS "ZONOOPT_BUILD_PYTHON_BINDINGS set to: ${ZONOOPT_BUILD_PYTHON_BINDINGS}")

option(ZONOOPT_INSTALL_BUILD "Build with install rules" OFF)
if(ZONOOPT_BUILD_PYTHON_BINDINGS)
    set(ZONOOPT_INSTALL_BUILD OFF CACHE BOOL "Disable install build when building Python bindings" FORCE)
endif()
message(STATUS "ZONOOPT_INSTALL_BUILD set to: ${ZONOOPT_INSTALL_BUILD}")

option(ZONOOPT_USE_NATIVE "Enable native vectorization (march=native)" ON)
if(ZONOOPT_INSTALL_BUILD)
    set(ZONOOPT_USE_NATIVE OFF CACHE BOOL "Disable march=native for install builds" FORCE)
endif()
message(STATUS "ZONOOPT_USE_NATIVE set to: ${ZONOOPT_USE_NATIVE}")

option(ZONOOPT_BUILD_EXAMPLE "Build example executable" OFF)
if(ZONOOPT_BUILD_PYTHON_BINDINGS)
    set(ZONOOPT_BUILD_EXAMPLE OFF CACHE BOOL "Disable example build when building Python bindings" FORCE)
endif()
message(STATUS "ZONOOPT_BUILD_EXAMPLE set to: ${ZONOOPT_BUILD_EXAMPLE}")

set(ZONO_FLOAT "double" CACHE STRING "The underlying floating-point type for ZonoOpt library.")
message(STATUS "ZONO_FLOAT set to: ${ZONO_FLOAT}")

set(ZONO_EPS "1e-12" CACHE STRING "Numerical precision for floating point comparisons in ZonoOpt.")
message(STATUS "ZONO_EPS set to: ${ZONO_EPS}")

# project name
if (ZONOOPT_BUILD_PYTHON_BINDINGS)
    # Scikit-build-core sets these values for you, or you can just hard-code the
    # name and version.
    project(
        ${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX
    )
    
    set(TARGET_VAR _core)
else()
    project(ZonoOpt
        VERSION 2.0.1
        LANGUAGES CXX
    )
    set(TARGET_VAR ZonoOpt)
endif()

# release / debug
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'Release' by default.")
  set(CMAKE_BUILD_TYPE Release)
endif()

if (ZONOOPT_BUILD_PYTHON_BINDINGS)
    # Find the module development requirements (requires FindPython from 3.17 or
    # scikit-build-core's built-in backport)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
endif()

# submodule
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    message(STATUS "Updating Git submodules...")

    # Execute 'git submodule update --init --recursive'
    execute_process(
        COMMAND git submodule update --init --recursive
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE GIT_SUBMODULE_RESULT
        OUTPUT_VARIABLE GIT_SUBMODULE_OUTPUT
        ERROR_VARIABLE GIT_SUBMODULE_ERROR
    )

    if (NOT GIT_SUBMODULE_RESULT EQUAL "0")
        message(STATUS "Failed to update Git submodules: ${GIT_SUBMODULE_ERROR}")
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
if (ZONOOPT_BUILD_PYTHON_BINDINGS)
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
)

# Eigen
add_library(Eigen_Headers INTERFACE)

target_include_directories(Eigen_Headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen>
    $<INSTALL_INTERFACE:include/extern/eigen>
)

target_link_libraries(${TARGET_VAR} PUBLIC Eigen_Headers)

# C++ standard
target_compile_features(${TARGET_VAR} 
    PUBLIC cxx_std_17
)

# preprocessor definitions
target_compile_definitions(${TARGET_VAR} PRIVATE
    zono_float=${ZONO_FLOAT}
    zono_eps=${ZONO_EPS}
)

# debug flags
if (MSVC)
    target_compile_options(${TARGET_VAR} PRIVATE /bigobj /W3)
else()
    target_compile_options(${TARGET_VAR} PRIVATE -Wall -Wextra -Wpedantic)
endif()

# march native
if (ZONOOPT_USE_NATIVE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${TARGET_VAR} PUBLIC -march=native)
    elseif(MSVC)
        target_compile_options(${TARGET_VAR} PUBLIC /arch:AVX2)
    endif()
endif()

# C++ example
if (ZONOOPT_BUILD_EXAMPLE)
    # build test executable
    add_subdirectory(test)
endif()

if (ZONOOPT_INSTALL_BUILD)
    include(CMakePackageConfigHelpers)

    # Install the target
    install(TARGETS Eigen_Headers EXPORT ZonoOptTargets)
    install(TARGETS ${TARGET_VAR}
            EXPORT ZonoOptTargets
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin
            INCLUDES DESTINATION include
    )

    # Install headers
    install(DIRECTORY include/ DESTINATION include)
    install(DIRECTORY extern/eigen/ DESTINATION include/extern/eigen)

    # Export the targets
    install(EXPORT ZonoOptTargets
            FILE ZonoOptTargets.cmake
            DESTINATION lib/cmake/ZonoOpt
    )

    # Create and install the config files
    configure_package_config_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ZonoOptConfig.cmake.in"
            "${CMAKE_CURRENT_BINARY_DIR}/ZonoOptConfig.cmake"
            INSTALL_DESTINATION lib/cmake/ZonoOpt
    )

    write_basic_package_version_file(
            "${CMAKE_CURRENT_BINARY_DIR}/ZonoOptConfigVersion.cmake"
            VERSION ${PROJECT_VERSION}
            COMPATIBILITY SameMajorVersion
    )

    install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/ZonoOptConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/ZonoOptConfigVersion.cmake"
            DESTINATION lib/cmake/ZonoOpt
    )
endif()