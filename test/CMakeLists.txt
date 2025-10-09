cmake_minimum_required(VERSION 3.12)
project(test_zonoopt)

# compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

if (MSVC)
    add_compile_options(/bigobj /W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wsign-compare)
endif()

# include path
include_directories(
        ../include
        ../extern/eigen
)

# add source files
add_executable(example_operations
        src/example_operations.cpp
)

set_target_properties(example_operations PROPERTIES COMPILE_WARNING_AS_ERROR ON)