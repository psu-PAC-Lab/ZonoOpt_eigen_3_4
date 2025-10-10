cmake_minimum_required(VERSION 3.12)
project(test_zonoopt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# include path
include_directories(
        ../include
        ../extern/eigen
)

# add source files
add_executable(example_operations
        src/example_operations.cpp
)

# compiler flags
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    target_compile_options(example_operations PRIVATE -march=native)
endif()

target_compile_options(example_operations PRIVATE
    # Non-MSVC flags (e.g., GCC/Clang)
    $<$<NOT:$<BOOL:MSVC>>:-Wall -Wextra>
    
    # MSVC flags
    $<$<BOOL:MSVC>:/bigobj /W3>
)

set_target_properties(example_operations PROPERTIES COMPILE_WARNING_AS_ERROR ON)