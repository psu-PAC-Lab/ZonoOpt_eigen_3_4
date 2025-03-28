cmake_minimum_required(VERSION 3.12)
project(test_zonoopt)

# compiler flags
set(CMAKE_BUILD_TYPE Release) # Release, Debug
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# include path
include_directories(
    ../include
    ../extern/eigen
)

# add source files
add_executable(test_polymorphism
    src/test_polymorphism.cpp
)
