cmake_minimum_required(VERSION 3.12)
project(test_zonoopt)

# set mode (Debug or Release)
set(CMAKE_BUILD_TYPE Debug)

include_directories(
    ../include
    ../extern/eigen
)

# add source files
add_executable(test_polymorphism
    src/test_polymorphism.cpp
)
