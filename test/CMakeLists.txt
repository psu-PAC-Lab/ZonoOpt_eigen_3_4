cmake_minimum_required(VERSION 3.15)
project(example_operations)

add_executable(example_operations
    src/example_operations.cpp
)
target_link_libraries(example_operations PRIVATE ZonoOpt)