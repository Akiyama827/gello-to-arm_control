cmake_minimum_required(VERSION 3.16)
project(arm_rt_servo LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(arm_rt_servo servo_law_py.cpp)
target_include_directories(arm_rt_servo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
install(TARGETS arm_rt_servo DESTINATION .)
