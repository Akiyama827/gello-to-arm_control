cmake_minimum_required(VERSION 3.16)
project(arm_rt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
add_compile_options(-Wall -Wextra)

# libfranka is optional so the fake/dm paths (and the selfcheck) build on any
# machine. ON requires the libfranka headers + shared object.
option(WITH_FRANKA "Build the FR3 backend against libfranka" OFF)

add_executable(arm_rt_server
  src/main.cpp
  src/rt_loop.cpp
  src/udp_link.cpp
  src/control_session.cpp
  src/backend_fake.cpp
  src/backend_franka.cpp
  src/backend_dm.cpp
)
target_include_directories(arm_rt_server PRIVATE include)
find_package(Threads REQUIRED)
target_link_libraries(arm_rt_server PRIVATE Threads::Threads)

if(WITH_FRANKA)
  find_package(Franka REQUIRED)
  target_compile_definitions(arm_rt_server PRIVATE ARM_RT_WITH_FRANKA)
  target_link_libraries(arm_rt_server PRIVATE Franka::Franka)
endif()

add_executable(protocol_selfcheck src/selfcheck.cpp)
target_include_directories(protocol_selfcheck PRIVATE include)
