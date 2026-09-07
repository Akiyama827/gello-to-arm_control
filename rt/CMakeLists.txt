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
# Deployed layout on the RT box: binary + its shared libs in ./lib (the box
# has no toolchain and no root-installed libfranka). $ORIGIN must reach the
# ELF verbatim — passing it via -DCMAKE_EXE_LINKER_FLAGS gets eaten by the
# shell inside the ninja link command.
set_target_properties(arm_rt_server PROPERTIES
  BUILD_RPATH "\$ORIGIN/lib"
  INSTALL_RPATH "\$ORIGIN/lib")
# Old dtags = DT_RPATH, which unlike RUNPATH also covers TRANSITIVE loads —
# libfranka's own Poco/fmt dependencies must resolve from ./lib too.
target_link_options(arm_rt_server PRIVATE -Wl,--disable-new-dtags)
target_include_directories(arm_rt_server PRIVATE include)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
target_link_libraries(arm_rt_server PRIVATE Threads::Threads Eigen3::Eigen)

if(WITH_FRANKA)
  find_package(Franka REQUIRED)
  target_compile_definitions(arm_rt_server PRIVATE ARM_RT_WITH_FRANKA)
  target_link_libraries(arm_rt_server PRIVATE Franka::Franka)

  # Franka Hand daemon for the box: the Hand's state stream is server-push
  # UDP, which cannot cross the PC-side NAT — the Hand must be owned from
  # the robot LAN, like the torque loop. See src/hand_bridge.cpp.
  add_executable(hand_bridge src/hand_bridge.cpp)
  target_link_libraries(hand_bridge PRIVATE Franka::Franka)
  set_target_properties(hand_bridge PROPERTIES
    BUILD_RPATH "\$ORIGIN/lib"
    INSTALL_RPATH "\$ORIGIN/lib")
  target_link_options(hand_bridge PRIVATE -Wl,--disable-new-dtags)
endif()

add_executable(protocol_selfcheck src/selfcheck.cpp)
target_include_directories(protocol_selfcheck PRIVATE include)

add_executable(pose_hold_selfcheck src/pose_hold_selfcheck.cpp)
target_include_directories(pose_hold_selfcheck PRIVATE include)
target_link_libraries(pose_hold_selfcheck PRIVATE Eigen3::Eigen)
# Assertions and Eigen's allocation guard must survive release builds.
target_compile_options(pose_hold_selfcheck PRIVATE -UNDEBUG)
