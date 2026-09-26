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
  # Read-only kinematics probe: no control path, so nothing to deploy to the
  # box -- it runs wherever libfranka can reach the FCI.
  add_executable(flange_probe src/flange_probe.cpp)
  target_link_libraries(flange_probe PRIVATE Franka::Franka Eigen3::Eigen)
  # Same deployed layout as arm_rt_server: the box has no toolchain and no
  # root-installed libfranka, so it must find its shared libs in ./lib.
  set_target_properties(flange_probe PROPERTIES
    BUILD_RPATH "\$ORIGIN/lib"
    INSTALL_RPATH "\$ORIGIN/lib")
  target_link_options(flange_probe PRIVATE -Wl,--disable-new-dtags)

  add_executable(hand_bridge src/hand_bridge.cpp)
  add_executable(hand_bridge_selfcheck src/hand_bridge_selfcheck.cpp)
  target_link_libraries(hand_bridge_selfcheck PRIVATE Franka::Franka Threads::Threads)
  target_compile_options(hand_bridge_selfcheck PRIVATE -UNDEBUG)
  target_link_libraries(hand_bridge PRIVATE Franka::Franka Threads::Threads)
  set_target_properties(hand_bridge PROPERTIES
    BUILD_RPATH "\$ORIGIN/lib"
    INSTALL_RPATH "\$ORIGIN/lib")
  target_link_options(hand_bridge PRIVATE -Wl,--disable-new-dtags)
endif()

add_executable(protocol_selfcheck src/selfcheck.cpp)
add_executable(hand_admission_selfcheck src/hand_admission_selfcheck.cpp)
target_compile_options(hand_admission_selfcheck PRIVATE -UNDEBUG)
target_include_directories(protocol_selfcheck PRIVATE include)

add_executable(pose_hold_selfcheck src/pose_hold_selfcheck.cpp)
target_include_directories(pose_hold_selfcheck PRIVATE include)
target_link_libraries(pose_hold_selfcheck PRIVATE Eigen3::Eigen)
# Assertions and Eigen's allocation guard must survive release builds.
target_compile_options(pose_hold_selfcheck PRIVATE -UNDEBUG)

add_executable(torque_cap_selfcheck src/torque_cap_selfcheck.cpp src/rt_loop.cpp)
target_include_directories(torque_cap_selfcheck PRIVATE include)
target_link_libraries(torque_cap_selfcheck PRIVATE Threads::Threads Eigen3::Eigen)
target_compile_options(torque_cap_selfcheck PRIVATE -UNDEBUG)

add_executable(dm_command_selfcheck src/dm_command_selfcheck.cpp src/rt_loop.cpp)
target_include_directories(dm_command_selfcheck PRIVATE include)
target_link_libraries(dm_command_selfcheck PRIVATE Threads::Threads Eigen3::Eigen)
target_compile_options(dm_command_selfcheck PRIVATE -UNDEBUG)

# FR3 backend + rt_loop against a scripted franka:: surface: builds and runs
# without libfranka or a robot (exception containment, fault attribution).
add_executable(franka_backend_selfcheck src/franka_backend_selfcheck.cpp src/rt_loop.cpp
  src/udp_link.cpp)
target_include_directories(franka_backend_selfcheck PRIVATE include)
target_link_libraries(franka_backend_selfcheck PRIVATE Threads::Threads Eigen3::Eigen)
target_compile_options(franka_backend_selfcheck PRIVATE -UNDEBUG)

# Fault journal ring: causes, truncation, full ring, concurrent producers.
add_executable(fault_journal_selfcheck src/fault_journal_selfcheck.cpp)
target_include_directories(fault_journal_selfcheck PRIVATE include)
target_link_libraries(fault_journal_selfcheck PRIVATE Threads::Threads)
target_compile_options(fault_journal_selfcheck PRIVATE -UNDEBUG)

# RT <-> non-RT transport: mailbox, command admission, envelope freshness.
# Build it with -fsanitize=thread as well (see README); no sockets are opened.
add_executable(ipc_selfcheck src/ipc_selfcheck.cpp src/udp_link.cpp)
target_include_directories(ipc_selfcheck PRIVATE include)
target_link_libraries(ipc_selfcheck PRIVATE Threads::Threads)
target_compile_options(ipc_selfcheck PRIVATE -UNDEBUG)

add_executable(dm_backend_selfcheck src/dm_backend_selfcheck.cpp src/backend_dm.cpp)
target_include_directories(dm_backend_selfcheck PRIVATE include)
target_compile_options(dm_backend_selfcheck PRIVATE -UNDEBUG)
foreach(call socket setsockopt ioctl bind close write recv ppoll)
  target_link_options(dm_backend_selfcheck PRIVATE "-Wl,--wrap=${call}")
endforeach()

add_executable(reference_protocol_selfcheck src/reference_protocol_selfcheck.cpp)
target_include_directories(reference_protocol_selfcheck PRIVATE include)
target_compile_options(reference_protocol_selfcheck PRIVATE -UNDEBUG)

# Numerical parity wrapper delegates to the same pure evaluator as RT.
add_library(reference_probe SHARED src/reference_probe.cpp)
set_target_properties(reference_probe PROPERTIES PREFIX "")
target_include_directories(reference_probe PRIVATE include)
target_compile_options(reference_probe PRIVATE -ffp-contract=off)

find_package(OpenSSL REQUIRED COMPONENTS Crypto)
target_link_libraries(arm_rt_server PRIVATE OpenSSL::Crypto)
add_executable(reference_upload_selfcheck src/reference_upload_selfcheck.cpp)
target_include_directories(reference_upload_selfcheck PRIVATE include)
target_link_libraries(reference_upload_selfcheck PRIVATE OpenSSL::Crypto)
target_compile_options(reference_upload_selfcheck PRIVATE -UNDEBUG -ffp-contract=off)

add_executable(reference_session_selfcheck src/reference_session_selfcheck.cpp)
target_include_directories(reference_session_selfcheck PRIVATE include)
target_link_libraries(reference_session_selfcheck PRIVATE OpenSSL::Crypto Threads::Threads)

add_executable(reference_selection_selfcheck src/reference_selection_selfcheck.cpp)
target_include_directories(reference_selection_selfcheck PRIVATE include)
target_compile_options(reference_selection_selfcheck PRIVATE -UNDEBUG -ffp-contract=off)

add_executable(reference_runtime_selfcheck src/reference_runtime_selfcheck.cpp
  src/rt_loop.cpp src/udp_link.cpp src/control_session.cpp src/backend_fake.cpp)
target_include_directories(reference_runtime_selfcheck PRIVATE include)
target_link_libraries(reference_runtime_selfcheck PRIVATE OpenSSL::Crypto Threads::Threads Eigen3::Eigen)
