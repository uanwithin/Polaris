cmake_minimum_required(VERSION 3.28)
project(Polaris VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# 整个项目统一 C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(POLARIS_WITH_RERUN "Build the Rerun visualization" ON)
option(POLARIS_FETCH_DEPENDENCIES "Download missing dependencies" ON)
option(POLARIS_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)

add_subdirectory(third_party)

add_library(polaris_core
  src/geometry.cpp
  src/grid_map.cpp
  src/reference_path.cpp
  src/mppc.cpp
  src/simulation.cpp)
add_library(polaris::core ALIAS polaris_core)
target_include_directories(polaris_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(polaris_core PRIVATE polaris::nlopt)
target_compile_features(polaris_core PUBLIC cxx_std_20)

add_executable(polaris_demo src/main.cpp)
target_link_libraries(polaris_demo PRIVATE polaris::core)

if(POLARIS_WITH_RERUN)
  target_sources(polaris_demo PRIVATE src/visualizer.cpp)
  target_link_libraries(polaris_demo PRIVATE polaris::rerun)
  target_compile_definitions(polaris_demo PRIVATE POLARIS_WITH_RERUN=1)
  configure_file(resources/polaris.rbl polaris.rbl COPYONLY)
  target_compile_definitions(polaris_demo PRIVATE
    POLARIS_BLUEPRINT_PATH="${CMAKE_CURRENT_BINARY_DIR}/polaris.rbl")
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(polaris_tests tests/tests.cpp)
  target_link_libraries(polaris_tests PRIVATE polaris::core)
  add_test(NAME geometry_and_planner COMMAND polaris_tests)
  add_test(NAME simulation_empty COMMAND polaris_tests empty)
  add_test(NAME simulation_avoidance COMMAND polaris_tests avoidance)
  add_test(NAME simulation_blocked COMMAND polaris_tests blocked)
  set_tests_properties(simulation_empty simulation_avoidance simulation_blocked PROPERTIES TIMEOUT 180)
endif()

set(polaris_targets polaris_core polaris_demo)
if(BUILD_TESTING)
  list(APPEND polaris_targets polaris_tests)
endif()
foreach(target IN LISTS polaris_targets)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    if(POLARIS_SANITIZERS)
      target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
      target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
  endif()
endforeach()

find_program(CLANG_FORMAT NAMES clang-format-23 clang-format-22 clang-format)
if(CLANG_FORMAT)
  file(GLOB_RECURSE polaris_format_files CONFIGURE_DEPENDS include/polaris/*.hpp src/*.cpp tests/*.cpp)
  add_custom_target(format COMMAND ${CLANG_FORMAT} -i ${polaris_format_files} VERBATIM)
  add_custom_target(format-check COMMAND ${CLANG_FORMAT} --dry-run --Werror ${polaris_format_files} VERBATIM)
endif()
