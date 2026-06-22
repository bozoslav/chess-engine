cmake_minimum_required(VERSION 3.20)

project(chess_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CHESS_ENGINE_NATIVE_M4
       "Tune Release engine code for the local Apple Silicon CPU" ON)
option(CHESS_ENGINE_IPO "Enable Release link-time optimization" ON)
set(CHESS_ENGINE_PGO "OFF" CACHE STRING
    "Clang instrumentation PGO mode: OFF, GENERATE, or USE")
set_property(CACHE CHESS_ENGINE_PGO PROPERTY STRINGS OFF GENERATE USE)
set(CHESS_ENGINE_PGO_PROFILE "" CACHE FILEPATH
    "PGO raw-profile pattern (GENERATE) or merged .profdata file (USE)")

if(CHESS_ENGINE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT CHESS_ENGINE_IPO_SUPPORTED
                      OUTPUT CHESS_ENGINE_IPO_ERROR)
  if(CHESS_ENGINE_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  else()
    message(WARNING "IPO/LTO unavailable: ${CHESS_ENGINE_IPO_ERROR}")
  endif()
endif()

set(CHESS_ENGINE_DEFAULT_EVALFILE
    "${CMAKE_SOURCE_DIR}/data/stockfish/export/latest.nnue"
    CACHE STRING "Default Stockfish-format NNUE file loaded by the UCI engine")

string(TOUPPER "${CHESS_ENGINE_PGO}" CHESS_ENGINE_PGO_MODE)
if(NOT CHESS_ENGINE_PGO_MODE STREQUAL "OFF")
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang)$")
    message(FATAL_ERROR "CHESS_ENGINE_PGO requires Clang or AppleClang")
  endif()
  if(CHESS_ENGINE_PGO_PROFILE STREQUAL "")
    message(FATAL_ERROR "CHESS_ENGINE_PGO_PROFILE must be set for PGO")
  endif()

  if(CHESS_ENGINE_PGO_MODE STREQUAL "GENERATE")
    add_compile_options(
      $<$<CONFIG:Release>:-fprofile-instr-generate=${CHESS_ENGINE_PGO_PROFILE}>
    )
    add_link_options(
      $<$<CONFIG:Release>:-fprofile-instr-generate=${CHESS_ENGINE_PGO_PROFILE}>
    )
  elseif(CHESS_ENGINE_PGO_MODE STREQUAL "USE")
    if(NOT EXISTS "${CHESS_ENGINE_PGO_PROFILE}")
      message(FATAL_ERROR
              "PGO profile does not exist: ${CHESS_ENGINE_PGO_PROFILE}")
    endif()
    add_compile_options(
      $<$<CONFIG:Release>:-fprofile-instr-use=${CHESS_ENGINE_PGO_PROFILE}>
      $<$<CONFIG:Release>:-Wno-profile-instr-unprofiled>
      $<$<CONFIG:Release>:-Wno-profile-instr-out-of-date>
    )
    add_link_options(
      $<$<CONFIG:Release>:-fprofile-instr-use=${CHESS_ENGINE_PGO_PROFILE}>
    )
  else()
    message(FATAL_ERROR
            "CHESS_ENGINE_PGO must be OFF, GENERATE, or USE")
  endif()
endif()

add_library(chess_engine_core
  src/attacks.cpp
  src/board.cpp
  src/evaluate.cpp
  src/nnue.cpp
  src/see.cpp
  src/search.cpp
  src/transposition_table.cpp
  src/types.cpp
  src/movegen.cpp
  src/uci.cpp
)

target_include_directories(chess_engine_core PUBLIC include)

target_compile_definitions(chess_engine_core PRIVATE
  CHESS_ENGINE_DEFAULT_EVALFILE=\"${CHESS_ENGINE_DEFAULT_EVALFILE}\"
)

if(MSVC)
  target_compile_options(chess_engine_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(chess_engine_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CHESS_ENGINE_NATIVE_M4 AND APPLE AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  target_compile_options(chess_engine_core PRIVATE
    $<$<CONFIG:Release>:-mcpu=native>
  )
endif()

add_executable(chess_engine src/main.cpp)
target_link_libraries(chess_engine PRIVATE chess_engine_core)

include(CTest)

if(BUILD_TESTING)
  add_executable(chess_engine_attacks_tests tests/attacks_tests.cpp)
  target_link_libraries(chess_engine_attacks_tests PRIVATE chess_engine_core)

  add_executable(chess_engine_perft_tests tests/perft_tests.cpp)
  target_link_libraries(chess_engine_perft_tests PRIVATE chess_engine_core)
  target_compile_definitions(chess_engine_perft_tests PRIVATE
    CHESS_ENGINE_TEST_EVALFILE=\"${CMAKE_SOURCE_DIR}/data/stockfish/export/latest.nnue\"
  )

  add_test(NAME chess_engine_runs COMMAND chess_engine)
  add_test(NAME attacks_correctness COMMAND chess_engine_attacks_tests)
  add_test(NAME perft_correctness COMMAND chess_engine_perft_tests)
endif()

add_executable(chess_engine_perft_benchmark benchmarks/perft_benchmark.cpp)
target_link_libraries(chess_engine_perft_benchmark PRIVATE chess_engine_core)

add_executable(chess_engine_search_benchmark benchmarks/search_benchmark.cpp)
target_link_libraries(chess_engine_search_benchmark PRIVATE chess_engine_core)

add_executable(chess_engine_scaling_benchmark benchmarks/scaling_benchmark.cpp)
target_link_libraries(chess_engine_scaling_benchmark PRIVATE chess_engine_core)

add_executable(chess_engine_nnue_benchmark benchmarks/nnue_benchmark.cpp)
target_link_libraries(chess_engine_nnue_benchmark PRIVATE chess_engine_core)

add_executable(chess_engine_allocation_benchmark
  benchmarks/allocation_benchmark.cpp
)
target_link_libraries(chess_engine_allocation_benchmark PRIVATE chess_engine_core)
if(BUILD_TESTING)
  add_test(
    NAME search_allocation_free
    COMMAND chess_engine_allocation_benchmark
            ${CMAKE_SOURCE_DIR}/data/stockfish/export/latest.nnue
  )
endif()

add_executable(chess_engine_attacks_benchmark benchmarks/attacks_benchmark.cpp)
target_link_libraries(chess_engine_attacks_benchmark PRIVATE chess_engine_core)

add_executable(chess_engine_magic_generator EXCLUDE_FROM_ALL
  tools/magic_generator.cpp
)
