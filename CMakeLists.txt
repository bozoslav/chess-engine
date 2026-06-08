cmake_minimum_required(VERSION 3.20)

project(chess_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(chess_engine_core
  src/board.cpp
  src/types.cpp
  src/movegen.cpp
)

target_include_directories(chess_engine_core PUBLIC include)

if(MSVC)
  target_compile_options(chess_engine_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(chess_engine_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(chess_engine src/main.cpp)
target_link_libraries(chess_engine PRIVATE chess_engine_core)

include(CTest)

if(BUILD_TESTING)
  add_executable(chess_engine_perft_tests tests/perft_tests.cpp)
  target_link_libraries(chess_engine_perft_tests PRIVATE chess_engine_core)

  add_test(NAME chess_engine_runs COMMAND chess_engine)
  add_test(NAME perft_correctness COMMAND chess_engine_perft_tests)
endif()

add_executable(chess_engine_perft_benchmark benchmarks/perft_benchmark.cpp)
target_link_libraries(chess_engine_perft_benchmark PRIVATE chess_engine_core)
