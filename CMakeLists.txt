cmake_minimum_required(VERSION 3.20)

project(chess_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(chess_engine
  src/main.cpp
  src/board.cpp
)

target_include_directories(chess_engine PRIVATE include)

if(MSVC)
  target_compile_options(chess_engine PRIVATE /W4 /permissive-)
else()
  target_compile_options(chess_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)

if(BUILD_TESTING)
  add_test(NAME chess_engine_runs COMMAND chess_engine)
endif()
