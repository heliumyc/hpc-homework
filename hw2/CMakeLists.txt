cmake_minimum_required(VERSION 3.15)
project(hpc_homework)

set(CMAKE_CXX_STANDARD 14)
set(EXE_NAME main)
set(SOURCE_FILE main.cpp)

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")

add_executable(${EXE_NAME} ${SOURCE_FILE})
add_executable(MMulti1 MMult1.cpp)
add_executable(jacobi jacobi2D-omp.cpp)
add_executable(gs gs2D-omp.cpp)
