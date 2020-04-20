cmake_minimum_required(VERSION 3.15)
project(hw4)

set(CMAKE_CXX_STANDARD 14)

add_executable(hw4 main.cpp inner_product.cu vec_mat_mul.cu)