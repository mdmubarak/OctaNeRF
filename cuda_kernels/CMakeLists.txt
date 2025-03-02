cmake_minimum_required(VERSION 3.10)
project(RayTraversalCUDA)
find_package(CUDA REQUIRED)
cuda_add_library(ray_traversal SHARED ray_traversal.cu)
