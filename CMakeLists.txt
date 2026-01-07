cmake_minimum_required(VERSION 3.20)
project(gpu_ll_infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CXX>:-O3>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo> --extended-lambda --expt-relaxed-constexpr)


set(_DEFAULT_ARCHS 70 75 80 86 89)

add_executable(bench_empty src/bench_main.cu)
target_include_directories(bench_empty PRIVATE include)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET bench_empty PROPERTY CUDA_ARCHITECTURES ${_DEFAULT_ARCHS})
endif()

add_executable(bench_cublas src/bench_cublas_mlp.cu)
target_include_directories(bench_cublas PRIVATE include)
target_link_libraries(bench_cublas PRIVATE cublas)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET bench_cublas PROPERTY CUDA_ARCHITECTURES ${_DEFAULT_ARCHS})
endif()

add_executable(bench_mbyv src/bench_custom_mbyv_mlp.cu)
target_include_directories(bench_mbyv PRIVATE include)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET bench_mbyv PROPERTY CUDA_ARCHITECTURES ${_DEFAULT_ARCHS})
endif()

add_executable(bench_warp src/bench_custom_warp_mlp.cu)
target_include_directories(bench_warp PRIVATE include)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET bench_warp PROPERTY CUDA_ARCHITECTURES ${_DEFAULT_ARCHS})
endif()

add_executable(bench_cublas_graph src/bench_cublas_mlp_graph.cu)
target_include_directories(bench_cublas_graph PRIVATE include)
target_link_libraries(bench_cublas_graph PRIVATE cublas)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET bench_cublas_graph PROPERTY CUDA_ARCHITECTURES ${_DEFAULT_ARCHS})
endif()

add_executable(bench_warp_graph src/bench_custom_warp_mlp_graph.cu)
target_include_directories(bench_warp_graph PRIVATE include)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET bench_warp_graph PROPERTY CUDA_ARCHITECTURES ${_DEFAULT_ARCHS})
endif()

add_executable(bench_mbyv_graph src/bench_custom_mbyv_mlp_graph.cu)
target_include_directories(bench_mbyv_graph PRIVATE include)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET bench_mbyv_graph PROPERTY CUDA_ARCHITECTURES ${_DEFAULT_ARCHS})
endif()

add_executable(bench_warp_e2e src/bench_custom_warp_e2e.cu)
target_include_directories(bench_warp_e2e PRIVATE include)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET bench_warp_e2e PROPERTY CUDA_ARCHITECTURES ${_DEFAULT_ARCHS})
endif()

add_executable(bench_mbyv_packed_graph src/bench_custom_mbyv_packed_graph.cu)
target_include_directories(bench_mbyv_packed_graph PRIVATE include)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET bench_mbyv_packed_graph PROPERTY CUDA_ARCHITECTURES ${_DEFAULT_ARCHS})
endif()

add_executable(bench_mbyv_persistent src/bench_persistent_mbyv_mlp.cu)
target_include_directories(bench_mbyv_persistent PRIVATE include)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET bench_mbyv_persistent PROPERTY CUDA_ARCHITECTURES ${_DEFAULT_ARCHS})
endif()

add_executable(bench_warp_persistent_mlp src/bench_persistent_mlp.cu)
target_include_directories(bench_warp_persistent_mlp PRIVATE include)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET bench_warp_persistent_mlp PROPERTY CUDA_ARCHITECTURES ${_DEFAULT_ARCHS})
endif()

add_executable(bench_cublas_persistent_mlp src/bench_persistent_cublas_mlp.cu)
target_include_directories(bench_cublas_persistent_mlp PRIVATE include)
target_link_libraries(bench_cublas_persistent_mlp PRIVATE cublas)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET bench_cublas_persistent_mlp PROPERTY CUDA_ARCHITECTURES ${_DEFAULT_ARCHS})
endif()

