cmake_minimum_required (VERSION 3.22)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)



make_benchmark (file-benchmark-module file-bench "main.cu;")

make_benchmark_choice (file-benchmark file-benchmark-smartio file-benchmark-module)
