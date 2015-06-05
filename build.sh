#!/bin/bash

nim c -d:atlas -d:cublas --parallelBuild:1 --cincludes:"/usr/local/cuda-7.0/targets/x86_64-linux/include" --clibdir:"/usr/local/cuda-7.0/targets/x86_64-linux/lib" -d:release main.nim
# nim c -d:mkl -d:release main.nim
# nim c -d:mkl -d:threaded -d:release main.nim
# nim c -d:atlas -d:release --profiler:on --stackTrace:on blas.nim