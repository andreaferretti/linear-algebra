#!/bin/sh

# gcc -w -O3 -fno-strict-aliasing /home/papillon/esperimenti/blas/matrices.c -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm
gcc -w -O3 -fno-strict-aliasing /home/papillon/esperimenti/blas/matrices.c -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lgomp -lm