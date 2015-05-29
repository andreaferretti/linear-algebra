#!/bin/bash

nim c -d:atlas -d:release blas.nim
# nim c -d:mkl -d:threaded -d:release blas.nim
# nim c --clibdir:"/usr/lib/atlas-base" --passL:"-lblas" --listCmd --verbosity:3 -d:release blas.nim
