# Copyright 2015 UniCredit S.p.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

type
  cublasTransposeType = enum
    cuNoTranspose = 0, cuTranspose = 1, cuConjTranspose = 2

proc cudaMalloc(p: ptr pointer, size: int): cudaError
  {. header: "cublas_v2.h", importc: "cudaMalloc" .}

proc cudaMalloc32(size: int): ptr float32 =
  let s = size * sizeof(float32)
  check cudaMalloc(cast[ptr pointer](addr result), s)

proc cudaMalloc64(size: int): ptr float64 =
  let s = size * sizeof(float64)
  check cudaMalloc(cast[ptr pointer](addr result), s)

proc rawCudaFree(p: pointer): cublasStatus
  {. header: "cublas_v2.h", importc: "cudaFree" .}

proc cudaFree(p: ptr float32 or ptr float64) =
  check rawCudaFree(p)

proc freeDeviceMemory(p: ref[ptr float32]) = cudaFree(p[])

proc freeDeviceMemory(p: ref[ptr float64]) = cudaFree(p[])

proc cublasCreate_v2(h: ptr cublasHandle): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasCreate_v2" .}

proc cublasCreate(): cublasHandle =
  check cublasCreate_v2(addr result)

# y is on device
proc cublasSetVector(n, elemSize: int, x: pointer, incx: int,
  y: pointer, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSetVector" .}

# x is on device
proc cublasGetVector(n, elemSize: int, x: pointer, incx: int,
  y: pointer, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasGetVector" .}

# b is on device
proc cublasSetMatrix(rows, cols, elemSize: int, a: pointer, lda: int,
  b: pointer, ldb: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSetMatrix" .}

# a is on device
proc cublasGetMatrix(rows, cols, elemSize: int, a: pointer, lda: int,
  b: pointer, ldb: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasGetMatrix" .}

proc cublasCopy(handle: cublasHandle, n: int, x: ptr float32, incx: int,
  y: ptr float32, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasScopy" .}

proc cublasCopy(handle: cublasHandle, n: int, x: ptr float64, incx: int,
  y: ptr float64, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasDcopy" .}

proc rawCublasScal(handle: cublasHandle, n: int, alpha: ptr float32, x: ptr float32,
  incx: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSscal" .}

proc rawCublasScal(handle: cublasHandle, n: int, alpha: ptr float64, x: ptr float64,
  incx: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasDscal" .}

proc cublasScal(handle: cublasHandle, n: int, alpha: float32, x: ptr float32): cublasStatus =
  rawCublasScal(handle, n, unsafeAddr(alpha), x, 1)

proc cublasScal(handle: cublasHandle, n: int, alpha: float64, x: ptr float64): cublasStatus =
  rawCublasScal(handle, n, unsafeAddr(alpha), x, 1)

proc rawCublasAxpy(handle: cublasHandle, n: int, alpha: ptr float32, x: ptr float32, incx: int,
  y: ptr float32, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSaxpy" .}

proc rawCublasAxpy(handle: cublasHandle, n: int, alpha: ptr float64, x: ptr float64, incx: int,
  y: ptr float64, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasDaxpy" .}

proc cublasAxpy(handle: cublasHandle, n: int, alpha: float32, x, y: ptr float32): cublasStatus =
  rawCublasAxpy(handle, n, unsafeAddr(alpha), x, 1, y, 1)

proc cublasAxpy(handle: cublasHandle, n: int, alpha: float64, x, y: ptr float64): cublasStatus =
  rawCublasAxpy(handle, n, unsafeAddr(alpha), x, 1, y, 1)

proc cublasNrm2(handle: cublasHandle, n: int, x: ptr float32,
  incx: int, res: ptr float32): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSnrm2" .}

proc cublasAsum(handle: cublasHandle, n: int, x: ptr float32,
  incx: int, res: ptr float32): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSasum" .}

proc cublasDot(handle: cublasHandle, n: int, x: ptr float32, incx: int,
  y: ptr float32, incy: int, res: ptr float32): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSdot" .}

proc rawCublasGemv(handle: cublasHandle, trans: cublasTransposeType,
  m, n: int, alpha: ptr float32, A: ptr float32, lda: int, x: ptr float32,
  incx: int, beta: ptr float32, y: ptr float32, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSgemv" .}

proc cublasGemv(handle: cublasHandle, trans: cublasTransposeType,
  m, n: int, alpha: float32, A: ptr float32, lda: int, x: ptr float32, incx: int,
  beta: float32, y: ptr float32, incy: int): cublasStatus =
  rawCublasGemv(handle, trans, m, n, unsafeAddr(alpha), A, lda, x, incx, unsafeAddr(beta), y, incy)

proc rawCublasGemm(handle: cublasHandle, transa, transb: cublasTransposeType,
  m, n, k: int, alpha: ptr float32, A: ptr float32, lda: int, B: ptr float32,
  ldb: int, beta: ptr float32, C: ptr float32, ldc: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSgemm" .}

proc cublasGemm(handle: cublasHandle, transa, transb: cublasTransposeType,
  m, n, k: int, alpha: float32, A: ptr float32, lda: int, B: ptr float32,
  ldb: int, beta: float32, C: ptr float32, ldc: int): cublasStatus =
  rawCublasGemm(handle, transa, transb, m, n, k, unsafeAddr(alpha), A, lda, B, ldb, unsafeAddr(beta), C, ldc)