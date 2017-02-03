# Copyright 2016 UniCredit S.p.A.
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

proc cublasSscal(handle: cublasHandle, n: int, alpha: ptr float32, x: ptr float32,
  incx: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSscal" .}

proc cublasDscal(handle: cublasHandle, n: int, alpha: ptr float64, x: ptr float64,
  incx: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasDscal" .}

proc cublasScal(handle: cublasHandle, n: int, alpha: float32, x: ptr float32): cublasStatus =
  cublasSscal(handle, n, unsafeAddr(alpha), x, 1)

proc cublasScal(handle: cublasHandle, n: int, alpha: float64, x: ptr float64): cublasStatus =
  cublasDscal(handle, n, unsafeAddr(alpha), x, 1)

proc cublasSaxpy(handle: cublasHandle, n: int, alpha: ptr float32, x: ptr float32, incx: int,
  y: ptr float32, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSaxpy" .}

proc cublasDaxpy(handle: cublasHandle, n: int, alpha: ptr float64, x: ptr float64, incx: int,
  y: ptr float64, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasDaxpy" .}

proc cublasAxpy(handle: cublasHandle, n: int, alpha: float32, x, y: ptr float32): cublasStatus =
  cublasSaxpy(handle, n, unsafeAddr(alpha), x, 1, y, 1)

proc cublasAxpy(handle: cublasHandle, n: int, alpha: float64, x, y: ptr float64): cublasStatus =
  cublasDaxpy(handle, n, unsafeAddr(alpha), x, 1, y, 1)

proc cublasNrm2(handle: cublasHandle, n: int, x: ptr float32,
  incx: int, res: ptr float32): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSnrm2" .}

proc cublasNrm2(handle: cublasHandle, n: int, x: ptr float64,
  incx: int, res: ptr float64): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasDnrm2" .}

proc cublasAsum(handle: cublasHandle, n: int, x: ptr float32,
  incx: int, res: ptr float32): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSasum" .}

proc cublasAsum(handle: cublasHandle, n: int, x: ptr float64,
  incx: int, res: ptr float64): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasDasum" .}

proc cublasDot(handle: cublasHandle, n: int, x: ptr float32, incx: int,
  y: ptr float32, incy: int, res: ptr float32): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSdot" .}

proc cublasDot(handle: cublasHandle, n: int, x: ptr float64, incx: int,
  y: ptr float64, incy: int, res: ptr float64): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasDdot" .}

proc cublasSgemv(handle: cublasHandle, trans: cublasTransposeType,
  m, n: int, alpha: ptr float32, A: ptr float32, lda: int, x: ptr float32,
  incx: int, beta: ptr float32, y: ptr float32, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSgemv" .}

proc cublasDgemv(handle: cublasHandle, trans: cublasTransposeType,
  m, n: int, alpha: ptr float64, A: ptr float64, lda: int, x: ptr float64,
  incx: int, beta: ptr float64, y: ptr float64, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasDgemv" .}

proc cublasGemv(handle: cublasHandle, trans: cublasTransposeType,
  m, n: int, alpha: float32, A: ptr float32, lda: int, x: ptr float32, incx: int,
  beta: float32, y: ptr float32, incy: int): cublasStatus =
  cublasSgemv(handle, trans, m, n, unsafeAddr(alpha), A, lda, x, incx, unsafeAddr(beta), y, incy)

proc cublasGemv(handle: cublasHandle, trans: cublasTransposeType,
  m, n: int, alpha: float64, A: ptr float64, lda: int, x: ptr float64, incx: int,
  beta: float64, y: ptr float64, incy: int): cublasStatus =
  cublasDgemv(handle, trans, m, n, unsafeAddr(alpha), A, lda, x, incx, unsafeAddr(beta), y, incy)

proc cublasSgemm(handle: cublasHandle, transa, transb: cublasTransposeType,
  m, n, k: int, alpha: ptr float32, A: ptr float32, lda: int, B: ptr float32,
  ldb: int, beta: ptr float32, C: ptr float32, ldc: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSgemm" .}

proc cublasDgemm(handle: cublasHandle, transa, transb: cublasTransposeType,
  m, n, k: int, alpha: ptr float64, A: ptr float64, lda: int, B: ptr float64,
  ldb: int, beta: ptr float64, C: ptr float64, ldc: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasDgemm" .}

proc cublasGemm(handle: cublasHandle, transa, transb: cublasTransposeType,
  m, n, k: int, alpha: float32, A: ptr float32, lda: int, B: ptr float32,
  ldb: int, beta: float32, C: ptr float32, ldc: int): cublasStatus =
  cublasSgemm(handle, transa, transb, m, n, k, unsafeAddr(alpha), A, lda, B, ldb, unsafeAddr(beta), C, ldc)

proc cublasGemm(handle: cublasHandle, transa, transb: cublasTransposeType,
  m, n, k: int, alpha: float64, A: ptr float64, lda: int, B: ptr float64,
  ldb: int, beta: float64, C: ptr float64, ldc: int): cublasStatus =
  cublasDgemm(handle, transa, transb, m, n, k, unsafeAddr(alpha), A, lda, B, ldb, unsafeAddr(beta), C, ldc)