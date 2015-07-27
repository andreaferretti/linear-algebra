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

proc cudaMalloc(size: int): ptr float32 =
  var error: cudaError
  {.emit: """error = cudaMalloc((void**)&`result`, `size`); """.}
  if error != cudaSuccess:
    quit($(error))

proc cudaFree(p: ptr float32) =
  var error: cudaError
  {.emit: """error = cudaFree(p); """.}
  if error != cudaSuccess:
    quit($(error))

proc freeDeviceMemory(p: ref[ptr float32]) = cudaFree(p[])

proc cublasCreate(): cublasHandle =
  var stat: cublasStatus
  {.emit: """stat = cublasCreate_v2(& `result`); """.}
  if stat != cublasStatusSuccess:
    quit($(stat))

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

proc cublasScopy(handle: cublasHandle, n: int, x: ptr float32, incx: int,
  y: ptr float32, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasScopy" .}

proc rawCublasSscal(handle: cublasHandle, n: int, alpha: ptr float32, x: ptr float32,
  incx: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSscal" .}

proc cublasSscal(handle: cublasHandle, n: int, alpha: float32, x: ptr float32): cublasStatus =
  var al: ptr float32
  {.emit: """al = &alpha; """.}
  rawCublasSscal(handle, n, al, x, 1)

proc rawCublasSaxpy(handle: cublasHandle, n: int, alpha: ptr float32, x: ptr float32, incx: int,
  y: ptr float32, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSaxpy" .}

proc cublasSaxpy(handle: cublasHandle, n: int, alpha: float32, x, y: ptr float32): cublasStatus =
  var al: ptr float32
  {.emit: """al = &alpha; """.}
  rawCublasSaxpy(handle, n, al, x, 1, y, 1)

proc cublasSnrm2(handle: cublasHandle, n: int, x: ptr float32,
  incx: int, res: ptr float32): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSnrm2" .}

proc cublasSasum(handle: cublasHandle, n: int, x: ptr float32,
  incx: int, res: ptr float32): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSasum" .}

proc cublasSdot(handle: cublasHandle, n: int, x: ptr float32, incx: int,
  y: ptr float32, incy: int, res: ptr float32): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSdot" .}

proc rawCublasSgemv(handle: cublasHandle, trans: cublasTransposeType,
  m, n: int, alpha: ptr float32, A: ptr float32, lda: int, x: ptr float32,
  incx: int, beta: ptr float32, y: ptr float32, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSgemv" .}

proc cublasSgemv(handle: cublasHandle, trans: cublasTransposeType,
  m, n: int, alpha: float32, A: ptr float32, lda: int, x: ptr float32, incx: int,
  beta: float32, y: ptr float32, incy: int): cublasStatus =
  var al, be: ptr float32
  {.emit: """al = &alpha; be = &beta; """.}
  rawCublasSgemv(handle, trans, m, n, al, A, lda, x, incx, be, y, incy)