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

proc cudaMalloc(size: int): ptr float32 =
  var error: cudaError
  {.emit: """error = cudaMalloc((void**)&`result`, `size`); """.}
  if error != cudaSuccess:
    quit($(error))

proc cublasCreate(): cublasHandle =
  var stat: cublasStatus
  {.emit: """stat = cublasCreate_v2(& `result`); """.}
  if stat != cublasStatusSuccess:
    quit($(stat))

proc cublasSetVector(n, elemSize: int, x: pointer, incx: int,
  devicePtr: pointer, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSetVector" .}

proc cublasGetVector(n, elemSize: int, devicePtr: pointer, incx: int,
  x: pointer, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasGetVector" .}

proc rawCublasSaxpy(handle: cublasHandle, n: int, alpha: ptr float32, x: ptr float32, incx: int,
  y: ptr float32, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSaxpy" .}

proc cublasSaxpy(handle: cublasHandle, n: int, alpha: float32, x, y: ptr float32): cublasStatus =
  var al: ptr float32
  {.emit: """al = &alpha; """.}
  rawCublasSaxpy(handle, n, al, x, 1, y, 1)

proc cublasScopy(handle: cublasHandle, n: int, x: ptr float32, incx: int,
  y: ptr float32, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasScopy" .}

# proc rawCudaMalloc(p: ptr ptr, size: int): cudaError
#   {. header: "cuda_runtime_api.h", importc: "cudaMalloc" .}

# proc rawCublasCreate(h: object): cublasStatus
#   {. header: "cublas_api.h", importc: "cublasCreate_v2" .}