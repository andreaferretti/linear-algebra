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

let handle {.global.} = cublasCreate()

proc `*=`*[N: static[int]](v: var CudaVector[N], k: float32) {. inline .} =
  check cublasSscal(handle, N, k, v[])

proc `*`*[N: static[int]](v: CudaVector[N], k: float32): CudaVector[N]  {. inline .} =
  new result, freeDeviceMemory
  result[] = cudaMalloc(N * sizeof(float32))
  check cublasScopy(handle, N, v[], 1, result[], 1)
  check cublasSscal(handle, N, k, result[])

proc `+=`*[N: static[int]](v: var CudaVector[N], w: CudaVector[N]) {. inline .} =
  check cublasSaxpy(handle, N, 1, w[], v[])

proc `+`*[N: static[int]](v, w: CudaVector[N]): CudaVector[N] {. inline .} =
  new result, freeDeviceMemory
  result[] = cudaMalloc(N * sizeof(float32))
  check cublasScopy(handle, N, v[], 1, result[], 1)
  check cublasSaxpy(handle, N, 1, w[], result[])

proc `-=`*[N: static[int]](v: var CudaVector[N], w: CudaVector[N]) {. inline .} =
  check cublasSaxpy(handle, N, -1, w[], v[])

proc `-`*[N: static[int]](v, w: CudaVector[N]): CudaVector[N] {. inline .} =
  new result, freeDeviceMemory
  result[] = cudaMalloc(N * sizeof(float32))
  check cublasScopy(handle, N, v[], 1, result[], 1)
  check cublasSaxpy(handle, N, -1, w[], result[])

proc `*`*[N: static[int]](v, w: CudaVector[N]): float32 {. inline .} =
  check cublasSdot(handle, N, v[], 1, w[], 1, addr(result))

proc l_2*[N: static[int]](v: CudaVector[N]): float32 {. inline .} =
  check cublasSnrm2(handle, N, v[], 1, addr(result))

proc l_1*[N: static[int]](v: CudaVector[N]): float32 {. inline .} =
  check cublasSasum(handle, N, v[], 1, addr(result))

proc `==`*[N: static[int]](v, w: CudaVector[N]): bool =
  v.cpu() == w.cpu()

proc compareApprox(a, b: CudaVector or CudaMatrix): bool =
  mixin l_1
  const epsilon = 0.000001
  let
    aNorm = l_1(a)
    bNorm = l_1(b)
    dNorm = l_1(a - b)
  return (dNorm / (aNorm + bNorm)) < epsilon


proc `=~`*[N: static[int]](v, w: CudaVector[N]): bool = compareApprox(v, w)

proc `*`*[M, N: static[int]](a: CudaMatrix[M, N], v: CudaVector[N]): CudaVector[M]  {. inline .} =
  new result, freeDeviceMemory
  result[] = cudaMalloc(M * sizeof(float32))
  check cublasSgemv(handle, cuNoTranspose, M, N, 1, a.fp, M, v[], 1, 0, result[], 1)

proc `*=`*[M, N: static[int]](m: var CudaMatrix[M, N], k: float32) {. inline .} =
  check cublasSscal(handle, M * N, k, m.fp)

proc `==`*[M, N: static[int]](m, n: CudaMatrix[M, N]): bool =
  m.cpu() == n.cpu()

proc `*`*[M, N: static[int]](m: CudaMatrix[M, N], k: float32): CudaMatrix[M, N]  {. inline .} =
  new result.data, freeDeviceMemory
  result.data[] = cudaMalloc(M * N * sizeof(float32))
  check cublasScopy(handle, M * N, m.fp, 1, result.fp, 1)
  check cublasSscal(handle, M * N, k, result.fp)

template `*`*(k: float32, v: CudaVector or CudaMatrix): expr = v * k

template `/`*(v: CudaVector or CudaMatrix, k: float32): expr = v * (1 / k)

template `/=`*(v: var CudaVector or var CudaMatrix, k: float32): expr = v *= (1 / k)

proc `+=`*[M, N: static[int]](a: var CudaMatrix[M, N], b: CudaMatrix[M, N]) {. inline .} =
  check cublasSaxpy(handle, M * N, 1, b.fp, a.fp)

proc `+`*[M, N: static[int]](a, b: CudaMatrix[M, N]): CudaMatrix[M, N]  {. inline .} =
  new result.data, freeDeviceMemory
  result.data[] = cudaMalloc(M * N * sizeof(float32))
  check cublasScopy(handle, M * N, a.fp, 1, result.fp, 1)
  check cublasSaxpy(handle, M * N, 1, b.fp, result.fp)

proc `-=`*[M, N: static[int]](a: var CudaMatrix[M, N], b: CudaMatrix[M, N]) {. inline .} =
  check cublasSaxpy(handle, M * N, -1, b.fp, a.fp)

proc `-`*[M, N: static[int]](a, b: CudaMatrix[M, N]): CudaMatrix[M, N]  {. inline .} =
  new result.data, freeDeviceMemory
  result.data[] = cudaMalloc(M * N * sizeof(float32))
  check cublasScopy(handle, M * N, a.fp, 1, result.fp, 1)
  check cublasSaxpy(handle, M * N, -1, b.fp, result.fp)

proc `*`*[M, N, K: static[int]](a: CudaMatrix[M, K], b: CudaMatrix[K, N]): CudaMatrix[M, N] {. inline .} =
  new result.data, freeDeviceMemory
  result.data[] = cudaMalloc(M * N * sizeof(float32))
  check cublasSgemm(handle, cuNoTranspose, cuNoTranspose, M, N, K, 1,
    a.fp, M, b.fp, K, 0, result.fp, M)

proc l_2*[M, N: static[int]](m: CudaMatrix[M, N]): float32 {. inline .} =
  check cublasSnrm2(handle, M * N, m.fp, 1, addr(result))

proc l_1*[M, N: static[int]](m: CudaMatrix[M, N]): float32 {. inline .} =
  check cublasSasum(handle, M * N, m.fp, 1, addr(result))

proc `=~`*[M, N: static[int]](m, n: CudaMatrix[M, N]): bool = compareApprox(m, n)

template `!=~`*(a, b: CudaVector or CudaMatrix): bool = not (a =~ b)