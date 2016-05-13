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

proc `*=`*[N: static[int]](v: var CudaVector32[N], k: float32) {. inline .} =
  check cublasScal(handle, N, k, v[])

proc `*=`*[N: static[int]](v: var CudaVector64[N], k: float64) {. inline .} =
  check cublasScal(handle, N, k, v[])

proc `*`*[N: static[int]](v: CudaVector32[N], k: float32): CudaVector32[N]  {. inline .} =
  new result, freeDeviceMemory
  result[] = cudaMalloc32(N)
  check cublasCopy(handle, N, v[], 1, result[], 1)
  check cublasScal(handle, N, k, result[])

proc `*`*[N: static[int]](v: CudaVector64[N], k: float64): CudaVector64[N]  {. inline .} =
  new result, freeDeviceMemory
  result[] = cudaMalloc64(N)
  check cublasCopy(handle, N, v[], 1, result[], 1)
  check cublasScal(handle, N, k, result[])

proc `+=`*[N: static[int]](v: var CudaVector32[N], w: CudaVector32[N]) {. inline .} =
  check cublasAxpy(handle, N, 1, w[], v[])

proc `+=`*[N: static[int]](v: var CudaVector64[N], w: CudaVector64[N]) {. inline .} =
  check cublasAxpy(handle, N, 1, w[], v[])

proc `+`*[N: static[int]](v, w: CudaVector32[N]): CudaVector32[N] {. inline .} =
  new result, freeDeviceMemory
  result[] = cudaMalloc32(N)
  check cublasCopy(handle, N, v[], 1, result[], 1)
  check cublasAxpy(handle, N, 1, w[], result[])

proc `+`*[N: static[int]](v, w: CudaVector64[N]): CudaVector64[N] {. inline .} =
  new result, freeDeviceMemory
  result[] = cudaMalloc64(N)
  check cublasCopy(handle, N, v[], 1, result[], 1)
  check cublasAxpy(handle, N, 1, w[], result[])

proc `-=`*[N: static[int]](v: var CudaVector32[N], w: CudaVector32[N]) {. inline .} =
  check cublasAxpy(handle, N, -1, w[], v[])

proc `-=`*[N: static[int]](v: var CudaVector64[N], w: CudaVector64[N]) {. inline .} =
  check cublasAxpy(handle, N, -1, w[], v[])

proc `-`*[N: static[int]](v, w: CudaVector32[N]): CudaVector32[N] {. inline .} =
  new result, freeDeviceMemory
  result[] = cudaMalloc32(N)
  check cublasCopy(handle, N, v[], 1, result[], 1)
  check cublasAxpy(handle, N, -1, w[], result[])

proc `-`*[N: static[int]](v, w: CudaVector64[N]): CudaVector64[N] {. inline .} =
  new result, freeDeviceMemory
  result[] = cudaMalloc64(N)
  check cublasCopy(handle, N, v[], 1, result[], 1)
  check cublasAxpy(handle, N, -1, w[], result[])

proc `*`*[N: static[int]](v, w: CudaVector32[N]): float32 {. inline .} =
  check cublasDot(handle, N, v[], 1, w[], 1, addr(result))

proc l_2*[N: static[int]](v: CudaVector32[N]): float32 {. inline .} =
  check cublasNrm2(handle, N, v[], 1, addr(result))

proc l_1*[N: static[int]](v: CudaVector32[N]): float32 {. inline .} =
  check cublasAsum(handle, N, v[], 1, addr(result))

proc `==`*[N: static[int]](v, w: CudaVector32[N]): bool =
  v.cpu() == w.cpu()

proc compareApprox(a, b: CudaVector32 or CudaMatrix32): bool =
  mixin l_1
  const epsilon = 0.000001
  let
    aNorm = l_1(a)
    bNorm = l_1(b)
    dNorm = l_1(a - b)
  return (dNorm / (aNorm + bNorm)) < epsilon


proc `=~`*[N: static[int]](v, w: CudaVector32[N]): bool = compareApprox(v, w)

proc `*`*[M, N: static[int]](a: CudaMatrix32[M, N], v: CudaVector32[N]): CudaVector32[M]  {. inline .} =
  new result, freeDeviceMemory
  result[] = cudaMalloc32(M)
  check cublasGemv(handle, cuNoTranspose, M, N, 1, a.fp, M, v[], 1, 0, result[], 1)

proc `*=`*[M, N: static[int]](m: var CudaMatrix32[M, N], k: float32) {. inline .} =
  check cublasScal(handle, M * N, k, m.fp)

proc `==`*[M, N: static[int]](m, n: CudaMatrix32[M, N]): bool =
  m.cpu() == n.cpu()

proc `*`*[M, N: static[int]](m: CudaMatrix32[M, N], k: float32): CudaMatrix32[M, N]  {. inline .} =
  new result.data, freeDeviceMemory
  result.data[] = cudaMalloc32(M * N)
  check cublasCopy(handle, M * N, m.fp, 1, result.fp, 1)
  check cublasScal(handle, M * N, k, result.fp)

template `*`*(k: float32, v: CudaVector32 or CudaMatrix32): expr = v * k

template `*`*(k: float64, v: CudaVector64 or CudaMatrix64): expr = v * k

template `/`*(v: CudaVector32 or CudaMatrix32, k: float32): expr = v * (1 / k)

template `/=`*(v: var CudaVector32 or var CudaMatrix32, k: float32): expr = v *= (1 / k)

proc `+=`*[M, N: static[int]](a: var CudaMatrix32[M, N], b: CudaMatrix32[M, N]) {. inline .} =
  check cublasAxpy(handle, M * N, 1, b.fp, a.fp)

proc `+`*[M, N: static[int]](a, b: CudaMatrix32[M, N]): CudaMatrix32[M, N]  {. inline .} =
  new result.data, freeDeviceMemory
  result.data[] = cudaMalloc32(M * N)
  check cublasCopy(handle, M * N, a.fp, 1, result.fp, 1)
  check cublasAxpy(handle, M * N, 1, b.fp, result.fp)

proc `-=`*[M, N: static[int]](a: var CudaMatrix32[M, N], b: CudaMatrix32[M, N]) {. inline .} =
  check cublasAxpy(handle, M * N, -1, b.fp, a.fp)

proc `-`*[M, N: static[int]](a, b: CudaMatrix32[M, N]): CudaMatrix32[M, N]  {. inline .} =
  new result.data, freeDeviceMemory
  result.data[] = cudaMalloc32(M * N)
  check cublasCopy(handle, M * N, a.fp, 1, result.fp, 1)
  check cublasAxpy(handle, M * N, -1, b.fp, result.fp)

proc `*`*[M, N, K: static[int]](a: CudaMatrix32[M, K], b: CudaMatrix32[K, N]): CudaMatrix32[M, N] {. inline .} =
  new result.data, freeDeviceMemory
  result.data[] = cudaMalloc32(M * N)
  check cublasGemm(handle, cuNoTranspose, cuNoTranspose, M, N, K, 1,
    a.fp, M, b.fp, K, 0, result.fp, M)

proc l_2*[M, N: static[int]](m: CudaMatrix32[M, N]): float32 {. inline .} =
  check cublasNrm2(handle, M * N, m.fp, 1, addr(result))

proc l_1*[M, N: static[int]](m: CudaMatrix32[M, N]): float32 {. inline .} =
  check cublasAsum(handle, M * N, m.fp, 1, addr(result))

proc `=~`*[M, N: static[int]](m, n: CudaMatrix32[M, N]): bool = compareApprox(m, n)

template `!=~`*(a, b: CudaVector32 or CudaMatrix32): bool = not (a =~ b)