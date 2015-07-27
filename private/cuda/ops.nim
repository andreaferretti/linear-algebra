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
  new result
  result[] = cudaMalloc(N * sizeof(float32))
  check cublasScopy(handle, N, v[], 1, result[], 1)
  check cublasSscal(handle, N, k, result[])

template `*`*(k: float32, v: CudaVector): expr = v * k

proc `+=`*[N: static[int]](v: var CudaVector[N], w: CudaVector[N]) {. inline .} =
  check cublasSaxpy(handle, N, 1, w[], v[])

proc `+`*[N: static[int]](v, w: CudaVector[N]): CudaVector[N] {. inline .} =
  new result
  result[] = cudaMalloc(N * sizeof(float32))
  check cublasScopy(handle, N, v[], 1, result[], 1)
  check cublasSaxpy(handle, N, 1, w[], result[])

proc `-=`*[N: static[int]](v: var CudaVector[N], w: CudaVector[N]) {. inline .} =
  check cublasSaxpy(handle, N, -1, w[], v[])

proc `-`*[N: static[int]](v, w: CudaVector[N]): CudaVector[N] {. inline .} =
  new result
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

proc `=~`*[N: static[int]](v, w: CudaVector[N]): bool =
  mixin l_1
  const epsilon = 0.000001
  let
    vNorm = l_1(v)
    wNorm = l_1(w)
    dNorm = l_1(v - w)
  return (dNorm / (vNorm + wNorm)) < epsilon

template `!=~`*(a, b: CudaVector): bool = not (a =~ b)