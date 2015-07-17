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