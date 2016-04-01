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

proc gpu*[N: static[int]](v: Vector32[N]): CudaVector32[N] =
  new result, freeDeviceMemory
  result[] = cudaMalloc32(N)
  check cublasSetVector(N, sizeof(float32), v.fp, 1, result[], 1)

proc gpu*[N: static[int]](v: Vector64[N]): CudaVector64[N] =
  new result, freeDeviceMemory
  result[] = cudaMalloc64(N)
  check cublasSetVector(N, sizeof(float64), v.fp, 1, result[], 1)

proc cpu*[N: static[int]](v: CudaVector32[N]): Vector32[N] =
  new result
  check cublasGetVector(N, sizeof(float32), v[], 1, result.fp, 1)

proc cpu*[N: static[int]](v: CudaVector64[N]): Vector64[N] =
  new result
  check cublasGetVector(N, sizeof(float64), v[], 1, result.fp, 1)

proc gpu*[M, N: static[int]](m: Matrix32[M, N]): CudaMatrix32[M, N] =
  if m.order == rowMajor: quit("wrong order")
  new result.data, freeDeviceMemory
  result.data[] = cudaMalloc32(M * N)
  check cublasSetMatrix(M, N, sizeof(float32), m.fp, M, result.fp, M)

proc gpu*[M, N: static[int]](m: Matrix64[M, N]): CudaMatrix64[M, N] =
  if m.order == rowMajor: quit("wrong order")
  new result.data, freeDeviceMemory
  result.data[] = cudaMalloc64(M * N)
  check cublasSetMatrix(M, N, sizeof(float64), m.fp, M, result.fp, M)

proc cpu*[M, N: static[int]](m: CudaMatrix32[M, N]): Matrix32[M, N] =
  result.order = colMajor
  new result.data
  check cublasGetMatrix(M, N, sizeof(float32), m.fp, M, result.fp, M)

proc cpu*[M, N: static[int]](m: CudaMatrix64[M, N]): Matrix64[M, N] =
  result.order = colMajor
  new result.data
  check cublasGetMatrix(M, N, sizeof(float64), m.fp, M, result.fp, M)