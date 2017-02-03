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

template transposeS(a, result: expr) =
  result.order = if a.order == rowMajor: colMajor else: rowMajor
  result.data = a.data

template transposeD(a, result: expr) =
  new result
  result.M = a.N
  result.N = a.M
  result.order = if a.order == rowMajor: colMajor else: rowMajor
  shallowCopy(result.data, a.data)

proc t*[M, N: static[int]](a: Matrix32[M, N]): Matrix32[N, M] =
  transposeS(a, result)

proc t*(a: DMatrix32): DMatrix32 = transposeD(a, result)

proc t*[M, N: static[int]](a: Matrix64[M, N]): Matrix64[N, M] =
  transposeS(a, result)

proc t*(a: DMatrix64): DMatrix64 = transposeD(a, result)

template reshapeS(M, N, A, B, m , result: expr) =
  static: doAssert(M * N == A * B, "The dimensions do not match: M = " & $(M) & ", N = " & $(N) & ", A = " & $(A) & ", B = " & $(B))
  result.order = m.order
  result.data = m.data

template reshapeD(A, B, m , result: expr) =
  assert(m.M * m.N == A * B, "The dimensions do not match: M = " & $(m.M) & ", N = " & $(m.N) & ", A = " & $(A) & ", B = " & $(B))
  new result
  result.M = A
  result.N = B
  result.order = m.order
  shallowCopy(result.data, m.data)

proc reshape*[M, N: static[int]](m: Matrix32[M, N], A, B: static[int]): Matrix32[A, B] =
  reshapeS(M, N, A, B, m, result)

proc reshape*(m: DMatrix32, A, B: int): DMatrix32 = reshapeD(A, B, m, result)

proc reshape*[M, N: static[int]](m: Matrix64[M, N], A, B: static[int]): Matrix64[A, B] =
  reshapeS(M, N, A, B, m, result)

proc reshape*(m: DMatrix64, A, B: int): DMatrix64 = reshapeD(A, B, m, result)

template asMatrixS(N, A, B, v, order, result: expr) =
  static: doAssert(N == A * B, "The dimensions do not match: N = " & $(N) & ", A = " & $(A) & ", B = " & $(B))
  result.order = order
  result.data = v

template asMatrixD(A, B, v, order, result: expr) =
  assert(v.len == A * B, "The dimensions do not match: N = " & $(v.len) & ", A = " & $(A) & ", B = " & $(B))
  new result
  result.order = order
  shallowCopy(result.data, v)
  result.M = A
  result.N = B

proc asMatrix*[N: static[int]](v: Vector32[N], A, B: static[int], order: OrderType = colMajor): Matrix32[A, B] =
  asMatrixS(N, A, B, v, order, result)

proc asMatrix*(v: DVector32, A, B: int, order: OrderType = colMajor): DMatrix32 =
  asMatrixD(A, B, v, order, result)

proc asMatrix*[N: static[int]](v: Vector64[N], A, B: static[int], order: OrderType = colMajor): Matrix64[A, B] =
  asMatrixS(N, A, B, v, order, result)

proc asMatrix*(v: DVector64, A, B: int, order: OrderType = colMajor): DMatrix64 =
  asMatrixD(A, B, v, order, result)

proc asVector*[M, N: static[int]](m: Matrix32[M, N]): Vector32[M * N] = m.data

proc asVector*[M, N: static[int]](m: Matrix64[M, N]): Vector64[M * N] = m.data

proc asVector*(m: DMatrix32): DVector32 =
  shallowCopy(result, m.data)

proc asVector*(m: DMatrix64): DVector64 =
  shallowCopy(result, m.data)
