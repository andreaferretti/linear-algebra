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

proc t*[M, N: static[int]](a: Matrix32[M, N]): Matrix32[N, M] =
  result.order = if a.order == rowMajor: colMajor else: rowMajor
  result.data = a.data

proc t*(a: DMatrix32): DMatrix32 =
  result.M = a.M
  result.N = a.N
  result.order = if a.order == rowMajor: colMajor else: rowMajor
  result.data = a.data

proc t*[M, N: static[int]](a: Matrix64[M, N]): Matrix64[N, M] =
  result.order = if a.order == rowMajor: colMajor else: rowMajor
  result.data = a.data

proc t*(a: DMatrix64): DMatrix64 =
  result.M = a.M
  result.N = a.N
  result.order = if a.order == rowMajor: colMajor else: rowMajor
  result.data = a.data

proc reshape*[M, N: static[int]](m: Matrix32[M, N], A, B: static[int]): Matrix32[A, B] =
  static: doAssert(M * N == A * B, "The dimensions do not match: M = " & $(M) & ", N = " & $(N) & ", A = " & $(A) & ", B = " & $(B))
  result.order = m.order
  result.data = m.data

proc reshape*(m: DMatrix32, A, B: int): DMatrix32 =
  assert(m.M * m.N == A * B, "The dimensions do not match: M = " & $(m.M) & ", N = " & $(m.N) & ", A = " & $(A) & ", B = " & $(B))
  result.M = A
  result.N = B
  result.order = m.order
  result.data = m.data

proc reshape*[M, N: static[int]](m: Matrix64[M, N], A, B: static[int]): Matrix64[A, B] =
  static: doAssert(M * N == A * B, "The dimensions do not match: M = " & $(M) & ", N = " & $(N) & ", A = " & $(A) & ", B = " & $(B))
  result.order = m.order
  result.data = m.data

proc reshape*(m: DMatrix64, A, B: int): DMatrix64 =
  assert(m.M * m.N == A * B, "The dimensions do not match: M = " & $(m.M) & ", N = " & $(m.N) & ", A = " & $(A) & ", B = " & $(B))
  result.M = A
  result.N = B
  result.order = m.order
  result.data = m.data

proc asMatrix*[N: static[int]](v: Vector32[N], A, B: static[int], order: OrderType = colMajor): Matrix32[A, B] =
  static: doAssert(N == A * B, "The dimensions do not match: N = " & $(N) & ", A = " & $(A) & ", B = " & $(B))
  result.order = order
  result.data = v

proc asMatrix*(v: DVector32, A, B: int, order: OrderType = colMajor): DMatrix32 =
  assert(v.len == A * B, "The dimensions do not match: N = " & $(v.len) & ", A = " & $(A) & ", B = " & $(B))
  result.order = order
  result.data = v
  result.M = A
  result.N = B

proc asMatrix*[N: static[int]](v: Vector64[N], A, B: static[int], order: OrderType = colMajor): Matrix64[A, B] =
  static: doAssert(N == A * B, "The dimensions do not match: N = " & $(N) & ", A = " & $(A) & ", B = " & $(B))
  result.order = order
  result.data = v

proc asMatrix*(v: DVector64, A, B: int, order: OrderType = colMajor): DMatrix64 =
  assert(v.len == A * B, "The dimensions do not match: N = " & $(v.len) & ", A = " & $(A) & ", B = " & $(B))
  result.order = order
  result.data = v
  result.M = A
  result.N = B

proc asVector*[M, N: static[int]](m: Matrix32[M, N]): Vector32[M * N] = m.data

proc asVector*[M, N: static[int]](m: Matrix64[M, N]): Vector64[M * N] = m.data

proc asVector*(m: DMatrix32 or DMatrix64): auto = m.data