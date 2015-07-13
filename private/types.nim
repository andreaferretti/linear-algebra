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
  Vector32*[N: static[int]] = ref array[N, float32]
  Vector64*[N: static[int]] = ref array[N, float64]
  Matrix32*[M, N: static[int]] = object
    order: OrderType
    data: ref array[N * M, float32]
  Matrix64*[M, N: static[int]] = object
    order: OrderType
    data: ref array[M * N, float64]

# Float pointers
template fp(v: Vector32): ptr float32 = cast[ptr float32](addr(v[]))

template fp(v: Vector64): ptr float64 = cast[ptr float64](addr(v[]))

template fp(m: Matrix32): ptr float32 = cast[ptr float32](addr(m.data[]))

template fp(m: Matrix64): ptr float64 = cast[ptr float64](addr(m.data[]))

# Equality

proc `==`*(u, v: Vector32 or Vector64): bool = u[] == v[]

template slowEqPrivate(M, N, m, n: expr, A: typedesc) =
  let
    mData = cast[ref array[N, array[M, A]]](m.data)
    nData = cast[ref array[M, array[N, A]]](n.data)
  for i in 0 .. < M:
    for j in 0 .. < N:
      if mData[j][i] != nData[i][j]:
        return false
  return true

proc slowEq[M, N: static[int]](m, n: Matrix32[M, N]): bool = slowEqPrivate(M, N, m, n, float32)

proc slowEq[M, N: static[int]](m, n: Matrix64[M, N]): bool = slowEqPrivate(M, N, m, n, float64)

proc `==`*(m, n: Matrix32 or Matrix64): bool =
  if m.order == n.order: m.data[] == n.data[]
  elif m.order == colMajor: slowEq(m, n)
  else: slowEq(n, m)

# Conversion

proc to32*[N: static[int]](v: Vector64[N]): Vector32[N] =
  new result
  for i in 0 .. < N:
    result[i] = v[i].float32

proc to64*[N: static[int]](v: Vector32[N]): Vector64[N] =
  new result
  for i in 0 .. < N:
    result[i] = v[i].float64

proc to32*[M, N: static[int]](m: Matrix64[M, N]): Matrix32[M, N] =
  new result.data
  result.order = m.order
  for i in 0 .. < (M * N):
    result.data[i] = m.data[i].float32

proc to64*[M, N: static[int]](m: Matrix32[M, N]): Matrix64[M, N] =
  new result.data
  result.order = m.order
  for i in 0 .. < (M * N):
    result.data[i] = m.data[i].float64