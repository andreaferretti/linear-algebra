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
  DVector32* = seq[float32]
  DVector64* = seq[float64]
  DMatrix32* = ref object
    order: OrderType
    M, N: int
    data: seq[float32]
  DMatrix64* = ref object
    order: OrderType
    M, N: int
    data: seq[float64]

# Float pointers
template fp(v: Vector32): ptr float32 = cast[ptr float32](addr(v[]))

template fp(v: Vector64): ptr float64 = cast[ptr float64](addr(v[]))

template fp(m: Matrix32): ptr float32 = cast[ptr float32](addr(m.data[]))

template fp(m: Matrix64): ptr float64 = cast[ptr float64](addr(m.data[]))

template fp(v: DVector32): ptr float32 = cast[ptr float32](unsafeAddr(v[0]))

template fp(v: DVector64): ptr float64 = cast[ptr float64](unsafeAddr(v[0]))

template fp(m: DMatrix32): ptr float32 = cast[ptr float32](unsafeAddr(m.data[0]))

template fp(m: DMatrix64): ptr float64 = cast[ptr float64](unsafeAddr(m.data[0]))

# Equality

proc `==`*(u, v: Vector32 or Vector64): bool = u[] == v[]

template elem(m, i, j: expr): auto =
  if m.order == colMajor: m.data[j * m.M + i]
  else: m.data[i * m.N + j]

template slowEqPrivate(m, n: expr) =
  if m.M != n.M or m.N != n.N:
    return false
  for i in 0 .. < m.M:
    for j in 0 .. < m.N:
      if elem(m, i, j) != elem(n, i, j):
        return false
  return true

proc slowEq[M, N: static[int]](m, n: Matrix32[M, N]): bool = slowEqPrivate(m, n)

proc slowEq[M, N: static[int]](m, n: Matrix64[M, N]): bool = slowEqPrivate(m, n)

proc slowEq(m, n: DMatrix32): bool = slowEqPrivate(m, n)

proc slowEq(m, n: DMatrix64): bool = slowEqPrivate(m, n)

proc `==`*(m, n: Matrix32 or Matrix64): bool =
  if m.order == n.order: m.data[] == n.data[]
  elif m.order == colMajor: slowEq(m, n)
  else: slowEq(n, m)

proc `==`*(m, n: DMatrix32 or DMatrix64): bool =
  if m.order == n.order: m.data == n.data
  elif m.order == colMajor: slowEq(m, n)
  else: slowEq(m, n)

# Conversion

proc to32*[N: static[int]](v: Vector64[N]): Vector32[N] =
  new result
  for i in 0 .. < N:
    result[i] = v[i].float32

proc to64*[N: static[int]](v: Vector32[N]): Vector64[N] =
  new result
  for i in 0 .. < N:
    result[i] = v[i].float64

proc to32*(v: DVector64): DVector32 = v.mapIt(float32, it.float32)

proc to64*(v: DVector32): DVector64 = v.mapIt(float64, it.float64)

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

proc to32*(m: DMatrix64): DMatrix32 =
  result = DMatrix32(data:m.data.mapIt(float32, it.float32), order:m.order, M:m.M, N:m.N)

proc to64*(m: DMatrix32): DMatrix64 =
  result = DMatrix64(data:m.data.mapIt(float64, it.float64), order:m.order, M:m.M, N:m.N)

proc toDynamic*[N: static[int]](v: Vector32[N] or Vector64[N]): auto = @(v[])

proc toDynamic*[M, N: static[int]](m: Matrix32[M, N]): DMatrix32 =
  let data = @(m.data[])
  result = DMatrix32(data:data, order:m.order, M:m.M, N:m.N)

proc toDynamic*[M, N: static[int]](m: Matrix64[M, N]): DMatrix64 =
  let data = @(m.data[])
  result = DMatrix64(data:data, order:m.order, M:m.M, N:m.N)

proc toStatic*(v: DVector32, N: static[int]): Vector32[N] =
  new result
  for i in 0 .. < N:
    result[i] = v[i]

proc toStatic*(v: DVector64, N: static[int]): Vector64[N] =
  new result
  for i in 0 .. < N:
    result[i] = v[i]

proc toStatic*(m: DMatrix32, M, N: static[int]): Matrix32[M, N] =
  result.order = m.order
  new result.data
  for i in 0 .. < (M * N):
    result.data[i] = m.data[i]

proc toStatic*(m: DMatrix64, M, N: static[int]): Matrix64[M, N] =
  result.order = m.order
  new result.data
  for i in 0 .. < (M * N):
    result.data[i] = m.data[i]
