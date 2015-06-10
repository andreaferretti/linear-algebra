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
template fp(v: Vector64): ptr float64 = cast[ptr float64](addr(v[]))

template fp(m: Matrix64): ptr float64 = cast[ptr float64](addr(m.data[]))

proc `==`*(u, v: Vector64): bool = u[] == v[]

proc slowEq[M, N: static[int]](m, n: Matrix64[M, N]): bool =
  if m.order == colMajor:
    let
      mData = cast[ref array[N, array[M, float64]]](m.data)
      nData = cast[ref array[M, array[N, float64]]](m.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        if mData[j][i] != nData[i][j]:
          return false
    return true
  else:
    return slowEq(n, m)

proc `==`*(m, n: Matrix64): bool =
  if m.order == n.order: m.data[] == n.data[]
  else: slowEq(m, n)