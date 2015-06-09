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

proc makeVector*(N: static[int], f: proc (i: int): float64): Vector64[N] =
  new result
  for i in 0 .. < N:
    result[i] = f(i)

proc randomVector*(N: static[int], max: float64 = 1): Vector64[N] =
  makeVector(N, proc(i: int): float64 = random(max))

proc constantVector*(N: static[int], x: float64): Vector64[N] =
  new result
  for i in 0 .. < N:
    result[i] = x

proc zeros*(N: static[int]): Vector64[N] = constantVector(N, 0)

proc ones*(N: static[int]): Vector64[N] = constantVector(N, 1)

type Array[N: static[int], A] = array[N, A]

proc vector*[N: static[int]](xs: Array[N, float64]): Vector64[N] =
  new result
  for i in 0 .. < N:
    result[i] = xs[i]

proc dvector*(N: static[int], xs: seq[float64]): Vector64[N] =
  makeVector(N, proc(i: int): float64 = xs[i])

proc makeMatrix*(M, N: static[int], f: proc (i, j: int): float64, order: OrderType = colMajor): Matrix64[M, N] =
  new result.data
  result.order = order
  if order == colMajor:
    var data = cast[ref array[N, array[M, float64]]](result.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        data[j][i] = f(i, j)
  else:
    var data = cast[ref array[M, array[N, float64]]](result.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        data[i][j] = f(i, j)

proc randomMatrix*(M, N: static[int], max: float64 = 1, order: OrderType = colMajor): Matrix64[M, N] =
  makeMatrix(M, N, proc(i, j: int): float64 = random(max), order)

proc constantMatrix*(M, N: static[int], x: float64, order: OrderType = colMajor): Matrix64[M, N] =
  new result.data
  result.order = order
  if order == colMajor:
    var data = cast[ref array[N, array[M, float64]]](result.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        data[j][i] = x
  else:
    var data = cast[ref array[M, array[N, float64]]](result.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        data[i][j] = x

proc zeros*(M, N: static[int], order: OrderType = colMajor): Matrix64[M, N] = constantMatrix(M, N, 0, order)

proc ones*(M, N: static[int], order: OrderType = colMajor): Matrix64[M, N] = constantMatrix(M, N, 1, order)

proc eye*(N: static[int], order: OrderType = colMajor): Matrix64[N, N] =
  new result.data
  result.order = order
  var data = cast[ref array[N, array[N, float64]]](result.data)
  for i in 0 .. < N:
    for j in 0 .. < N:
      data[i][j] = if i == j: 1 else: 0

proc dmatrix*(M, N: static[int], xs: seq[seq[float64]], order: OrderType = colMajor): Matrix64[M, N] =
  makeMatrix(M, N, proc(i, j: int): float64 = xs[i][j], order)