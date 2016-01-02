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

template isStatic(a: typed): bool =
  compiles(proc() =
    const v = a)

template makeSVectorPrivate(N, f, result: expr) =
  new result
  for i in 0 .. < N:
    result[i] = f(i)

template makeDVectorPrivate(N, f, T, result: expr) =
  result = newSeq[T](N)
  for i in 0 .. < N:
    result[i] = f(i)

proc makeSVector(N: static[int], f: proc (i: int): float64): Vector64[N] =
  makeSVectorPrivate(N, f, result)

proc makeSVector(N: static[int], f: proc (i: int): float32): Vector32[N] =
  makeSVectorPrivate(N, f, result)

proc makeDVector(N: int, f: proc (i: int): float64): DVector64 =
  makeDVectorPrivate(N, f, float64, result)

proc makeDVector(N: int, f: proc (i: int): float32): DVector32 =
  makeDVectorPrivate(N, f, float32, result)

proc makeVector*(N: int or static[int], f: proc (i: int): float64): auto =
  when N.isStatic: makeSVector(N, f)
  else: makeDVector(N, f)

proc makeVector*(N: int or static[int], f: proc (i: int): float32): auto =
  when N.isStatic: makeSVector(N, f)
  else: makeDVector(N, f)

proc randomVector*(N: int or static[int], max: float64 = 1): auto =
  makeVector(N, proc(i: int): float64 = random(max))

proc randomVector*(N: int or static[int], max: float32): auto =
  makeVector(N, proc(i: int): float32 = random(max).float32)

template constantVectorPrivate(N, x, result: expr) =
  new result
  for i in 0 .. < N:
    result[i] = x

template constantDVectorPrivate(N, x, T, result: expr) =
  result = newSeq[T](N)
  for i in 0 .. < N:
    result[i] = x

proc constantSVector(N: static[int], x: float64): Vector64[N] = constantVectorPrivate(N, x, result)

proc constantSVector(N: static[int], x: float32): Vector32[N] = constantVectorPrivate(N, x, result)

proc constantDVector(N: int, x: float64): DVector64 = constantDVectorPrivate(N, x, float64, result)

proc constantDVector(N: int, x: float32): DVector32 = constantDVectorPrivate(N, x, float32, result)

proc constantVector*(N: int or static[int], x: float32 or float64): auto =
  when N.isStatic: constantSVector(N, x)
  else: constantDVector(N, x)

proc zeros*(N: int or static[int]): auto = constantVector(N, 0'f64)

proc zeros*(N: int or static[int], A: typedesc[float32]): auto = constantVector(N, 0'f32)

proc ones*(N: int or static[int]): auto = constantVector(N, 1'f64)

proc ones*(N: int or static[int], A: typedesc[float32]): auto = constantVector(N, 1'f32)

type Array[N: static[int], A] = array[N, A]
type DoubleArray32[M, N: static[int]] = array[M, array[N, float32]]
type DoubleArray64[M, N: static[int]] = array[M, array[N, float64]]

proc vector*[N: static[int]](xs: Array[N, float64]): Vector64[N] =
  new result
  for i in 0 .. < N:
    result[i] = xs[i]

proc vector*[N: static[int]](xs: Array[N, float32], A: typedesc[float32]): Vector32[N] =
  new result
  for i in 0 .. < N:
    result[i] = xs[i]

proc vector32*[N: static[int]](xs: Array[N, float32]): Vector32[N] =
  new result
  for i in 0 .. < N:
    result[i] = xs[i]

template makeSMatrixPrivate(M, N, f, order, result: expr, A: typedesc) =
  new result.data
  result.order = order
  if order == colMajor:
    var data = cast[ref array[N, array[M, A]]](result.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        data[j][i] = f(i, j)
  else:
    var data = cast[ref array[M, array[N, A]]](result.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        data[i][j] = f(i, j)

template makeDMatrixPrivate(M, N, f, order, result: expr, A: typedesc) =
  result.data = newSeq[A](M * N)
  result.order = order
  result.M = M
  result.N = N
  if order == colMajor:
    for i in 0 .. < M:
      for j in 0 .. < N:
        result.data[j * M + i] = f(i, j)
  else:
    for i in 0 .. < M:
      for j in 0 .. < N:
        result.data[i * N + j] = f(i, j)

proc makeSMatrix(M, N: static[int], f: proc (i, j: int): float64, order: OrderType): Matrix64[M, N] =
  makeSMatrixPrivate(M, N, f, order, result, float64)

proc makeDMatrix(M, N: int, f: proc (i, j: int): float64, order: OrderType): DMatrix64 =
  makeDMatrixPrivate(M, N, f, order, result, float64)

proc makeMatrix*(M: int or static[int], N: int or static[int], f: proc (i, j: int): float64, order: OrderType = colMajor): auto =
  when M.isStatic and N.isStatic: makeSMatrix(M, N, f, order)
  else: makeDMatrix(M, N, f, order)

proc makeSMatrix(M, N: static[int], f: proc (i, j: int): float32, order: OrderType): Matrix32[M, N] =
  makeSMatrixPrivate(M, N, f, order, result, float32)

proc makeDMatrix(M, N: int, f: proc (i, j: int): float32, order: OrderType): DMatrix32 =
  makeDMatrixPrivate(M, N, f, order, result, float32)

proc makeMatrix*(M: int or static[int], N: int or static[int], f: proc (i, j: int): float32, order: OrderType = colMajor): auto =
  when M.isStatic and N.isStatic: makeSMatrix(M, N, f, order)
  else: makeDMatrix(M, N, f, order)

proc randomMatrix*(M: int or static[int], N: int or static[int], max: float64 = 1, order: OrderType = colMajor): auto =
  makeMatrix(M, N, proc(i, j: int): float64 = random(max), order)

proc randomMatrix*(M: int or static[int], N: int or static[int], max: float32, order: OrderType = colMajor): auto =
  makeMatrix(M, N, proc(i, j: int): float32 = random(max).float32, order)

template constantMatrixPrivate(M, N, x, order, result: expr, A: typedesc) =
  new result.data
  result.order = order
  for i in 0 .. < (M * N):
    result.data[i] = x

template constantDMatrixPrivate(M, N, x, order, result: expr, A: typedesc) =
  result.data = newSeq[A](M * N)
  result.order = order
  result.M = M
  result.N = N
  for i in 0 .. < (M * N):
    result.data[i] = x

proc constantSMatrix(M: static[int], N: static[int], x: float64, order: OrderType = colMajor): Matrix64[M, N] =
  constantMatrixPrivate(M, N, x, order, result, float64)

proc constantDMatrix(M, N: int, x: float64, order: OrderType = colMajor): DMatrix64 =
  constantDMatrixPrivate(M, N, x, order, result, float64)

proc constantSMatrix(M, N: static[int], x: float32, order: OrderType = colMajor): Matrix32[M, N] =
  constantMatrixPrivate(M, N, x, order, result, float32)

proc constantDMatrix(M, N: int, x: float32, order: OrderType = colMajor): DMatrix32 =
  constantDMatrixPrivate(M, N, x, order, result, float32)

proc constantMatrix*(M: int or static[int], N: int or static[int], x: float64, order: OrderType = colMajor): auto =
  when M.isStatic and N.isStatic: constantSMatrix(M, N, x, order)
  else: constantDMatrix(M, N, x, order)

proc constantMatrix*(M: int or static[int], N: int or static[int], x: float32, order: OrderType = colMajor): auto =
  when M.isStatic and N.isStatic: constantSMatrix(M, N, x, order)
  else: constantDMatrix(M, N, x, order)

proc zeros*(M: int or static[int], N: int or static[int], order: OrderType = colMajor): auto =
  constantMatrix(M, N, 0'f64, order)

proc zeros*(M: int or static[int], N: int or static[int], A: typedesc[float32], order: OrderType = colMajor): auto =
  constantMatrix(M, N, 0'f32, order)

proc ones*(M: int or static[int], N: int or static[int], order: OrderType = colMajor): auto =
  constantMatrix(M, N, 1'f64, order)

proc ones*(M: int or static[int], N: int or static[int], A: typedesc[float32], order: OrderType = colMajor): auto =
  constantMatrix(M, N, 1'f32, order)

template eyePrivateS(N, order, result: expr, A: typedesc) =
  new result.data
  result.order = order
  var data = cast[ref array[N, array[N, A]]](result.data)
  for i in 0 .. < N:
    for j in 0 .. < N:
      data[i][j] = if i == j: 1 else: 0

proc eye32S(N: static[int], order: OrderType): Matrix32[N, N] = eyePrivateS(N, order, result, float32)

proc eye64S(N: static[int], order: OrderType): Matrix64[N, N] = eyePrivateS(N, order, result, float64)

proc eye32D(N: int, order: OrderType): DMatrix32 =
  result = zeros(N, N, float32, order)
  for i in 0 .. < N:
    result.data[i + N * i] = 1'f32

proc eye64D(N: int, order: OrderType): DMatrix64 =
  result = zeros(N, N, order)
  for i in 0 .. < N:
    result.data[i + N * i] = 1'f64

proc eye*(N: int or static[int], order: OrderType = colMajor): auto =
  when N.isStatic: eye64S(N, order)
  else: eye64D(N, order)

proc eye*(N: int or static[int], A: typedesc[float32], order: OrderType = colMajor): auto =
  when N.isStatic: eye32S(N, order)
  else: eye32D(N, order)

proc matrix*(xs: seq[seq[float32]], order: OrderType = colMajor): DMatrix32 =
  makeMatrix(xs.len, xs[0].len, proc(i, j: int): float32= xs[i][j], order)

proc matrix*(xs: seq[seq[float64]], order: OrderType = colMajor): DMatrix64 =
  makeMatrix(xs.len, xs[0].len, proc(i, j: int): float64 = xs[i][j], order)

proc matrix*[M, N: static[int]](xs: DoubleArray32[M, N], order: OrderType = colMajor): Matrix32[M, N] =
  makeMatrix(M, N, proc(i, j: int): float32 = xs[i][j], order)

proc matrix*[M, N: static[int]](xs: DoubleArray64[M, N], order: OrderType = colMajor): Matrix64[M, N] =
  makeMatrix(M, N, proc(i, j: int): float64 = xs[i][j], order)

proc matrix*(xs: seq[seq[float32]], M, N: static[int], order: OrderType = colMajor): Matrix32[M, N] =
  makeMatrix(M, N, proc(i, j: int): float32 = xs[i][j], order)

proc matrix*(xs: seq[seq[float64]], M, N: static[int], order: OrderType = colMajor): Matrix64[M, N] =
  makeMatrix(M, N, proc(i, j: int): float64 = xs[i][j], order)