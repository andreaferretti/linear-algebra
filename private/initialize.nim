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

template makeVectorPrivate(N, f, result: expr) =
  new result
  for i in 0 .. < N:
    result[i] = f(i)

template makeDVectorPrivate(N, f, T, result: expr) =
  result = newSeq[T](N)
  for i in 0 .. < N:
    result[i] = f(i)

proc makeDVector*(N: int, f: proc (i: int): float64): DVector64 = makeDVectorPrivate(N, f, float64, result)

proc makeDVector*(N: int, f: proc (i: int): float32): DVector32 = makeDVectorPrivate(N, f, float32, result)

proc makeVector*(N: static[int], f: proc (i: int): float64): Vector64[N] = makeVectorPrivate(N, f, result)

proc makeVector*(N: static[int], f: proc (i: int): float32): Vector32[N] = makeVectorPrivate(N, f, result)

proc randomDVector*(N: int, max: float64 = 1): DVector64 =
  makeDVector(N, proc(i: int): float64 = random(max))

proc randomDVector*(N: int, max: float32): DVector32 =
  makeDVector(N, proc(i: int): float32 = random(max))

proc randomVector*(N: static[int], max: float64 = 1): Vector64[N] =
  makeVector(N, proc(i: int): float64 = random(max))

proc randomVector*(N: static[int], max: float32): Vector32[N] =
  makeVector(N, proc(i: int): float32 = random(max).float32)

template constantVectorPrivate(N, x, result: expr) =
  new result
  for i in 0 .. < N:
    result[i] = x

template constantDVectorPrivate(N, x, T, result: expr) =
  result = newSeq[T](N)
  for i in 0 .. < N:
    result[i] = x

proc constantVector*(N: static[int], x: float64): Vector64[N] = constantVectorPrivate(N, x, result)

proc constantVector*(N: static[int], x: float32): Vector32[N] = constantVectorPrivate(N, x, result)

proc constantVector*(N: int, x: float64): DVector64 = constantDVectorPrivate(N, x, float64, result)

proc constantVector*(N: int, x: float32): DVector32 = constantDVectorPrivate(N, x, float32, result)

proc zeros*(N: int): DVector64 = constantVector(N, 0'f64)

proc zeros*(N: int, A: typedesc): auto =
  when A is float64: constantVector(N, 0'f64)
  else:
    when A is float32: constantVector(N, 0'f32)

proc zeros*(N: static[int]): Vector64[N] = constantVector(N, 0'f64)

proc zeros*(N: static[int], A: typedesc): auto =
  when A is float64: constantVector(N, 0'f64)
  else:
    when A is float32: constantVector(N, 0'f32)

proc ones*(N: int): DVector64 = constantVector(N, 1'f64)

proc ones*(N: int, A: typedesc): auto =
  when A is float64: constantVector(N, 1'f64)
  else:
    when A is float32: constantVector(N, 1'f32)

proc ones*(N: static[int]): Vector64[N] = constantVector(N, 1'f64)

proc ones*(N: static[int], A: typedesc): auto =
  when A is float64: constantVector(N, 1'f64)
  else:
    when A is float32: constantVector(N, 1'f32)

type Array[N: static[int], A] = array[N, A]

proc vector*[N: static[int]](xs: Array[N, float64]): Vector64[N] =
  new result
  for i in 0 .. < N:
    result[i] = xs[i]

proc vector*[N: static[int]](xs: Array[N, float32], A: typedesc): Vector32[N] =
  when A is float32:
    new result
    for i in 0 .. < N:
      result[i] = xs[i]

proc vector32*[N: static[int]](xs: Array[N, float32]): Vector32[N] =
  new result
  for i in 0 .. < N:
    result[i] = xs[i]

proc dvector*(N: static[int], xs: seq[float64]): Vector64[N] =
  makeVector(N, proc(i: int): float64 = xs[i])

proc dvector*(N: static[int], xs: seq[float32]): Vector32[N] =
  makeVector(N, proc(i: int): float32 = xs[i])

template makeMatrixPrivate(M, N, f, order, result: expr, A: typedesc) =
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

proc makeMatrix*(M, N: static[int], f: proc (i, j: int): float64, order: OrderType = colMajor): Matrix64[M, N] =
  makeMatrixPrivate(M, N, f, order, result, float64)

proc makeMatrix*(M, N: static[int], f: proc (i, j: int): float32, order: OrderType = colMajor): Matrix32[M, N] =
  makeMatrixPrivate(M, N, f, order, result, float32)

proc makeDMatrix*(M, N: int, f: proc (i, j: int): float64, order: OrderType = colMajor): DMatrix64 =
  makeDMatrixPrivate(M, N, f, order, result, float64)

proc makeDMatrix*(M, N: int, f: proc (i, j: int): float32, order: OrderType = colMajor): DMatrix32 =
  makeDMatrixPrivate(M, N, f, order, result, float32)

proc randomMatrix*(M, N: static[int], max: float64 = 1, order: OrderType = colMajor): Matrix64[M, N] =
  makeMatrix(M, N, proc(i, j: int): float64 = random(max), order)

proc randomMatrix*(M, N: static[int], max: float32, order: OrderType = colMajor): Matrix32[M, N] =
  makeMatrix(M, N, proc(i, j: int): float32 = random(max).float32, order)

proc randomDMatrix*(M, N: int, max: float64 = 1, order: OrderType = colMajor): DMatrix64 =
  makeDMatrix(M, N, proc(i, j: int): float64 = random(max), order)

proc randomDMatrix*(M, N: int, max: float32, order: OrderType = colMajor): DMatrix32 =
  makeDMatrix(M, N, proc(i, j: int): float32 = random(max).float32, order)

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

proc constantMatrix*(M, N: static[int], x: float64, order: OrderType = colMajor): Matrix64[M, N] =
  constantMatrixPrivate(M, N, x, order, result, float64)

proc constantMatrix*(M, N: static[int], x: float32, order: OrderType = colMajor): Matrix32[M, N] =
  constantMatrixPrivate(M, N, x, order, result, float32)

proc constantDMatrix*(M, N: int, x: float64, order: OrderType = colMajor): DMatrix64 =
  constantDMatrixPrivate(M, N, x, order, result, float64)

proc constantDMatrix*(M, N: int, x: float32, order: OrderType = colMajor): DMatrix32 =
  constantDMatrixPrivate(M, N, x, order, result, float32)

proc zeros*(M: int or static[int], N: int or static[int], order: OrderType = colMajor): auto =
  when M.isStatic and N.isStatic:
    constantMatrix(M, N, 0'f64, order)
  else:
    constantDMatrix(M, N, 0'f64, order)

proc zeros*(M: int or static[int], N: int or static[int], A: typedesc[float32], order: OrderType = colMajor): auto =
  when M.isStatic and N.isStatic:
    constantMatrix(M, N, 0'f32, order)
  else:
    constantDMatrix(M, N, 0'f32, order)

proc ones*(M: int or static[int], N: int or static[int], order: OrderType = colMajor): auto =
  when M.isStatic and N.isStatic:
    constantMatrix(M, N, 1'f64, order)
  else:
    constantDMatrix(M, N, 1'f64, order)

proc ones*(M: int or static[int], N: int or static[int], A: typedesc[float32], order: OrderType = colMajor): auto =
  when M.isStatic and N.isStatic:
    constantMatrix(M, N, 1'f32, order)
  else:
    constantDMatrix(M, N, 1'f32, order)

template eyePrivate(N, order, result: expr, A: typedesc) =
  new result.data
  result.order = order
  var data = cast[ref array[N, array[N, A]]](result.data)
  for i in 0 .. < N:
    for j in 0 .. < N:
      data[i][j] = if i == j: 1 else: 0


proc eye32(N: static[int], order: OrderType): Matrix32[N, N] = eyePrivate(N, order, result, float32)

proc eye64(N: static[int], order: OrderType): Matrix64[N, N] = eyePrivate(N, order, result, float64)

proc eye*(N: static[int], order: OrderType = colMajor): Matrix64[N, N] = eye64(N, order)

proc eye*(N: static[int], A: typedesc, order: OrderType = colMajor): auto =
  when A is float64: eye64(N, order)
  else:
    when A is float32: eye32(N, order)

proc dmatrix*(M, N: static[int], xs: seq[seq[float64]], order: OrderType = colMajor): Matrix64[M, N] =
  makeMatrix(M, N, proc(i, j: int): float64 = xs[i][j], order)

proc dmatrix*(M, N: static[int], xs: seq[seq[float32]], order: OrderType = colMajor): Matrix32[M, N] =
  makeMatrix(M, N, proc(i, j: int): float32 = xs[i][j], order)