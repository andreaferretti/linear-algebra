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

proc `*=`*[N: static[int]](v: var Vector64[N], k: float64) {. inline .} = dscal(N, k, v.fp, 1)

proc `*`*[N: static[int]](v: Vector64[N], k: float64): Vector64[N]  {. inline .} =
  new result
  dcopy(N, v.fp, 1, result.fp, 1)
  dscal(N, k, result.fp, 1)

proc `*=`*[N: static[int]](v: var Vector32[N], k: float32) {. inline .} = sscal(N, k, v.fp, 1)

proc `*`*[N: static[int]](v: Vector32[N], k: float32): Vector32[N]  {. inline .} =
  new result
  scopy(N, v.fp, 1, result.fp, 1)
  sscal(N, k, result.fp, 1)

proc `+=`*[N: static[int]](v: var Vector64[N], w: Vector64[N]) {. inline .} =
  daxpy(N, 1, w.fp, 1, v.fp, 1)

proc `+`*[N: static[int]](v, w: Vector64[N]): Vector64[N]  {. inline .} =
  new result
  dcopy(N, v.fp, 1, result.fp, 1)
  daxpy(N, 1, w.fp, 1, result.fp, 1)

proc `-=`*[N: static[int]](v: var Vector64[N], w: Vector64[N]) {. inline .} =
  daxpy(N, -1, w.fp, 1, v.fp, 1)

proc `-`*[N: static[int]](v, w: Vector64[N]): Vector64[N]  {. inline .} =
  new result
  dcopy(N, v.fp, 1, result.fp, 1)
  daxpy(N, -1, w.fp, 1, result.fp, 1)

proc `*`*[N: static[int]](v, w: Vector64[N]): float64 {. inline .} = ddot(N, v.fp, 1, w.fp, 1)

proc l_2*[N: static[int]](v: Vector64[N]): float64 {. inline .} = dnrm2(N, v.fp, 1)

proc l_1*[N: static[int]](v: Vector64[N]): float64 {. inline .} = dasum(N, v.fp, 1)

proc maxIndex*[N: static[int]](v: Vector64[N]): tuple[i: int, val: float64] =
  var
    j = 0
    m = v[0]
  for i, val in v:
    if val > m:
      j = i
      m = val
  return (j, m)

template max*(v: Vector64): float64 = maxIndex(v).val

proc minIndex*[N: static[int]](v: Vector64[N]): tuple[i: int, val: float64] =
  var
    j = 0
    m = v[0]
  for i, val in v:
    if val < m:
      j = i
      m = val
  return (j, m)

template min*(v: Vector64): float64 = minIndex(v).val

proc `~=`*[N: static[int]](v, w: Vector64[N]): bool =
  const epsilon = 0.000001
  let
    vNorm = l_1(v)
    wNorm = l_1(w)
    dNorm = l_1(v - w)
  return (dNorm / (vNorm + wNorm)) < epsilon

proc `*`*[M, N: static[int]](a: Matrix64[M, N], v: Vector64[N]): Vector64[M]  {. inline .} =
  new result
  dgemv(a.order, noTranspose, M, N, 1, a.fp, M, v.fp, 1, 0, result.fp, 1)

proc `*=`*[M, N: static[int]](m: var Matrix64[M, N], k: float64) {. inline .} = dscal(M * N, k, m.fp, 1)

proc `*`*[M, N: static[int]](m: Matrix64[M, N], k: float64): Matrix64[M, N]  {. inline .} =
  new result.data
  result.order = m.order
  dcopy(M * N, m.fp, 1, result.fp, 1)
  dscal(M * N, k, result.fp, 1)

template `*`*(k: float64, v: Vector64 or Matrix64): expr = v * k

template `*`*(k: float32, v: Vector32 or Matrix32): expr = v * k

proc `+=`*[M, N: static[int]](a: var Matrix64[M, N], b: Matrix64[M, N]) {. inline .} =
  if a.order == b.order:
    daxpy(M * N, 1, b.fp, 1, a.fp, 1)
  elif a.order == colMajor and b.order == rowMajor:
    let
      a_data = cast[ref array[N, array[M, float64]]](a.data)
      b_data = cast[ref array[M, array[N, float64]]](b.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        a_data[j][i] += b_data[i][j]
  else:
    let
      a_data = cast[ref array[M, array[N, float64]]](a.data)
      b_data = cast[ref array[N, array[M, float64]]](b.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        a_data[i][j] += b_data[j][i]

proc `+`*[M, N: static[int]](a, b: Matrix64[M, N]): Matrix64[M, N]  {. inline .} =
  new result.data
  result.order = a.order
  dcopy(M * N, a.fp, 1, result.fp, 1)
  result += b

proc `-=`*[M, N: static[int]](a: var Matrix64[M, N], b: Matrix64[M, N]) {. inline .} =
  if a.order == b.order:
    daxpy(M * N, -1, b.fp, 1, a.fp, 1)
  elif a.order == colMajor and b.order == rowMajor:
    let
      a_data = cast[ref array[N, array[M, float64]]](a.data)
      b_data = cast[ref array[M, array[N, float64]]](b.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        a_data[j][i] -= b_data[i][j]
  else:
    let
      a_data = cast[ref array[M, array[N, float64]]](a.data)
      b_data = cast[ref array[N, array[M, float64]]](b.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        a_data[i][j] -= b_data[j][i]

proc `-`*[M, N: static[int]](a, b: Matrix64[M, N]): Matrix64[M, N]  {. inline .} =
  new result.data
  result.order = a.order
  dcopy(M * N, a.fp, 1, result.fp, 1)
  result -= b

proc l_2*[M, N: static[int]](m: Matrix64[M, N]): float64 {. inline .} = dnrm2(M * N, m.fp, 1)

proc l_1*[M, N: static[int]](m: Matrix64[M, N]): float64 {. inline .} = dasum(M * N, m.fp, 1)

proc `~=`*[M, N: static[int]](m, n: Matrix64[M, N]): bool =
  const epsilon = 0.000001
  let
    mNorm = l_1(m)
    nNorm = l_1(n)
    dNorm = l_1(m - n)
  return (dNorm / (mNorm + nNorm)) < epsilon

template `~!=`*(v, w: Vector64 or Matrix64): bool = not (v ~= w)

template max*(m: Matrix64): float64 = max(m.data)

template min*(m: Matrix64): float64 = min(m.data)

proc `*`*[M, N, K: static[int]](a: Matrix64[M, K], b: Matrix64[K, N]): Matrix64[M, N] {. inline .} =
  new result.data
  if a.order == b.order:
    result.order = a.order
    dgemm(a.order, noTranspose, noTranspose, M, N, K, 1, a.fp, M, b.fp, K, 0, result.fp, M)
  elif a.order == colMajor and b.order == rowMajor:
    result.order = colMajor
    dgemm(colMajor, noTranspose, transpose, M, N, K, 1, a.fp, M, b.fp, N, 0, result.fp, M)
  else:
    result.order = colMajor
    dgemm(colMajor, transpose, noTranspose, M, N, K, 1, a.fp, K, b.fp, K, 0, result.fp, M)