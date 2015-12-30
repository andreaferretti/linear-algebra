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

template len(m: DMatrix32 or DMatrix64): int = m.M * m.N

template initLike(r, m: typed) =
  when m is DMatrix32:
    r.data = newSeq[float32](m.len)
  when m is DMatrix64:
    r.data = newSeq[float64](m.len)
  r.order = m.order
  r.M = m.M
  r.N = m.N

proc `*=`*[N: static[int]](v: var Vector32[N], k: float32) {. inline .} = scal(N, k, v.fp, 1)

proc `*=`*(v: var DVector32, k: float32) {. inline .} = scal(v.len, k, v.fp, 1)

proc `*`*[N: static[int]](v: Vector32[N], k: float32): Vector32[N]  {. inline .} =
  new result
  copy(N, v.fp, 1, result.fp, 1)
  scal(N, k, result.fp, 1)

proc `*`*(v: DVector32, k: float32): DVector32 {. inline .} =
  let N = v.len
  result = newSeq[float32](N)
  copy(N, v.fp, 1, result.fp, 1)
  scal(N, k, result.fp, 1)

proc `*=`*[N: static[int]](v: var Vector64[N], k: float64) {. inline .} = scal(N, k, v.fp, 1)

proc `*=`*(v: var DVector64, k: float64) {. inline .} = scal(v.len, k, v.fp, 1)

proc `*`*[N: static[int]](v: Vector64[N], k: float64): Vector64[N]  {. inline .} =
  new result
  copy(N, v.fp, 1, result.fp, 1)
  scal(N, k, result.fp, 1)

proc `*`*(v: DVector64, k: float64): DVector64 {. inline .} =
  let N = v.len
  result = newSeq[float64](N)
  copy(N, v.fp, 1, result.fp, 1)
  scal(N, k, result.fp, 1)

proc `+=`*[N: static[int]](v: var Vector32[N], w: Vector32[N]) {. inline .} =
  axpy(N, 1, w.fp, 1, v.fp, 1)

proc `+=`*(v: var DVector32, w: DVector32) {. inline .} =
  assert(v.len == w.len)
  let N = v.len
  axpy(N, 1, w.fp, 1, v.fp, 1)

proc `+`*[N: static[int]](v, w: Vector32[N]): Vector32[N]  {. inline .} =
  new result
  copy(N, v.fp, 1, result.fp, 1)
  axpy(N, 1, w.fp, 1, result.fp, 1)

proc `+`*(v, w: DVector32): DVector32  {. inline .} =
  assert(v.len == w.len)
  let N = v.len
  result = newSeq[float32](N)
  copy(N, v.fp, 1, result.fp, 1)
  axpy(N, 1, w.fp, 1, result.fp, 1)

proc `+=`*[N: static[int]](v: var Vector64[N], w: Vector64[N]) {. inline .} =
  axpy(N, 1, w.fp, 1, v.fp, 1)

proc `+=`*(v: var DVector64, w: DVector64) {. inline .} =
  assert(v.len == w.len)
  let N = v.len
  axpy(N, 1, w.fp, 1, v.fp, 1)

proc `+`*[N: static[int]](v, w: Vector64[N]): Vector64[N]  {. inline .} =
  new result
  copy(N, v.fp, 1, result.fp, 1)
  axpy(N, 1, w.fp, 1, result.fp, 1)

proc `+`*(v, w: DVector64): DVector64  {. inline .} =
  assert(v.len == w.len)
  let N = v.len
  result = newSeq[float64](N)
  copy(N, v.fp, 1, result.fp, 1)
  axpy(N, 1, w.fp, 1, result.fp, 1)

proc `-=`*[N: static[int]](v: var Vector32[N], w: Vector32[N]) {. inline .} =
  axpy(N, -1, w.fp, 1, v.fp, 1)

proc `-=`*(v: var DVector32, w: DVector32) {. inline .} =
  assert(v.len == w.len)
  let N = v.len
  axpy(N, -1, w.fp, 1, v.fp, 1)

proc `-`*[N: static[int]](v, w: Vector32[N]): Vector32[N]  {. inline .} =
  new result
  copy(N, v.fp, 1, result.fp, 1)
  axpy(N, -1, w.fp, 1, result.fp, 1)

proc `-`*(v, w: DVector32): DVector32  {. inline .} =
  assert(v.len == w.len)
  let N = v.len
  result = newSeq[float32](N)
  copy(N, v.fp, 1, result.fp, 1)
  axpy(N, -1, w.fp, 1, result.fp, 1)

proc `-=`*[N: static[int]](v: var Vector64[N], w: Vector64[N]) {. inline .} =
  axpy(N, -1, w.fp, 1, v.fp, 1)

proc `-=`*(v: var DVector64, w: DVector64) {. inline .} =
  assert(v.len == w.len)
  let N = v.len
  axpy(N, -1, w.fp, 1, v.fp, 1)

proc `-`*[N: static[int]](v, w: Vector64[N]): Vector64[N]  {. inline .} =
  new result
  copy(N, v.fp, 1, result.fp, 1)
  axpy(N, -1, w.fp, 1, result.fp, 1)

proc `-`*(v, w: DVector64): DVector64  {. inline .} =
  assert(v.len == w.len)
  let N = v.len
  result = newSeq[float64](N)
  copy(N, v.fp, 1, result.fp, 1)
  axpy(N, -1, w.fp, 1, result.fp, 1)

proc `*`*[N: static[int]](v, w: Vector32[N]): float32 {. inline .} = dot(N, v.fp, 1, w.fp, 1)

proc `*`*[N: static[int]](v, w: Vector64[N]): float64 {. inline .} = dot(N, v.fp, 1, w.fp, 1)

proc `*`*(v, w: DVector32): float32 {. inline .} =
  assert(v.len == w.len)
  let N = v.len
  dot(N, v.fp, 1, w.fp, 1)

proc `*`*(v, w: DVector64): float64 {. inline .} =
  assert(v.len == w.len)
  let N = v.len
  dot(N, v.fp, 1, w.fp, 1)

proc l_2*[N: static[int]](v: Vector32[N] or Vector64[N]): auto {. inline .} = nrm2(N, v.fp, 1)

proc l_2*(v: DVector32 or DVector64): auto {. inline .} = nrm2(v.len, v.fp, 1)

proc l_1*[N: static[int]](v: Vector32[N] or Vector64[N]): auto {. inline .} = asum(N, v.fp, 1)

proc l_1*(v: DVector32 or DVector64): auto {. inline .} = asum(v.len, v.fp, 1)

template maxIndexPrivate(N, v: expr): auto =
  var
    j = 0
    m = v[0]
  for i, val in v:
    if val > m:
      j = i
      m = val
  (j, m)

proc maxIndex*[N: static[int]](v: Vector32[N]): tuple[i: int, val: float32] =
  maxIndexPrivate(N, v)

proc maxIndex*[N: static[int]](v: Vector64[N]): tuple[i: int, val: float64] =
  maxIndexPrivate(N, v)

proc maxIndex*(v: DVector32): tuple[i: int, val: float32] =
  maxIndexPrivate(v.len, v)

proc maxIndex*(v: DVector64): tuple[i: int, val: float64] =
  maxIndexPrivate(v.len, v)

template max*(v: Vector32 or Vector64): auto = maxIndex(v).val

template minIndexPrivate(N, v: expr): auto =
  var
    j = 0
    m = v[0]
  for i, val in v:
    if val < m:
      j = i
      m = val
  return (j, m)

proc minIndex*[N: static[int]](v: Vector32[N]): tuple[i: int, val: float32] =
  minIndexPrivate(N, v)

proc minIndex*[N: static[int]](v: Vector64[N]): tuple[i: int, val: float64] =
  minIndexPrivate(N, v)

proc minIndex*(v: DVector32): tuple[i: int, val: float32] =
  minIndexPrivate(v.len, v)

proc minIndex*(v: DVector64): tuple[i: int, val: float64] =
  minIndexPrivate(v.len, v)

template min*(v: Vector32 or Vector64): auto = minIndex(v).val

proc `*`*[M, N: static[int]](a: Matrix32[M, N], v: Vector32[N]): Vector32[M]  {. inline .} =
  new result
  let lda = if a.order == colMajor: M.int else: N.int
  gemv(a.order, noTranspose, M, N, 1, a.fp, lda, v.fp, 1, 0, result.fp, 1)

proc `*`*(a: DMatrix32, v: DVector32): DVector32  {. inline .} =
  result = newSeq[float32](a.M)
  let lda = if a.order == colMajor: a.M.int else: a.N.int
  gemv(a.order, noTranspose, a.M, a.N, 1, a.fp, lda, v.fp, 1, 0, result.fp, 1)

proc `*`*[M, N: static[int]](a: Matrix64[M, N], v: Vector64[N]): Vector64[M]  {. inline .} =
  new result
  let lda = if a.order == colMajor: M.int else: N.int
  gemv(a.order, noTranspose, M, N, 1, a.fp, lda, v.fp, 1, 0, result.fp, 1)

proc `*`*(a: DMatrix64, v: DVector64): DVector64  {. inline .} =
  result = newSeq[float64](a.M)
  let lda = if a.order == colMajor: a.M.int else: a.N.int
  gemv(a.order, noTranspose, a.M, a.N, 1, a.fp, lda, v.fp, 1, 0, result.fp, 1)

proc `*=`*[M, N: static[int]](m: var Matrix32[M, N], k: float32) {. inline .} = scal(M * N, k, m.fp, 1)

proc `*`*[M, N: static[int]](m: Matrix32[M, N], k: float32): Matrix32[M, N]  {. inline .} =
  new result.data
  result.order = m.order
  copy(M * N, m.fp, 1, result.fp, 1)
  scal(M * N, k, result.fp, 1)

proc `*=`*(m: var DMatrix32, k: float32) {. inline .} = scal(m.M * m.N, k, m.fp, 1)

proc `*`*(m: DMatrix32, k: float32): DMatrix32  {. inline .} =
  result.initLike(m)
  copy(m.len, m.fp, 1, result.fp, 1)
  scal(m.len, k, result.fp, 1)

proc `*=`*[M, N: static[int]](m: var Matrix64[M, N], k: float64) {. inline .} = scal(M * N, k, m.fp, 1)

proc `*`*[M, N: static[int]](m: Matrix64[M, N], k: float64): Matrix64[M, N]  {. inline .} =
  new result.data
  result.order = m.order
  copy(M * N, m.fp, 1, result.fp, 1)
  scal(M * N, k, result.fp, 1)

proc `*=`*(m: var DMatrix64, k: float64) {. inline .} = scal(m.M * m.N, k, m.fp, 1)

proc `*`*(m: DMatrix64, k: float64): DMatrix64  {. inline .} =
  result.initLike(m)
  copy(m.len, m.fp, 1, result.fp, 1)
  scal(m.len, k, result.fp, 1)

template `*`*(k: float32, v: Vector32 or Matrix32 or DVector32 or DMatrix32): expr = v * k

template `/`*(v: Vector32 or Matrix32 or DVector32 or DMatrix32, k: float32): expr = v * (1 / k)

template `/=`*(v: var Vector32 or var Matrix32 or var DVector32 or var DMatrix32, k: float32): expr = v *= (1 / k)

template `*`*(k: float64, v: Vector64 or Matrix64 or DVector64 or DMatrix64): expr = v * k

template `/`*(v: Vector64 or Matrix64 or DVector64 or DMatrix64, k: float64): expr = v * (1 / k)

template `/=`*(v: var Vector64 or var Matrix64 or var DVector64 or var DMatrix64, k: float64): expr = v *= (1 / k)

template matrixAdd(M, N, a, b: expr, A: typedesc) =
  if a.order == b.order:
    axpy(M * N, 1, b.fp, 1, a.fp, 1)
  elif a.order == colMajor and b.order == rowMajor:
    let
      a_data = cast[ref array[N, array[M, A]]](a.data)
      b_data = cast[ref array[M, array[N, A]]](b.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        a_data[j][i] += b_data[i][j]
  else:
    let
      a_data = cast[ref array[M, array[N, A]]](a.data)
      b_data = cast[ref array[N, array[M, A]]](b.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        a_data[i][j] += b_data[j][i]

template matrixDAdd(a, b: expr) =
  assert a.M == b.M and a.N == a.N
  if a.order == b.order:
    axpy(a.M * a.N, 1, b.fp, 1, a.fp, 1)
  elif a.order == colMajor and b.order == rowMajor:
    for i in 0 .. < a.M:
      for j in 0 .. < a.N:
        a.data[j * a.M + i] += b.data[i * b.N + j]
  else:
    for i in 0 .. < a.M:
      for j in 0 .. < a.N:
        a.data[i * a.N + j] += b.data[j * b.M + i]

proc `+=`*[M, N: static[int]](a: var Matrix32[M, N], b: Matrix32[M, N]) {. inline .} =
  matrixAdd(M, N, a, b, float32)

proc `+=`*(a: var DMatrix32, b: DMatrix32) {. inline .} = matrixDAdd(a, b)

proc `+=`*[M, N: static[int]](a: var Matrix64[M, N], b: Matrix64[M, N]) {. inline .} =
  matrixAdd(M, N, a, b, float64)

proc `+=`*(a: var DMatrix64, b: DMatrix64) {. inline .} = matrixDAdd(a, b)

proc `+`*[M, N: static[int]](a, b: Matrix32[M, N]): Matrix32[M, N] {. inline .} =
  new result.data
  result.order = a.order
  copy(M * N, a.fp, 1, result.fp, 1)
  result += b

proc `+`*(a, b: DMatrix32): DMatrix32 {. inline .} =
  result.initLike(a)
  copy(a.len, a.fp, 1, result.fp, 1)
  result += b

proc `+`*[M, N: static[int]](a, b: Matrix64[M, N]): Matrix64[M, N]  {. inline .} =
  new result.data
  result.order = a.order
  copy(M * N, a.fp, 1, result.fp, 1)
  result += b

proc `+`*(a, b: DMatrix64): DMatrix64 {. inline .} =
  result.initLike(a)
  copy(a.len, a.fp, 1, result.fp, 1)
  result += b

template matrixSub(M, N, a, b: expr, A: typedesc) =
  if a.order == b.order:
    axpy(M * N, -1, b.fp, 1, a.fp, 1)
  elif a.order == colMajor and b.order == rowMajor:
    let
      a_data = cast[ref array[N, array[M, A]]](a.data)
      b_data = cast[ref array[M, array[N, A]]](b.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        a_data[j][i] -= b_data[i][j]
  else:
    let
      a_data = cast[ref array[M, array[N, A]]](a.data)
      b_data = cast[ref array[N, array[M, A]]](b.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        a_data[i][j] -= b_data[j][i]

template matrixDSub(a, b: expr) =
  assert a.M == b.M and a.N == a.N
  if a.order == b.order:
    axpy(a.M * a.N, -1, b.fp, 1, a.fp, 1)
  elif a.order == colMajor and b.order == rowMajor:
    for i in 0 .. < a.M:
      for j in 0 .. < a.N:
        a.data[j * a.M + i] -= b.data[i * b.N + j]
  else:
    for i in 0 .. < a.M:
      for j in 0 .. < a.N:
        a.data[i * a.N + j] -= b.data[j * b.M + i]

proc `-=`*[M, N: static[int]](a: var Matrix32[M, N], b: Matrix32[M, N]) {. inline .} =
  matrixSub(M, N, a, b, float32)

proc `-=`*(a: var DMatrix32, b: DMatrix32) {. inline .} = matrixDSub(a, b)

proc `-`*[M, N: static[int]](a, b: Matrix32[M, N]): Matrix32[M, N]  {. inline .} =
  new result.data
  result.order = a.order
  copy(M * N, a.fp, 1, result.fp, 1)
  result -= b

proc `-`*(a, b: DMatrix32): DMatrix32 {. inline .} =
  result.initLike(a)
  copy(a.len, a.fp, 1, result.fp, 1)
  result -= b

proc `-=`*[M, N: static[int]](a: var Matrix64[M, N], b: Matrix64[M, N]) {. inline .} =
  matrixSub(M, N, a, b, float64)

proc `-=`*(a: var DMatrix64, b: DMatrix64) {. inline .} = matrixDSub(a, b)

proc `-`*[M, N: static[int]](a, b: Matrix64[M, N]): Matrix64[M, N]  {. inline .} =
  new result.data
  result.order = a.order
  copy(M * N, a.fp, 1, result.fp, 1)
  result -= b

proc `-`*(a, b: DMatrix64): DMatrix64 {. inline .} =
  result.initLike(a)
  copy(a.len, a.fp, 1, result.fp, 1)
  result -= b

proc l_2*[M, N: static[int]](m: Matrix32[M, N] or Matrix64[M, N]): auto {. inline .} =
  nrm2(M * N, m.fp, 1)

proc l_2*(m: DMatrix32 or DMatrix64): auto {. inline .} = nrm2(m.len, m.fp, 1)

proc l_1*[M, N: static[int]](m: Matrix32[M, N] or Matrix64[M, N]): auto {. inline .} =
  asum(M * N, m.fp, 1)

proc l_1*(m: DMatrix32 or DMatrix64): auto {. inline .} = asum(m.len, m.fp, 1)

template max*(m: Matrix32 or Matrix64 or DMatrix32 or DMatrix64): auto = max(m.data)

template min*(m: Matrix32 or Matrix64 or DMatrix32 or DMatrix64): auto = min(m.data)

template matrixMult(M, N, K, a, b, result: expr): auto =
  new result.data
  if a.order == colMajor and b.order == colMajor:
    result.order = colMajor
    gemm(colMajor, noTranspose, noTranspose, M, N, K, 1, a.fp, M, b.fp, K, 0, result.fp, M)
  elif a.order == rowMajor and b.order == rowMajor:
    result.order = rowMajor
    gemm(rowMajor, noTranspose, noTranspose, M, N, K, 1, a.fp, K, b.fp, N, 0, result.fp, N)
  elif a.order == colMajor and b.order == rowMajor:
    result.order = colMajor
    gemm(colMajor, noTranspose, transpose, M, N, K, 1, a.fp, M, b.fp, N, 0, result.fp, M)
  else:
    result.order = colMajor
    gemm(colMajor, transpose, noTranspose, M, N, K, 1, a.fp, K, b.fp, K, 0, result.fp, M)

proc `*`*[M, N, K: static[int]](a: Matrix32[M, K], b: Matrix32[K, N]): Matrix32[M, N] {. inline .} =
  matrixMult(M, N, K, a, b, result)

proc `*`*[M, N, K: static[int]](a: Matrix64[M, K], b: Matrix64[K, N]): Matrix64[M, N] {. inline .} =
  matrixMult(M, N, K, a, b, result)

proc compareApprox(a, b: Vector32 or Vector64 or DVector32 or DVector64 or Matrix32 or Matrix64): bool =
  const epsilon = 0.000001
  let
    aNorm = l_1(a)
    bNorm = l_1(b)
    dNorm = l_1(a - b)
  return (dNorm / (aNorm + bNorm)) < epsilon

template `=~`*[N: static[int]](v, w: Vector32[N]): bool = compareApprox(v, w)

template `=~`*[N: static[int]](v, w: Vector64[N]): bool = compareApprox(v, w)

template `=~`*(v, w: DVector32): bool = compareApprox(v, w)

template `=~`*(v, w: DVector64): bool = compareApprox(v, w)

proc `=~`*[M, N: static[int]](m, n: Matrix32[M, N]): bool = compareApprox(m, n)

proc `=~`*[M, N: static[int]](m, n: Matrix64[M, N]): bool = compareApprox(m, n)

template `!=~`*(a, b: Vector32 or Vector64 or DVector32 or DVector64 or Matrix32 or Matrix64): bool = not (a =~ b)