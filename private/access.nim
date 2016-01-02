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

proc len*[N: static[int]](v: Vector32[N] or Vector64[N]): int = N

proc clone*[N: static[int]](v: Vector64[N]): Vector64[N] =
  new result
  copyMem(result.fp, v.fp, N * sizeof(float64))

proc clone*[N: static[int]](v: Vector32[N]): Vector32[N] =
  new result
  copyMem(result.fp, v.fp, N * sizeof(float32))

proc clone*(v: DVector64 or DVector32): auto =
  result = v

proc map*[N: static[int]](v: Vector32[N], f: proc(x: float32): float32): Vector32[N] =
  new result
  for i in 0 .. < N:
    result[i] = f(v[i])

proc map*[N: static[int]](v: Vector64[N], f: proc(x: float64): float64): Vector64[N] =
  new result
  for i in 0 .. < N:
    result[i] = f(v[i])

template atPrivate(M, N, m, i, j: expr, A: typedesc): auto =
  if m.order == colMajor:
    let data = cast[ref array[N, array[M, A]]](m.data)
    data[j][i]
  else:
    let data = cast[ref array[M, array[N, A]]](m.data)
    data[i][j]

template atPrivateD(m, i, j: expr): auto =
  if m.order == colMajor: m.data[j * m.M + i]
  else: m.data[i * m.N + j]

proc at*[M, N: static[int]](m: Matrix64[M, N], i, j: int): float64 {. inline .} = atPrivate(M, N, m, i, j, float64)

proc at*(m: DMatrix64, i, j: int): float64 {. inline .} = atPrivateD(m, i, j)

template `[]`*(m: Matrix64 or DMatrix64, i, j: int): float64 = m.at(i, j)

proc at*[M, N: static[int]](m: Matrix32[M, N], i, j: int): float32 {. inline .} = atPrivate(M, N, m, i, j, float32)

proc at*(m: DMatrix32, i, j: int): float32 {. inline .} = atPrivateD(m, i, j)

template `[]`*(m: Matrix32 or DMatrix32, i, j: int): float32 = m.at(i, j)

template putPrivate(M, N, m, i, j, val: expr, A: typedesc) =
  if m.order == colMajor:
    var data = cast[ref array[N, array[M, A]]](m.data)
    data[j][i] = val
  else:
    var data = cast[ref array[M, array[N, A]]](m.data)
    data[i][j] = val

template putPrivateD(m, i, j, val: expr) =
  if m.order == colMajor:
    m.data[j * m.M + i] = val
  else:
    m.data[i * m.N + j] = val

proc put*[M, N: static[int]](m: var Matrix64[M, N], i, j: int, val: float64) {. inline .} =
  putPrivate(M, N, m, i, j, val, float64)

proc put*(m: var DMatrix64, i, j: int, val: float64) {. inline .} =
  putPrivateD(m, i, j, val)

proc `[]=`*(m: var Matrix64, i, j: int, val: float64) {. inline .} = m.put(i, j, val)

proc `[]=`*(m: var DMatrix64, i, j: int, val: float64) {. inline .} = m.put(i, j, val)

proc put*[M, N: static[int]](m: var Matrix32[M, N], i, j: int, val: float32) {. inline .} =
  putPrivate(M, N, m, i, j, val, float32)

proc put*(m: var DMatrix32, i, j: int, val: float32) {. inline .} =
  putPrivateD(m, i, j, val)

proc `[]=`*(m: var Matrix32, i, j: int, val: float32) {. inline .} = m.put(i, j, val)

proc `[]=`*(m: var DMatrix32, i, j: int, val: float32) {. inline .} = m.put(i, j, val)

proc column*[M, N: static[int]](m: Matrix32[M, N], j: int): Vector32[M] {. inline .} =
  new result
  for i in 0 .. < M:
    result[i] = m.at(i, j)

proc row*[M, N: static[int]](m: Matrix32[M, N], i: int): Vector32[N] {. inline .} =
  new result
  for j in 0 .. < N:
    result[j] = m.at(i, j)

proc column*(m: DMatrix32, j: int): DVector32 {. inline .} =
  result = newSeq[float32](m.M)
  for i in 0 .. < m.M:
    result[i] = m.at(i, j)

proc row*(m: DMatrix32, i: int): DVector32 {. inline .} =
  result = newSeq[float32](m.N)
  for j in 0 .. < m.N:
    result[j] = m.at(i, j)

# proc columnUnsafe*[M, N: static[int]](m: Matrix32[M, N], j: int): Vector32[M] {. inline .} =
#   if m.order == colMajor:
#     return cast[ref array[M, float32]](addr(m.data[j * M]))
#   else:
#     raise newException(AccessViolationError, "Cannot access columns in an unsafe way")
#
# proc rowUnsafe*[M, N: static[int]](m: Matrix32[M, N], i: int): Vector32[N] {. inline .} =
#   if m.order == rowMajor:
#     return cast[ref array[N, float32]](addr(m.data[i * N]))
#   else:
#     raise newException(AccessViolationError, "Cannot access rows in an unsafe way")

proc column*[M, N: static[int]](m: Matrix64[M, N], j: int): Vector64[M] {. inline .} =
  new result
  for i in 0 .. < M:
    result[i] = m.at(i, j)

proc row*[M, N: static[int]](m: Matrix64[M, N], i: int): Vector64[N] {. inline .} =
  new result
  for j in 0 .. < N:
    result[j] = m.at(i, j)

proc column*(m: DMatrix64, j: int): DVector64 {. inline .} =
  result = newSeq[float64](m.M)
  for i in 0 .. < m.M:
    result[i] = m.at(i, j)

proc row*(m: DMatrix64, i: int): DVector64 {. inline .} =
  result = newSeq[float64](m.N)
  for j in 0 .. < m.N:
    result[j] = m.at(i, j)

# proc columnUnsafe*[M, N: static[int]](m: Matrix64[M, N], j: int): Vector64[M] {. inline .} =
#   if m.order == colMajor:
#     return cast[ref array[M, float64]](addr(m.data[j * M]))
#   else:
#     raise newException(AccessViolationError, "Cannot access columns in an unsafe way")
#
# proc rowUnsafe*[M, N: static[int]](m: Matrix64[M, N], i: int): Vector64[N] {. inline .} =
#   if m.order == rowMajor:
#     return cast[ref array[N, float64]](addr(m.data[i * N]))
#   else:
#     raise newException(AccessViolationError, "Cannot access rows in an unsafe way")

proc dim*[M, N: static[int]](m: Matrix32[M, N] or Matrix64[M, N]): tuple[rows, columns: int] = (M, N)

proc dim*(m: DMatrix32 or DMatrix64): tuple[rows, columns: int] = (m.M, m.N)

proc clone*[M, N: static[int]](m: Matrix32[M, N]): Matrix32[M, N] =
  result.order = m.order
  new result.data
  copyMem(result.fp, m.fp, M * N * sizeof(float32))

proc clone*[M, N: static[int]](m: Matrix64[M, N]): Matrix64[M, N] =
  result.order = m.order
  new result.data
  copyMem(result.fp, m.fp, M * N * sizeof(float64))

proc clone*(m: DMatrix32): DMatrix32 =
  result.order = m.order
  result.M = m.M
  result.N = m.N
  result.data = m.data

proc clone*(m: DMatrix64): DMatrix64 =
  result.order = m.order
  result.M = m.M
  result.N = m.N
  result.data = m.data

proc map*[M, N: static[int]](m: Matrix32[M, N], f: proc(x: float32): float32): Matrix32[M, N] =
  result.order = m.order
  new result.data
  for i in 0 .. < (M * N):
    result.data[i] = f(m.data[i])

proc map*[M, N: static[int]](m: Matrix64[M, N], f: proc(x: float64): float64): Matrix64[M, N] =
  result.order = m.order
  new result.data
  for i in 0 .. < (M * N):
    result.data[i] = f(m.data[i])

proc map*(m: DMatrix32, f: proc(x: float32): float32): DMatrix32 =
  result.order = m.order
  result.M = m.M
  result.N = m.N
  result.data = newSeq[float32](m.M * m.N)
  for i in 0 .. < (m.M * m.N):
    result.data[i] = f(m.data[i])

proc map*(m: DMatrix64, f: proc(x: float64): float64): DMatrix64 =
  result.order = m.order
  result.M = m.M
  result.N = m.N
  result.data = newSeq[float64](m.M * m.N)
  for i in 0 .. < (m.M * m.N):
    result.data[i] = f(m.data[i])