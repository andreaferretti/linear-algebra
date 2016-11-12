# Copyright 2016 UniCredit S.p.A.
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

iterator items*[N: static[int]](v: Vector32[N] or Vector64[N]): auto {. inline .} =
  for i in 0 .. < N:
    yield v[i]

iterator pairs*[N: static[int]](v: Vector32[N] or Vector64[N]): auto {. inline .} =
  for i in 0 .. < N:
    yield (i, v[i])

iterator columns*[M, N: static[int]](m: Matrix32[M, N] or Matrix64[M, N]): auto {. inline .} =
  for i in 0 .. < N:
    yield m.column(i)

iterator columns*(m: DMatrix32 or DMatrix64): auto {. inline .} =
  for i in 0 .. < m.N:
    yield m.column(i)

# iterator columnsUnsafe*[M, N: static[int]](m: Matrix32[M, N]): Vector32[M] {. inline .} =
#   if m.order == colMajor:
#     for i in 0 .. < N:
#       yield cast[ref array[M, float32]](addr(m.data[i * M]))
#   else:
#     raise newException(AccessViolationError, "Cannot access columns in an unsafe way")
#
# iterator columnsUnsafe*[M, N: static[int]](m: Matrix64[M, N]): Vector64[M] {. inline .} =
#   if m.order == colMajor:
#     for i in 0 .. < N:
#       yield cast[ref array[M, float64]](addr(m.data[i * M]))
#   else:
#     raise newException(AccessViolationError, "Cannot access columns in an unsafe way")

iterator rows*[M, N: static[int]](m: Matrix32[M, N] or Matrix64[M, N]): auto {. inline .} =
  for i in 0 .. < M:
    yield m.row(i)

iterator rows*(m: DMatrix32 or DMatrix64): auto {. inline .} =
  for i in 0 .. < m.M:
    yield m.row(i)

# iterator rowsUnsafe*[M, N: static[int]](m: Matrix32[M, N]): Vector32[N] {. inline .} =
#   if m.order == rowMajor:
#     for i in 0 .. < M:
#       yield cast[ref array[N, float32]](addr(m.data[i * N]))
#   else:
#     raise newException(AccessViolationError, "Cannot access rows in an unsafe way")
#
# iterator rowsUnsafe*[M, N: static[int]](m: Matrix64[M, N]): Vector64[N] {. inline .} =
#   if m.order == rowMajor:
#     for i in 0 .. < M:
#       yield cast[ref array[N, float64]](addr(m.data[i * N]))
#   else:
#     raise newException(AccessViolationError, "Cannot access rows in an unsafe way")

iterator items*[M, N: static[int]](m: Matrix32[M, N] or Matrix64[M, N]): auto {. inline .} =
  for i in 0 .. < M:
    for j in 0 .. < N:
      yield m[i, j]

iterator items*(m: DMatrix32 or DMatrix64): auto {. inline .} =
  for i in 0 .. < m.M:
    for j in 0 .. < m.N:
      yield m[i, j]

iterator pairs*[M, N: static[int]](m: Matrix32[M, N] or Matrix64[M, N]): auto {. inline .} =
  for i in 0 .. < M:
    for j in 0 .. < N:
      yield ((i, j), m[i, j])

iterator pairs*(m: DMatrix32 or DMatrix64): auto {. inline .} =
  for i in 0 .. < m.M:
    for j in 0 .. < m.N:
      yield ((i, j), m[i, j])