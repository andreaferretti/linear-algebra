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

iterator items*[N: static[int]](v: Vector32[N] or Vector64[N]): auto {. inline .} =
  for i in 0 .. < N:
    yield v[i]

iterator pairs*[N: static[int]](v: Vector32[N] or Vector64[N]): auto {. inline .} =
  for i in 0 .. < N:
    yield (i, v[i])

iterator columns*[M, N: static[int]](m: Matrix64[M, N]): Vector64[M] {. inline .} =
  for i in 0 .. < N:
    yield m.column(i)

iterator rows*[M, N: static[int]](m: Matrix64[M, N]): Vector64[N] {. inline .} =
  for i in 0 .. < M:
    yield m.row(i)

iterator items*[M, N: static[int]](m: Matrix64[M, N]): float64 {. inline .} =
  for i in 0 .. < M:
    for j in 0 .. < N:
      yield m[i, j]

iterator pairs*[M, N: static[int]](m: Matrix64[M, N]): tuple[indices: tuple[i, j: int], val: float64] {. inline .} =
  for i in 0 .. < M:
    for j in 0 .. < N:
      yield ((i, j), m[i, j])