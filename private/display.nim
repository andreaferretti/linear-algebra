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

proc `$`*[N: static[int]](v: Vector32[N] or Vector64[N]): string =
  result = "[ "
  for i in 0 .. < N - 1:
    result &= $(v[i]) & "\n  "
  result &= $(v[N - 1]) & " ]"

proc toStringHorizontal[N: static[int]](v: Vector32[N] or Vector64[N]): string =
  result = "[ "
  for i in 0 .. < N - 1:
    result &= $(v[i]) & "\t"
  result &= $(v[N - 1]) & " ]"

proc `$`*[M, N: static[int]](m: Matrix64[M, N]): string =
  result = "[ "
  for i in 0 .. < M - 1:
    result &= toStringHorizontal(m.row(i)) & "\n  "
  result &= toStringHorizontal(m.row(M - 1)) & " ]"

proc `$`*[M, N: static[int]](m: Matrix32[M, N]): string =
  result = "[ "
  for i in 0 .. < M - 1:
    result &= toStringHorizontal(m.row(i)) & "\n  "
  result &= toStringHorizontal(m.row(M - 1)) & " ]"