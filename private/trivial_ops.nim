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

proc t*[M, N: static[int]](a: Matrix64[M, N]): Matrix64[N, M] =
  result.order = if a.order == rowMajor: colMajor else: rowMajor
  result.data = a.data

proc reshape*[M, N: static[int]](m: Matrix64[M, N], A, B: static[int]): Matrix64[A, B] =
  static: doAssert(M * N == A * B, "The dimensions do not match: M = " & $(M) & ", N = " & $(N) & ", A = " & $(A) & ", B = " & $(B))
  result.order = m.order
  result.data = m.data