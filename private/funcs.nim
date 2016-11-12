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

#####TODO: Add template and corresponding procedures for dynamic matrices/vectors

template MatrixSolve(M, N, a, b: expr, A: typedesc): auto =
  var ipvt: array[M, int32]
  var ipvt_ptr = cast[ptr int32](addr(ipvt))
  var info: cint
  var m: cint = M
  var n: cint = N
  var mptr = addr(m)
  var nptr = addr(n)
  if a.order == colMajor and b.order == colMajor:
    gesv(mptr, nptr, a.fp, mptr, ipvt_ptr, b.fp, mptr, addr(info))
  elif a.order == rowMajor and b.order == rowMajor:
    gesv(mptr, nptr, a.t.fp, mptr, ipvt_ptr, b.t.fp, mptr, addr(info))
  elif a.order == colMajor and b.order == rowMajor:
    gesv(mptr, nptr, a.fp, mptr, ipvt_ptr, b.t.fp, mptr, addr(info))
  else:
    gesv(mptr, nptr, a.t.fp, mptr, ipvt_ptr, b.fp, mptr, addr(info))
  if info > 0:
    raise newException( FloatingPointError, "Left hand matrix is singular or factorization failed")

template MatrixVectorSolve(M, a, b: expr, A: typedesc): auto =
  var ipvt: array[M, int32]
  var ipvt_ptr = cast[ptr int32](addr(ipvt))
  var info: cint
  var m: cint = M
  var n: cint = 1
  var mptr = addr(m)
  var nptr = addr(n)
  if a.order == colMajor:
    gesv(mptr, nptr, a.fp, mptr, ipvt_ptr, b.fp, mptr, addr(info))
  else:
    gesv(mptr, nptr, a.t.fp, mptr, ipvt_ptr, b.fp, mptr, addr(info))
  if info > 0:
    raise newException( FloatingPointError, "Left hand matrix is singular or factorization failed")

proc solve*[M, N: static[int]](a: Matrix64[M, M], b: Matrix64[M, N]): Matrix64[M, N] {.inline.} =
  new result.data
  result.order = b.order
  var acopy = a.clone
  copy(M*N, b.fp, 1, result.fp, 1)
  MatrixSolve(M, N, acopy, result, float64)

proc solve*[M, N: static[int]](a: Matrix32[M, M], b: Matrix32[M, N]): Matrix32[M, N] {.inline.} =
  new result.data
  result.order = b.order
  var acopy = a.clone
  copy(M*N, b.fp, 1, result.fp, 1)
  MatrixSolve(M, N, acopy, result, float32)

proc solve*[M: static[int]](a: Matrix64[M, M], b: Vector64[M]): Vector64[M] {.inline.} =
  new result
  var acopy = a.clone
  copy(M, b.fp, 1, result.fp, 1)
  MatrixVectorSolve(M, acopy, result, float64)

proc solve*[M: static[int]](a: Matrix32[M, M], b: Vector32[M]): Vector32[M] {.inline.} =
  new result
  var acopy = a.clone
  copy(M, b.fp, 1, result.fp, 1)
  MatrixVectorSolve(M, acopy, result, float32)

proc inv*[M: static[int]](a: Matrix64[M, M]): Matrix64[M, M] {.inline.} =
  result = eye(M.int)
  var acopy = a.clone
  MatrixSolve(M, M, acopy, result, float64)

proc inv*[M: static[int]](a: Matrix32[M, M]): Matrix32[M, M] {.inline.} =
  result = eye(M.int, float32)
  var acopy = a.clone
  MatrixSolve(M, M, acopy, result, float64)
