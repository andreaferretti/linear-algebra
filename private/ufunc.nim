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


template makeUniversal*(fname: expr) =
  proc fname*(x: float32): float32 = fname(x.float64).float32

  proc fname*[N: static[int]](v: Vector32[N]): Vector32[N] =
    new result
    for i in 0 .. < N:
      result[i] = fname(v[i])

  proc fname*(v: DVector32): DVector32 =
    result = newSeq[float32](v.len)
    for i in 0 .. < (v.len):
      result[i] = fname(v[i])

  proc fname*[M, N: static[int]](m: Matrix32[M, N]): Matrix32[M, N] =
    new result.data
    result.order = m.order
    for i in 0 .. < (M * N):
      result.data[i] = fname(m.data[i])

  proc fname*(m: DMatrix32): DMatrix32 =
    result.data = newSeq[float32](m.len)
    result.order = m.order
    result.M = m.M
    result.N = m.N
    for i in 0 .. < (m.len):
      result.data[i] = fname(m.data[i])

  proc fname*[N: static[int]](v: Vector64[N]): Vector64[N] =
    new result
    for i in 0 .. < N:
      result[i] = fname(v[i])

  proc fname*(v: DVector64): DVector64 =
    result = newSeq[float64](v.len)
    for i in 0 .. < (v.len):
      result[i] = fname(v[i])

  proc fname*[M, N: static[int]](m: Matrix64[M, N]): Matrix64[M, N] =
    new result.data
    result.order = m.order
    for i in 0 .. < (M * N):
      result.data[i] = fname(m.data[i])

  proc fname*(m: DMatrix64): DMatrix64 =
    result.data = newSeq[float64](m.len)
    result.order = m.order
    result.M = m.M
    result.N = m.N
    for i in 0 .. < (m.len):
      result.data[i] = fname(m.data[i])

  export fname


template makeUniversalLocal*(fname: expr) =
  proc fname(x: float32): float32 = fname(x.float64).float32

  proc fname[N: static[int]](v: Vector32[N]): Vector32[N] =
    new result
    for i in 0 .. < N:
      result[i] = fname(v[i])

  proc fname(v: DVector32): DVector32 =
    result = newSeq[float32](v.len)
    for i in 0 .. < (v.len):
      result[i] = fname(v[i])

  proc fname[M, N: static[int]](m: Matrix32[M, N]): Matrix32[M, N] =
    new result.data
    result.order = m.order
    for i in 0 .. < (M * N):
      result.data[i] = fname(m.data[i])

  proc fname(m: DMatrix32): DMatrix32 =
    result.data = newSeq[float32](m.len)
    result.order = m.order
    result.M = m.M
    result.N = m.N
    for i in 0 .. < (m.len):
      result.data[i] = fname(m.data[i])

  proc fname[N: static[int]](v: Vector64[N]): Vector64[N] =
    new result
    for i in 0 .. < N:
      result[i] = fname(v[i])

  proc fname(v: DVector64): DVector64 =
    result = newSeq[float64](v.len)
    for i in 0 .. < (v.len):
      result[i] = fname(v[i])

  proc fname[M, N: static[int]](m: Matrix64[M, N]): Matrix64[M, N] =
    new result.data
    result.order = m.order
    for i in 0 .. < (M * N):
      result.data[i] = fname(m.data[i])

  proc fname(m: DMatrix64): DMatrix64 =
    result.data = newSeq[float64](m.len)
    result.order = m.order
    result.M = m.M
    result.N = m.N
    for i in 0 .. < (m.len):
      result.data[i] = fname(m.data[i])

makeUniversal(sqrt)
makeUniversal(cbrt)
makeUniversal(log10)
makeUniversal(log2)
makeUniversal(log)
makeUniversal(exp)
makeUniversal(arccos)
makeUniversal(arcsin)
makeUniversal(arctan)
makeUniversal(cos)
makeUniversal(cosh)
makeUniversal(sin)
makeUniversal(sinh)
makeUniversal(tan)
makeUniversal(tanh)
makeUniversal(erf)
makeUniversal(erfc)
makeUniversal(lgamma)
makeUniversal(tgamma)
makeUniversal(trunc)
makeUniversal(floor)
makeUniversal(ceil)
makeUniversal(degToRad)
makeUniversal(radToDeg)