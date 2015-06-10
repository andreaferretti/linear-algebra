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

import unittest, linalg


suite "vector operations":
  test "scalar multiplication":
    let v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
    check((v * 2.0) == vector([2.0, 6.0, 4.0, 16.0, -4.0]))
    check((-1.0 * v) == vector([-1.0, -3.0, -2.0, -8.0, 2.0]))
  test "mutating scalar multiplication":
    var v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
    v *= 2.0
    check v == vector([2.0, 6.0, 4.0, 16.0, -4.0])
  test "vector sum":
    let
      v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
      w = vector([2.0, -1.0, 2.0, 0.0, 4.0])
    check((v + w) == vector([3.0, 2.0, 4.0, 8.0, 2.0]))
  test "mutating vector sum":
    var v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
    let w = vector([2.0, -1.0, 2.0, 0.0, 4.0])
    v += w
    check v == vector([3.0, 2.0, 4.0, 8.0, 2.0])
  test "vector difference":
    let
      v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
      w = vector([2.0, -1.0, 2.0, 0.0, 4.0])
    check((v - w) == vector([-1.0, 4.0, 0.0, 8.0, -6.0]))
  test "mutating vector difference":
    var v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
    let w = vector([2.0, -1.0, 2.0, 0.0, 4.0])
    v -= w
    check v == vector([-1.0, 4.0, 0.0, 8.0, -6.0])
  test "dot product":
    let
      v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
      w = vector([2.0, -1.0, 2.0, 0.0, 4.0])
    check(v * w == -5.0)
  test "ℓ² norm":
    let v = vector([1.0, 1.0, 2.0, 3.0, -7.0])
    check l_2(v) == 8.0
  test "ℓ¹ norm":
    let v = vector([1.0, 1.0, 2.0, 3.0, -7.0])
    check l_1(v) == 14.0
  test "max and min of vectors":
    let v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
    check max(v) == 8.0
    check maxIndex(v) == (3, 8.0)
    check min(v) == -2.0
    check minIndex(v) == (4, -2.0)

suite "matrix/vector operations":
  test "multiplication of matrix and vector":
    let
      m = dmatrix(3, 4, @[
        @[1.0, 0.0, 2.0, -1.0],
        @[-1.0, 1.0, 3.0, 1.0],
        @[3.0, 2.0, 2.0, 4.0]
      ])
      v = vector([1.0, 3.0, 2.0, -2.0])
    check((m * v) == vector([7.0, 6.0, 5.0]))

suite "trivial operations":
  test "reshape of matrices":
    let
      m1 = dmatrix(3, 4, @[
        @[1.0, 0.0, 2.0, -1.0],
        @[-1.0, 1.0, 3.0, 1.0],
        @[3.0, 2.0, 2.0, 4.0]
      ])
      m2 = dmatrix(4, 3, @[
        @[1.0, 1.0, 2.0],
        @[-1.0, 2.0, -1.0],
        @[3.0, 2.0, 1.0],
        @[0.0, 3.0, 4.0]
      ])
    check m1.reshape(4, 3) == m2