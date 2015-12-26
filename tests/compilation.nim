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


suite "compilation errors":
  test "vector dimension should agree in a sum":
    let
      u = vector([1.0, 2.0, 3.0, 4.0, 5.0])
      v = vector([1.0, 2.0, 3.0, 4.0])
    when compiles(u + v): fail()
    when compiles(u - v): fail()
  test "vector dimension should agree in a mutating sum":
    var u = vector([1.0, 2.0, 3.0, 4.0, 5.0])
    let v = vector([1.0, 2.0, 3.0, 4.0])
    when compiles(u += v): fail()
    when compiles(u -= v): fail()
  test "in place sum should not work for immutable vectors":
    let
      u = vector([1.0, 2.0, 3.0, 4.0])
      v = vector([1.0, 2.0, 3.0, 4.0])
    when compiles(u += v): fail()
    when compiles(u -= v): fail()
  test "in place sum should not work for immutable dynamic vectors":
    let
      u = @[1.0, 2.0, 3.0, 4.0]
      v = @[1.0, 2.0, 3.0, 4.0]
    when compiles(u += v): fail()
    when compiles(u -= v): fail()
  test "making an array into a matrix should not work for wrong dimensions":
    let u = vector([1.0, 2.0, 3.0, 4.0])
    when compiles(u.asMatrix(3, 5)): fail()
  test "reshaping a matrix should not work for wrong dimensions":
    let m = randomMatrix(2, 3)
    when compiles(m.reshape(3, 5)): fail()
  test "matrix/vector multiplication should not work for wrong dimensions":
    let
      m = randomMatrix(6, 7)
      v = randomVector(6)
    when compiles(m * v): fail()
  test "matrix dimensions should agree in a sum":
    let
      m = randomMatrix(3, 6)
      n = randomMatrix(4, 5)
    when compiles(m + n): fail()
    when compiles(m - n): fail()
  test "matrix dimensions should agree in an in place sum":
    var m = randomMatrix(3, 6)
    let n = randomMatrix(4, 5)
    when compiles(m += n): fail()
    when compiles(m -= n): fail()
  test "in place sum should not work for immutable matrices":
    let
      m = randomMatrix(3, 6)
      n = randomMatrix(3, 6)
    when compiles(m += n): fail()
    when compiles(m -= n): fail()
  test "matrix multiplication should not work for wrong dimensions":
    let
      m = randomMatrix(6, 7)
      n = randomMatrix(8, 18)
    when compiles(m * n): fail()