# Copyright 'f3215 UniCredit S.p.A.
#
# Licensed under the Apache License, Version 2'f32 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2'f32
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
      u = vector([1'f32, 2'f32, 3'f32, 4'f32, 5'f32], float32)
      v = vector([1'f32, 2'f32, 3'f32, 4'f32], float32)
      u1 = u.gpu()
      v1 = v.gpu()
    when compiles(u1 + v1): fail()
    when compiles(u1 - v1): fail()
  test "vector dimension should agree in a mutating sum":
    let
      u = vector([1'f32, 2'f32, 3'f32, 4'f32, 5'f32], float32)
      v = vector([1'f32, 2'f32, 3'f32, 4'f32], float32)
      v1 = v.gpu()
    var u1 = u.gpu()
    when compiles(u1 += v1): fail()
    when compiles(u1 -= v1): fail()
  test "in place sum should not work for immutable vectors":
    let
      u = vector([1'f32, 2'f32, 3'f32, 4'f32], float32)
      v = vector([1'f32, 2'f32, 3'f32, 4'f32], float32)
      u1 = u.gpu()
      v1 = v.gpu()
    when compiles(u1 += v1): fail()
    when compiles(u1 -= v1): fail()
  test "vector dimension should agree in a dot product":
    let
      u = vector([1'f32, 2'f32, 3'f32, 4'f32, 5'f32], float32).gpu()
      v = vector([1'f32, 2'f32, 3'f32, 4'f32], float32).gpu()
    when compiles(u * v): fail()
  test "matrix dimensions should agree in a sum":
    let
      m = randomMatrix(3, 6, max = 1'f32).gpu()
      n = randomMatrix(4, 5, max = 1'f32).gpu()
    when compiles(m + n): fail()
    when compiles(m - n): fail()
  test "matrix dimensions should agree in an in place sum":
    var m = randomMatrix(3, 6, max = 1'f32).gpu()
    let n = randomMatrix(4, 5, max = 1'f32).gpu()
    when compiles(m += n): fail()
    when compiles(m -= n): fail()
  test "in place sum should not work for immutable matrices":
    let
      m = randomMatrix(3, 6, max = 1'f32).gpu()
      n = randomMatrix(3, 6, max = 1'f32).gpu()
    when compiles(m += n): fail()
    when compiles(m -= n): fail()
  test "matrix multiplication should not work for wrong dimensions":
    let
      m = randomMatrix(6, 7, max = 1'f32).gpu()
      n = randomMatrix(8, 18, max = 1'f32).gpu()
    when compiles(m * n): fail()