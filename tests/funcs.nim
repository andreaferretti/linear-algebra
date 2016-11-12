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

import unittest, linalg

suite "lapack funcs":
  test "matrix-matrix solver":
    let a = matrix(@[
      @[3.0, 1.0],
      @[1.0, -2.0]
    ], 2, 2)
    let b = matrix(@[@[1.0],
                     @[0.0]],
                     2, 1)
    let x = solve(a, b)
    let expected = matrix(@[@[2.0 / 7.0],
                            @[1.0 / 7.0]],
                            2, 1)
    check expected =~ x

  test "singular matrix error":
    let a = matrix(@[
      @[3.0, 3.0],
      @[1.0, 1.0]
    ], 2, 2)
    let b = matrix(@[@[1.0],
                     @[0.0]],
                     2, 1)
    expect FloatingPointError:
      discard solve(a, b)

  test "matrix-matrix solver 32":
    let a = matrix(@[
      @[3'f32, 1'f32],
      @[1'f32, -2'f32]
    ], 2, 2)
    let b = matrix(@[@[1'f32],
                     @[0'f32]],
                     2, 1)
    let x = solve(a, b)
    let expected = matrix(@[@[2'f32 / 7'f32],
                            @[1'f32 / 7'f32]],
                            2, 1)
    check expected =~ x

  test "matrix-vector solver":
    let a = matrix(@[
      @[3.0, 1.0],
      @[1.0, -2.0]
    ], 2, 2)
    let b = vector([1.0, 0.0])
    let x = solve(a, b)
    let expected = vector([2.0/7.0, 1.0/7.0])
    check expected =~ x

  test "matrix-vector singular matrix error":
    let a = matrix(@[
      @[0.0, 0.0],
      @[0.0, 0.0]
    ], 2, 2)
    let b = vector([1.0, 0.0])
    expect FloatingPointError:
      discard solve(a, b)

  test "matrix-vector solver 32":
    let a = matrix(@[
      @[3.0'f32, 1.0'f32],
      @[1.0'f32, -2.0'f32]
    ], 2, 2)
    let b = vector([1.0'f32, 0.0'f32])
    let x = solve(a, b)
    let expected = vector([2.0'f32/7.0'f32, 1.0'f32/7.0'f32])
    check expected =~ x

  test "matrix inverse":
    let a = matrix(@[
      @[4.0, 3.0],
      @[3.0, 2.0]
    ], 2, 2)
    let expected = matrix(@[
      @[-2.0, 3.0],
      @[3.0, -4.0]
    ], 2, 2)
    let ainv = inv(a)
    check expected =~ ainv

  test "matrix inverse 32":
    let a = matrix(@[
      @[4.0'f32, 3.0'f32],
      @[3.0'f32, 2.0'f32]
    ], 2, 2)
    let expected = matrix(@[
      @[-2.0'f32, 3.0'f32],
      @[3.0'f32, -4.0'f32]
    ], 2, 2)
    let ainv = inv(a)
    check expected =~ ainv
