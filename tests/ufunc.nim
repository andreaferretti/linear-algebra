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


suite "universal functions":
  test "universal logarithm on static vectors":
    let u = vector([1.0, 2.0, 4.0, 8.0])
    check log2(u) == vector([0.0, 1.0, 2.0, 3.0])
  test "universal sqrt on dynamic vectors":
    let u = @[1.0, 4.0, 9.0, 16.0]
    check sqrt(u) == @[1.0, 2.0, 3.0, 4.0]
  test "universal cosine on static matrices":
    let m = Matrix(2, 2, @[@[1.0, 2.0], @[4.0, 8.0]])
    check cos(m) == Matrix(2, 2, @[@[cos(1.0), cos(2.0)], @[cos(4.0), cos(8.0)]])
  test "universal sine on dynamic matrices":
    let m = matrix(@[@[1.0, 2.0], @[4.0, 8.0]])
    check sin(m) == matrix(@[@[sin(1.0), sin(2.0)], @[sin(4.0), sin(8.0)]])
  test "defining a new universal function":
    proc plusFive(x: float64): float64 = x + 5
    makeUniversalLocal(plusFive)
    let v = @[1.0, 4.0, 9.0, 16.0]
    check plusFive(v) == @[6.0, 9.0, 14.0, 21.0]


suite "32-bit universal functions":
  test "universal logarithm on static vectors":
    let u = vector([1'f32, 2'f32, 4'f32, 8'f32], float32)
    check log2(u) == vector([0'f32, 1'f32, 2'f32, 3'f32], float32)
  test "universal sqrt on dynamic vectors":
    let u = @[1'f32, 4'f32, 9'f32, 16'f32]
    check sqrt(u) == @[1'f32, 2'f32, 3'f32, 4'f32]
  test "universal cosine on static matrices":
    let m = Matrix(2, 2, @[@[1'f32, 2'f32], @[4'f32, 8'f32]])
    check cos(m) == Matrix(2, 2, @[@[cos(1'f32), cos(2'f32)], @[cos(4'f32), cos(8'f32)]])
  test "universal sine on dynamic matrices":
    let m = matrix(@[@[1'f32, 2'f32], @[4'f32, 8'f32]])
    check sin(m) == matrix(@[@[sin(1'f32), sin(2'f32)], @[sin(4'f32), sin(8'f32)]])
  test "defining a new universal function":
    proc plusFive(x: float64): float64 = x + 5
    makeUniversalLocal(plusFive)
    let v = @[1'f32, 4'f32, 9'f32, 16'f32]
    check plusFive(v) == @[6'f32, 9'f32, 14'f32, 21'f32]