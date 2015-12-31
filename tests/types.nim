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


suite "types of created vectors":
  test "constant vectors should be of expected types":
    let
      N = 5
      u = constantVector(5, 1.0)
      v = constantVector(N, 1.0)
    when not (u is Vector64[5]): fail()
    when not (v is DVector64): fail()
  test "zero vectors should be of expected types":
    let
      N = 5
      u = zeros(5)
      v = zeros(N)
    when not (u is Vector64[5]): fail()
    when not (v is DVector64): fail()
  test "one vectors should be of expected types":
    let
      N = 5
      u = ones(5)
      v = ones(N)
    when not (u is Vector64[5]): fail()
    when not (v is DVector64): fail()
  test "literal vectors should be of expected types":
    let
      u = vector([1.0, 2.0, 3.0, 4.0, 5.0])
      v = @[1.0, 2.0, 3.0, 4.0, 5.0]
    when not (u is Vector64[5]): fail()
    when not (v is DVector64): fail()
  test "random vectors should be of expected types":
    let
      N = 5
      u = randomVector(5)
      v = randomVector(N)
    when not (u is Vector64[5]): fail()
    when not (v is DVector64): fail()
  test "proc vectors should be of expected types":
    let
      N = 5
      u = makeVector(5, proc(i: int): float64 = i.float64)
      v = makeVector(N, proc(i: int): float64 = i.float64)
    when not (u is Vector64[5]): fail()
    when not (v is DVector64): fail()

suite "types of created vectors (32-bit)":
  test "constant vectors should be of expected types":
    let
      N = 5
      u = constantVector(5, 1'f32)
      v = constantVector(N, 1'f32)
    when not (u is Vector32[5]): fail()
    when not (v is DVector32): fail()
  test "zero vectors should be of expected types":
    let
      N = 5
      u = zeros(5, float32)
      v = zeros(N, float32)
    when not (u is Vector32[5]): fail()
    when not (v is DVector32): fail()
  test "one vectors should be of expected types":
    let
      N = 5
      u = ones(5, float32)
      v = ones(N, float32)
    when not (u is Vector32[5]): fail()
    when not (v is DVector32): fail()
  test "literal vectors should be of expected types":
    let
      u = vector([1'f32, 2'f32, 3'f32, 4'f32, 5'f32], float32)
      v = @[1'f32, 2'f32, 3'f32, 4'f32, 5'f32]
    when not (u is Vector32[5]): fail()
    when not (v is DVector32): fail()
  test "random vectors should be of expected types":
    let
      N = 5
      u = randomVector(5, max = 1'f32)
      v = randomVector(N, max = 1'f32)
    when not (u is Vector32[5]): fail()
    when not (v is DVector32): fail()
  test "proc vectors should be of expected types":
    let
      N = 5
      u = makeVector(5, proc(i: int): float32 = i.float32)
      v = makeVector(N, proc(i: int): float32 = i.float32)
    when not (u is Vector32[5]): fail()
    when not (v is DVector32): fail()

suite "types of created matrices":
  test "constant matrices should be of expected types":
    let
      M = 4
      N = 5
      u = constantMatrix(4, 5, 1.5)
      v = constantMatrix(M, N, 1.5)
    when not (u is Matrix64[4, 5]): fail()
    when not (v is DMatrix64): fail()
  test "zero matrices should be of expected types":
    let
      M = 4
      N = 5
      u = zeros(4, 5)
      v = zeros(M, N)
    when not (u is Matrix64[4, 5]): fail()
    when not (v is DMatrix64): fail()
  test "one matrices should be of expected types":
    let
      M = 4
      N = 5
      u = ones(4, 5)
      v = ones(M, N)
    when not (u is Matrix64[4, 5]): fail()
    when not (v is DMatrix64): fail()
  test "literal matrices should be of expected types":
    let
      u = Matrix(2, 3, @[@[1.0, 2.0, 3.0], @[4.0, 5.0, 6.0]])
      v = matrix(@[@[1.0, 2.0, 3.0], @[4.0, 5.0, 6.0]])
    when not (u is Matrix64[2, 3]): fail()
    when not (v is DMatrix64): fail()
  test "identity matrices should be of expected types":
    let
      M = 5
      u = eye(5)
      v = eye(M)
    when not (u is Matrix64[5, 5]): fail()
    when not (v is DMatrix64): fail()
  test "random matrices should be of expected types":
    let
      M = 4
      N = 5
      u = randomMatrix(4, 5)
      v = randomMatrix(M, N)
    when not (u is Matrix64[4, 5]): fail()
    when not (v is DMatrix64): fail()
  test "proc matrices should be of expected types":
    let
      M = 4
      N = 5
      u = makeMatrix(4, 5, proc(i, j: int): float64 = (i + j).float64)
      v = makeMatrix(M, N, proc(i, j: int): float64 = (i + j).float64)
    when not (u is Matrix64[4, 5]): fail()
    when not (v is DMatrix64): fail()

suite "types of created 32-bit matrices":
  test "constant matrices should be of expected types":
    let
      M = 4
      N = 5
      u = constantMatrix(4, 5, 1.5'f32)
      v = constantMatrix(M, N, 1.5'f32)
    when not (u is Matrix32[4, 5]): fail()
    when not (v is DMatrix32): fail()
  test "zero matrices should be of expected types":
    let
      M = 4
      N = 5
      u = zeros(4, 5, float32)
      v = zeros(M, N, float32)
    when not (u is Matrix32[4, 5]): fail()
    when not (v is DMatrix32): fail()
  test "one matrices should be of expected types":
    let
      M = 4
      N = 5
      u = ones(4, 5, float32)
      v = ones(M, N, float32)
    when not (u is Matrix32[4, 5]): fail()
    when not (v is DMatrix32): fail()
  test "literal matrices should be of expected types":
    let
      u = Matrix(2, 3, @[@[1'f32, 2'f32, 3'f32], @[4'f32, 5'f32, 6'f32]])
      v = matrix(@[@[1'f32, 2'f32, 3'f32], @[4'f32, 5'f32, 6'f32]])
    when not (u is Matrix32[2, 3]): fail()
    when not (v is DMatrix32): fail()
  test "identity matrices should be of expected types":
    let
      M = 5
      u = eye(5, float32)
      v = eye(M, float32)
    when not (u is Matrix32[5, 5]): fail()
    when not (v is DMatrix32): fail()
  test "random matrices should be of expected types":
    let
      M = 4
      N = 5
      u = randomMatrix(4, 5, max = 1'f32)
      v = randomMatrix(M, N, max = 1'f32)
    when not (u is Matrix32[4, 5]): fail()
    when not (v is DMatrix32): fail()
  test "proc matrices should be of expected types":
    let
      M = 4
      N = 5
      u = makeMatrix(4, 5, proc(i, j: int): float32 = (i + j).float32)
      v = makeMatrix(M, N, proc(i, j: int): float32 = (i + j).float32)
    when not (u is Matrix32[4, 5]): fail()
    when not (v is DMatrix32): fail()