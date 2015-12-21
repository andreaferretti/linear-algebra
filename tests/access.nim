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


suite "vector accessors":
  test "reading vector length":
    let v = randomVector(10)
    check v.len == 10
  test "reading vector elements":
    let v = makeVector(5, proc(i: int): float64 = (3 * i - 2).float64)
    check v[0] == -2.0
    check v[1] == 1.0
    check v[2] == 4.0
    check v[3] == 7.0
    check v[4] == 10.0
  test "writing vector elements":
    var v = zeros(3)
    v[0] = -2.1
    v[1] = 1.0
    check v[0] == -2.1
    check v[1] == 1.0
  test "cloning vectors":
    var v = randomVector(5)
    let
      w = v.clone
      f = w[0]
    check v == w
    v[0] = v[0] + 1
    check w[0] == f
  test "mapping vectors":
    var v = vector([1.0, 2.0, 3.0, 4.0, 5.0])
    check v.map(proc(x: float64): float64 = 2 * x) ==
      vector([2.0, 4.0, 6.0, 8.0, 10.0])

suite "32-bit vector accessors":
  test "reading vector length":
    let v = randomVector(10, max = 1'f32)
    check v.len == 10
  test "reading vector elements":
    let v = makeVector(5, proc(i: int): float32 = (3 * i - 2).float32)
    check v[0] == -2.0
    check v[1] == 1.0
    check v[2] == 4.0
    check v[3] == 7.0
    check v[4] == 10.0
  test "writing vector elements":
    var v = zeros(3, float32)
    v[0] = -2.5
    v[1] = 1.0
    check v[0] == -2.5
    check v[1] == 1.0
  test "cloning vectors":
    var v = randomVector(5, max = 1'f32)
    let
      w = v.clone
      f = w[0]
    check v == w
    v[0] = v[0] + 1
    check w[0] == f
  test "mapping vectors":
    var v = vector([1'f32, 2'f32, 3'f32, 4'f32, 5'f32], float32)
    check v.map(proc(x: float32): float32 = 2 * x) ==
      vector([2'f32, 4'f32, 6'f32, 8'f32, 10'f32], float32)

suite "matrix accessors":
  test "reading matrix dimensions":
    let m = randomMatrix(3, 7)
    check m.dim == (3, 7)
  test "reading matrix elements":
    let m = makeMatrix(2, 2, proc(i, j: int): float64 = (3 * i - 2 * j).float64)
    check m[0, 0] == 0.0
    check m[0, 1] == -2.0
    check m[1, 0] == 3.0
    check m[1, 1] == 1.0
  test "writing matrix elements":
    var m = zeros(3, 3)
    m[0, 2] = -2.1
    m[1, 1] = 1.0
    check m[0, 2] == -2.1
    check m[1, 1] == 1.0
  test "reading matrix rows":
    let
      m = makeMatrix(2, 2, proc(i, j: int): float64 = (3 * i - 2 * j).float64)
      r = m.row(1)
    check r[0] == 3.0
    check r[1] == 1.0
  test "reading matrix columns":
    let
      m = makeMatrix(2, 2, proc(i, j: int): float64 = (3 * i - 2 * j).float64)
      c = m.column(1)
    check c[0] == -2.0
    check c[1] == 1.0
  test "reading matrix rows without copy":
    let
      m = makeMatrix(4, 3, proc(i, j: int): float64 = (3 * i - 2 * j).float64, rowMajor)
      r = m.row(1)
      ru = m.rowUnsafe(1)
    check r == ru
  test "reading matrix columns without copy":
    let
      m = makeMatrix(3, 5, proc(i, j: int): float64 = (3 * i - 2 * j).float64)
      c = m.column(1)
      cu = m.columnUnsafe(1)
    check c == cu
  test "cloning matrices":
    var m = randomMatrix(5, 5)
    let
      n = m.clone
      f = n[2, 2]
    check m == n
    m[2, 2] = m[2, 2] + 1
    check n[2, 2] == f
  test "mapping matrices":
    let
      m = makeMatrix(2, 2, proc(i, j: int): float64 = (3 * i - 2 * j).float64)
      n = makeMatrix(2, 2, proc(i, j: int): float64 = (6 * i - 4 * j).float64)
    proc double(x: float64): float64 = 2 * x
    check m.map(double) == n

suite "32-bit matrix accessors":
  test "reading matrix dimensions":
    let m = randomMatrix(3, 7, max = 1'f32)
    check m.dim == (3, 7)
  test "reading matrix elements":
    let m = makeMatrix(2, 2, proc(i, j: int): float32 = (3 * i - 2 * j).float32)
    check m[0, 0] == 0'f32
    check m[0, 1] == -2'f32
    check m[1, 0] == 3'f32
    check m[1, 1] == 1'f32
  test "writing matrix elements":
    var m = zeros(3, 3, float32)
    m[0, 2] = -2.5'f32
    m[1, 1] = 1'f32
    check m[0, 2] == -2.5'f32
    check m[1, 1] == 1'f32
  test "reading matrix rows":
    let
      m = makeMatrix(2, 2, proc(i, j: int): float32 = (3 * i - 2 * j).float32)
      r = m.row(1)
    check r[0] == 3'f32
    check r[1] == 1'f32
  test "reading matrix columns":
    let
      m = makeMatrix(2, 2, proc(i, j: int): float32 = (3 * i - 2 * j).float32)
      c = m.column(1)
    check c[0] == -2'f32
    check c[1] == 1'f32
  test "reading matrix rows without copy":
    let
      m = makeMatrix(4, 3, proc(i, j: int): float32 = (3 * i - 2 * j).float32, rowMajor)
      r = m.row(1)
      ru = m.rowUnsafe(1)
    check r == ru
  test "reading matrix columns without copy":
    let
      m = makeMatrix(3, 5, proc(i, j: int): float32 = (3 * i - 2 * j).float32)
      c = m.column(1)
      cu = m.columnUnsafe(1)
    check c == cu
  test "cloning matrices":
    var m = randomMatrix(5, 5, max = 1'f32)
    let
      n = m.clone
      f = n[2, 2]
    check m == n
    m[2, 2] = m[2, 2] + 1
    check n[2, 2] == f
  test "mapping matrices":
    let
      m = makeMatrix(2, 2, proc(i, j: int): float32 = (3 * i - 2 * j).float32)
      n = makeMatrix(2, 2, proc(i, j: int): float32 = (6 * i - 4 * j).float32)
    proc double(x: float32): float32 = 2 * x
    check m.map(double) == n

suite "dynamic vector accessors":
  test "reading vector length":
    let n = 10
    let v = randomDVector(n)
    check v.len == 10
  test "reading vector elements":
    let n = 5
    let v = makeDVector(n, proc(i: int): float64 = (3 * i - 2).float64)
    check v[0] == -2.0
    check v[1] == 1.0
    check v[2] == 4.0
    check v[3] == 7.0
    check v[4] == 10.0
  test "writing vector elements":
    let n = 3
    var v = zeros(n)
    v[0] = -2.1
    v[1] = 1.0
    check v[0] == -2.1
    check v[1] == 1.0
  test "cloning vectors":
    let n = 5
    var v = randomDVector(n)
    let
      w = v.clone
      f = w[0]
    check v == w
    v[0] = v[0] + 1
    check w[0] == f

suite "dynamic 32-bit vector accessors":
  test "reading vector length":
    let n = 10
    let v = randomDVector(n, max = 1'f32)
    check v.len == 10
  test "reading vector elements":
    let n = 5
    let v = makeDVector(n, proc(i: int): float32 = (3 * i - 2).float32)
    check v[0] == -2'f32
    check v[1] == 1'f32
    check v[2] == 4'f32
    check v[3] == 7'f32
    check v[4] == 10'f32
  test "writing vector elements":
    let n = 3
    var v = zeros(n, float32)
    v[0] = -2.5'f32
    v[1] = 1'f32
    check v[0] == -2.5'f32
    check v[1] == 1'f32
  test "cloning vectors":
    let n = 5
    var v = randomDVector(n, max = 1'f32)
    let
      w = v.clone
      f = w[0]
    check v == w
    v[0] = v[0] + 1
    check w[0] == f