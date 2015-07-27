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


suite "vector 32 operations":
  test "scalar vector multiplication":
    let
      v1 = randomVector(10, max=1'f32)
      p1 = v1.gpu()
    check(v1 * 3 == (p1 * 3).cpu())
    check(2 * v1 == (2 * p1).cpu())
  test "in place scalar vector multiplication":
    var
      v1 = randomVector(10, max=1'f32)
      p1 = v1.gpu()
    v1 *= 5
    p1 *= 5
    check(v1 == p1.cpu())
  test "vector sum":
    let
      v1 = randomVector(10, max=1'f32)
      v2 = randomVector(10, max=1'f32)
      p1 = v1.gpu()
      p2 = v2.gpu()
      p3 = p1 + p2
      v3 = p3.cpu()
    check(v1 + v2 == v3)
  test "in place vector sum":
    var v1 = randomVector(10, max=1'f32)
    let v2 = randomVector(10, max=1'f32)
    var p1 = v1.gpu()
    let p2 = v2.gpu()
    v1 += v2
    p1 += p2
    check(v1 == p1.cpu())
  test "vector difference":
    let
      v1 = randomVector(10, max=1'f32)
      v2 = randomVector(10, max=1'f32)
      p1 = v1.gpu()
      p2 = v2.gpu()
      p3 = p1 - p2
      v3 = p3.cpu()
    check(v1 - v2 == v3)
  test "in place vector difference":
    var v1 = randomVector(10, max=1'f32)
    let v2 = randomVector(10, max=1'f32)
    var p1 = v1.gpu()
    let p2 = v2.gpu()
    v1 -= v2
    p1 -= p2
    check(v1 == p1.cpu())
  test "dot product":
    let
      v = vector([1.0, 3.0, 2.0, 8.0, -2.0]).to32().gpu()
      w = vector([2.0, -1.0, 2.0, 0.0, 4.0]).to32().gpu()
    check(v * w == -5.0)
  test "ℓ² norm":
    let v = vector([1.0, 1.0, 2.0, 3.0, -7.0]).to32().gpu()
    check l_2(v) == 8.0
  test "ℓ¹ norm":
    let v = vector([1.0, 1.0, 2.0, 3.0, -7.0]).to32().gpu()
    check l_1(v) == 14.0