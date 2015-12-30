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