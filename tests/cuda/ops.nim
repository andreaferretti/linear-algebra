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
  test "vector sum":
    let
      v1 = randomVector(10, max=1'f32)
      v2 = randomVector(10, max=1'f32)

    let
      p1 = v1.gpu()
      p2 = v2.gpu()
      p3 = p1 + p2
      v3 = p3.cpu()

    check(v1 + v2 == v3)

  test "vector difference":
    let
      v1 = randomVector(10, max=1'f32)
      v2 = randomVector(10, max=1'f32)

    let
      p1 = v1.gpu()
      p2 = v2.gpu()
      p3 = p1 - p2
      v3 = p3.cpu()

    check(v1 - v2 == v3)