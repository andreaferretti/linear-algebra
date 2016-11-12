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


suite "collection operations":
  test "cumulative sum over Vector32":
    let
      v = vector([1'f32, 3.5'f32, 2'f32, 4.5'f32])
      w = cumsum(v)
    check w == vector([1'f32, 4.5'f32, 6.5'f32, 11'f32])
  test "cumulative sum over Vector64":
    let
      v = vector([1.0, 3.5, 2.0, 4.5])
      w = cumsum(v)
    check w == vector([1.0, 4.5, 6.5, 11.0])
  test "cumulative sum over DVector32":
    let
      v = @[1'f32, 3.5'f32, 2'f32, 4.5'f32]
      w = cumsum(v)
    check w == @[1'f32, 4.5'f32, 6.5'f32, 11'f32]
  test "cumulative sum over DVector64":
    let
      v = @[1.0, 3.5, 2.0, 4.5]
      w = cumsum(v)
    check w == @[1.0, 4.5, 6.5, 11.0]