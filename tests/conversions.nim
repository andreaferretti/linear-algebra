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


suite "conversions":
  test "Vector64 to Vector32":
    let v = vector([1.0, 3.5, 2.0, 4.5])
    check v.to32 == vector([1'f32, 3.5'f32, 2'f32, 4.5'f32], float32)
  test "Vector32 to Vector64":
    let v = vector([1'f32, 3.5'f32, 2'f32, 4.5'f32], float32)
    check v.to64 == vector([1.0, 3.5, 2.0, 4.5])