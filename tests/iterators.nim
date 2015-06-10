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


suite "iterators on vectors":
  test "item vector iterators":
    let v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
    var
      sum = 0.0
      count = 0
    for x in v:
      sum += x
      count += 1
    check sum == 12.0
    check count == 5
  test "pairs vector iterators":
    let v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
    var
      sum = 0.0
      sumI = 0
    for i, x in v:
      sum += x
      sumI += i
    check sum == 12.0
    check sumI == 10