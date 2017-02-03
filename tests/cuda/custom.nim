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


suite "custom operations":
  test "Hadamard product on CudaVector32":
    let
      v = @[1.0, 2.5, 3.2].to32()
      w = @[3.0, 1.5, 3.0].to32()
      vg = v.gpu()
      wg = w.gpu()
    check((v |*| w) == (vg |*| wg).cpu())
  test "Hadamard product on CudaVector64":
    let
      v = @[1.0, 2.5, 3.2]
      w = @[3.0, 1.5, 3.0]
      vg = v.gpu()
      wg = w.gpu()
    check((v |*| w) == (vg |*| wg).cpu())