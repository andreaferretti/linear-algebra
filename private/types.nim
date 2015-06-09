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

type
  Vector32*[N: static[int]] = ref array[N, float32]
  Vector64*[N: static[int]] = ref array[N, float64]
  Matrix32*[M, N: static[int]] = object
    order: OrderType
    data: ref array[N * M, float32]
  Matrix64*[M, N: static[int]] = object
    order: OrderType
    data: ref array[M * N, float64]

# Float pointers
template fp(v: Vector64): ptr float64 = cast[ptr float64](addr(v[]))

template fp(m: Matrix64): ptr float64 = cast[ptr float64](addr(m.data[]))