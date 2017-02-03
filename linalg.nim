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
import math, sequtils, nimblas, random, linalg/nimlapack

when defined(js):
  {.fatal: "linalg is only available for native backends".}

include linalg/private/types
include linalg/private/initialize
include linalg/private/access
include linalg/private/iterators
include linalg/private/display
include linalg/private/trivial_ops
include linalg/private/ops
include linalg/private/ufunc
include linalg/private/funcs
include linalg/private/collection
include linalg/private/cublas
include linalg/private/rewrites

export nimblas.OrderType
