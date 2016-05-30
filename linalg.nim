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
import math, nimblas
from sequtils import mapIt

when defined(js):
  {.fatal: "linalg is only available for native backends".}

include private/types
include private/cublas
include private/initialize
include private/access
include private/iterators
include private/display
include private/trivial_ops
include private/ops
include private/ufunc
# include private/rewrites

export nimblas.OrderType