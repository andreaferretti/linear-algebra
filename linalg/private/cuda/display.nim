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

proc `$`*[N: static[int]](v: CudaVector32[N]): string = $(v.cpu())

proc `$`*[N: static[int]](v: CudaVector64[N]): string = $(v.cpu())

proc `$`*[M, N: static[int]](m: CudaMatrix32[M, N]): string = $(m.cpu())

proc `$`*[M, N: static[int]](m: CudaMatrix64[M, N]): string = $(m.cpu())

proc `$`*(v: CudaDVector32): string = $(v.cpu())

proc `$`*(v: CudaDVector64): string = $(v.cpu())

proc `$`*(m: CudaDMatrix32): string = $(m.cpu())

proc `$`*(m: CudaDMatrix64): string = $(m.cpu())