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
  TransposeType = enum
    noTranspose = 111, transpose = 112, conjTranspose = 113
  OrderType* = enum
    rowMajor = 101, colMajor = 102

proc sscal(N: int, ALPHA: float32, X: ptr float32, INCX: int)
  {. header: header, importc: "cblas_sscal" .}
proc dscal(N: int, ALPHA: float64, X: ptr float64, INCX: int)
  {. header: header, importc: "cblas_dscal" .}
proc scopy(N: int, X: ptr float32, INCX: int, Y: ptr float32, INCY: int)
  {. header: header, importc: "cblas_scopy" .}
proc dcopy(N: int, X: ptr float64, INCX: int, Y: ptr float64, INCY: int)
  {. header: header, importc: "cblas_dcopy" .}
proc saxpy(N: int, ALPHA: float32, X: ptr float32, INCX: int, Y: ptr float32, INCY: int)
  {. header: header, importc: "cblas_saxpy" .}
proc daxpy(N: int, ALPHA: float64, X: ptr float64, INCX: int, Y: ptr float64, INCY: int)
  {. header: header, importc: "cblas_daxpy" .}
proc sdot(N: int, X: ptr float32, INCX: int, Y: ptr float32, INCY: int): float32
  {. header: header, importc: "cblas_sdot" .}
proc ddot(N: int, X: ptr float64, INCX: int, Y: ptr float64, INCY: int): float64
  {. header: header, importc: "cblas_ddot" .}
proc nrm2(N: int, X: ptr float32, INCX: int): float32
  {. header: header, importc: "cblas_snrm2" .}
proc nrm2(N: int, X: ptr float64, INCX: int): float64
  {. header: header, importc: "cblas_dnrm2" .}
proc asum(N: int, X: ptr float32, INCX: int): float32
  {. header: header, importc: "cblas_sasum" .}
proc asum(N: int, X: ptr float64, INCX: int): float64
  {. header: header, importc: "cblas_dasum" .}
proc sgemv(ORDER: OrderType, TRANS: TransposeType, M, N: int, ALPHA: float32, A: ptr float32,
  LDA: int, X: ptr float32, INCX: int, BETA: float32, Y: ptr float32, INCY: int)
  {. header: header, importc: "cblas_sgemv" .}
proc dgemv(ORDER: OrderType, TRANS: TransposeType, M, N: int, ALPHA: float64, A: ptr float64,
  LDA: int, X: ptr float64, INCX: int, BETA: float64, Y: ptr float64, INCY: int)
  {. header: header, importc: "cblas_dgemv" .}
proc dgemm(ORDER: OrderType, TRANSA, TRANSB: TransposeType, M, N, K: int, ALPHA: float64,
  A: ptr float64, LDA: int, B: ptr float64, LDB: int, BETA: float64, C: ptr float64, LDC: int)
  {. header: header, importc: "cblas_dgemm" .}