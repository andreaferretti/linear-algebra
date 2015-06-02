type
  TransposeType = enum
    noTranspose = 111, transpose = 112, conjTranspose = 113
  OrderType = enum
    rowMajor = 101, colMajor = 102

proc dscal(N: int, ALPHA: float64, X: ptr float64, INCX: int)
  {. header: header, importc: "cblas_dscal" .}
proc dcopy(N: int, X: ptr float64, INCX: int, Y: ptr float64, INCY: int)
  {. header: header, importc: "cblas_dcopy" .}
proc ddot(N: int, X: ptr float64, INCX: int, Y: ptr float64, INCY: int): float64
  {. header: header, importc: "cblas_ddot" .}
proc dnrm2(N: int, X: ptr float64, INCX: int): float64
  {. header: header, importc: "cblas_dnrm2" .}
proc dasum(N: int, X: ptr float64, INCX: int): float64
  {. header: header, importc: "cblas_dasum" .}
proc dgemv(ORDER: OrderType, TRANS: TransposeType, M, N: int, ALPHA: float64, A: ptr float64,
  LDA: int, X: ptr float64, INCX: int, BETA: float64, Y: ptr float64, INCY: int)
  {. header: header, importc: "cblas_dgemv" .}
proc dgemm(ORDER: OrderType, TRANSA, TRANSB: TransposeType, M, N, K: int, ALPHA: float64,
  A: ptr float64, LDA: int, B: ptr float64, LDB: int, BETA: float64, C: ptr float64, LDC: int)
  {. header: header, importc: "cblas_dgemm" .}
proc mkl_malloc(size, align: int): ptr float64
  {. header: header, importc: "mkl_malloc" .}