#http://www.netlib.org/blas/#_blas_routines
when defined(mkl):
  const header = "mkl.h"
  when defined(threaded):
    {. passl: "-lmkl_intel_lp64" passl: "-lmkl_core" passl: "-lmkl_gnu_thread" passl: "-lgomp" .}
  # {. passl: "-lmkl_intel_lp64" passl: "-lmkl_core" passl: "-lmkl_intel_thread" passl: "-lmpi" .}
    static: echo "--USING MKL THREADED--"
  else:
    {.passl: "-lmkl_intel_lp64" passl: "-lmkl_core" passl: "-lmkl_sequential" passl: "-lpthread" .}
    static: echo "--USING MKL SEQUENTIAL--"
else:
  when defined(atlas):
    {.passl: "-lcblas".}
    const header = "atlas/cblas.h"
    static: echo "--USING ATLAS--"
  else:
    {.passl: "-lblas".}
    const header = "cblas.h"
    static: echo "--USING DEFAULT BLAS--"


type
  Vect32*[N: static[int]] = array[N, float32]
  Vect64*[N: static[int]] = array[N, float64]
  Vect*[N: static[int]] = Vect64[N]
  Matrix32*[M, N: static[int]] = array[N, array[M, float32]]
  Matrix64*[M, N: static[int]] = array[N, array[M, float64]]
  Matrix*[M, N: static[int]] = Matrix64[M, N]
  TransposeType = enum
    noTranspose = 111, transpose = 112, conjTranspose = 113
  OrderType = enum
    rowMajor = 101, colMajor = 102

# Raw BLAS operations

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

# Internal functions

template asPtr[N: static[int]](v: Vect64[N]): ptr float64 = cast[ptr float64](v.addr)

template asPtr[M, N: static[int]](a: Matrix64[M, N]): ptr float64 = cast[ptr float64](a.addr)

# Public API - Initialization

proc makeVect(N: static[int], f: proc (i: int): float64): Vect64[N] =
  for i in 0 .. < N:
    result[i] = f(i)

proc makeMatrix(M, N: static[int], f: proc (i, j: int): float64): Matrix64[M, N] =
  for i in 0 .. < N:
    for j in 0 .. < M:
      result[i][j] = f(i, j)

# Public API - Display

proc `$`*(v: Vect64): string =
  let s = $(@(v))
  return s[1 .. high(s)]

proc `$`*(m: Matrix64): string = $(@(m))

# Public API - Iterators

iterator items*[M, N: static[int]](m: Matrix64[M, N]): float64 {. inline .} =
  for i in 0 .. < N:
    for j in 0 .. < M:
      yield m[i][j]

iterator pairs*[M, N: static[int]](m: Matrix64[M, N]): tuple[indices: tuple[i, j: int], val: float64] {. inline .} =
  for i in 0 .. < N:
    for j in 0 .. < M:
      yield ((i, j), m[i][j])

# Public API - Linear algebra

proc `*=`*[N: static[int]](v: var Vect64[N], k: float64) {. inline .} =
  dscal(N, k, v.asPtr, 1)

proc `*`*[N: static[int]](v: var Vect64[N], k: float64): Vect64[N]  {. inline .} =
  dcopy(N, v.asPtr, 1, result.asPtr, 1)
  dscal(N, k, result.asPtr, 1)

template `*`*[N: static[int]](k: float64, v: var Vect64[N]): expr = v * k

proc `*`*[N: static[int]](v, w: var Vect64[N]): float64 {. inline .} =
  ddot(N, v.asPtr, 1, w.asPtr, 1)

proc l_2*[N: static[int]](v: var Vect64[N]): float64 {. inline .} = dnrm2(N, v.asPtr, 1)

proc l_1*[N: static[int]](v: var Vect64[N]): float64 {. inline .} = dasum(N, v.asPtr, 1)

proc `*`*[M, N: static[int]](a: var Matrix64[M, N], v: var Vect64[N]): Vect64[M]  {. inline .} =
  dgemv(colMajor, noTranspose, M, N, 1, a.asPtr, M, v.asPtr, 1, 0, result.asPtr, 1)

proc `*`*[M, N, K: static[int]](a: var Matrix64[M, K], b: var Matrix64[K, N]): Matrix64[M, N] {. inline .} =
  dgemm(colMajor, noTranspose, noTranspose, M, N, K, 1, a.asPtr, M, b.asPtr, K, 0, result.asPtr, M)

when isMainModule:
  import math, times

  var
    # xs = [1.0, 2.0, 3.5]
    # ys = [2.0, 3.0, 3.5]
    # m = [
    #   [1.2, 3.4],
    #   [1.1, 2.1],
    #   [0.6, -3.1]
    # ]
    mat1 = makeMatrix(1000, 987, proc(i, j: int): float64 = random(1.0))
    mat2 = makeMatrix(987, 876, proc(i, j: int): float64 = random(1.0))
    # vec= makeVect(987, proc(i: int): float64 = random(1.0))

  # let startTime = epochTime()
  # for i in 0 .. 100:
  #   discard mat1 * vec
  # let endTime = epochTime()
  # echo "We have required ", endTime - startTime, " seconds to do 100 multiplications."

  let startTime1 = epochTime()
  for i in 0 .. < 10:
    discard mat1 * mat2
  let endTime1 = epochTime()
  echo "We have required ", endTime1 - startTime1, " seconds to do multiply matrices 10 times."

  # echo((mat1 * vec)[1..10])
  # echo(xs * 5.3)
  # echo(5.3 * xs)
  # xs *= 5.3
  # echo(xs)
  # echo(xs * ys)
  # echo(xs * xs)
  # echo(l_2(xs))
  # echo(l_1(xs))
  # echo(m * ys)

  # template optMul{`*`(a, 2)}(a: int): int =
  #   echo "hi"
  #   a*3

  # let b = 55

  # echo b * 2