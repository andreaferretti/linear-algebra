proc `*=`*[N: static[int]](v: var Vector64[N], k: float64) {. inline .} = dscal(N, k, v.fp, 1)

proc `*`*[N: static[int]](v: Vector64[N], k: float64): Vector64[N]  {. inline .} =
  initVector(result)
  dcopy(N, v.fp, 1, result.fp, 1)
  dscal(N, k, result.fp, 1)

template `*`*[N: static[int]](k: float64, v: Vector64[N]): expr = v * k

proc `*`*[N: static[int]](v, w: Vector64[N]): float64 {. inline .} = ddot(N, v.fp, 1, w.fp, 1)

proc l_2*[N: static[int]](v: Vector64[N]): float64 {. inline .} = dnrm2(N, v.fp, 1)

proc l_1*[N: static[int]](v: Vector64[N]): float64 {. inline .} = dasum(N, v.fp, 1)

proc `*`*[M, N: static[int]](a: Matrix64[M, N], v: Vector64[N]): Vector64[M]  {. inline .} =
  initVector(result)
  dgemv(colMajor, noTranspose, M, N, 1, a.fp, M, v.fp, 1, 0, result.fp, 1)

proc `*`*[M, N, K: static[int]](a: Matrix64[M, K], b: Matrix64[K, N]): Matrix64[M, N] {. inline .} =
  initMatrix(result)
  dgemm(colMajor, noTranspose, noTranspose, M, N, K, 1, a.fp, M, b.fp, K, 0, result.fp, M)