# proc `*=`*[N: static[int]](v: var Vect64[N], k: float64) {. inline .} =
#   dscal(N, k, v.asPtr, 1)

# proc `*`*[N: static[int]](v: var Vect64[N], k: float64): Vect64[N]  {. inline .} =
#   dcopy(N, v.asPtr, 1, result.asPtr, 1)
#   dscal(N, k, result.asPtr, 1)

# template `*`*[N: static[int]](k: float64, v: var Vect64[N]): expr = v * k

# proc `*`*[N: static[int]](v, w: var Vect64[N]): float64 {. inline .} =
#   ddot(N, v.asPtr, 1, w.asPtr, 1)

proc l_2*[N: static[int]](v: Vector64[N]): float64 {. inline .} = dnrm2(N, v.fp, 1)

proc l_1*[N: static[int]](v: Vector64[N]): float64 {. inline .} = dasum(N, v.fp, 1)

# proc `*`*[M, N: static[int]](a: var Matrix64[M, N], v: var Vect64[N]): Vect64[M]  {. inline .} =
#   dgemv(colMajor, noTranspose, M, N, 1, a.p, M, v.p, 1, 0, result.p, 1)

proc `*`*[M, N, K: static[int]](a: Matrix64[M, K], b: Matrix64[K, N]): Matrix64[M, N] {. inline .} =
  initMatrix(result)
  dgemm(colMajor, noTranspose, noTranspose, M, N, K, 1, a.fp, M, b.fp, K, 0, result.fp, M)