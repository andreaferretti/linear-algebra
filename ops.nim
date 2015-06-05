proc `*=`*[N: static[int]](v: var Vector64[N], k: float64) {. inline .} = dscal(N, k, v.fp, 1)

proc `*`*[N: static[int]](v: Vector64[N], k: float64): Vector64[N]  {. inline .} =
  new result
  dcopy(N, v.fp, 1, result.fp, 1)
  dscal(N, k, result.fp, 1)

proc `+=`*[N: static[int]](v: var Vector64[N], w: Vector64[N]) {. inline .} =
  daxpy(N, 1, w.fp, 1, v.fp, 1)

proc `+`*[N: static[int]](v, w: Vector64[N]): Vector64[N]  {. inline .} =
  new result
  dcopy(N, v.fp, 1, result.fp, 1)
  daxpy(N, 1, w.fp, 1, result.fp, 1)

proc `-=`*[N: static[int]](v: var Vector64[N], w: Vector64[N]) {. inline .} =
  daxpy(N, -1, w.fp, 1, v.fp, 1)

proc `-`*[N: static[int]](v, w: Vector64[N]): Vector64[N]  {. inline .} =
  new result
  dcopy(N, v.fp, 1, result.fp, 1)
  daxpy(N, -1, w.fp, 1, result.fp, 1)

template `*`*[N: static[int]](k: float64, v: Vector64[N]): expr = v * k

proc `*`*[N: static[int]](v, w: Vector64[N]): float64 {. inline .} = ddot(N, v.fp, 1, w.fp, 1)

proc l_2*[N: static[int]](v: Vector64[N]): float64 {. inline .} = dnrm2(N, v.fp, 1)

proc l_1*[N: static[int]](v: Vector64[N]): float64 {. inline .} = dasum(N, v.fp, 1)

proc `*`*[M, N: static[int]](a: Matrix64[M, N], v: Vector64[N]): Vector64[M]  {. inline .} =
  new result
  dgemv(a.order, noTranspose, M, N, 1, a.fp, M, v.fp, 1, 0, result.fp, 1)

proc `+=`*[M, N: static[int]](a: var Matrix64[M, N], b: Matrix64[M, N]) {. inline .} =
  if a.order == b.order:
    daxpy(M * N, 1, b.fp, 1, a.fp, 1)
  elif a.order == colMajor and b.order == rowMajor:
    let
      a_data = cast[ref array[N, array[M, float64]]](a.data)
      b_data = cast[ref array[M, array[N, float64]]](b.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        a_data[j][i] += b_data[i][j]
  else:
    let
      a_data = cast[ref array[M, array[N, float64]]](a.data)
      b_data = cast[ref array[N, array[M, float64]]](b.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        a_data[i][j] += b_data[j][i]

proc `+`*[M, N: static[int]](a, b: Matrix64[M, N]): Matrix64[M, N]  {. inline .} =
  new result.data
  result.order = a.order
  dcopy(M * N, a.fp, 1, result.fp, 1)
  result += b

proc `-=`*[M, N: static[int]](a: var Matrix64[M, N], b: Matrix64[M, N]) {. inline .} =
  if a.order == b.order:
    daxpy(M * N, -1, b.fp, 1, a.fp, 1)
  elif a.order == colMajor and b.order == rowMajor:
    let
      a_data = cast[ref array[N, array[M, float64]]](a.data)
      b_data = cast[ref array[M, array[N, float64]]](b.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        a_data[j][i] -= b_data[i][j]
  else:
    let
      a_data = cast[ref array[M, array[N, float64]]](a.data)
      b_data = cast[ref array[N, array[M, float64]]](b.data)
    for i in 0 .. < M:
      for j in 0 .. < N:
        a_data[i][j] -= b_data[j][i]

proc `-`*[M, N: static[int]](a, b: Matrix64[M, N]): Matrix64[M, N]  {. inline .} =
  new result.data
  result.order = a.order
  dcopy(M * N, a.fp, 1, result.fp, 1)
  result -= b

proc `*`*[M, N, K: static[int]](a: Matrix64[M, K], b: Matrix64[K, N]): Matrix64[M, N] {. inline .} =
  new result.data
  if a.order == b.order:
    result.order = a.order
    dgemm(a.order, noTranspose, noTranspose, M, N, K, 1, a.fp, M, b.fp, K, 0, result.fp, M)
  elif a.order == colMajor and b.order == rowMajor:
    result.order = colMajor
    dgemm(colMajor, noTranspose, transpose, M, N, K, 1, a.fp, M, b.fp, N, 0, result.fp, M)
  else:
    result.order = colMajor
    dgemm(colMajor, transpose, noTranspose, M, N, K, 1, a.fp, K, b.fp, K, 0, result.fp, M)

proc t*[M, N: static[int]](a: Matrix64[M, N]): Matrix64[N, M] =
  result.order = if a.order == rowMajor: colMajor else: rowMajor
  result.data = a.data