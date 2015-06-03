proc initVector*[N: static[int]](v: var Vector64[N]) =
  new v, finalizeVector64
  v.p = cast[ptr array[N, float64]](la_malloc(N * sizeof(float64)))

proc makeVector*(N: static[int], f: proc (i: int): float64): Vector64[N] =
  initVector(result)
  for i in 0 .. < N:
    result.p[i] = f(i)

proc randomVector*(N: static[int], max: float64 = 1): Vector64[N] =
  makeVector(N, proc(i: int): float64 = random(max))

proc constant*(N: static[int], x: float64): Vector64[N] =
  initVector(result)
  for i in 0 .. < N:
    result.p[i] = x

proc zeros*(N: static[int]): Vector64[N] = constant(N, 0)

proc ones*(N: static[int]): Vector64[N] = constant(N, 1)

proc initMatrix*[M, N: static[int]](m: var Matrix64[M, N]) =
  new m, finalizeMatrix64
  m.p = cast[ptr array[N, array[M, float64]]](la_malloc(M * N * sizeof(float64)))

proc makeMatrix*(M, N: static[int], f: proc (i, j: int): float64): Matrix64[M, N] =
  initMatrix(result)
  for i in 0 .. < N:
    for j in 0 .. < M:
      result.p[i][j] = f(i, j)

proc randomMatrix*(M, N: static[int], max: float64 = 1): Matrix64[M, N] =
  makeMatrix(M, N, proc(i, j: int): float64 = random(max))

proc constant*(M, N: static[int], x: float64): Matrix64[M, N] =
  initMatrix(result)
  for i in 0 .. < N:
    for j in 0 .. < M:
      result.p[i][j] = x

proc zeros*(M, N: static[int]): Matrix64[M, N] = constant(M, N, 0)

proc ones*(M, N: static[int]): Matrix64[M, N] = constant(M, N, 1)

proc eye*(N: static[int]): Matrix64[N, N] =
  initMatrix(result)
  for i in 0 .. < N:
    for j in 0 .. < N:
      result.p[i][j] = if i == j: 1 else: 0