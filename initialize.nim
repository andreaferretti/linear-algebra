proc makeVector*(N: static[int], f: proc (i: int): float64): Vector64[N] =
  new result
  for i in 0 .. < N:
    result[i] = f(i)

proc randomVector*(N: static[int], max: float64 = 1): Vector64[N] =
  makeVector(N, proc(i: int): float64 = random(max))

proc constant*(N: static[int], x: float64): Vector64[N] =
  new result
  for i in 0 .. < N:
    result[i] = x

proc zeros*(N: static[int]): Vector64[N] = constant(N, 0)

proc ones*(N: static[int]): Vector64[N] = constant(N, 1)

proc makeMatrix*(M, N: static[int], f: proc (i, j: int): float64): Matrix64[M, N] =
  new result
  for i in 0 .. < N:
    for j in 0 .. < M:
      result[i][j] = f(i, j)

proc randomMatrix*(M, N: static[int], max: float64 = 1): Matrix64[M, N] =
  makeMatrix(M, N, proc(i, j: int): float64 = random(max))

proc constant*(M, N: static[int], x: float64): Matrix64[M, N] =
  new result
  for i in 0 .. < N:
    for j in 0 .. < M:
      result[i][j] = x

proc zeros*(M, N: static[int]): Matrix64[M, N] = constant(M, N, 0)

proc ones*(M, N: static[int]): Matrix64[M, N] = constant(M, N, 1)

proc eye*(N: static[int]): Matrix64[N, N] =
  new result
  for i in 0 .. < N:
    for j in 0 .. < N:
      result[i][j] = if i == j: 1 else: 0