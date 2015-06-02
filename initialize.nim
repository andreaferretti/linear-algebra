proc makeVect(N: static[int], f: proc (i: int): float64): Vect64[N] =
  for i in 0 .. < N:
    result[i] = f(i)

proc initMatrix*[M, N: static[int]](m: var Matrix[M, N]) =
  new m
  m.p = cast[ptr array[N, array[M, float64]]](mkl_malloc(M * N * sizeof(float64), 64))

proc makeMatrix*(M, N: static[int], f: proc (i, j: int): float64): Matrix[M, N] =
  initMatrix(result)
  var m = cast[ptr Matrix64[M, N]](result.p)
  for i in 0 .. < N:
    for j in 0 .. < M:
      m[i][j] = f(i, j)