# proc makeVect(N: static[int], f: proc (i: int): float64): Vect64[N] =
#   for i in 0 .. < N:
#     result[i] = f(i)

proc initVector*[N: static[int]](v: var Vector64[N]) =
  new v, finalizeVector64
  v.p = cast[ptr array[N, float64]](malloc(N * sizeof(float64)))

proc initMatrix*[M, N: static[int]](m: var Matrix64[M, N]) =
  new m, finalizeMatrix64
  m.p = cast[ptr array[N, array[M, float64]]](malloc(M * N * sizeof(float64)))
  echo "M = ", M, "; N = ", N, "; pointer is ", cast[int](m.p)

proc makeMatrix*(M, N: static[int], f: proc (i, j: int): float64): Matrix64[M, N] =
  initMatrix(result)
  var m = cast[ptr array[N, array[M, float64]]](result.p)
  for i in 0 .. < N:
    for j in 0 .. < M:
      m[i][j] = f(i, j)