iterator items*[M, N: static[int]](m: Matrix64[M, N]): float64 {. inline .} =
  for i in 0 .. < N:
    for j in 0 .. < M:
      yield m[i][j]

iterator pairs*[M, N: static[int]](m: Matrix64[M, N]): tuple[indices: tuple[i, j: int], val: float64] {. inline .} =
  for i in 0 .. < N:
    for j in 0 .. < M:
      yield ((i, j), m[i][j])