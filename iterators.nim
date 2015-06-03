iterator items*[N: static[int]](v: Vector64[N]): float64 {. inline .} =
  for i in 0 .. < N:
    yield v[i]

iterator pairs*[N: static[int]](v: Vector64[N]): tuple[i: int, val: float64] {. inline .} =
  for i in 0 .. < N:
    yield (i, v[i])

iterator columns*[M, N: static[int]](m: Matrix64[M, N]): Vector64[N] {. inline .} =
  for i in 0 .. < M:
    yield m.column(i)

iterator rows*[M, N: static[int]](m: Matrix64[M, N]): Vector64[M] {. inline .} =
  for i in 0 .. < N:
    yield m.row(i)

iterator items*[M, N: static[int]](m: Matrix64[M, N]): float64 {. inline .} =
  for i in 0 .. < N:
    for j in 0 .. < M:
      yield m[i, j]

iterator pairs*[M, N: static[int]](m: Matrix64[M, N]): tuple[indices: tuple[i, j: int], val: float64] {. inline .} =
  for i in 0 .. < N:
    for j in 0 .. < M:
      yield ((i, j), m[i, j])