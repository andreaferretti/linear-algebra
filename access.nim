proc `[]`*(v: Vector64, i: int): float64 {. inline .} = v.p[i]

proc `[]`*(m: Matrix64, i, j: int): float64 {. inline .} = m.p[j][i]

proc column*[M, N: static[int]](m: Matrix64[M, N], j: int): Vector64[N] {. inline .} =
  initVector(result)
  for i in 0 .. < N:
    result.p[i] = m[i, j]

proc row*[M, N: static[int]](m: Matrix64[M, N], i: int): Vector64[M] {. inline .} =
  initVector(result)
  for j in 0 .. < M:
    result.p[j] = m[i, j]