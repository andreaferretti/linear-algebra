proc `[]`*(m: Matrix64, i, j: int): float64 {. inline .} = m[j][i]

proc `[]=`*(m: var Matrix64, i, j: int, val: float64) {. inline .} =
  m[j][i] = val

proc column*[M, N: static[int]](m: Matrix64[M, N], j: int): Vector64[N] {. inline .} =
  new result
  for i in 0 .. < N:
    result[i] = m[i, j]

proc row*[M, N: static[int]](m: Matrix64[M, N], i: int): Vector64[M] {. inline .} =
  new result
  for j in 0 .. < M:
    result[j] = m[i, j]