type
  Vector32*[N: static[int]] = ref array[N, float32]
  Vector64*[N: static[int]] = ref array[N, float64]
  Matrix32*[M, N: static[int]] = ref array[N, array[M, float32]]
  Matrix64*[M, N: static[int]] = ref array[N, array[M, float64]]

template fp(v: Vector64): ptr float64 = cast[ptr float64](addr(v[]))

template fp(m: Matrix64): ptr float64 = cast[ptr float64](addr(m[]))