type
  anyFloat = float32 or float64
  Vector32*[N: static[int]] = ref array[N, float32]
  Vector64*[N: static[int]] = ref array[N, float64]
  Matrix32*[M, N: static[int]] = object
    order: OrderType
    data: ref array[N * M, float32]
  Matrix64*[M, N: static[int]] = object
    order: OrderType
    data: ref array[M * N, float64]

# Float pointers
template fp(v: Vector64): ptr float64 = cast[ptr float64](addr(v[]))

template fp(m: Matrix64): ptr float64 = cast[ptr float64](addr(m.data[]))