type
  Vect32*[N: static[int]] = array[N, float32]
  Vect64*[N: static[int]] = array[N, float64]
  Vect*[N: static[int]] = Vect64[N]
  Matrix32*[M, N: static[int]] = array[N, array[M, float32]]
  Matrix64*[M, N: static[int]] = array[N, array[M, float64]]
  Matrix*[M, N: static[int]] = ref tuple[p: ptr array[N, array[M, float64]]]

# Internal functions

# template asPtr[N: static[int]](v: Vect64[N]): ptr float64 = cast[ptr float64](v.addr)

# template asPtr[M, N: static[int]](a: Matrix64[M, N]): ptr float64 = cast[ptr float64](a.addr)

# template asPtr[M, N: static[int]](a: Matrix[M, N]): ptr float64 = cast[ptr float64](a.p)

proc `+`[A](p: ptr A, x: int): ptr A = cast[ptr[A]](cast[int](p) + x * sizeof(A))

template fp(m: Matrix): ptr float64 = cast[ptr float64](m.p)