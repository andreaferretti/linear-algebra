type
  Vect32*[N: static[int]] = array[N, float32]
  Vect64*[N: static[int]] = array[N, float64]
  Vect*[N: static[int]] = Vect64[N]
  Matrix32*[M, N: static[int]] = array[N, array[M, float32]]
  Matrix64*[M, N: static[int]] = array[N, array[M, float64]]
  Matrix*[M, N: static[int]] = ref object
    p: ptr float64