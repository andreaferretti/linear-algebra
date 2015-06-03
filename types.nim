# I types are for internal use
type
  IVector32[N: static[int]] = object
    p: ptr array[N, float32]
  Vector32*[N: static[int]] = ref IVector32[N]
  IVector64[N: static[int]] = object
    p: ptr array[N, float64]
  Vector64*[N: static[int]] = ref IVector64[N]
  IMatrix32[M, N: static[int]] = object
    p: ptr array[N, array[M, float32]]
  Matrix32*[M, N: static[int]] = ref IMatrix32[M, N]
  IMatrix64[M, N: static[int]] = object
    p: ptr array[N, array[M, float64]]
  Matrix64*[M, N: static[int]] = ref IMatrix64[M, N]

proc finalizeVector64[N: static[int]](v: Vector64[N]) {. nimcall .} =
  if v.p != nil:
    echo "case != nil"
    free(v.p)
    v.p = nil
  else:
    echo "case == nil"

proc finalizeMatrix64[M, N: static[int]](m: Matrix64[M, N]) {. nimcall .} =
  echo "finalizing ", M, "x", N, "; pointer is ", cast[int](m.p)
  if m.p != nil:
    echo "case != nil"
    echo m.p[0][0]
    free(m.p)
    m.p = nil
  else:
    echo "case == nil"

proc `+`[A](p: ptr A, x: int): ptr A = cast[ptr[A]](cast[int](p) + x * sizeof(A))

template fp(m: Matrix64): ptr float64 = cast[ptr float64](m.p)