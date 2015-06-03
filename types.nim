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

template fp(v: Vector64): ptr float64 = cast[ptr float64](v.p)

template fp(m: Matrix64): ptr float64 = cast[ptr float64](m.p)

proc finalizeVector64[N: static[int]](v: Vector64[N]) {. nimcall .} =
  if v.p != nil:
    la_free(v.fp)
    v.p = nil

proc finalizeMatrix64[M, N: static[int]](m: Matrix64[M, N]) {. nimcall .} =
  echo "finalizing ", M, "x", N
  if m.p != nil:
    la_free(m.fp)
    m.p = nil