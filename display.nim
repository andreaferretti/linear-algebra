proc `$`*[N: static[int]](v: Vector64[N]): string =
  result = "[ "
  for i in 0 .. < N - 1:
    result &= $(v[i]) & "\n  "
  result &= $(v[N - 1]) & " ]"

proc toStringHorizontal[N: static[int]](v: Vector64[N]): string =
  result = "[ "
  for i in 0 .. < N - 1:
    result &= $(v[i]) & "\t"
  result &= $(v[N - 1]) & " ]"

proc `$`*[M, N: static[int]](m: Matrix64[M, N]): string =
  result = "[ "
  for i in 0 .. < M - 1:
    result &= toStringHorizontal(m.row(i)) & "\n  "
  result &= toStringHorizontal(m.row(M - 1)) & " ]"