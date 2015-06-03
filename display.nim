proc `$`*(v: Vector64): string =
  result = "[ "
  for i in 0 .. < Vector64.N - 1:
    result &= $(v[i]) & "\n  "
  result &= $(v[Vector64.N - 1]) & " ]"

proc toStringHorizontal(v: Vector64): string =
  result = "[ "
  for i in 0 .. < Vector64.N - 1:
    result &= $(v[i]) & "\t"
  result &= $(v[Vector64.N - 1]) & " ]"

proc `$`*(m: Matrix64): string =
  result = "[ "
  for i in 0 .. < Matrix64.M - 1:
    result &= toStringHorizontal(m.row(i)) & "\n  "
  result &= toStringHorizontal(m.row(Matrix64.M - 1)) & " ]"