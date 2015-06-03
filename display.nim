proc `$`*(v: Vector64): string =
  result = "[ "
  for i in 0 .. < Vector64.N - 1:
    result &= $(v[i]) & "\n  "
  result &= $(v[Vector64.N - 1]) & " ]"

proc `$`*(m: Matrix64): string = repr(m.p[])