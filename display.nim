proc `$`*(v: Vect64): string =
  let s = $(@(v))
  return s[1 .. high(s)]

proc `$`*(m: Matrix64): string = $(@(m))

proc `$`*(m: Matrix): string = $(cast[ptr Matrix64](m.p)[])