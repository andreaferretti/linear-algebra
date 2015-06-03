proc `$`*(v: Vector64): string = repr(v.p[])

proc `$`*(m: Matrix64): string = repr(m.p[])