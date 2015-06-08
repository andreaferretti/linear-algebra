proc linearCombination[N: static[int]](a: float64, v, w: Vector64[N]): Vector64[N]  {. inline .} =
  new result
  dcopy(N, v.fp, 1, result.fp, 1)
  daxpy(N, a, w.fp, 1, result.fp, 1)

proc linearCombinationMut[N: static[int]](a: float64, v: var Vector64[N], w: Vector64[N])  {. inline .} =
  daxpy(N, a, w.fp, 1, v.fp, 1)

template rewriteLinearCombination*{v + `*`(w, a)}(a: float64, v, w: Vector64): auto =
  linearCombination(a, v, w)

template rewriteLinearCombinationMut*{v += `*`(w, a)}(a: float64, v: var Vector64, w: Vector64): auto =
  linearCombinationMut(a, v, w)