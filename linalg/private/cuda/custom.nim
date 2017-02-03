proc hadamard(in1, in2, `out`: ptr float64, count: cint)
  {. cdecl, importc: "hadamard_d" , dynlib: "liblinalg.so" .}

proc hadamard(in1, in2, `out`: ptr float32, count: cint)
  {. cdecl, importc: "hadamard_s" , dynlib: "liblinalg.so" .}

proc `|*|`*(v, w: CudaDVector32): CudaDVector32 {. inline .} =
  assert(v.N == w.N)
  initDynamic(result, v.N)
  hadamard(v.fp, w.fp, result.fp, v.N.cint)

proc `|*|`*(v, w: CudaDVector64): CudaDVector64 {. inline .} =
  assert(v.N == w.N)
  initDynamic(result, v.N)
  hadamard(v.fp, w.fp, result.fp, v.N.cint)