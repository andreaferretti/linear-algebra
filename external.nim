when defined(mkl):
  const header = "mkl.h"
  when defined(threaded):
    {. passl: "-lmkl_intel_lp64" passl: "-lmkl_core" passl: "-lmkl_gnu_thread" passl: "-lgomp" passl: "-lm" .}
  # {. passl: "-lmkl_intel_lp64" passl: "-lmkl_core" passl: "-lmkl_intel_thread" passl: "-lmpi" .}
    static: echo "--USING MKL THREADED--"
  else:
    {.passl: "-lmkl_intel_lp64" passl: "-lmkl_core" passl: "-lmkl_sequential" passl: "-lpthread" passl: "-lm" .}
    static: echo "--USING MKL SEQUENTIAL--"

  proc mkl_malloc(size, align: int): ptr float64 {. header: header, importc: "mkl_malloc" .}
  proc free(p: ptr float64) {. header: header, importc: "mkl_free" .}
  template malloc(n: int): ptr float64 = mkl_malloc(n, 64)
else:
  when defined(atlas):
    {.passl: "-lcblas".}
    const header = "atlas/cblas.h"
    static: echo "--USING ATLAS--"
  else:
    {.passl: "-lblas".}
    const header = "cblas.h"
    static: echo "--USING DEFAULT BLAS--"
  proc malloc(size: int): ptr float64 {. header: "stdlib.h", importc: "malloc" .}
  proc free(p: ptr float64) {. header: "stdlib.h", importc: "free" .}
