when defined(mkl):
  const header = "mkl.h"
  when defined(threaded):
    {. passl: "-lmkl_intel_lp64" passl: "-lmkl_core" passl: "-lmkl_gnu_thread" passl: "-lgomp" .}
  # {. passl: "-lmkl_intel_lp64" passl: "-lmkl_core" passl: "-lmkl_intel_thread" passl: "-lmpi" .}
    static: echo "--USING MKL THREADED--"
  else:
    {.passl: "-lmkl_intel_lp64" passl: "-lmkl_core" passl: "-lmkl_sequential" passl: "-lpthread" .}
    static: echo "--USING MKL SEQUENTIAL--"
else:
  when defined(atlas):
    {.passl: "-lcblas".}
    const header = "atlas/cblas.h"
    static: echo "--USING ATLAS--"
  else:
    {.passl: "-lblas".}
    const header = "cblas.h"
    static: echo "--USING DEFAULT BLAS--"