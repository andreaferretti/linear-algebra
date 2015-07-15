proc cudaMalloc(size: int): ptr float32 =
  var error: cudaError
  {.emit: """error = cudaMalloc((void**)&`result`, `size`); """.}
  if error != cudaSuccess:
    quit($(error))

proc cublasCreate(): cublasHandle =
  var stat: cublasStatus
  {.emit: """stat = cublasCreate_v2(& `result`); """.}
  if stat != cublasStatusSuccess:
    quit($(stat))

proc cublasSetVector(n, elemSize: int, x: pointer, incx: int,
  devicePtr: pointer, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSetVector" .}

proc cublasGetVector(n, elemSize: int, devicePtr: pointer, incx: int,
  x: pointer, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasGetVector" .}

proc rawCublasSaxpy(handle: cublasHandle, n: int, alpha: ptr float32, x: ptr float32, incx: int,
  y: ptr float32, incy: int): cublasStatus
  {. header: "cublas_v2.h", importc: "cublasSaxpy" .}

proc cublasSaxpy(handle: cublasHandle, n: int, alpha: float32, x, y: ptr float32): cublasStatus =
  var al: ptr float32
  {.emit: """al = &alpha; """.}
  rawCublasSaxpy(handle, n, al, x, 1, y, 1)

# proc rawCudaMalloc(p: ptr ptr, size: int): cudaError
#   {. header: "cuda_runtime_api.h", importc: "cudaMalloc" .}

# proc rawCublasCreate(h: object): cublasStatus
#   {. header: "cublas_api.h", importc: "cublasCreate_v2" .}