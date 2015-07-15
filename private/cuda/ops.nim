let handle {.global.} = cublasCreate()

template check(stat: cublasStatus): stmt =
  if stat != cublasStatusSuccess:
    quit($(stat))

proc gpu*[N: static[int]](v: Vector32[N]): CudaVector[N] =
  new result
  result[] = cudaMalloc(N * sizeof(float32))
  check cublasSetVector(N, sizeof(float32), v.fp, 1, result[], 1)

proc cpu*[N: static[int]](v: CudaVector[N]): Vector32[N] =
  new result
  check cublasGetVector(N, sizeof(float32), v[], 1, result.fp, 1)

proc `+=`*[N: static[int]](v: var CudaVector[N], w: CudaVector[N]) {. inline .} =
  check cublasSaxpy(handle, N, 1, w[], v[])

proc `+`*[N: static[int]](v, w: CudaVector[N]): CudaVector[N] {. inline .} =
  new result
  result[] = cudaMalloc(N * sizeof(float32))
  check cublasScopy(handle, N, v[], 1, result[], 1)
  check cublasSaxpy(handle, N, 1, w[], result[])