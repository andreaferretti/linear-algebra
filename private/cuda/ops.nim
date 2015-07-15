let handle {.global.} = cublasCreate()

proc gpu*[N: static[int]](v: Vector32[N]): CudaVector[N] =
  new result
  result[] = cudaMalloc(N * sizeof(float32))
  let stat = cublasSetVector(N, sizeof(float32), v.fp, 1, result[], 1)
  if stat != cublasStatusSuccess:
    quit($(stat))

proc cpu*[N: static[int]](v: CudaVector[N]): Vector32[N] =
  new result
  let stat = cublasGetVector(N, sizeof(float32), v[], 1, result.fp, 1)
  if stat != cublasStatusSuccess:
    quit($(stat))

proc `+=`*[N: static[int]](v: var CudaVector[N], w: CudaVector[N]) {. inline .} =
  let stat = cublasSaxpy(handle, N, 1, w[], v[])
  if stat != cublasStatusSuccess:
    quit($(stat))