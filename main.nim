import times, linalg

proc main() =
  let
    v1 = randomVector(10, max=1'f32)
    v2 = randomVector(10, max=1'f32)
  var
    p1 = cudaMalloc(10 * sizeof(ptr float32))
    p2 = cudaMalloc(10 * sizeof(ptr float32))
    handle = cublasCreate()
  var stat = cublasSetVector(10, sizeof(float32), addr(v1[]), 1, p1, 1)
  stat = cublasSetVector(10, sizeof(float32), addr(v2[]), 1, p2, 1)
  stat = cublasSaxpy(handle, 10, 1.0, p1, p2)
  var v3 = zeros(10, float32)
  stat = cublasGetVector(10, sizeof(float32), p2, 1, addr(v3[]), 1)

  let
    q1 = v1.gpu()
    q2 = q1.cpu()

  echo "v3 = ", v3
  echo "v1 + v2 = ", v1 + v2
  echo "v1 = ", v1
  echo "q2 = ", q2


when isMainModule:
  main()
  GC_fullCollect()