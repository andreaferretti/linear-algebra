import times, linalg

proc main() =
  let
    v1 = randomVector(10, max=1'f32)
    v2 = randomVector(10, max=1'f32)
  var
    p1 = cudaMalloc(10 * sizeof(ptr float32))
    p2 = cudaMalloc(10 * sizeof(ptr float32))
    handler = cublasCreate()
  var stat = cublasSetVector(10, sizeof(float32), addr(v1[]), 1, p1, 1)
  echo stat
  stat = cublasSetVector(10, sizeof(float32), addr(v2[]), 1, p2, 1)
  echo stat


when isMainModule:
  main()
  GC_fullCollect()