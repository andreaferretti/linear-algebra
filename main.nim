import times, linalg

proc main() =
  let v = randomVector(10)

  # var p: ptr ptr float32
  var p = cudaMalloc(10 * sizeof(ptr float32))
  var h = cublasCreate()
  # let stat = cublasSetVector(10, sizeof(float64), addr(v[]), 1, p, 1)
  # echo stat

when isMainModule:
  main()
  GC_fullCollect()