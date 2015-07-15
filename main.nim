import times, linalg

proc main() =
  let
    v1 = randomVector(10, max=1'f32)
    v2 = randomVector(10, max=1'f32)

  let
    p1 = v1.gpu()
    p2 = v2.gpu()
    p3 = p1 + p2
    v3 = p3.cpu()

  echo "v1 + v2 = ", v1 + v2
  echo "p1 + p2 = ", v3


when isMainModule:
  main()
  GC_fullCollect()