import times, linalg

proc main() =
  let
    mat1 = randomMatrix(1000, 987)
    mat2 = randomMatrix(987, 876)
    mat3 = randomMatrix(3, 3)
    v1 = randomVector(10)
    v2 = vector([1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 2.0, 5.0])

  let startTime = epochTime()
  for i in 0 .. < 10:
    discard mat1 * mat2
  let endTime = epochTime()
  echo "We have required ", endTime - startTime, " seconds to multiply matrices 10 times."

  echo "v1 = ", v1
  echo "v2 = ", v2
  echo "v1 * v2 = ", v1 * v2

when isMainModule:
  main()
  GC_fullCollect()