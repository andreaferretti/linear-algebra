import times, linalg

proc main() =
  let
    mat1 = randomMatrix(1000, 987)
    mat2 = randomMatrix(987, 876)
    mat3 = randomMatrix(3, 3)
    v = randomVector(10)

  let startTime = epochTime()
  for i in 0 .. < 10:
    discard mat1 * mat2
  let endTime = epochTime()
  echo "We have required ", endTime - startTime, " seconds to multiply matrices 10 times."

  echo v
  echo maxIndex(v)
  echo min(v)
  echo mat3
  echo min(mat3)

when isMainModule:
  main()
  GC_fullCollect()