import times, linalg

proc main() =
  let
    mat1 = randomMatrix(1000, 987)
    mat2 = randomMatrix(987, 876)
    mat3 = randomMatrix(3, 3)
    mat4 = dmatrix(3, 3, @[
      @[1.0, 2.0, 3.0],
      @[1.2, 2.1, 3.1],
      @[2.3, 4.5, 3.2]
    ])
    v1 = vector([1.2, 3.5, 4.3])
    v2 = dvector(3, @[1.2, 3.5, 4.3])


  let startTime = epochTime()
  for i in 0 .. < 10:
    discard mat1 * mat2
  let endTime = epochTime()
  echo "We have required ", endTime - startTime, " seconds to multiply matrices 10 times."

  echo "mat4 * v1 = ", mat4 * v1
  echo "v2 * v1 = ", v2 * v1

when isMainModule:
  main()
  GC_fullCollect()