import times, linalg

proc main() =
  let
    mat1 = randomMatrix(1000, 987)
    mat2 = randomMatrix(987, 876)
    mat3 = randomMatrix(4, 3)
    mat4 = randomMatrix(4, 3)
    v1 = randomVector(10)

  let startTime = epochTime()
  for i in 0 .. < 10:
    discard mat1 * mat2
  let endTime = epochTime()
  echo "We have required ", endTime - startTime, " seconds to multiply matrices 10 times."

  for c in columns(mat3):
    echo c

  echo "v1 = ", v1
  echo "2 * v1 = ", 2 * v1
  echo "id * v1 = ", eye(10) * v1
  echo "l_1(v1) = ", l_1(v1)
  echo "l_2(v1) = ", l_2(v1)
  echo "l_2(v1)^2 = ", v1 * v1

  echo mat3 * mat4.t
  echo mat3.t * mat4

when isMainModule:
  main()
  GC_fullCollect()