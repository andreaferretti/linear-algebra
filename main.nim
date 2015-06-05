import times, linalg

proc main() =
  let
    mat1 = randomMatrix(1000, 987)
    mat2 = randomMatrix(987, 876)
    mat3 = randomMatrix(4, 3)
    mat4 = randomMatrix(4, 3)
    mat5 = randomMatrix(3, 3)
    v = randomVector(10)

  let startTime = epochTime()
  for i in 0 .. < 10:
    discard mat1 * mat2
  let endTime = epochTime()
  echo "We have required ", endTime - startTime, " seconds to multiply matrices 10 times."

  for c in columns(mat3):
    echo c

  echo "v = ", v
  echo "2 * v = ", 2 * v
  echo "v - v = ", v - v
  echo "id * v = ", eye(10) * v
  echo "l_1(v) = ", l_1(v)
  echo "l_2(v) = ", l_2(v)
  echo "l_2(v)^2 = ", v * v

  echo mat3 * mat4.t
  echo mat3.t * mat4
  echo mat3 + mat4
  echo mat5 + mat5.t
  echo mat5 - mat5.t

  let v2 = randomVector(10)
  let v3 = v + 3.4 * v2
  echo v3
  var v4 = randomVector(10)
  v4 += 2 * v2

when isMainModule:
  main()
  GC_fullCollect()