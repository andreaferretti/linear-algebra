import math, times, linalg

proc main() =

  let
    mat1 = makeMatrix(1000, 987, proc(i, j: int): float64 = random(1.0))
    mat2 = makeMatrix(987, 876, proc(i, j: int): float64 = random(1.0))
    mat3 = makeMatrix(4, 4, proc(i, j: int): float64 = random(1.0))

  let startTime1 = epochTime()
  for i in 0 .. < 10:
    discard mat1 * mat2
  let endTime1 = epochTime()
  echo "We have required ", endTime1 - startTime1, " seconds to multiply matrices 10 times."

  for c in columns(mat3):
    echo c

  echo mat3

  let v1 = makeVector(10, proc(i: int): float64 = random(1.0))
  echo "v1 = ", v1
  echo "2 * v1 = ", 2 * v1
  echo "id * v1 = ", eye(10) * v1
  echo "l_1(v1) = ", l_1(v1)
  echo "l_2(v1) = ", l_2(v1)
  echo "l_2(v1)^2 = ", v1 * v1
  var mat4 = eye(4)
  mat4[1, 1] = 3.0
  echo mat4

when isMainModule:
  main()
  GC_fullCollect()
  echo "done"