import times, linalg

proc main() =
  let
    mat3 = randomMatrix(3, 3)
    mat4 = dmatrix(3, 3, @[
      @[1.0, 2.0, 3.0],
      @[1.2, 2.1, 3.1],
      @[2.3, 4.5, 3.2]
    ])
    v1 = vector([1.2, 3.5, 4.3])
    v2 = dvector(3, @[1.2, 3.5, 4.3])

  echo "mat4 * v1 = ", mat4 * v1
  echo "v2 * v1 = ", v2 * v1

when isMainModule:
  main()
  GC_fullCollect()