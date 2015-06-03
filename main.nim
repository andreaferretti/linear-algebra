import math, times, linalg

proc main() =

  let
    # xs = [1.0, 2.0, 3.5]
    # ys = [2.0, 3.0, 3.5]
    # m = [
    #   [1.2, 3.4],
    #   [1.1, 2.1],
    #   [0.6, -3.1]
    # ]
    mat1 = makeMatrix(1000, 987, proc(i, j: int): float64 = random(1.0))
    mat2 = makeMatrix(987, 876, proc(i, j: int): float64 = random(1.0))
    mat3 = makeMatrix(4, 4, proc(i, j: int): float64 = random(1.0))
    # vec= makeVect(987, proc(i: int): float64 = random(1.0))

  # echo mat3

  # echo(cast[ptr Matrix64[4, 4]](mat1.p)[])
  # echo(cast[ptr Matrix64[4, 4]](mat2.p)[])
  # echo(cast[ptr Matrix64[4, 4]]((mat1 * mat2).p)[])

  # let startTime = epochTime()
  # for i in 0 .. 100:
  #   discard mat1 * vec
  # let endTime = epochTime()
  # echo "We have required ", endTime - startTime, " seconds to do 100 multiplications."

  let startTime1 = epochTime()
  # for i in 0 .. < 10:
  #   discard mat1 * mat2
  let m4 = mat1 * mat2
  let endTime1 = epochTime()
  echo "We have required ", endTime1 - startTime1, " seconds to do multiply matrices 10 times."

  for c in columns(mat3):
    echo c

  echo mat3

  # echo((mat1 * vec)[1..10])
  # echo(xs * 5.3)
  # echo(5.3 * xs)
  # xs *= 5.3
  # echo(xs)
  # echo(xs * ys)
  # echo(xs * xs)
  # echo(l_2(xs))
  # echo(l_1(xs))
  # echo(m * ys)

  # template optMul{`*`(a, 2)}(a: int): int =
  #   echo "hi"
  #   a*3

  # let b = 55

  # echo b * 2

when isMainModule:
  main()
  GC_fullCollect()
  echo "done"