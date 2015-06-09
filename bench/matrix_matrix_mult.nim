import times, linalg

proc main() =
  let
    m1 = randomMatrix(1000, 987)
    m2 = randomMatrix(987, 876)
    startTime = epochTime()
  
  for i in 0 .. < 10:
    discard m1 * m2
  let endTime = epochTime()

  echo "We have required ", endTime - startTime, " seconds to multiply matrices 10 times."

when isMainModule:
  main()