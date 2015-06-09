import times, linalg

proc main() =
  let v = constant(4, 5.6)

  echo v

when isMainModule:
  main()
  GC_fullCollect()