mode = ScriptMode.Verbose

version       = "0.1.5"
author        = "Andrea Ferretti"
description   = "Linear Algebra for Nim"
license       = "Apache2"
skipDirs      = @["tests", "bench"]

requires "nim >= 0.11.2"

--forceBuild

proc configForTests() =
  --hints: off
  --linedir: on
  --stacktrace: on
  --linetrace: on
  --debuginfo
  --run

proc configForBenchmarks() =
  --define: "release"
  --run

task tests, "run standard tests":
  #exec "nim c -r tests/all"
  #path = "$projectPath/.."
  configForTests()
  setCommand "c", "tests/all"

task testscuda, "run tests for the cuda implementation":
  exec """nim c -d:cublas \
    --cincludes:/usr/local/cuda/targets/x86_64-linux/include \
    --clibdir:/usr/local/cuda/targets/x86_64-linux/lib \
    -r tests/cublas"""
  setCommand "nop"

task bench, "run standard benchmarks":
  #exec "nim c -r bench/matrix_matrix_mult"
  configForBenchmarks()
  setCommand "c", "bench/matrix_matrix_mult"

task benchcuda, "run benchmarks for the cuda implementation":
  exec """nim c -d:cublas \
    --cincludes:/usr/local/cuda/targets/x86_64-linux/include \
    --clibdir:/usr/local/cuda/targets/x86_64-linux/lib \
    -r bench/cuda/matrix_vector_mult"""
  setCommand "nop"

task gendoc, "generate documentation":
  #exec """nim doc2 -d:cublas \
  #  --docSeeSrcUrl:https://github.com/unicredit/linear-algebra/blob/master \
  #  linalg.nim"""
  --define: "cublas"
  --docSeeSrcUrl: "https://github.com/unicredit/linear-algebra/blob/master"
  setCommand "doc2", "linalg"