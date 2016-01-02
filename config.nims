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
  --path: "."
  --run

proc configForBenchmarks() =
  --define: release
  --path: "."
  --run

proc configForCuda() =
  switch("cincludes", "/usr/local/cuda/targets/x86_64-linux/include")
  switch("clibdir", "/usr/local/cuda/targets/x86_64-linux/lib")
  --define: cublas

task tests, "run standard tests":
  configForTests()
  setCommand "c", "tests/all"

task testscuda, "run tests for the cuda implementation":
  configForTests()
  configForCuda()
  setCommand "c", "tests/cublas"

task bench, "run standard benchmarks":
  configForBenchmarks()
  setCommand "c", "bench/matrix_matrix_mult"

task benchcuda, "run benchmarks for the cuda implementation":
  configForBenchmarks()
  configForCuda()
  setCommand "c", "bench/cuda/matrix_vector_mult"

task gendoc, "generate documentation":
  --define: cublas
  --docSeeSrcUrl: https://github.com/unicredit/linear-algebra/blob/master
  setCommand "doc2", "linalg"