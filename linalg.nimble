mode = ScriptMode.Verbose

packageName   = "linalg"
version       = "0.4.0"
author        = "Andrea Ferretti"
description   = "Linear Algebra for Nim"
license       = "Apache2"
skipDirs      = @["tests", "bench"]
skipFiles     = @["linalg.html"]

requires "nim >= 0.11.2", "nimblas >= 0.1.1"

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

task test, "run standard tests":
  configForTests()
  setCommand "c", "tests/all"

task testopenblas, "run standard tests on openblas":
  configForTests()
  --define: openblas
  setCommand "c", "tests/all"

task testmkl, "run standard tests on mkl":
  configForTests()
  --dynlibOverride:mkl_intel_lp64
  --passL:"/home/papillon/.intel/mkl/lib/intel64/libmkl_intel_lp64.a"
  --define: mkl
  setCommand "c", "tests/all"

task testcuda, "run tests for the cuda implementation":
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