mode = ScriptMode.Verbose

packageName   = "linalg"
version       = "0.6.6"
author        = "Andrea Ferretti"
description   = "Linear Algebra for Nim"
license       = "Apache2"
skipDirs      = @["tests", "bench"]
skipFiles     = @["linalg.html"]

requires "nim >= 0.13.1", "nimblas >= 0.1.3"

--forceBuild

when defined(nimdistros):
  import distros
  if detectOs(Ubuntu) or detectOs(Debian):
    foreignDep "libblas-dev"
    foreignDep "libopenblas-dev"
    foreignDep "liblapack-dev"
  else:
    foreignDep "libblas"
    foreignDep "liblapack"

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
  setCommand "c", "tests/all.nim"

task testopenblas, "run standard tests on openblas":
  configForTests()
  --define: openblas
  setCommand "c", "tests/all.nim"

task testmkl, "run standard tests on mkl":
  configForTests()
  --dynlibOverride:mkl_intel_lp64
  --passL:"/home/papillon/.intel/mkl/lib/intel64/libmkl_intel_lp64.a"
  --define: mkl
  setCommand "c", "tests/all.nim"

task testcuda, "run tests for the cuda implementation":
  configForTests()
  configForCuda()
  setCommand "c", "tests/cublas.nim"

task bench, "run standard benchmarks":
  configForBenchmarks()
  setCommand "c", "bench/matrix_matrix_mult.nim"

task benchcuda, "run benchmarks for the cuda implementation":
  configForBenchmarks()
  configForCuda()
  setCommand "c", "bench/cuda/matrix_vector_mult.nim"

task gendoc, "generate documentation":
  --define: cublas
  --docSeeSrcUrl: https://github.com/andreaferretti/linear-algebra/blob/master
  setCommand "doc2", "linalg.nim"
