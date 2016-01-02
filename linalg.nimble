[Package]
name          = "linalg"
version       = "0.2.1"
author        = "Andrea Ferretti"
description   = "Linear Algebra for Nim"
license       = "Apache2"
SkipDirs      = "tests,bench"
SkipFiles     = "test,test_cuda,benchmark,main,main.nim,doc"

[Deps]
Requires: "nim >= 0.11.2"