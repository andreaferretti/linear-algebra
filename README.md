Nim Linear Algebra
==================

The library defines types `Matrix64[M, N]` and `Vector64[N]` and related operations.

In all examples, types are inferred, and are shown just for explanatory purposes.

The library has been tested on Ubuntu Linux 14.10 and 15.04 64-bit with ATLAS, OpenBlas and Intel MKL.

The GPU support has been tested using NVIDIA CUDA 7.0.

API documentation is [here](http://unicredit.github.io/linear-algebra/)

Initialization
--------------

Here we show a few ways to initialize matrices and vectors. All matrices method accept a parameter to
define whether to store the matrix in row-major or column-major order (default: column-major).

```nim
import linalg

let
  v1: Vector64[5] = makeVector(5, proc(i: int): float64 = (i * i).float64)
  v2: Vector64[7] = randomVector(7, max = 3) # max is optional, default 1
  v3: Vector64[5] = constantVector(5, 3.5)
  v4: Vector64[8] = zeros(8)
  v5: Vector64[9] = ones(9)
  v6: Vector64[5] = vector([1.0, 2.0, 3.0, 4.0, 5.0]) # initialize from an array...
  v7: Vector64[4] = dvector(4, @[1.0, 2.0, 3.0, 4.0]) # ...or from a seq
  m1: Matrix64[6, 3] = makeMatrix(6, 3, proc(i, j: int): float64 = (i + j).float64)
  m2: Matrix64[2, 8] = randomMatrix(2, 8, max = 1.6) # max is optional, default 1
  m3: Matrix64[3, 5] = constantMatrix(3, 5, 1.8, order = rowMajor) # order is optional, default colMajor
  m4: Matrix64[3, 6] = ones(3, 6)
  m5: Matrix64[5, 2] = zeros(5, 2)
  m6: Matrix64[7, 7] = eye(7)
  m7: Matrix64[2, 3] = dmatrix(2, 3, @[
    @[1.2, 3.5, 4.3],
    @[1.1, 4.2, 1.7]
  ])
```

For some reason that has to do with type inference, there is no `matrix` constructor that
takes an array of arrays. The `dmatrix` constructor (d stand for dynamic) that requires
statically passing the dimensions has to be used. All constructors that take as input an
existing array or seq (such as `vector`, `dvector` and `dmatrix`) perform a copy of the data
for memory safety.

Working with 32-bit vectors and matrices
----------------------------------------

One can also instantiate 32-bit matrices and vectors. Examples are given below

```nim
import linalg

let
  v1: Vector32[5] = makeVector(5, proc(i: int): float32 = (i * i).float32)
  v2: Vector32[7] = randomVector(7, max = 3'f32) # max is no longer optional, to distinguis 32/64 bit
  v3: Vector32[5] = constantVector(5, 3.5'f32)
  v4: Vector32[8] = zeros(8, float32)
  v5: Vector32[9] = ones(9, float32)
  v6: Vector32[5] = vector([1'f32, 2'f32, 3'f32, 4'f32, 5'f32], float32) # unfortunately, here float32 is required
  v7: Vector32[4] = dvector(4, @[1'f32, 2'f32, 3'f32, 4'f32]) # but not required here
  m1: Matrix32[6, 3] = makeMatrix(6, 3, proc(i, j: int): float32 = (i + j).float32)
  m2: Matrix32[2, 8] = randomMatrix(2, 8, max = 1.6'f32)
  m3: Matrix32[3, 5] = constantMatrix(3, 5, 1.8'f32, order = rowMajor) # order is optional, default colMajor
  m4: Matrix32[3, 6] = ones(3, 6, float32)
  m5: Matrix32[5, 2] = zeros(5, 2, float32)
  m6: Matrix32[7, 7] = eye(7, float32)
  m7: Matrix32[2, 3] = dmatrix(2, 3, @[
    @[1.2'f32, 3.5'f32, 4.3'f32],
    @[1.1'f32, 4.2'f32, 1.7'f32]
  ])
```

One can convert precision with `to32` or `to64`:

```nim
let
  v64: Vector64[10] = randomVector(10)
  v32: Vector32[10] = v64.to32()
  m32: Matrix32[3, 8] = randomMatrix(3, 8, max = 1'f32)
  m64: Matrix64[3, 6] = m32.to64()
```

Once vectors and matrices are created, everything is inferred, so there are no differences in working
with 32-bit or 64-bit. All examples that follow are for 64-bit, but they would work as well for 32-bit.

Accessors
---------

Vectors can be accessed as expected

```nim
var v = randomVector(6)
v[4] = 1.2
echo v[3]
```

Same for matrices, where `m[i, j]` denotes the item on row `i` and column `j`, regardless of the matrix order

```nim
var m = randomMatrix(3, 7)
m[1, 3] = 0.8
echo m[2, 2]
```

Also one can see rows and columns as vectors

```nim
let
  r2: Vector64[7] = m.row(2)
  c5: Vector64[3] = m.column(5)
```

For memory safety, this performs a **copy** of the row or column values, at least for now. One can also map vectors and matrices via a proc:

```nim
let
  v1 = v.map(proc(x: float64): float64 = 2 - 3 * x)
  m1 = m.map(proc(x: float64): float64 = 1 / x)
```

Iterators
---------

One can iterate over vector or matrix elements, as well as over rows and columns

```nim
let
  v = randomVector(6)
  m = randomMatrix(3, 5)
for x in v: echo x
for i, x in v: echo i, x
for x in m: echo x
for t, x in m:
  let (i, j) = t
  echo i, j, x
for row in m.rows:
  echo row[0]
for column in m.columns:
  echo column[1]
```

Equality
--------

There are two kinds of equality. The usual `==` operator will compare the contents of vector and matrices exactly

```nim
let
  u = vector([1.0, 2.0, 3.0, 4.0])
  v = vector([1.0, 2.0, 3.0, 4.0])
  w = vector([1.0, 3.0, 3.0, 4.0])
u == v # true
u == w # false
```

Usually, though, one wants to take into account the errors introduced by floating point operations. To do this,
use the `=~` operator, or its negation `!=~`:

    let
      u = vector([1.0, 2.0, 3.0, 4.0])
      v = vector([1.0, 2.000000001, 2.99999999, 4.0])
    u == v # false
    u =~ v # true

Pretty-print
------------

Both vectors and matrix have a pretty-print operation, so one can do

```nim
let m = randomMatrix(3, 7)
echo m8
```

and get something like

    [ [ 0.5024584865674662  0.0798945419892334  0.7512423051567048  0.9119041361916302  0.5868388894943912  0.3600554448403415  0.4419034543022882 ]
      [ 0.8225964245706265  0.01608615513584155 0.1442007939324697  0.7623388321096165  0.8419745686508193  0.08792951865247645 0.2902529012579151 ]
      [ 0.8488187232786935  0.422866666087792 0.1057975175658363  0.07968277822379832 0.7526946339452074  0.7698915909784674  0.02831893268471575 ] ]

Operations
----------

A few linear algebra operations are available, wrapping BLAS:

```nim
var v1 = randomVector(7)
let
  v2 = randomVector(7)
  m1 = randomMatrix(6, 9)
  m2 = randomMatrix(9, 7)
echo 3.5 * v1
v1 *= 2.3
echo v1 + v2
echo v1 - v2
echo v1 * v2 # dot product
echo l_1(v1) # l_1 norm
echo l_2(v1) # l_2 norm
echo m2 * v1 # matrix-vector product
echo m1 * m2 # matrix-matrix product
echo max(m1)
echo min(v2)
```

Trivial operations
------------------

The following operations do not change the underlying memory layout of matrices and vectors.
This means they run in very little time even on big matrices, but you have to pay attention
when mutating matrices and vectors produced in this way, since the underyling data is shared.

```nim
let
  m1 = randomMatrix(6, 9)
  m2 = randomMatrix(9, 6)
  v1 = randomVector(9)
echo m1.t # transpose, done in constant time without copying
echo m1 + m2.t
let m3: Matrix64[9, 6] = m1.reshape(9, 6)
let m4: Matrix64[3, 3] = v1.asMatrix(3, 3)
let v2: Vector64[54] = m2.asVector
```

In case you need to allocate a copy of the original data, say in order to transpose a matrix
and then mutate the transpose without altering the original matrix, a `clone` operation is
available:

```nim
let m5 = m1.clone
```

Rewrite rules
-------------

A few rewrite rules allow to optimize a chain of linear algebra operations into a single BLAS call. For instance, if you try

```nim
echo v1 + 5.3 * v2
```

this is not implemented as a scalar multiplication followed by a sum, but it is turned into a single function call.

Type safety guarantees
----------------------

The library is designed with the use case of having dimensions known at compile time, and
leverages the compiles to ensure that dimensions match when performing the appropriate
operations - for instance in matrix multiplication.

To see some examples where the compiler avoids malformed operations, look inside `tests/compilation`
(yes, in Nim one can actually test that some operations do not compile!).

Support for matrices and vectors whose size is only known at runtime will be added, but is not
there yet.

Linking BLAS implementations
----------------------------

A few compile flags are available to link specific BLAS implementations

    -d:atlas
    -d:openblas
    -d:mkl
    -d:mkl -d:threaded

GPU support
-----------

It is possible to delegate work to the GPU using CUDA. The library has been tested to work with NVIDIA
CUDA 7.0, but it is possible that earlier versions work as well. In order to compile and link against
CUDA, you should make the appropriate headers and libraries available. If they are not globally set,
you can pass suitable options to the Nim compiler, such as

    --cincludes:"/usr/local/cuda-7.0/targets/x86_64-linux/include" \
    --clibdir:"/usr/local/cuda-7.0/targets/x86_64-linux/lib"

You will also need to explicitly add `linalg` support for CUDA with the flag

    -d:cublas

Support is currently limited to 32-bit operations, which is the most common case, but 64-bit will
also be implemented soon.

If you have a 32-bit matrix or vector, you can move it on the GPU, and back like this

```nim
let
  v: Vector32[12] = randomVector(12, max=1'f32)
  vOnTheGpu: CudaVector[12] = v.gpu()
  vBackOnTheCpu: Vector32[12] = vOnTheGpu.cpu()
```

Vectors and matrices on the GPU support linear-algebraic operations via cuBLAS, exactly like their
CPU counterparts. A few operation - such as reading a single element - are not supported, as it
does not make much sense to copy a single value back and forth from the GPU. Usually it is advisable
to move vectors and matrices to the GPU, make as man computations as possible there, and finally
move the result back to the CPU. The following are all valid operations, assuming `v` and `w` are
vectors on the GPU, and `m` and `n` are matrices on the GPU, and the dimensions are compatible:

```nim
v * 3'f32
v + w
v -= w
m * v
m - n
m * n
```

For more information, look at the tests in `tests/cuda`.

TODO
----

* Add support for matrices and vectors whose size is only known at runtime
* Add support for matrices and vector on the stack, since dimensions are known at compile time anyway
* Use rewrite rules to optimize complex operations into a single BLAS call
* 64-bit GPU support
* More specialized BLAS operations
* Add operations from LAPACK
* Support slicing/nonconstant steps
* Make `row` and `column` operations non-copying
* Better types to avoid out of bounds exceptions when statically checkable
* Add a fallback Nim implementation of most operations, that is valid over other rings
* Try on more platforms/configurations
* Make a proper benchmark
* Improve documentation