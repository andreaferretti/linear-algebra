#Linear Algebra for Nim

This library is meant to provide basic linear algebra operations for Nim
applications. The ambition would be to become a stable basis on which to
develop a scientific ecosystem for Nim, much like Numpy does for Python.

The library has been tested on Ubuntu Linux 14.10 through 15.10 64-bit using
either ATLAS, OpenBlas or Intel MKL. The GPU support has been tested using
NVIDIA CUDA 7.0 and 7.5.

API documentation is [here](http://unicredit.github.io/linear-algebra/api.html)

Table of contents
-----------------
<!-- TOC depthFrom:2 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Introduction](#introduction)
- [Initialization](#initialization)
- [Accessors](#accessors)
- [Iterators](#iterators)
- [Equality](#equality)
- [Pretty-print](#pretty-print)
- [Operations](#operations)
- [Trivial operations](#trivial-operations)
- [Universal functions](#universal-functions)
- [Rewrite rules](#rewrite-rules)
- [Type safety guarantees](#type-safety-guarantees)
- [Linking BLAS implementations](#linking-blas-implementations)
- [GPU support](#gpu-support)
- [TODO](#todo)
- [Contributing](#contributing)

<!-- /TOC -->

##Introduction

The library revolves around operations on vectors and matrices of floating
point numbers. It allows to compute operations either on the CPU or on the
GPU offering identical APIs. Also, whenever possible, the dimension of vectors
and matrices are encoded into the types in the form of `static[int]` type
parameters. This allow to check dimensions at compile time and refuse to
compile invalid operations, such as summing two vectors of different sizes,
or multiplying two matrices of incompatible dimensions.

The library defines types `Matrix64[M, N]` and `Vector64[N]` for 64-bit matrices
and vectors of static dimension, as well as their 32-bit counterparts
`Matrix32[M, N]` and `Vector32[N]`.

For the case where the dimension is not known at compile time, one can use
so-called *dynamic* vectors and matrices, whose types are `DVector64` and
`DMatrix64` (resp. `DVector32` and `DMatrix32`). Note that `DVector64` is just
and alias for `seq[float64]` (and similarly for 32-bit), which allows to
perform linear algebra operations on standard Nim sequences.

In all examples, types are inferred, and are shown just for explanatory purposes.

##Initialization

Here we show a few ways to create matrices and vectors. All matrices methods
accept a parameter to define whether to store the matrix in row-major (that is,
data are laid out in memory row by row) or column-major order (that is, data
are laid out in memory column by column). The default is in each case
column-major.

Whenever possible, we try to deduce whether to use 32 or 64 bits by appropriate
parameters. When this is not possible, there is an optional parameter `float32`
that can be passed to specify the precision (the default is 64 bit).

Static matrices and vectors can be created like this:

```nim
import linalg

let
  v1: Vector64[5] = makeVector(5, proc(i: int): float64 = (i * i).float64)
  v2: Vector64[7] = randomVector(7, max = 3.0) # max is optional, default 1
  v3: Vector64[5] = constantVector(5, 3.5)
  v4: Vector64[8] = zeros(8)
  v5: Vector64[9] = ones(9)
  v6: Vector64[5] = vector([1.0, 2.0, 3.0, 4.0, 5.0]) # initialize from an array...
  m1: Matrix64[6, 3] = makeMatrix(6, 3, proc(i, j: int): float64 = (i + j).float64)
  m2: Matrix64[2, 8] = randomMatrix(2, 8, max = 1.6) # max is optional, default 1
  m3: Matrix64[3, 5] = constantMatrix(3, 5, 1.8, order = rowMajor) # order is optional, default colMajor
  m4: Matrix64[3, 6] = ones(3, 6)
  m5: Matrix64[5, 2] = zeros(5, 2)
  m6: Matrix64[7, 7] = eye(7)
  m7: Matrix64[2, 3] = matrix([
    [1.2, 3.5, 4.3],
    [1.1, 4.2, 1.7]
  ])
  m8: Matrix64[2, 3] = matrix(@[
    @[1.2, 3.5, 4.3],
    @[1.1, 4.2, 1.7]
  ], 2, 3)
```

The last `matrix` constructor takes a `seq` of `seq`s, but also requires
statically passing the dimensions to be used. The following are equivalent
when `xs` is a `seq[seq[float64]]` and `M`, `N` are integers known at compile
time:

```nim
let
  m1 = matrix(xs).toStatic(M, N)
  m2 = matrix(xs, M, N)
```

but the latter form avoids the construction of an intermediate matrix.

All constructors that take as input an existing array or seq perform a copy of
the data for memory safety.

Dynamic matrices and vectors have similar constructors - the difference is that
the dimension parameter are not known at compile time:

```nim
import linalg

let
  M = 5
  N = 7
  v1: DVector64 = makeVector(M, proc(i: int): float64 = (i * i).float64)
  v2: DVector64 = randomVector(N, max = 3.0) # max is optional, default 1
  v3: DVector64 = constantVector(M, 3.5)
  v4: DVector64 = zeros(M)
  v5: DVector64 = ones(N)
  v6: DVector64 = @[1.0, 2.0, 3.0, 4.0, 5.0] # DVectors are just seqs...
  m1: DMatrix64 = makeMatrix(M, N, proc(i, j: int): float64 = (i + j).float64)
  m2: DMatrix64 = randomMatrix(M, N, max = 1.6) # max is optional, default 1
  m3: DMatrix64 = constantMatrix(M, N, 1.8, order = rowMajor) # order is optional, default colMajor
  m4: DMatrix64 = ones(M, N)
  m5: DMatrix64 = zeros(M, N)
  m6: DMatrix64 = eye(M)
  m7: DMatrix64 = matrix(@[
    @[1.2, 3.5, 4.3],
    @[1.1, 4.2, 1.7]
  ])
```

If for some reason you want to create a dynamic vector of matrix, but you want
to write literal dimensions, you can either assign these numbers to variables
or use the `toDynamic` proc to convert the static instances to dynamic ones:

```nim
import linalg

let
  M = 5
  v1 = makeVector(M, proc(i: int): float64 = (i * i).float64)
  v2 = makeVector(5, proc(i: int): float64 = (i * i).float64)

assert v1.toStatic(5) == v2
assert v2.toDynamic == v1
```

Working with 32-bit
-------------------

One can also instantiate 32-bit matrices and vectors. Whenever possible, the
API is identical. In cases that would be ambiguous (such as `zeros`), one can
explicitly specify the `float32` parameter.

```nim
import linalg

let
  v1: Vector32[5] = makeVector(5, proc(i: int): float32 = (i * i).float32)
  v2: Vector32[7] = randomVector(7, max = 3'f32) # max is no longer optional, to distinguish 32/64 bit
  v3: Vector32[5] = constantVector(5, 3.5'f32)
  v4: Vector32[8] = zeros(8, float32)
  v5: Vector32[9] = ones(9, float32)
  v6: Vector32[5] = vector([1'f32, 2'f32, 3'f32, 4'f32, 5'f32])
  m1: Matrix32[6, 3] = makeMatrix(6, 3, proc(i, j: int): float32 = (i + j).float32)
  m2: Matrix32[2, 8] = randomMatrix(2, 8, max = 1.6'f32)
  m3: Matrix32[3, 5] = constantMatrix(3, 5, 1.8'f32, order = rowMajor) # order is optional, default colMajor
  m4: Matrix32[3, 6] = ones(3, 6, float32)
  m5: Matrix32[5, 2] = zeros(5, 2, float32)
  m6: Matrix32[7, 7] = eye(7, float32)
  m7: Matrix32[2, 3] = matrix([
    [1.2'f32, 3.5'f32, 4.3'f32],
    [1.1'f32, 4.2'f32, 1.7'f32]
  ])
  m8: Matrix32[2, 3] = matrix(@[
    @[1.2'f32, 3.5'f32, 4.3'f32],
    @[1.1'f32, 4.2'f32, 1.7'f32]
  ], 2, 3)
```

Similarly,

```nim
import linalg

let
  M = 5
  N = 7
  v1: DVector32 = makeVector(M, proc(i: int): float32 = (i * i).float32)
  v2: DVector32 = randomVector(N, max = 3'f32) # max is not optional
  v3: DVector32 = constantVector(M, 3.5'f32)
  v4: DVector32 = zeros(M, float32)
  v5: DVector32 = ones(N, float32)
  v6: DVector32 = @[1'f32, 2'f32, 3'f32, 4'f32, 5'f32] # DVectors are just seqs...
  m1: DMatrix32 = makeMatrix(M, N, proc(i, j: int): float32 = (i + j).float32)
  m2: DMatrix32 = randomMatrix(M, N, max = 1.6'f32) # max is not optional
  m3: DMatrix32 = constantMatrix(M, N, 1.8'f32, order = rowMajor) # order is optional, default colMajor
  m4: DMatrix32 = ones(M, N, float32)
  m5: DMatrix32 = zeros(M, N, float32)
  m6: DMatrix32 = eye(M, float32)
  m7: DMatrix32 = matrix(@[
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
  m64: Matrix64[3, 8] = m32.to64()
```

Once vectors and matrices are created, everything is inferred, so there are no
differences in working with 32-bit or 64-bit. All examples that follow are for
64-bit, but they would work as well for 32-bit.

##Accessors

Vectors can be accessed as expected:

```nim
var v = randomVector(6)
v[4] = 1.2
echo v[3]
```

Same for matrices, where `m[i, j]` denotes the item on row `i` and column `j`,
regardless of the matrix order:

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

For memory safety, this performs a **copy** of the row or column values, at
least for now. One can also map vectors and matrices via a proc:

```nim
let
  v1 = v.map(proc(x: float64): float64 = 2 - 3 * x)
  m1 = m.map(proc(x: float64): float64 = 1 / x)
```

Similar operations are available for dynamic matrices and vectors as well.

##Iterators

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

##Equality

There are two kinds of equality. The usual `==` operator will compare the
contents of vector and matrices exactly

```nim
let
  u = vector([1.0, 2.0, 3.0, 4.0])
  v = vector([1.0, 2.0, 3.0, 4.0])
  w = vector([1.0, 3.0, 3.0, 4.0])
u == v # true
u == w # false
```

Usually, though, one wants to take into account the errors introduced by
floating point operations. To do this, use the `=~` operator, or its
negation `!=~`:

```nim
let
  u = vector([1.0, 2.0, 3.0, 4.0])
  v = vector([1.0, 2.000000001, 2.99999999, 4.0])
u == v # false
u =~ v # true
```

##Pretty-print

Both vectors and matrix have a pretty-print operation, so one can do

```nim
let m = randomMatrix(3, 7)
echo m8
```

and get something like

    [ [ 0.5024584865674662  0.0798945419892334  0.7512423051567048  0.9119041361916302  0.5868388894943912  0.3600554448403415  0.4419034543022882 ]
      [ 0.8225964245706265  0.01608615513584155 0.1442007939324697  0.7623388321096165  0.8419745686508193  0.08792951865247645 0.2902529012579151 ]
      [ 0.8488187232786935  0.422866666087792 0.1057975175658363  0.07968277822379832 0.7526946339452074  0.7698915909784674  0.02831893268471575 ] ]

##Operations

A few linear algebra operations are available, wrapping BLAS libraries:

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

##Trivial operations

The following operations do not change the underlying memory layout of matrices
and vectors. This means they run in very little time even on big matrices, but
you have to pay attention when mutating matrices and vectors produced in this
way, since the underlying data is shared.

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

In case you need to allocate a copy of the original data, say in order to
transpose a matrix and then mutate the transpose without altering the original
matrix, a `clone` operation is available:

```nim
let m5 = m1.clone
```

##Universal functions

Universal functions are real-valued functions that are extended to vectors
and matrices by working element-wise. There are many common functions that are
implemented as universal functions:

```nim
sqrt
cbrt
log10
log2
log
exp
arccos
arcsin
arctan
cos
cosh
sin
sinh
tan
tanh
erf
erfc
lgamma
tgamma
trunc
floor
ceil
degToRad
radToDeg
```

This means that, for instance, the following check passes:

```nim
  let
    v1 = vector([1.0, 2.3, 4.5, 3.2, 5.4])
    v2 = log(v1)
    v3 = v1.map(log)

  assert v2 == v3
```

Universal functions work both on 32 and 64 bit precision, on vectors and
matrices, both static and dynamic.

If you have a function `f` of type `proc(x: float64): float64` you can use

```nim
makeUniversal(f)
```

to turn `f` into a (public) universal function. If you do not want to export
`f`, there is the equivalent template `makeUniversalLocal`.

##Rewrite rules

A few rewrite rules allow to optimize a chain of linear algebra operations
into a single BLAS call. For instance, if you try

```nim
echo v1 + 5.3 * v2
```

this is not implemented as a scalar multiplication followed by a sum, but it
is turned into a single function call.

##Type safety guarantees

The library is designed with the use case of having dimensions known at compile
time, and leverages the compiles to ensure that dimensions match when performing
the appropriate operations - for instance in matrix multiplication.

To see some examples where the compiler avoids malformed operations, look
inside `tests/compilation` (yes, in Nim one can actually test that some
operations do not compile!).

Also, operations that mutate a vector of matrix in place are only available if
that vector of matrix is defined via `var` instead of `let`.

##Linking BLAS implementations

The library requires to link some BLAS implementation to perform the actual
linear algebra operations. By default, it tries to link whatever is the default
system-wide BLAS implementation.

A few compile flags are available to link specific BLAS implementations

    -d:atlas
    -d:openblas
    -d:mkl
    -d:mkl -d:threaded

##GPU support

It is possible to delegate work to the GPU using CUDA. The library has been
tested to work with NVIDIA CUDA 7.0 and 7.5, but it is possible that earlier
versions will work as well. In order to compile and link against CUDA, you
should make the appropriate headers and libraries available. If they are not
globally set, you can pass suitable options to the Nim compiler, such as

    --cincludes:"/usr/local/cuda/targets/x86_64-linux/include" \
    --clibdir:"/usr/local/cuda/targets/x86_64-linux/lib"

You will also need to explicitly add `linalg` support for CUDA with the flag

    -d:cublas

Support is currently limited to 32-bit operations on static matrices and
vectors, which is the most common case, but 64-bit and dynamic instances will
also be implemented soon.

If you have a 32-bit matrix or vector, you can move it on the GPU, and back
like this

```nim
let
  v: Vector32[12] = randomVector(12, max=1'f32)
  vOnTheGpu: CudaVector[12] = v.gpu()
  vBackOnTheCpu: Vector32[12] = vOnTheGpu.cpu()
```

Vectors and matrices on the GPU support linear-algebraic operations via cuBLAS,
exactly like their CPU counterparts. A few operation - such as reading a single
element - are not supported, as it does not make much sense to copy a single
value back and forth from the GPU. Usually it is advisable to move vectors
and matrices to the GPU, make as man computations as possible there, and
finally move the result back to the CPU.

The following are all valid operations, assuming `v` and `w` are vectors on the
GPU, `m` and `n` are matrices on the GPU and the dimensions are compatible:

```nim
v * 3'f32
v + w
v -= w
m * v
m - n
m * n
```

For more information, look at the tests in `tests/cuda`.

##TODO

* Add support for matrices and vector on the stack
* Use rewrite rules to optimize complex operations into a single BLAS call
* 64-bit and dynamic GPU support
* More specialized BLAS operations
* Add operations from LAPACK
* Support slicing/nonconstant steps
* Make `row` and `column` operations non-copying
* Better types to avoid out of bounds exceptions when statically checkable
* Add a fallback Nim implementation, that is valid over other rings
* Try on more platforms/configurations
* Make a proper benchmark
* Improve documentation
* Better pretty-print

##Contributing

Every contribution is very much appreciated! This can range from:

* using the library and reporting any issues and any configuration on which
  it works fine
* building other parts of the scientific environment on top of it
* writing blog posts and tutorials
* contributing actual code (see the **TODO** section)

The library has to cater many different use cases, hence the vector and matrix
types differ in various axes:

* 32/64 bit
* CPU/GPU
* static/dynamic
* (on the stack? non-contiguous? non GC pointers?)

In order to avoid a combinatorial explosion of operations, a judicious use of
templates and union types helps to limit the actual implementations that have
to be written.