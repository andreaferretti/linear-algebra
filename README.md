Nim Linear Algebra
==================

The library defines types `Matrix64[M, N]` and `Vector64[N]` and related operations.

In all examples, types are inferred, and are shown just for explanatory purposes.

The library has been tested on Ubuntu Linux 14.10 and 15.04 64-bit.

Initialization
--------------

Here we show a few ways to initialize matrices and vectors. All matrices method accept a parameter to
define whether to store the matrix in row-major or column-major order (default: column-major).

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

For some reason that has to do with type inference, there is no `matrix` constructor that
takes an array of arrays. The `dmatrix` constructor (d stand for dynamic) that requires
statically passing the dimensions has to be used. All constructors that take as input an
existing array or seq (such as `vector`, `dvector` and `dmatrix`) perform a copy of the data
for memory safety.

Accessors
---------

Vectors can be accessed as expected

    var v8 = randomVector(6)
    v8[4] = 1.2
    echo v8[3]

Same for matrices, where `m[i, j]` denotes the item on row `i` and column `j`, regardless of the matrix order

    var m8 = randomMatrix(3, 7)
    m8[1, 3] = 0.8
    echo m8[2, 2]

Also one can see rows and columns as vectors

    let
      r2: Vector64[7] = m8.row(2)
      c5: Vector64[3] = m8.column(5)

For memory safety, this performs a **copy** of the row or column values, at least for now.

Iterators
---------

One can iterate over vector or matrix elements, as well as over rows and columns

    for x in v8: echo x
    for i, x in v8: echo i, x
    for x in m8: echo x
    for t, x in m8:
      let (i, j) = t
      echo i, j, x
    for row in m8.rows:
      echo row[0]
    for column in m8.columns:
      echo column[1]

Equality
--------

There are two kinds of equality. The usual `==` operator will compare the contents of vector and matrices exactly

    let
      u = vector([1.0, 2.0, 3.0, 4.0])
      v = vector([1.0, 2.0, 3.0, 4.0])
      w = vector([1.0, 3.0, 3.0, 4.0])
    u == v # true
    u == w # false

Usually, though, one wants to take into account the errors introduced by floating point operations. To do this,
use the `~=` operator, or its negation `~!=`:

    let
      u = vector([1.0, 2.0, 3.0, 4.0])
      v = vector([1.0, 2.000000001, 2.99999999, 4.0])
    u == v # false
    u ~= v # true

Pretty-print
------------

Both vectors and matrix have a pretty-print operation, so one can do

    echo m8

and get something like

    [ [ 0.5024584865674662  0.0798945419892334  0.7512423051567048  0.9119041361916302  0.5868388894943912  0.3600554448403415  0.4419034543022882 ]
      [ 0.8225964245706265  0.01608615513584155 0.1442007939324697  0.7623388321096165  0.8419745686508193  0.08792951865247645 0.2902529012579151 ]
      [ 0.8488187232786935  0.422866666087792 0.1057975175658363  0.07968277822379832 0.7526946339452074  0.7698915909784674  0.02831893268471575 ] ]

Operations
----------

A few linear algebra operations are available, wrapping BLAS:

    echo 3.5 * v8
    v8 *= 2.3
    echo v1 + v3
    echo v1 - v3
    echo v1 * v3 # dot product
    echo l_1(v1) # l_1 norm
    echo l_2(v1) # l_2 norm
    echo m3 * v3 # matrix-vector product
    echo m4 * m1 # matrix-matrix product
    echo max(m1)
    echo min(v3)

Trivial operations
------------------

The following operations do not change the underlying memory layout of matrices and vectors.
This means they run in very little time even on big matrices, but you have to pay attention
when mutating matrices and vectors produced in this way, since the underyling data is shared.

    echo m4.t # transpose, done in constant time without copying
    echo m1 + m4.t
    let m9: Matrix64[5, 3] = m3.reshape(5, 3)
    let m10: Matrix64[3, 3] = v5.asMatrix(3, 3)
    let v9: Vector64[15] = m3.asVector

In case you need to allocate a copy of the original data, say in order to transpose a matrix
and then mutate the transpose without altering the original matrix, a `clone` operation is
available:

    let m11 = m10.clone

Rewrite rules
-------------

A few rewrite rules allow to optimize a chain of linear algebra operations into a single BLAS call. For instance, if you try

    echo v1 + 5.3 * v3

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

TODO
----

* Add support for `float32` matrices
* Add support for matrices and vectors whose size is only known at runtime
* Add support for matrices and vector on the stack, since dimensions are known at compile time anyway
* Use rewrite rules to optimize complex operations into a single BLAS call
* Move vectors and matrix to/from the GPU
* Run on the GPU via cuBLAS
* More specialized BLAS operations
* Add operations from LAPACK
* Support slicing/nonconstant steps
* Make `row` and `column` operations non-copying
* Better types to avoid out of bounds exceptions when statically checkable
* Add a fallback Nim implementation of most operations, that is valid over other rings
* Try on more platforms/configurations
* Make a proper benchmark
* Improve documentation