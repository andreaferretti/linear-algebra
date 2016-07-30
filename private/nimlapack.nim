########################### GESV #################################
#n is number of rows of A and B that are being used
#nrhs is number of columns of matrix B
#a is the matrix to be solved in data vector form (should be column-major)
#lda is number of rows of A
#ipvt is an integer array of size n
#b is the matrix B which gets filled with the X values
#ldb is the number of rows of B
#info is an integer with return OK or not
when defined(windows):
  const lapackSuffix = ".dll"
elif defined(macosx):
  const lapackSuffix = ".dylib"
else:
  const lapackSuffix = ".so"

const lapackPrefix = "liblapack"
const lapackName = lapackPrefix & lapackSuffix

proc gesv*(
  n: ptr cint,
  nrhs: ptr cint,
  a: ptr cfloat,
  lda: ptr cint,
  ipvt: ptr cint,
  b: ptr cfloat,
  ldb: ptr cint,
  info: ptr cint
  ) {.cdecl, importc: "sgesv_", dynlib: lapackName.}

proc gesv*(
  n: ptr cint,
  nrhs: ptr cint,
  a: ptr cdouble,
  lda: ptr cint,
  ipvt: ptr cint,
  b: ptr cdouble,
  ldb: ptr cint,
  info: ptr cint
  ) {.cdecl, importc: "dgesv_", dynlib: lapackName.}
