# Copyright 2015 UniCredit S.p.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

let handle {.global.} = cublasCreate()

template initDynamic(v, n: expr) =
  new v.data, freeDeviceMemory
  when v is CudaDVector32:
    v.data[] = cudaMalloc32(n)
  when result is CudaDVector64:
    v.data[] = cudaMalloc64(n)
  v.N = n

template initStatic(v, n: expr) =
  new v, freeDeviceMemory
  when v is CudaVector32:
    v[] = cudaMalloc32(n)
  when result is CudaVector64:
    v[] = cudaMalloc64(n)

template initMDynamic(v, m, n: expr) =
  new v.data, freeDeviceMemory
  when v is CudaDMatrix32:
    v.data[] = cudaMalloc32(m * n)
  when result is CudaDMatrix64:
    v.data[] = cudaMalloc64(m * n)
  v.M = m
  v.N = n

template initMStatic(v, m, n: expr) =
  new v.data, freeDeviceMemory
  when v is CudaMatrix32:
    v.data[] = cudaMalloc32(m * n)
  when result is CudaMatrix64:
    v.data[] = cudaMalloc64(m * n)

template init(v, N: expr) =
  when v is CudaDVector32 or v is CudaDVector64:
    initDynamic(v, N)
  when v is CudaVector32 or v is CudaVector64:
    initStatic(v, N)

template initM(v, M, N: expr) =
  when v is CudaDMatrix32 or v is CudaDMatrix64:
    initMDynamic(v, M, N)
  when v is CudaMatrix32 or v is CudaMatrix64:
    initMStatic(v, M, N)

proc `*=`*[N: static[int]](v: var CudaVector32[N], k: float32) {. inline .} =
  check cublasScal(handle, N, k, v.fp)

proc `*=`*(v: var CudaDVector32, k: float32) {. inline .} =
  check cublasScal(handle, v.N, k, v.fp)

proc `*=`*[N: static[int]](v: var CudaVector64[N], k: float64) {. inline .} =
  check cublasScal(handle, N, k, v.fp)

proc `*=`*(v: var CudaDVector64, k: float64) {. inline .} =
  check cublasScal(handle, v.N, k, v.fp)

template multiply(result, v, k, N: expr) =
  init(result, N)
  check cublasCopy(handle, N, v.fp, 1, result.fp, 1)
  check cublasScal(handle, N, k, result.fp)

proc `*`*[N: static[int]](v: CudaVector32[N], k: float32): CudaVector32[N]  {. inline .} =
  multiply(result, v, k, N)

proc `*`*(v: CudaDVector32, k: float32): CudaDVector32  {. inline .} =
  multiply(result, v, k, v.N)

proc `*`*[N: static[int]](v: CudaVector64[N], k: float64): CudaVector64[N]  {. inline .} =
  multiply(result, v, k, N)

proc `*`*(v: CudaDVector64, k: float64): CudaDVector64  {. inline .} =
  multiply(result, v, k, v.N)

proc `+=`*[N: static[int]](v: var CudaVector32[N], w: CudaVector32[N]) {. inline .} =
  check cublasAxpy(handle, N, 1, w.fp, v.fp)

proc `+=`*(v: var CudaDVector32, w: CudaDVector32) {. inline .} =
  assert(v.N == w.N)
  check cublasAxpy(handle, v.N, 1, w.fp, v.fp)

proc `+=`*[N: static[int]](v: var CudaVector64[N], w: CudaVector64[N]) {. inline .} =
  check cublasAxpy(handle, N, 1, w.fp, v.fp)

proc `+=`*(v: var CudaDVector64, w: CudaDVector64) {. inline .} =
  assert(v.N == w.N)
  check cublasAxpy(handle, v.N, 1, w.fp, v.fp)

template sum(result, v, w, N: expr) =
  init(result, N)
  check cublasCopy(handle, N, v.fp, 1, result.fp, 1)
  check cublasAxpy(handle, N, 1, w.fp, result.fp)

proc `+`*[N: static[int]](v, w: CudaVector32[N]): CudaVector32[N] {. inline .} =
  sum(result, v, w, N)

proc `+`*(v, w: CudaDVector32): CudaDVector32 {. inline .} =
  assert(v.N == w.N)
  sum(result, v, w, v.N)

proc `+`*[N: static[int]](v, w: CudaVector64[N]): CudaVector64[N] {. inline .} =
  sum(result, v, w, N)

proc `+`*(v, w: CudaDVector64): CudaDVector64 {. inline .} =
  assert(v.N == w.N)
  sum(result, v, w, v.N)

proc `-=`*[N: static[int]](v: var CudaVector32[N], w: CudaVector32[N]) {. inline .} =
  check cublasAxpy(handle, N, -1, w.fp, v.fp)

proc `-=`*(v: var CudaDVector32, w: CudaDVector32) {. inline .} =
  assert(v.N == w.N)
  check cublasAxpy(handle, v.N, -1, w.fp, v.fp)

proc `-=`*[N: static[int]](v: var CudaVector64[N], w: CudaVector64[N]) {. inline .} =
  check cublasAxpy(handle, N, -1, w.fp, v.fp)

proc `-=`*(v: var CudaDVector64, w: CudaDVector64) {. inline .} =
  assert(v.N == w.N)
  check cublasAxpy(handle, v.N, -1, w.fp, v.fp)

template diff(result, v, w, N: expr) =
  init(result, N)
  check cublasCopy(handle, N, v.fp, 1, result.fp, 1)
  check cublasAxpy(handle, N, -1, w.fp, result.fp)

proc `-`*[N: static[int]](v, w: CudaVector32[N]): CudaVector32[N] {. inline .} =
  diff(result, v, w, N)

proc `-`*(v, w: CudaDVector32): CudaDVector32 {. inline .} =
  assert(v.N == w.N)
  diff(result, v, w, v.N)

proc `-`*[N: static[int]](v, w: CudaVector64[N]): CudaVector64[N] {. inline .} =
  diff(result, v, w, N)

proc `-`*(v, w: CudaDVector64): CudaDVector64 {. inline .} =
  assert(v.N == w.N)
  diff(result, v, w, v.N)

proc `*`*[N: static[int]](v, w: CudaVector32[N]): float32 {. inline .} =
  check cublasDot(handle, N, v.fp, 1, w.fp, 1, addr(result))

proc `*`*(v, w: CudaDVector32): float32 {. inline .} =
  assert(v.N == w.N)
  check cublasDot(handle, v.N, v.fp, 1, w.fp, 1, addr(result))

proc `*`*[N: static[int]](v, w: CudaVector64[N]): float64 {. inline .} =
  check cublasDot(handle, N, v.fp, 1, w.fp, 1, addr(result))

proc `*`*(v, w: CudaDVector64): float64 {. inline .} =
  assert(v.N == w.N)
  check cublasDot(handle, v.N, v.fp, 1, w.fp, 1, addr(result))

proc l_2*[N: static[int]](v: CudaVector32[N]): float32 {. inline .} =
  check cublasNrm2(handle, N, v.fp, 1, addr(result))

proc l_2*(v: CudaDVector32): float32 {. inline .} =
  check cublasNrm2(handle, v.N, v.fp, 1, addr(result))

proc l_2*[N: static[int]](v: CudaVector64[N]): float64 {. inline .} =
  check cublasNrm2(handle, N, v.fp, 1, addr(result))

proc l_2*(v: CudaDVector64): float64 {. inline .} =
  check cublasNrm2(handle, v.N, v.fp, 1, addr(result))

proc l_1*[N: static[int]](v: CudaVector32[N]): float32 {. inline .} =
  check cublasAsum(handle, N, v.fp, 1, addr(result))

proc l_1*(v: CudaDVector32): float32 {. inline .} =
  check cublasAsum(handle, v.N, v.fp, 1, addr(result))

proc l_1*[N: static[int]](v: CudaVector64[N]): float64 {. inline .} =
  check cublasAsum(handle, N, v.fp, 1, addr(result))

proc l_1*(v: CudaDVector64): float64 {. inline .} =
  check cublasAsum(handle, v.N, v.fp, 1, addr(result))

template matVec(result, a, v, M, N: expr) =
  init(result, M)
  check cublasGemv(handle, cuNoTranspose, M, N, 1, a.fp, M, v.fp, 1, 0, result.fp, 1)

proc `*`*[M, N: static[int]](a: CudaMatrix32[M, N], v: CudaVector32[N]): CudaVector32[M]  {. inline .} =
  matVec(result, a, v, M, N)

proc `*`*(a: CudaDMatrix32, v: CudaDVector32): CudaDVector32  {. inline .} =
  assert(a.N == v.N)
  matVec(result, a, v, a.M, a.N)

proc `*`*[M, N: static[int]](a: CudaMatrix64[M, N], v: CudaVector64[N]): CudaVector64[M]  {. inline .} =
  matVec(result, a, v, M, N)

proc `*`*(a: CudaDMatrix64, v: CudaDVector64): CudaDVector64  {. inline .} =
  assert(a.N == v.N)
  matVec(result, a, v, a.M, a.N)

proc `*=`*[M, N: static[int]](m: var CudaMatrix32[M, N], k: float32) {. inline .} =
  check cublasScal(handle, M * N, k, m.fp)

proc `*=`*(m: var CudaDMatrix32, k: float32) {. inline .} =
  check cublasScal(handle, m.M * m.N, k, m.fp)

proc `*=`*[M, N: static[int]](m: var CudaMatrix64[M, N], k: float64) {. inline .} =
  check cublasScal(handle, M * N, k, m.fp)

proc `*=`*(m: var CudaDMatrix64, k: float64) {. inline .} =
  check cublasScal(handle, m.M * m.N, k, m.fp)

template matScal(result, m, k, M, N: expr) =
  initM(result, M, N)
  check cublasCopy(handle, M * N, m.fp, 1, result.fp, 1)
  check cublasScal(handle, M * N, k, result.fp)

proc `*`*[M, N: static[int]](m: CudaMatrix32[M, N], k: float32): CudaMatrix32[M, N]  {. inline .} =
  matScal(result, m, k, M, N)

proc `*`*(m: CudaDMatrix32, k: float32): CudaDMatrix32  {. inline .} =
  matScal(result, m, k, m.M, m.N)

proc `*`*[M, N: static[int]](m: CudaMatrix64[M, N], k: float64): CudaMatrix64[M, N]  {. inline .} =
  matScal(result, m, k, M, N)

proc `*`*(m: CudaDMatrix64, k: float64): CudaDMatrix64  {. inline .} =
  matScal(result, m, k, m.M, m.N)

template `*`*(k: float32, v: CudaVector32 or CudaMatrix32 or CudaDVector32 or CudaDMatrix32): expr = v * k

template `*`*(k: float64, v: CudaVector64 or CudaMatrix64 or CudaDVector64 or CudaDMatrix64): expr = v * k

template `/`*(v: CudaVector32 or CudaMatrix32 or CudaDVector32 or CudaDMatrix32, k: float32): expr = v * (1 / k)

template `/`*(v: CudaVector64 or CudaMatrix64 or CudaDVector64 or CudaDMatrix64, k: float64): expr = v * (1 / k)

template `/=`*(v: var CudaVector32 or var CudaMatrix32 or var CudaDVector32 or var CudaDMatrix32, k: float32): expr = v *= (1 / k)

template `/=`*(v: var CudaVector64 or var CudaMatrix64 or var CudaDVector64 or var CudaDMatrix64, k: float64): expr = v *= (1 / k)

proc `+=`*[M, N: static[int]](a: var CudaMatrix32[M, N], b: CudaMatrix32[M, N]) {. inline .} =
  check cublasAxpy(handle, M * N, 1, b.fp, a.fp)

proc `+=`*(a: var CudaDMatrix32, b: CudaDMatrix32) {. inline .} =
  assert a.M == b.M and a.N == a.N
  check cublasAxpy(handle, a.M * a.N, 1, b.fp, a.fp)

proc `+=`*[M, N: static[int]](a: var CudaMatrix64[M, N], b: CudaMatrix64[M, N]) {. inline .} =
  check cublasAxpy(handle, M * N, 1, b.fp, a.fp)

proc `+=`*(a: var CudaDMatrix64, b: CudaDMatrix64) {. inline .} =
  assert a.M == b.M and a.N == a.N
  check cublasAxpy(handle, a.M * a.N, 1, b.fp, a.fp)

template matSum(result, a, b, M, N: expr) =
  initM(result, M, N)
  check cublasCopy(handle, M * N, a.fp, 1, result.fp, 1)
  check cublasAxpy(handle, M * N, 1, b.fp, result.fp)

proc `+`*[M, N: static[int]](a, b: CudaMatrix32[M, N]): CudaMatrix32[M, N]  {. inline .} =
  matSum(result, a, b, M, N)

proc `+`*(a, b: CudaDMatrix32): CudaDMatrix32  {. inline .} =
  assert a.M == b.M and a.N == a.N
  matSum(result, a, b, a.M, a.N)

proc `+`*[M, N: static[int]](a, b: CudaMatrix64[M, N]): CudaMatrix64[M, N]  {. inline .} =
  matSum(result, a, b, M, N)

proc `+`*(a, b: CudaDMatrix64): CudaDMatrix64  {. inline .} =
  assert a.M == b.M and a.N == a.N
  matSum(result, a, b, a.M, a.N)

proc `-=`*[M, N: static[int]](a: var CudaMatrix32[M, N], b: CudaMatrix32[M, N]) {. inline .} =
  check cublasAxpy(handle, M * N, -1, b.fp, a.fp)

proc `-=`*(a: var CudaDMatrix32, b: CudaDMatrix32) {. inline .} =
  assert a.M == b.M and a.N == a.N
  check cublasAxpy(handle, a.M * a.N, -1, b.fp, a.fp)

proc `-=`*[M, N: static[int]](a: var CudaMatrix64[M, N], b: CudaMatrix64[M, N]) {. inline .} =
  check cublasAxpy(handle, M * N, -1, b.fp, a.fp)

proc `-=`*(a: var CudaDMatrix64, b: CudaDMatrix64) {. inline .} =
  assert a.M == b.M and a.N == a.N
  check cublasAxpy(handle, a.M * a.N, -1, b.fp, a.fp)

template matDiff(result, a, b, M, N: expr) =
  initM(result, M, N)
  check cublasCopy(handle, M * N, a.fp, 1, result.fp, 1)
  check cublasAxpy(handle, M * N, -1, b.fp, result.fp)

proc `-`*[M, N: static[int]](a, b: CudaMatrix32[M, N]): CudaMatrix32[M, N]  {. inline .} =
  matDiff(result, a, b, a.M, a.N)

proc `-`*(a, b: CudaDMatrix32): CudaDMatrix32  {. inline .} =
  assert a.M == b.M and a.N == a.N
  matDiff(result, a, b, a.M, a.N)

proc `-`*[M, N: static[int]](a, b: CudaMatrix64[M, N]): CudaMatrix64[M, N]  {. inline .} =
  matDiff(result, a, b, a.M, a.N)

proc `-`*(a, b: CudaDMatrix64): CudaDMatrix64  {. inline .} =
  assert a.M == b.M and a.N == a.N
  matDiff(result, a, b, a.M, a.N)

template matMul(result, a, b, M, K, N: expr) =
  initM(result, M, N)
  check cublasGemm(handle, cuNoTranspose, cuNoTranspose, M, N, K, 1,
    a.fp, M, b.fp, K, 0, result.fp, M)

proc `*`*[M, N, K: static[int]](a: CudaMatrix32[M, K], b: CudaMatrix32[K, N]): CudaMatrix32[M, N] {. inline .} =
  matMul(result, a, b, M, K, N)

proc `*`*(a: CudaDMatrix32, b: CudaDMatrix32): CudaDMatrix32 {. inline .} =
  assert a.N == b.M
  matMul(result, a, b, a.M, a.N, b.N)

proc `*`*[M, N, K: static[int]](a: CudaMatrix64[M, K], b: CudaMatrix64[K, N]): CudaMatrix64[M, N] {. inline .} =
  matMul(result, a, b, M, K, N)

proc `*`*(a: CudaDMatrix64, b: CudaDMatrix64): CudaDMatrix64 {. inline .} =
  assert a.N == b.M
  matMul(result, a, b, a.M, a.N, b.N)

proc l_2*[M, N: static[int]](m: CudaMatrix32[M, N]): float32 {. inline .} =
  check cublasNrm2(handle, M * N, m.fp, 1, addr(result))

proc l_2*(m: CudaDMatrix32): float32 {. inline .} =
  check cublasNrm2(handle, m.M * m.N, m.fp, 1, addr(result))

proc l_2*[M, N: static[int]](m: CudaMatrix64[M, N]): float64 {. inline .} =
  check cublasNrm2(handle, M * N, m.fp, 1, addr(result))

proc l_2*(m: CudaDMatrix64): float64 {. inline .} =
  check cublasNrm2(handle, m.M * m.N, m.fp, 1, addr(result))

proc l_1*[M, N: static[int]](m: CudaMatrix32[M, N]): float32 {. inline .} =
  check cublasAsum(handle, M * N, m.fp, 1, addr(result))

proc l_1*(m: CudaDMatrix32): float32 {. inline .} =
  check cublasAsum(handle, m.M * m.N, m.fp, 1, addr(result))

proc l_1*[M, N: static[int]](m: CudaMatrix64[M, N]): float64 {. inline .} =
  check cublasAsum(handle, M * N, m.fp, 1, addr(result))

proc l_1*(m: CudaDMatrix64): float64 {. inline .} =
  check cublasAsum(handle, m.M * m.N, m.fp, 1, addr(result))

proc `==`*[N: static[int]](v, w: CudaVector32[N]): bool =
  v.cpu() == w.cpu()

proc `==`*(m, n: CudaDVector32): bool =
  m.cpu() == n.cpu()

proc `==`*[N: static[int]](v, w: CudaVector64[N]): bool =
  v.cpu() == w.cpu()

proc `==`*(m, n: CudaDVector64): bool =
  m.cpu() == n.cpu()

proc `==`*[M, N: static[int]](m, n: CudaMatrix32[M, N]): bool =
  m.cpu() == n.cpu()

proc `==`*(m, n: CudaDMatrix32): bool =
  m.cpu() == n.cpu()

proc `==`*[M, N: static[int]](m, n: CudaMatrix64[M, N]): bool =
  m.cpu() == n.cpu()

proc `==`*(m, n: CudaDMatrix64): bool =
  m.cpu() == n.cpu()

type AnyCuda = CudaVector32 or CudaMatrix32 or CudaVector64 or
  CudaMatrix64 or CudaDVector32 or CudaDMatrix32 or CudaDVector64 or
  CudaDMatrix64

proc compareApprox(a, b: AnyCuda): bool =
  const epsilon = 0.000001
  let
    aNorm = l_1(a)
    bNorm = l_1(b)
    dNorm = l_1(a - b)
  return (dNorm / (aNorm + bNorm)) < epsilon

proc `=~`*[N: static[int]](v, w: CudaVector32[N]): bool = compareApprox(v, w)

proc `=~`*(v, w: CudaDVector32): bool = compareApprox(v, w)

proc `=~`*[N: static[int]](v, w: CudaVector64[N]): bool = compareApprox(v, w)

proc `=~`*(v, w: CudaDVector64): bool = compareApprox(v, w)

proc `=~`*[M, N: static[int]](m, n: CudaMatrix32[M, N]): bool = compareApprox(m, n)

proc `=~`*(v, w: CudaDMatrix32): bool = compareApprox(v, w)

proc `=~`*[M, N: static[int]](m, n: CudaMatrix64[M, N]): bool = compareApprox(m, n)

proc `=~`*(v, w: CudaDMatrix64): bool = compareApprox(v, w)

template `!=~`*(a, b: AnyCuda): bool = not (a =~ b)