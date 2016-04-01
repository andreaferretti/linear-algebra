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

type
  cudaError = enum
    cudaSuccess                           =      0
    cudaErrorMissingConfiguration         =      1
    cudaErrorMemoryAllocation             =      2
    cudaErrorInitializationError          =      3
    cudaErrorLaunchFailure                =      4
    cudaErrorPriorLaunchFailure           =      5
    cudaErrorLaunchTimeout                =      6
    cudaErrorLaunchOutOfResources         =      7
    cudaErrorInvalidDeviceFunction        =      8
    cudaErrorInvalidConfiguration         =      9
    cudaErrorInvalidDevice                =     10
    cudaErrorInvalidValue                 =     11
    cudaErrorInvalidPitchValue            =     12
    cudaErrorInvalidSymbol                =     13
    cudaErrorMapBufferObjectFailed        =     14
    cudaErrorUnmapBufferObjectFailed      =     15
    cudaErrorInvalidHostPointer           =     16
    cudaErrorInvalidDevicePointer         =     17
    cudaErrorInvalidTexture               =     18
    cudaErrorInvalidTextureBinding        =     19
    cudaErrorInvalidChannelDescriptor     =     20
    cudaErrorInvalidMemcpyDirection       =     21
    cudaErrorAddressOfConstant            =     22
    cudaErrorTextureFetchFailed           =     23
    cudaErrorTextureNotBound              =     24
    cudaErrorSynchronizationError         =     25
    cudaErrorInvalidFilterSetting         =     26
    cudaErrorInvalidNormSetting           =     27
    cudaErrorMixedDeviceExecution         =     28
    cudaErrorCudartUnloading              =     29
    cudaErrorUnknown                      =     30
    cudaErrorNotYetImplemented            =     31
    cudaErrorMemoryValueTooLarge          =     32
    cudaErrorInvalidResourceHandle        =     33
    cudaErrorNotReady                     =     34
    cudaErrorInsufficientDriver           =     35
    cudaErrorSetOnActiveProcess           =     36
    cudaErrorInvalidSurface               =     37
    cudaErrorNoDevice                     =     38
    cudaErrorECCUncorrectable             =     39
    cudaErrorSharedObjectSymbolNotFound   =     40
    cudaErrorSharedObjectInitFailed       =     41
    cudaErrorUnsupportedLimit             =     42
    cudaErrorDuplicateVariableName        =     43
    cudaErrorDuplicateTextureName         =     44
    cudaErrorDuplicateSurfaceName         =     45
    cudaErrorDevicesUnavailable           =     46
    cudaErrorInvalidKernelImage           =     47
    cudaErrorNoKernelImageForDevice       =     48
    cudaErrorIncompatibleDriverContext    =     49
    cudaErrorPeerAccessAlreadyEnabled     =     50
    cudaErrorPeerAccessNotEnabled         =     51
    cudaErrorDeviceAlreadyInUse           =     54
    cudaErrorProfilerDisabled             =     55
    cudaErrorProfilerNotInitialized       =     56
    cudaErrorProfilerAlreadyStarted       =     57
    cudaErrorProfilerAlreadyStopped       =     58
    cudaErrorAssert                       =     59
    cudaErrorTooManyPeers                 =     60
    cudaErrorHostMemoryAlreadyRegistered  =     61
    cudaErrorHostMemoryNotRegistered      =     62
    cudaErrorOperatingSystem              =     63
    cudaErrorPeerAccessUnsupported        =     64
    cudaErrorLaunchMaxDepthExceeded       =     65
    cudaErrorLaunchFileScopedTex          =     66
    cudaErrorLaunchFileScopedSurf         =     67
    cudaErrorSyncDepthExceeded            =     68
    cudaErrorLaunchPendingCountExceeded   =     69
    cudaErrorNotPermitted                 =     70
    cudaErrorNotSupported                 =     71
    cudaErrorHardwareStackError           =     72
    cudaErrorIllegalInstruction           =     73
    cudaErrorMisalignedAddress            =     74
    cudaErrorInvalidAddressSpace          =     75
    cudaErrorInvalidPc                    =     76
    cudaErrorIllegalAddress               =     77
    cudaErrorInvalidPtx                   =     78
    cudaErrorInvalidGraphicsContext       =     79
    cudaErrorStartupFailure               =   0x7f
    cudaErrorApiFailureBase               =  10000

  cublasStatus = enum
    cublasStatusSuccess                   =      0
    cublasStatusNotInitialized            =      1
    cublasStatusAllocFailed               =      3
    cublasStatusInvalidValue              =      7
    cublasStatusArchMismatch              =      8
    cublasStatusMappingError              =     11
    cublasStatusExecutionFailed           =     13
    cublasStatusInternalError             =     14
    cublasStatusNotSupported              =     15
    cublasStatusLicenseError              =     16

  cublasHandle = ptr object

  CudaException = object of IOError
    error: cudaError
  CublasException = object of IOError
    error: cublasStatus
  CudaVector32*[N: static[int]] = ref[ptr float32]
  CudaMatrix32*[M, N: static[int]] = object
    data: ref[ptr float32]

template fp(c: CudaMatrix32): ptr float32 = c.data[]

proc newCudaError(error: cudaError): ref CudaException =
  new result
  result.error = error
  result.msg = $(error)

proc newCublasError(error: cublasStatus): ref CublasException =
  new result
  result.error = error
  result.msg = $(error)

template check(error: cudaError): stmt =
  if error != cudaSuccess:
    raise newCudaError(error)

template check(stat: cublasStatus): stmt =
  if stat != cublasStatusSuccess:
    raise newCublasError(stat)