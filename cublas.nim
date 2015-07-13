when defined(cublas):
  {. passl: "-lcublas" passl: "-lcudart" .}

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

  proc cudaMalloc*(size: int): ptr float32 =
    var error: cudaError
    {.emit: """error = cudaMalloc((void**)&`result`, `size`); """.}
    if error != cudaSuccess:
      quit($(error))

  proc cublasCreate*(): cublasHandle =
    var stat: cublasStatus
    {.emit: """stat = cublasCreate_v2(& `result`); """.}
    if stat != cublasStatusSuccess:
      quit($(stat))

  proc cublasSetVector*(n, elemSize: int, x: pointer, incx: int,
    devicePtr: pointer, incy: int): cublasStatus
    {. header: "cublas_api.h", importc: "cublasSetVector" .}


  # proc rawCudaMalloc(p: ptr ptr, size: int): cudaError
  #   {. header: "cuda_runtime_api.h", importc: "cudaMalloc" .}

  # proc rawCublasCreate(h: object): cublasStatus
  #   {. header: "cublas_api.h", importc: "cublasCreate_v2" .}