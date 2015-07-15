when defined(cublas):
  {. passl: "-lcublas" passl: "-lcudart" .}

  include cuda/types
  include cuda/cublas
  include cuda/ops