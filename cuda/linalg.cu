__global__ void hadamard_kernel_d(double* in1, double* in2, double* out, int count)
{
  int index = threadIdx.x + blockIdx.x*blockDim.x;

  if (index < count) {
    out[index] = in1[index] * in2[index];
  }
}

__global__ void hadamard_kernel_s(float* in1, float* in2, float* out, int count)
{
  int index = threadIdx.x + blockIdx.x*blockDim.x;

  if (index < count) {
    out[index] = in1[index] * in2[index];
  }
}

extern "C" void hadamard_d(double* in1, double* in2, double* out, int count) {
    int dimGrid = 100;
    int dimBlock = 100;

    hadamard_kernel_d<<<dimGrid, dimBlock>>>(in1, in2, out, count);
}

extern "C" void hadamard_s(float* in1, float* in2, float* out, int count) {
    int dimGrid = 100;
    int dimBlock = 100;

    hadamard_kernel_s<<<dimGrid, dimBlock>>>(in1, in2, out, count);
}