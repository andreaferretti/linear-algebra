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

__global__ void relu_kernel_d(double* in, double* out, int count)
{
  int index = threadIdx.x + blockIdx.x*blockDim.x;

  if (index < count) {
    out[index] = (in[index] < 0) ? 0 : in[index];
  }
}

__global__ void relu_kernel_s(float* in, float* out, int count)
{
  int index = threadIdx.x + blockIdx.x*blockDim.x;

  if (index < count) {
    out[index] = (in[index] < 0) ? 0 : in[index];
  }
}

__global__ void relu_prime_kernel_d(double* in, double* out, int count)
{
  int index = threadIdx.x + blockIdx.x*blockDim.x;

  if (index < count) {
    out[index] = (in[index] < 0) ? 0 : 1;
  }
}

__global__ void relu_prime_kernel_s(float* in, float* out, int count)
{
  int index = threadIdx.x + blockIdx.x*blockDim.x;

  if (index < count) {
    out[index] = (in[index] < 0) ? 0 : 1;
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

extern "C" void relu_d(double* in, double* out, int count) {
    int dimGrid = 100;
    int dimBlock = 100;

    relu_kernel_d<<<dimGrid, dimBlock>>>(in, out, count);
}

extern "C" void relu_s(float* in, float* out, int count) {
    int dimGrid = 100;
    int dimBlock = 100;

    relu_kernel_s<<<dimGrid, dimBlock>>>(in, out, count);
}

extern "C" void relu_prime_d(double* in, double* out, int count) {
    int dimGrid = 100;
    int dimBlock = 100;

    relu_prime_kernel_d<<<dimGrid, dimBlock>>>(in, out, count);
}

extern "C" void relu_prime_s(float* in, float* out, int count) {
    int dimGrid = 100;
    int dimBlock = 100;

    relu_prime_kernel_s<<<dimGrid, dimBlock>>>(in, out, count);
}