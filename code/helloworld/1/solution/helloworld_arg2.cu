#include<stdio.h>
#include<stdlib.h>

struct Array {
  int data[5];
  __host__ __device__
  int operator[] (int i) {return data[i];}
};


__global__ void print_from_gpu(Array a) {
  printf("Hello World from GPU thread %d, block %d ! data = %d %d %d %d %d\n",
         threadIdx.x, blockIdx.x,
         a[0], a[1], a[2], a[3], a[4]);
}

int main(int argc, char* argv[]) 
{
  printf("Hello from CPU !\n");

  Array a = {0,1,2,3,4};

  print_from_gpu<<<1,1>>>(a);

  cudaDeviceSynchronize();
  return EXIT_SUCCESS;
}
