/*
 * How to build:
 *
 * nvcc -arch=sm_80 -o helloworld_block helloworld_block.cu
 *
 * Note that you need to adjust the architecture version to your current GPU hardware.
 * Hardware version can be probed with e.g. deviceQuery example (from Nvidia SDK samples).
 *
 */
#include <stdio.h>
#include <stdlib.h>

__global__ void hello()
{
  printf("I'm a thread %d in block %d\n",
         threadIdx.x, blockIdx.x);
}


int main(int argc,char **argv)
{

  // default values for 
  // - gridSize :  number of blocks
  // - blockSize : number of threads per block

  // unsigned int gridSize  = argc > 1 ? atoi(argv[1]) : 1;
  // unsigned int blockSize = argc > 2 ? atoi(argv[2]) : 16;

  dim3 blockSize (4, 4);
  dim3 gridSize (1);

  // launch the kernel
  hello<<<gridSize, blockSize>>>();
  
  // force the printf()s to flush
  cudaDeviceSynchronize();
  
  printf("That's all!\n");
  
  return EXIT_SUCCESS;
}
