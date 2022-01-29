#include <math.h> // for M_PI = 3.1415....

#include "lbmFlowUtils.h"

#include "lbmFlowUtils_kernels.h"
#include "cuda_error.h"

// ======================================================
// ======================================================
void macroscopic(const LBMParams& params, 
                 const velocity_array_t v,
                 const real_t* fin_d,
                 real_t* rho_d,
                 real_t* ux_d,
                 real_t* uy_d)
{

  const int nx = params.nx;
  const int ny = params.ny;
  //const int npop = LBMParams::npop;

  // TODO : call kernel
  unsigned int threadsPerBlockX=32;
  unsigned int threadsPerBlockY=8;

  dim3  threads(threadsPerBlockX, threadsPerBlockY, 1);
  dim3  gridSize( (nx+threads.x-1)/threads.x, (ny+threads.y-1)/threads.y , 1);

  macroscopic_kernel<<<gridSize,threads>>>(params, v, fin_d, rho_d, ux_d, uy_d);
  CUDA_KERNEL_CHECK("macroscopic_kernel");


} // macroscopic

// ======================================================
// ======================================================
void equilibrium(const LBMParams& params, 
                 const velocity_array_t v,
                 const weights_t t,
                 const real_t* rho_d,
                 const real_t* ux_d,
                 const real_t* uy_d,
                 real_t* feq_d)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int npop = LBMParams::npop;

  // TODO : call kernel
  unsigned int threadsPerBlockX=32;
  unsigned int threadsPerBlockY=8;
  //unsigned int threadsPerBlockZ=32;

  dim3  threads(threadsPerBlockX, threadsPerBlockY, 1);//threadsPerBlockZ);
  dim3  gridSize( (nx+threads.x-1)/threads.x, (ny+threads.y-1)/threads.y , 1);//(npop+threads.z-1)/threads.z);

  equilibrium_kernel<<<gridSize,threads>>>(params, v, t, rho_d, ux_d, uy_d, feq_d);
  CUDA_KERNEL_CHECK("equilibrium_kernel");

} // equilibrium

// ======================================================
// ======================================================
void init_obstacle_mask(const LBMParams& params, 
                        int* obstacle, 
                        int* obstacle_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  const real_t cx = params.cx;
  const real_t cy = params.cy;

  const real_t r = params.r;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int index = i + nx * j;

      real_t x = 1.0*i;
      real_t y = 1.0*j;

      obstacle[index] = (x-cx)*(x-cx) + (y-cy)*(y-cy) < r*r ? 1 : 0;

    } // end for i
  } // end for j

  // TODO : copy host to device
  CUDA_API_CHECK( cudaMemcpy( obstacle_d, obstacle, nx*ny*sizeof(int), cudaMemcpyHostToDevice ) );

} // init_obstacle_mask

// ======================================================
// ======================================================
__host__ __device__
real_t compute_vel(int dir, int i, int j, real_t uLB, real_t ly)
{

  // flow is along X axis
  // X component is non-zero
  // Y component is always zero

  return (1-dir) * uLB * (1 + 1e-4 * sin(j/ly*2*M_PI));

} // compute_vel

// ======================================================
// ======================================================
void initialize_macroscopic_variables(const LBMParams& params, 
                                      real_t* rho, real_t* rho_d,
                                      real_t* ux, real_t* ux_d,
                                      real_t* uy, real_t* uy_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int index = i + nx * j;

      rho[index] = 1.0;
      ux[index]  = compute_vel(0, i, j, params.uLB, params.ly);
      uy[index]  = compute_vel(1, i, j, params.uLB, params.ly);

    } // end for i
  } // end for j

  // TODO : copy host to device
  CUDA_API_CHECK( cudaMemcpy( rho_d, rho, nx*ny*sizeof(real_t), cudaMemcpyHostToDevice ) );
  CUDA_API_CHECK( cudaMemcpy( ux_d, ux, nx*ny*sizeof(real_t), cudaMemcpyHostToDevice ) );
  CUDA_API_CHECK( cudaMemcpy( uy_d, uy, nx*ny*sizeof(real_t), cudaMemcpyHostToDevice ) );


} // initialize_macroscopic_variables

// ======================================================
// ======================================================
void border_outflow(const LBMParams& params, real_t* fin_d)
{

  // TODO : call kernel
  const int ny = params.ny;
  unsigned int threadsPerBlock=256;
  dim3  threads(1, threadsPerBlock, 1);
  dim3  gridSize( 1, (ny+threads.y-1)/threads.y , 1);

  border_outflow_kernel<<<gridSize,threads>>>(params, fin_d);
  CUDA_KERNEL_CHECK("border_outflow_kernel");

} // border_outflow


// ======================================================
// ======================================================
void border_inflow(const LBMParams& params, const real_t* fin_d, 
                   real_t* rho_d, real_t* ux_d, real_t* uy_d)
{
  // TODO : call kernel
  const int ny = params.ny;
  unsigned int threadsPerBlock=256;
  dim3  threads(1, threadsPerBlock, 1);
  dim3  gridSize( 1, (ny+threads.y-1)/threads.y , 1);

  border_inflow_kernel<<<gridSize,threads>>>(params, fin_d, rho_d, ux_d, uy_d);
  CUDA_KERNEL_CHECK("border_inflow_kernel");

} // border_inflow


// ======================================================
// ======================================================
void update_fin_inflow(const LBMParams& params, const real_t* feq_d, 
                       real_t* fin_d)
{

  // TODO : call kernel
  const int ny = params.ny;
  unsigned int threadsPerBlock=256;
  dim3  threads(1, threadsPerBlock, 1);
  dim3  gridSize( 1, (ny+threads.y-1)/threads.y , 1);

  update_fin_inflow_kernel<<<gridSize,threads>>>(params, feq_d, fin_d);
  CUDA_KERNEL_CHECK("update_fin_inflow_kernel");

} // update_fin_inflow


// ======================================================
// ======================================================
void compute_collision(const LBMParams& params, 
                       const real_t* fin_d,
                       const real_t* feq_d,
                       real_t* fout_d)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int npop = LBMParams::npop;

  // TODO : call kernel
  unsigned int threadsPerBlockX=32;
  unsigned int threadsPerBlockY=8;
  //unsigned int threadsPerBlockZ=3;

  dim3  threads(threadsPerBlockX, threadsPerBlockY, 1);//threadsPerBlockZ);
  dim3  gridSize( (nx+threads.x-1)/threads.x, (ny+threads.y-1)/threads.y , 1);//(npop+threads.z-1)/threads.z);

  compute_collision_kernel<<<gridSize,threads>>>(params, fin_d, feq_d, fout_d);
  CUDA_KERNEL_CHECK("compute_collision_kernel");

} // compute_collision

// ======================================================
// ======================================================
void update_obstacle(const LBMParams &params, 
                     const real_t* fin_d,
                     const int* obstacle_d, 
                     real_t* fout_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  // TODO : call kernel
  unsigned int threadsPerBlockX=32;
  unsigned int threadsPerBlockY=8;

  dim3  threads(threadsPerBlockX, threadsPerBlockY, 1);
  dim3  gridSize( (nx+threads.x-1)/threads.x, (ny+threads.y-1)/threads.y , 1);

  update_obstacle_kernel<<<gridSize,threads>>>(params, fin_d, obstacle_d, fout_d);
  CUDA_KERNEL_CHECK("update_obstacle_kernel");

} // update_obstacle

// ======================================================
// ======================================================
void streaming(const LBMParams& params,
               const velocity_array_t v,
               const real_t* fout_d,
               real_t* fin_d)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int npop = LBMParams::npop;

  // TODO : call kernel
  unsigned int threadsPerBlockX=32;
  unsigned int threadsPerBlockY=8;
  //unsigned int threadsPerBlockZ=3;

  dim3  threads(threadsPerBlockX, threadsPerBlockY, 1);//threadsPerBlockZ);
  dim3  gridSize( (nx+threads.x-1)/threads.x, (ny+threads.y-1)/threads.y , 1);//(npop+threads.z-1)/threads.z);

  streaming_kernel<<<gridSize,threads>>>(params, v, fout_d, fin_d);
  CUDA_KERNEL_CHECK("streaming_kernel");

} // streaming
