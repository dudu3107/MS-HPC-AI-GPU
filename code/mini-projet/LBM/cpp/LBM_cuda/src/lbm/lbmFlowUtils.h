#ifndef LBM_FLOW_UTILS_H
#define LBM_FLOW_UTILS_H

#include "LBMParams.h"

//! define array type to store lattice velocity component
template<int npop, int dim>
struct velocity_array_tmpl {

  real_t data[npop*dim];

  // read only access
  __host__ __device__
  inline real_t operator () (int ipop, int dir) const noexcept { return data[dir + dim*ipop]; }

}; // struct velocity_array_t

using velocity_array_t = velocity_array_tmpl<LBMParams::npop, LBMParams::dim>;

//! define array type to store lattice weights
template <int npop>
struct weights_tmpl {

  real_t data[npop];

  // read only access
  __host__ __device__
  inline real_t operator () (int ipop) const noexcept { return data[ipop]; }

}; // struct weights_t

using weights_t = weights_tmpl<LBMParams::npop>;


/**
 * Compute macroscopic variables from distribution functions.
 *
 * fluid density is 0th moment of distribution functions
 * fluid velocity components are 1st order moment of dist. functions
 *
 * \param[in] params LBM parameters
 * \param[in] v lattice velocities
 * \param[in] fin distribution functions
 * \param[out] rho macroscopic density
 * \param[out] ux X-component of macroscopic velocity
 * \param[out] uy Y-component of macroscopic velocity
 */
void macroscopic(const LBMParams &params, const velocity_array_t v,
                 const real_t *fin_d, real_t *rho_d, real_t *ux_d, real_t *uy_d);

/**
 * Compute equilibrium disbution function for a given set of macroscopic
 * variables.
 *
 * \param[in] params LBM parameters
 * \param[in] v lattice velocities
 * \param[in] rho macroscopic density
 * \param[in] ux X-component of macroscopic velocity
 * \param[in] uy Y-component of macroscopic velocity
 * \param[out] feq distribution functions
 */
void equilibrium(const LBMParams &params, const velocity_array_t v,
                 const weights_t t, 
                 const real_t *rho_d,
                 const real_t *ux_d,
                 const real_t *uy_d,
                 real_t *feq_d);

/**
 * Setup: cylindrical obstacle mask.
 *
 * \param[in] params
 * \param[out] obstacle
 */
void init_obstacle_mask(const LBMParams &params, int *obstacle, int *obstacle_d);

/**
 * compute initial velocity at location (i,j)
 *
 * \param[in] dir (0 means X, 1 means Y)
 * \param[in] x node coordinate along X
 * \param[in] y node coordinate along Y
 */
__host__ __device__
real_t compute_vel(int dir, int i, int j, real_t uLB, real_t ly);

/**
 * initialize macroscopic variables.
 */
void initialize_macroscopic_variables(const LBMParams &params, 
                                      real_t *rho, real_t *rho_d,
                                      real_t *ux, real_t *ux_d, 
                                      real_t *uy, real_t *uy_d);

/**
 * border condition : outflow on the right interface
 *
 * Right wall: outflow condition.
 * we only need here to specify distrib. function for velocities
  that enter the domain (other that go out, are set by the streaming step)
 */
void border_outflow(const LBMParams &params, real_t *fin_d);


/**
 * border condition : inflow on the left interface
 */
void border_inflow(const LBMParams &params, const real_t *fin_d, 
                   real_t *rho_d, real_t *ux_d, real_t *uy_d);


/**
 * Update fin at inflow left border.
 */
void update_fin_inflow(const LBMParams& params, const real_t* feq_d, 
                       real_t* fin_d);


/**
 * Compute collision
 */
void compute_collision(const LBMParams& params, 
                       const real_t* fin_d,
                       const real_t* feq_d,
                       real_t* fout_d);

/**
 * Update distrib. function inside obstacle.
 */
 void update_obstacle(const LBMParams& params, 
                      const real_t* fin_d,
                      const int* obstacle_d, 
                      real_t* fout_d);

/**
 * Streaming.
 */
void streaming(const LBMParams& params,
               const velocity_array_t v,
               const real_t* fout,
               real_t* fin);

#endif // LBM_FLOW_UTILS_H
