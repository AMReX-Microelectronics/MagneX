#include <AMReX_Gpu.H>
// Algorithm to calculate 
// 1. Laplacian for exchange term in LLG equation
// 2. Divergence for demagnetization term
// 3. Spatial derivative for DMI term

/**
 * Perform derivative along x on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real UpwardDx (
    amrex::Array4<amrex::Real const> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dx = 1./dx[0];
    return inv_dx*( F(i+1,j,k) - F(i,j,k) );
}


/**
 * Perform derivative along x on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DownwardDx (
    amrex::Array4<amrex::Real const> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dx = 1./dx[0];
    return inv_dx*( F(i,j,k) - F(i-1,j,k) );
}


/**
 * Perform derivative along y on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real UpwardDy (
    amrex::Array4<amrex::Real const> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dy = 1./dx[1];
    return inv_dy*( F(i,j+1,k) - F(i,j,k) );
}


/**
 * Perform derivative along y on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DownwardDy (
    amrex::Array4<amrex::Real const> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dy = 1./dx[1];
    return inv_dy*( F(i,j,k) - F(i,j-1,k) );
}



/**
 * Perform derivative along z on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real UpwardDz (
    amrex::Array4<amrex::Real const> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dz = 1./dx[2];
    return inv_dz*( F(i,j,k+1) - F(i,j,k) );
}


/**
 * Perform derivative along z on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DownwardDz (
    amrex::Array4<amrex::Real const> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dz = 1./dx[2];
    return inv_dz*( F(i,j,k) - F(i,j,k-1) );
}

// 1. Laplacian for exchange term in LLG equation
/**
  * Perform divergence of gradient along x on M field */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real LaplacianDx_Mag (
    amrex::Array4<amrex::Real const> const& F,
    amrex::Real const Ms_lo_x, amrex::Real const Ms_hi_x, amrex::Real const BC_lo_x, amrex::Real const BC_hi_x, 
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dx = 1./dx[0];
    if ( Ms_hi_x == 0.) {
        return inv_dx*(BC_hi_x - DownwardDx(F, i, j, k, dx));
    } else if ( Ms_lo_x == 0.){
        return inv_dx*(UpwardDx(F, i, j, k, dx) - BC_lo_x);
    } else {
        return inv_dx*(UpwardDx(F, i, j, k, dx) - DownwardDx(F, i, j, k, dx));
    }

 }


/**
  * Perform divergence of gradient along y on M field */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real LaplacianDy_Mag (
    amrex::Array4<amrex::Real const> const& F,
    amrex::Real const Ms_lo_y, amrex::Real const Ms_hi_y, amrex::Real const BC_lo_y, amrex::Real const BC_hi_y,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dy = 1./dx[1];
    if ( Ms_hi_y == 0.) {
        return inv_dy*(BC_hi_y - DownwardDy(F, i, j, k, dx));
    } else if ( Ms_lo_y == 0.){
        return inv_dy*(UpwardDy(F, i, j, k, dx) - BC_lo_y);
    } else {
        return inv_dy*(UpwardDy(F, i, j, k, dx) - DownwardDy(F, i, j, k, dx));
    }

 }


/**
  * Perform divergence of gradient along z on M field */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real LaplacianDz_Mag (
    amrex::Array4<amrex::Real const> const& F,
    amrex::Real const Ms_lo_z, amrex::Real const Ms_hi_z, amrex::Real const BC_lo_z, amrex::Real const BC_hi_z,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dz = 1./dx[2];
    
    if ( Ms_hi_z == 0.) {
        return inv_dz*(BC_hi_z - DownwardDz(F, i, j, k, dx));
    } else if ( Ms_lo_z == 0.){
        return inv_dz*(UpwardDz(F, i, j, k, dx) - BC_lo_z);
    } else {
        return inv_dz*(UpwardDz(F, i, j, k, dx) - DownwardDz(F, i, j, k, dx));
    }

 }

/**
  * Compute the sum to get Laplacian of M field */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real Laplacian_Mag (
    amrex::Array4<amrex::Real const> const& F,
    amrex::Real const Ms_lo_x, amrex::Real const Ms_hi_x, amrex::Real BC_lo_x, amrex::Real BC_hi_x, 
    amrex::Real const Ms_lo_y, amrex::Real const Ms_hi_y, amrex::Real BC_lo_y, amrex::Real BC_hi_y, 
    amrex::Real const Ms_lo_z, amrex::Real const Ms_hi_z, amrex::Real BC_lo_z, amrex::Real BC_hi_z,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

        //amrex::Print() << "This is called " << Ms_lo_x << "\n";
        return LaplacianDx_Mag(F, Ms_lo_x, Ms_hi_x, BC_lo_x, BC_hi_x, i, j, k, dx) + LaplacianDy_Mag(F, Ms_lo_y, Ms_hi_y, BC_lo_y, BC_hi_y, i, j, k, dx) + LaplacianDz_Mag(F, Ms_lo_z, Ms_hi_z, BC_lo_z, BC_hi_z, i, j, k, dx);
 }

// 2. Divergence for demagnetization term
/**
  * Perform divergence  along x on M field */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DivergenceDx_Mag (
    amrex::Array4<amrex::Real const> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    return UpwardDx(F, i, j, k, dx);
 }


/**
  * Perform divergence along y on M field */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DivergenceDy_Mag (
    amrex::Array4<amrex::Real const> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    return UpwardDy(F, i, j, k, dx);
 }

 /**
  * Perform divergence along z on M field */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DivergenceDz_Mag (
    amrex::Array4<amrex::Real const> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    return UpwardDz(F, i, j, k, dx);
 }

 // 3. Spatial derivative for DMI term
/**
  * Perform divergence of gradient along x on M field */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DMDx_Mag (
    amrex::Array4<amrex::Real const> const& F,
    amrex::Real const Ms_lo_x, amrex::Real const Ms_hi_x, amrex::Real const BC_lo_x, amrex::Real const BC_hi_x,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dx = 1./dx[0];
    if ( Ms_hi_x == 0.) {
        return 0.5 *(BC_hi_x + DownwardDx(F, i, j, k, dx));
    } else if ( Ms_lo_x == 0.){
        return 0.5 *(UpwardDx(F, i, j, k, dx) + BC_lo_x);
    } else {
        return 0.5 *(UpwardDx(F, i, j, k, dx) + DownwardDx(F, i, j, k, dx));
    }

 }

 /**
  * Perform divergence of gradient along y on M field */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DMDy_Mag (
    amrex::Array4<amrex::Real const> const& F,
    amrex::Real const Ms_lo_y, amrex::Real const Ms_hi_y, amrex::Real const BC_lo_y, amrex::Real const BC_hi_y,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dy = 1./dx[1];
    if ( Ms_hi_y == 0.) {
        return 0.5 *(BC_hi_y + DownwardDy(F, i, j, k, dx));
    } else if ( Ms_lo_y == 0.){
        return 0.5 *(UpwardDy(F, i, j, k, dx) + BC_lo_y);
    } else {
        return 0.5 *(UpwardDy(F, i, j, k, dx) + DownwardDy(F, i, j, k, dx));
    }

 }


/**
  * Perform divergence of gradient along z on M field */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DMDz_Mag (
    amrex::Array4<amrex::Real const> const& F,
    amrex::Real const Ms_lo_z, amrex::Real const Ms_hi_z, amrex::Real const BC_lo_z, amrex::Real const BC_hi_z,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dz = 1./dx[2];
    if ( Ms_hi_z == 0.) {
        return 0.5 *(BC_hi_z + DownwardDz(F, i, j, k, dx));
    } else if ( Ms_lo_z == 0.){
        return 0.5 *(UpwardDz(F, i, j, k, dx) + BC_lo_z);
    } else {
        return 0.5 *(UpwardDz(F, i, j, k, dx) + DownwardDz(F, i, j, k, dx));
    }

 }

/** \brief
         * This GPU kernel averages a face-centered field onto a (possibly different nodality) face.
         * i,j,k are the spatial indices, n is the component index
         * iv_in is the nodality of the input field, iv_out is the nodality of the output.
         * Thus, if iv_in is the same as iv_out the value on the face is returned.
         * Some other examples when iv_in is different from iv_out:

         * Hy (0,1,0) components (in) to Hx face (1,0,0) (out)
         * return 0.25*(Hy(i,j,k)+Hy(i,j+1,k)+Hy(i-1,j,k)+Hy(i-1,j+1,k));
         *
         * \param[in] Fieldcomp  floating point data to be interpolated from one staggering location to another.
         * \param[in] iv_in           nodality of the input field, Fieldcomp
         * \param[in] iv_out         nodality of the output
         * \param[in] i                index along x of the Array4 Fieldcomp
         * \param[in] j                index along y of the Array4 Fieldcomp
         * \param[in] k                index along z of the Array4 Fieldcomp
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real face_avg_to_face (int i, int j, int k, int n,
                                    amrex::IntVect iv_in, amrex::IntVect iv_out,
                                    amrex::Array4<amrex::Real> const& Fieldcomp) {
    using namespace amrex;
    return ( 0.125_rt * ( Fieldcomp(i                   , j                   , k                   , n)
                        + Fieldcomp(i+iv_in[0]-iv_out[0], j                   , k                   , n)
                        + Fieldcomp(i                   , j+iv_in[1]-iv_out[1], k                   , n)
                        + Fieldcomp(i                   , j                   , k+iv_in[2]-iv_out[2], n)
                        + Fieldcomp(i+iv_in[0]-iv_out[0], j+iv_in[1]-iv_out[1], k                   , n)
                        + Fieldcomp(i+iv_in[0]-iv_out[0], j                   , k+iv_in[2]-iv_out[2], n)
                        + Fieldcomp(i                   , j+iv_in[1]-iv_out[1], k+iv_in[2]-iv_out[2], n)
                        + Fieldcomp(i+iv_in[0]-iv_out[0], j+iv_in[1]-iv_out[1], k+iv_in[2]-iv_out[2], n)
                        ));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real cc_avg_to_face (int i, int j, int k, int n,
                                    amrex::IntVect iv_in, amrex::IntVect iv_out,
                                    amrex::Array4<amrex::Real> const& Fieldcomp) {
    using namespace amrex;
    return ( 0.25_rt * ( Fieldcomp(i                   , j                   , k                   , n)
                        + Fieldcomp(i+iv_in[0]-iv_out[0], j+iv_in[1]-iv_out[1], k                   , n)
                        + Fieldcomp(i+iv_in[0]-iv_out[0], j                   , k+iv_in[2]-iv_out[2], n)
                        + Fieldcomp(i                   , j+iv_in[1]-iv_out[1], k+iv_in[2]-iv_out[2], n)
                        ));
}

/**
 update local M_field in the second-order time scheme
the objective is to output component n of the M_field
a and b have x,y,z components
**/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real updateM_field (int i, int j, int k, int n,
                            amrex::Array4<amrex::Real> const& a, amrex::Array4<amrex::Real> const& b) {
    using namespace amrex;
    amrex::Real a_square = pow(a(i, j, k, 0), 2.0) + pow(a(i, j, k, 1), 2.0) + pow(a(i, j, k, 2), 2.0);
    amrex::Real a_dot_b =  a(i, j, k, 0) * b(i, j, k, 0) +
                        a(i, j, k, 1) * b(i, j, k, 1) +
                        a(i, j, k, 2) * b(i, j, k, 2);
    // Initialize to 0.
    amrex::Real M_field = 0._rt;

    if(n==0){
        amrex::Real a_cross_b_x = a(i, j, k, 1) * b(i, j, k, 2) -
                                a(i, j, k, 2) * b(i, j, k, 1);
        M_field = ( b(i, j, k, 0) + a_dot_b * a(i, j, k, 0) - a_cross_b_x ) / ( 1.0 + a_square);
    }
    else if(n==1){
        amrex::Real a_cross_b_y = a(i, j, k, 2) * b(i, j, k, 0) -
                                a(i, j, k, 0) * b(i, j, k, 2);
        M_field = ( b(i, j, k, 1) + a_dot_b * a(i, j, k, 1) - a_cross_b_y ) / ( 1.0 + a_square);
    }
    else if(n==2){
        amrex::Real a_cross_b_z = a(i, j, k, 0) * b(i, j, k, 1) -
                                a(i, j, k, 1) * b(i, j, k, 0);
        M_field = ( b(i, j, k, 2) + a_dot_b * a(i, j, k, 2) - a_cross_b_z ) / ( 1.0 + a_square);
    }
    else{
        amrex::Abort("Wrong component n of the M_field");
    }
    return M_field;
}
