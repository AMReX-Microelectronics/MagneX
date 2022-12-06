//Algorithm to calculate Laplacian for exchange term in LLG equation

/**
 * Perform derivative along x on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real UpwardDx (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx, int const ncomp) {

    amrex::Real const inv_dx = 1./dx[0];
    return inv_dx*( F(i+1,j,k,ncomp) - F(i,j,k,ncomp) );
}


/**
 * Perform derivative along x on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DownwardDx (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx, int const ncomp) {

    amrex::Real const inv_dx = 1./dx[0];
    return inv_dx*( F(i,j,k,ncomp) - F(i-1,j,k,ncomp) );
}


/**
 * Perform derivative along y on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real UpwardDy (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx, int const ncomp) {

    amrex::Real const inv_dy = 1./dx[1];
    return inv_dy*( F(i,j+1,k,ncomp) - F(i,j,k,ncomp) );
}


/**
 * Perform derivative along y on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DownwardDy (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx, int const ncomp) {

    amrex::Real const inv_dy = 1./dx[1];
    return inv_dy*( F(i,j,k,ncomp) - F(i,j-1,k,ncomp) );
}



/**
 * Perform derivative along z on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real UpwardDz (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx, int const ncomp) {

    amrex::Real const inv_dz = 1./dx[2];
    return inv_dz*( F(i,j,k+1,ncomp) - F(i,j,k,ncomp) );
}


/**
 * Perform derivative along z on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DownwardDz (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx, int const ncomp) {

    amrex::Real const inv_dz = 1./dx[2];
    return inv_dz*( F(i,j,k,ncomp) - F(i,j,k-1,ncomp) );
}

/**
  * Perform divergence of gradient along x on M field */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real LaplacianDx_Mag (
    amrex::Array4<amrex::Real> const& F,
    amrex::Real const Ms_lo_x, amrex::Real const Ms_hi_x,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx, int const ncomp, int const nodality) {

    amrex::Real const inv_dx = 1./dx[0];
    if (nodality == 0){ // x face (normal face) dM/dx = 0
       if ( Ms_hi_x == 0.) {
           return 0.5 * inv_dx * inv_dx * (8. * F(i-1, j, k, ncomp) - F(i-2, j, k, ncomp) - 7. * F(i, j, k, ncomp));
       } else if ( Ms_lo_x == 0.){
           return 0.5 * inv_dx * inv_dx * (8. * F(i+1, j, k, ncomp) - F(i+2, j, k, ncomp) - 7. * F(i, j, k, ncomp));
       } else {
           return inv_dx*(UpwardDx(F, i, j, k, dx, ncomp) - DownwardDx(F, i, j, k, dx, ncomp));
       }

    } else { //y or z face
       if ( Ms_hi_x == 0.) {
           return inv_dx*(0. - DownwardDx(F, i, j, k, dx, ncomp));
       } else if ( Ms_lo_x == 0.){
           return inv_dx*(UpwardDx(F, i, j, k, dx, ncomp) - 0.);
       } else {
           return inv_dx*(UpwardDx(F, i, j, k, dx, ncomp) - DownwardDx(F, i, j, k, dx, ncomp));
       }
    }

 }


/**
  * Perform divergence of gradient along y on M field */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real LaplacianDy_Mag (
    amrex::Array4<amrex::Real> const& F,
    amrex::Real const Ms_lo_y, amrex::Real const Ms_hi_y,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx, int const ncomp, int const nodality) {

    amrex::Real const inv_dy = 1./dx[1];
    if (nodality == 1){ // y face (normal face) dM/dy = 0
       if ( Ms_hi_y == 0.) {
           return 0.5 * inv_dy * inv_dy * (8. * F(i, j-1, k, ncomp) - F(i, j-2, k, ncomp) - 7. * F(i, j, k, ncomp));
       } else if ( Ms_lo_y == 0.){
           return 0.5 * inv_dy * inv_dy * (8. * F(i, j+1, k, ncomp) - F(i, j+2, k, ncomp) - 7. * F(i, j, k, ncomp));
       } else {
           return inv_dy*(UpwardDy(F, i, j, k, dx, ncomp) - DownwardDy(F, i, j, k, dx, ncomp));
       }

    } else { // x or z face
       if ( Ms_hi_y == 0.) {
           return inv_dy*(0. - DownwardDy(F, i, j, k, dx, ncomp));
       } else if ( Ms_lo_y == 0.){
           return inv_dy*(UpwardDy(F, i, j, k, dx, ncomp) - 0.);
       } else {
           return inv_dy*(UpwardDy(F, i, j, k, dx, ncomp) - DownwardDy(F, i, j, k, dx, ncomp));
       }
    }

 }


/**
  * Perform divergence of gradient along z on M field */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real LaplacianDz_Mag (
    amrex::Array4<amrex::Real> const& F,
    amrex::Real const Ms_lo_z, amrex::Real const Ms_hi_z,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx, int const ncomp, int const nodality) {

    amrex::Real const inv_dz = 1./dx[2];
    if (nodality == 2){ // z face (normal face) dM/dz = 0
       if ( Ms_hi_z == 0.) {
           return 0.5 * inv_dz * inv_dz * (8. * F(i, j, k-1, ncomp) - F(i, j, k-2, ncomp) - 7. * F(i, j, k, ncomp));
       } else if ( Ms_lo_z == 0.){
           return 0.5 * inv_dz * inv_dz * (8. * F(i, j, k+1, ncomp) - F(i, j, k+2, ncomp) - 7. * F(i, j, k, ncomp));
       } else {
           return inv_dz*(UpwardDz(F, i, j, k, dx, ncomp) - DownwardDz(F, i, j, k, dx, ncomp));
       }

    } else { // x or y face
       if ( Ms_hi_z == 0.) {
           return inv_dz*(0. - DownwardDz(F, i, j, k, dx, ncomp));
       } else if ( Ms_lo_z == 0.){
           return inv_dz*(UpwardDz(F, i, j, k, dx, ncomp) - 0.);
       } else {
           return inv_dz*(UpwardDz(F, i, j, k, dx, ncomp) - DownwardDz(F, i, j, k, dx, ncomp));
       }
    }

 }

/**
  * Compute the sum to get Laplacian of M field */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real Laplacian_Mag (
     amrex::Array4<amrex::Real> const& F,
     amrex::Real const Ms_lo_x, amrex::Real const Ms_hi_x, amrex::Real const Ms_lo_y, amrex::Real const Ms_hi_y, amrex::Real const Ms_lo_z, amrex::Real const Ms_hi_z,
     int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx, int const ncomp, int const nodality) {

     //amrex::Print() << "This is called " << Ms_lo_x << "\n";
     return LaplacianDx_Mag(F, Ms_lo_x, Ms_hi_x, i, j, k, dx, ncomp, nodality) + LaplacianDy_Mag(F, Ms_lo_y, Ms_hi_y, i, j, k, dx, ncomp, nodality) + LaplacianDz_Mag(F, Ms_lo_z, Ms_hi_z, i, j, k, dx, ncomp, nodality);
 }

 /**
  * Perform divergence  along x on M field */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DivergenceDx_Mag (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx, int const ncomp) {

    return (UpwardDx(F, i, j, k, dx, ncomp) + DownwardDx(F, i, j, k, dx, ncomp))/2.;
 }


/**
  * Perform divergence along y on M field */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DivergenceDy_Mag (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx, int const ncomp) {

    return (UpwardDy(F, i, j, k, dx, ncomp) + DownwardDy(F, i, j, k, dx, ncomp))/2.;
 }

 /**
  * Perform divergence along z on M field */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DivergenceDz_Mag (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx, int const ncomp) {

    return (UpwardDz(F, i, j, k, dx, ncomp) + DownwardDz(F, i, j, k, dx, ncomp))/2.;
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