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
    amrex::Real const Ms_lo_x, amrex::Real const Ms_hi_x,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    return (UpwardDx(F, i, j, k, dx) + DownwardDx(F, i, j, k, dx))/2.;
 }


/**
  * Perform divergence along y on M field */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DivergenceDy_Mag (
    amrex::Array4<amrex::Real> const& F,
    amrex::Real const Ms_lo_y, amrex::Real const Ms_hi_y,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    return (UpwardDy(F, i, j, k, dx) + DownwardDy(F, i, j, k, dx))/2.;
 }

 /**
  * Perform divergence along z on M field */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DivergenceDz_Mag (
    amrex::Array4<amrex::Real> const& F,
    amrex::Real const Ms_lo_z, amrex::Real const Ms_hi_z,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    return (UpwardDz(F, i, j, k, dx) + DownwardDz(F, i, j, k, dx))/2.;
 }