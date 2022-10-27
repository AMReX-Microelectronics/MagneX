
/**
 * Compute the x component of the RHS of LLG equation given M, alpha, gamma, |M|, and H_eff*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real LLG_RHS_x(
    amrex::Array4<amrex::Real> const& M,
    amrex::Array4<amrex::Real> const& alpha,
    amrex::Array4<amrex::Real> const& gamma,
    Real const M_magnitude, Real const mu0,
    Real const Hx_eff, Real const Hy_eff, Real const Hz_eff,
    int const i, int const j, int const k) {

    amrex::Real mag_gammaL = gamma(i,j,k) / (1._rt + std::pow(alpha(i,j,k), 2._rt));
    return (mu0 * mag_gammaL) * (M(i, j, k, 1) * Hz_eff - M(i, j, k, 2) * Hy_eff
                              + alpha(i,j,k) / M_magnitude * (M(i, j, k, 1) * (M(i, j, k, 0) * Hy_eff - M(i, j, k, 1) * Hx_eff) - M(i, j, k, 2) * (M(i, j, k, 2) * Hx_eff - M(i, j, k, 0) * Hz_eff)));
}


/**
 * Compute the y component of the RHS of LLG equation given M, alpha, gamma, |M|, and H_eff*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real LLG_RHS_y(
    amrex::Array4<amrex::Real> const& M,
    amrex::Array4<amrex::Real> const& alpha,
    amrex::Array4<amrex::Real> const& gamma,
    Real const M_magnitude, Real const mu0,
    Real const Hx_eff, Real const Hy_eff, Real const Hz_eff,
    int const i, int const j, int const k) {

    amrex::Real mag_gammaL = gamma(i,j,k) / (1._rt + std::pow(alpha(i,j,k), 2._rt));
    return (mu0 * mag_gammaL) * (M(i, j, k, 2) * Hx_eff - M(i, j, k, 0) * Hz_eff
                              + alpha(i,j,k) / M_magnitude * (M(i, j, k, 2) * (M(i, j, k, 1) * Hz_eff - M(i, j, k, 2) * Hy_eff) - M(i, j, k, 0) * (M(i, j, k, 0) * Hy_eff - M(i, j, k, 1) * Hx_eff)));
}



/**
 * Compute the z component of the RHS of LLG equation given M, alpha, gamma, |M|, and H_eff*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real LLG_RHS_z(
    amrex::Array4<amrex::Real> const& M,
    amrex::Array4<amrex::Real> const& alpha,
    amrex::Array4<amrex::Real> const& gamma,
    Real const M_magnitude, Real const mu0,
    Real const Hx_eff, Real const Hy_eff, Real const Hz_eff,
    int const i, int const j, int const k) {

    amrex::Real mag_gammaL = gamma(i,j,k) / (1._rt + std::pow(alpha(i,j,k), 2._rt));
    return (mu0 * mag_gammaL) * (M(i, j, k, 0) * Hy_eff - M(i, j, k, 1) * Hx_eff
                              + alpha(i,j,k) / M_magnitude * (M(i, j, k, 0) * (M(i, j, k, 2) * Hx_eff - M(i, j, k, 0) * Hz_eff) - M(i, j, k, 1) * (M(i, j, k, 1) * Hz_eff - M(i, j, k, 2) * Hy_eff)));
}
