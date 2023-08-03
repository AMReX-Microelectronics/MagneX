/**
 * Compute the x component of the RHS of LLG equation given M, alpha, gamma, |M|, and H_eff*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real LLG_RHS_x(
    amrex::Array4<amrex::Real const> const& Mx,
    amrex::Array4<amrex::Real const> const& My,
    amrex::Array4<amrex::Real const> const& Mz,
    amrex::Array4<amrex::Real> const& alpha,
    amrex::Array4<amrex::Real> const& gamma,
    Real const M_magnitude, Real const mu0,
    Real const Hx_eff, Real const Hy_eff, Real const Hz_eff,
    int const i, int const j, int const k) {

    amrex::Real mag_gammaL = gamma(i,j,k) / (1._rt + std::pow(alpha(i,j,k), 2._rt));
    return (mu0 * mag_gammaL) * (My(i, j, k) * Hz_eff - Mz(i, j, k) * Hy_eff
                              + alpha(i,j,k) / M_magnitude * (My(i, j, k) * (Mx(i, j, k) * Hy_eff - My(i, j, k) * Hx_eff) - Mz(i, j, k) * (Mz(i, j, k) * Hx_eff - Mx(i, j, k) * Hz_eff)));
}


/**
 * Compute the y component of the RHS of LLG equation given M, alpha, gamma, |M|, and H_eff*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real LLG_RHS_y(
    amrex::Array4<amrex::Real const> const& Mx,
    amrex::Array4<amrex::Real const> const& My,
    amrex::Array4<amrex::Real const> const& Mz,
    amrex::Array4<amrex::Real> const& alpha,
    amrex::Array4<amrex::Real> const& gamma,
    Real const M_magnitude, Real const mu0,
    Real const Hx_eff, Real const Hy_eff, Real const Hz_eff,
    int const i, int const j, int const k) {

    amrex::Real mag_gammaL = gamma(i,j,k) / (1._rt + std::pow(alpha(i,j,k), 2._rt));
    return (mu0 * mag_gammaL) * (Mz(i, j, k) * Hx_eff - Mx(i, j, k) * Hz_eff
                              + alpha(i,j,k) / M_magnitude * (Mz(i, j, k) * (My(i, j, k) * Hz_eff - Mz(i, j, k) * Hy_eff) - Mx(i, j, k) * (Mx(i, j, k) * Hy_eff - My(i, j, k) * Hx_eff)));
}


/**
 * Compute the z component of the RHS of LLG equation given M, alpha, gamma, |M|, and H_eff*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real LLG_RHS_z(
    amrex::Array4<amrex::Real const> const& Mx,
    amrex::Array4<amrex::Real const> const& My,
    amrex::Array4<amrex::Real const> const& Mz,
    amrex::Array4<amrex::Real> const& alpha,
    amrex::Array4<amrex::Real> const& gamma,
    Real const M_magnitude, Real const mu0,
    Real const Hx_eff, Real const Hy_eff, Real const Hz_eff,
    int const i, int const j, int const k) {

    amrex::Real mag_gammaL = gamma(i,j,k) / (1._rt + std::pow(alpha(i,j,k), 2._rt));
    return (mu0 * mag_gammaL) * (Mx(i, j, k) * Hy_eff - My(i, j, k) * Hx_eff
                              + alpha(i,j,k) / M_magnitude * (Mx(i, j, k) * (Mz(i, j, k) * Hx_eff - Mx(i, j, k) * Hz_eff) - My(i, j, k) * (My(i, j, k) * Hz_eff - Mz(i, j, k) * Hy_eff)));
}
