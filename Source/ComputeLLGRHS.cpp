#include "MagneX.H"

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


void Compute_LLG_RHS(
                   Array< MultiFab, AMREX_SPACEDIM >&  LLG_RHS,
                   const Array< MultiFab, AMREX_SPACEDIM >&   Mfield_old,
                   Array< MultiFab, AMREX_SPACEDIM >&   H_demagfield,
                   Array< MultiFab, AMREX_SPACEDIM >&   H_biasfield,
                   Array< MultiFab, AMREX_SPACEDIM >&   H_exchangefield,
                   Array< MultiFab, AMREX_SPACEDIM >&   H_DMIfield,
                   Array< MultiFab, AMREX_SPACEDIM >&   H_anisotropyfield,
                   MultiFab&   alpha,
                   MultiFab&   Ms,
                   MultiFab&   gamma,
                   int demag_coupling,
                   int exchange_coupling,
                   int DMI_coupling,
                   int anisotropy_coupling,
                   int M_normalization, 
                   Real mu0)
{
    //for (MFIter mfi(*Mfield[0]); mfi.isValid(); ++mfi)
    for (MFIter mfi(LLG_RHS[0]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        // extract field data
        const Array4<Real>& Hx_demag= H_demagfield[0].array(mfi);
        const Array4<Real>& Hy_demag= H_demagfield[1].array(mfi);
        const Array4<Real>& Hz_demag= H_demagfield[2].array(mfi);
        const Array4<Real>& LLG_rhs_x = LLG_RHS[0].array(mfi);         
        const Array4<Real>& LLG_rhs_y = LLG_RHS[1].array(mfi);         
        const Array4<Real>& LLG_rhs_z = LLG_RHS[2].array(mfi);         
        const Array4<Real const>& Mx_old = Mfield_old[0].array(mfi); 
        const Array4<Real const>& My_old = Mfield_old[1].array(mfi); 
        const Array4<Real const>& Mz_old = Mfield_old[2].array(mfi); 
        const Array4<Real>& Hx_bias = H_biasfield[0].array(mfi);
        const Array4<Real>& Hy_bias = H_biasfield[1].array(mfi);
        const Array4<Real>& Hz_bias = H_biasfield[2].array(mfi);
      
        const Array4<Real>& Hx_exchange = H_exchangefield[0].array(mfi);
        const Array4<Real>& Hy_exchange = H_exchangefield[1].array(mfi);
        const Array4<Real>& Hz_exchange = H_exchangefield[2].array(mfi);
      
        const Array4<Real>& Hx_DMI = H_DMIfield[0].array(mfi);
        const Array4<Real>& Hy_DMI = H_DMIfield[1].array(mfi);
        const Array4<Real>& Hz_DMI = H_DMIfield[2].array(mfi);
      
        const Array4<Real>& Hx_anisotropy = H_anisotropyfield[0].array(mfi);
        const Array4<Real>& Hy_anisotropy = H_anisotropyfield[1].array(mfi);
        const Array4<Real>& Hz_anisotropy = H_anisotropyfield[2].array(mfi);
      
        const Array4<Real>& alpha_arr = alpha.array(mfi);
        const Array4<Real>& gamma_arr = gamma.array(mfi);
        const Array4<Real>& Ms_arr = Ms.array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (Ms_arr(i,j,k) > 0._rt)
            {
                amrex::Real Hx_eff = Hx_bias(i,j,k);
                amrex::Real Hy_eff = Hy_bias(i,j,k);
                amrex::Real Hz_eff = Hz_bias(i,j,k);
                if(demag_coupling == 1)
                {
                    Hx_eff += Hx_demag(i,j,k);
                    Hy_eff += Hy_demag(i,j,k);
                    Hz_eff += Hz_demag(i,j,k);
                }
                if(exchange_coupling == 1)
                { 
                    Hx_eff += Hx_exchange(i,j,k);
                    Hy_eff += Hy_exchange(i,j,k);
                    Hz_eff += Hz_exchange(i,j,k);
                }
             
                if(DMI_coupling == 1)
                { 
                    Hx_eff += Hx_DMI(i,j,k);
                    Hy_eff += Hy_DMI(i,j,k);
                    Hz_eff += Hz_DMI(i,j,k);
                }
             
                if(anisotropy_coupling == 1)
                {
                    Hx_eff += Hx_anisotropy(i,j,k);
                    Hy_eff += Hy_anisotropy(i,j,k);
                    Hz_eff += Hz_anisotropy(i,j,k);
                }
                //Update M
                // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(Mx_old(i, j, k), 2._rt) + std::pow(My_old(i, j, k), 2._rt) + std::pow(Mz_old(i, j, k), 2._rt))
                                                           : Ms_arr(i,j,k); 
                // x component on x-faces of grid
                LLG_rhs_x(i, j, k) = LLG_RHS_x(Mx_old, My_old, Mz_old, alpha_arr, gamma_arr, M_magnitude, mu0, Hx_eff, Hy_eff, Hz_eff, i, j, k);
                // y component on x-faces of grid
                LLG_rhs_y(i, j, k) = LLG_RHS_y(Mx_old, My_old, Mz_old, alpha_arr, gamma_arr, M_magnitude, mu0, Hx_eff, Hy_eff, Hz_eff, i, j, k);
                // z component on x-faces of grid
                LLG_rhs_z(i, j, k) = LLG_RHS_z(Mx_old, My_old, Mz_old, alpha_arr, gamma_arr, M_magnitude, mu0, Hx_eff, Hy_eff, Hz_eff, i, j, k);
            }   

        });     
    }
}
