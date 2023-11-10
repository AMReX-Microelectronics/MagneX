#include "EvolveM.H"
#include "CartesianAlgorithm.H"
#include "ComputeLLG_RHS.H"

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
                   Real mu0,
                   const Geometry& geom, const Real time)
{
    //for (MFIter mfi(*Mfield[0]); mfi.isValid(); ++mfi)
    for (MFIter mfi(LLG_RHS[0]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
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

void NormalizeM(Array< MultiFab, AMREX_SPACEDIM >& Mfield,
	       	MultiFab& Ms, int M_normalization)
{
    for (MFIter mfi(Mfield[0]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        // extract field data
        const Array4<Real>& Mx = Mfield[0].array(mfi);         
        const Array4<Real>& My = Mfield[1].array(mfi);         
        const Array4<Real>& Mz = Mfield[2].array(mfi);         
        const Array4<Real>& Ms_arr = Ms.array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (Ms_arr(i,j,k) > 0._rt)
            {
                // temporary normalized magnitude of M_xface field at the fixed point
                amrex::Real M_magnitude_normalized = std::sqrt(std::pow(Mx(i, j, k), 2._rt) + std::pow(My(i, j, k), 2._rt) + std::pow(Mz(i, j, k), 2._rt)) / Ms_arr(i,j,k);
                amrex::Real normalized_error = 0.1;
                if (M_normalization > 0)
                {
                    // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                    // check the normalized error
                    if (amrex::Math::abs(1._rt - M_magnitude_normalized) > normalized_error)
                    {
                        printf("M_magnitude_normalized = %g \n", M_magnitude_normalized);
                        printf("i = %d, j = %d, k = %d \n", i, j,k);
                        amrex::Abort("Saturated case: M has drifted from Ms by more than the normalized error threshold");
                    }
                    // normalize the M field
                    Mx(i, j, k) /= M_magnitude_normalized;
                    My(i, j, k) /= M_magnitude_normalized;
                    Mz(i, j, k) /= M_magnitude_normalized;
                }
                else if (M_normalization == 0)
                {
                    // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                    // check the normalized error
                    if (M_magnitude_normalized > (1._rt + normalized_error))
                    {
                        printf("M_magnitude_normalized = %g \n", M_magnitude_normalized);
                        printf("i = %d, j = %d, k = %d \n", i, j,k);
                        amrex::Abort("Unsaturated case: M has exceeded Ms by more than the normalized error threshold");
                    }
                    else if (M_magnitude_normalized > 1._rt && M_magnitude_normalized <= (1._rt + normalized_error) )
                    {
                        // normalize the M field
                        Mx(i, j, k) /= M_magnitude_normalized;
                        My(i, j, k) /= M_magnitude_normalized;
                        Mz(i, j, k) /= M_magnitude_normalized;
                    }
                }  
            }
        });             
    }
}
