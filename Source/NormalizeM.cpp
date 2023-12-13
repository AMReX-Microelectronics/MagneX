#include "MagneX.H"

void NormalizeM(Array< MultiFab, AMREX_SPACEDIM >& Mfield,
	       	MultiFab& Ms,
                const Geometry& geom)
{
    // timer for profiling
    BL_PROFILE_VAR("NormalizeM()",NormalizeM);

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

    // fill interior and periodic ghost cells
    for (int comp = 0; comp < 3; comp++) {
        Mfield[comp].FillBoundary(geom.periodicity());
    }
    
}
