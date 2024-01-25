#include "MagneX.H"
#include "CartesianAlgorithm_K.H"

using namespace amrex;

void CalculateH_DMI(std::array< MultiFab, AMREX_SPACEDIM> &   Mfield,
                    std::array< MultiFab, AMREX_SPACEDIM> &   H_DMIfield,
                    MultiFab&   Ms,
                    MultiFab&   exchange,
                    MultiFab&   DMI,
                    const Geometry& geom)
{
    // timer for profiling
    BL_PROFILE_VAR("CalculateH_DMI()",CalculateH_DMI);

    // calculate the b_temp_static, a_temp_static
    for (MFIter mfi(Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.validbox();

        // extract dd from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dd = geom.CellSizeArray();

        const Array4<Real>& Mx = Mfield[0].array(mfi); // note M_x include x components
        const Array4<Real>& My = Mfield[1].array(mfi); // note M_y include y components
        const Array4<Real>& Mz = Mfield[2].array(mfi); // note M_z include z components
        const Array4<Real>& Ms_arr = Ms.array(mfi);
        const Array4<Real>& DMI_arr = DMI.array(mfi);
        const Array4<Real>& exchange_arr = exchange.array(mfi);
        const Array4<Real>& Hx_DMI = H_DMIfield[0].array(mfi);   // x component
        const Array4<Real>& Hy_DMI = H_DMIfield[1].array(mfi);   // y component
        const Array4<Real>& Hz_DMI = H_DMIfield[2].array(mfi);   // z component

        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // determine if the material is magnetic or not
                if (Ms_arr(i,j,k) > 0.){

                    if (DMI_coupling == 1){
                        if (DMI_arr(i,j,k) == 0.) amrex::Abort("The DMI_arr(i,j,k) is 0.0 while including the DMI coupling term H_DMI for H_eff");
                        if (exchange_arr(i,j,k) == 0.) amrex::Abort("The exchange_arr(i,j,k) is 0.0 while DMI is turned on");

                        // H_DMI - use M^(old_time)
                        amrex::Real const H_DMI_coeff = 2.0 * DMI_arr(i,j,k) / mu0 / Ms_arr(i,j,k) / Ms_arr(i,j,k);
                        // Neumann boundary condition dM/dn = -1/xi (z x n) x M
                        amrex::Real const xi_DMI = 2.0*exchange_arr(i,j,k)/DMI_arr(i,j,k);

                        amrex::Real Ms_lo_x = Ms_arr(i-1, j, k);
                        amrex::Real Ms_hi_x = Ms_arr(i+1, j, k);
                        amrex::Real Ms_lo_y = Ms_arr(i, j-1, k);
                        amrex::Real Ms_hi_y = Ms_arr(i, j+1, k);

                        // Neumann boundary condition in scalar form, dMz/dx = +/- 1/xi*Mx 
                        
                        amrex::Real dMzdx_BC_lo_x =  1.0/xi_DMI*Mx(i,j,k); // lower x BC: dMz/dx = - 1/xi*Mx 
                        amrex::Real dMzdx_BC_hi_x =  1.0/xi_DMI*Mx(i,j,k); // higher x BC: dMz/dx = 1/xi*Mx 
                        Hx_DMI(i,j,k) = H_DMI_coeff * DMDx_Mag(Mz, Ms_lo_x, Ms_hi_x, dMzdx_BC_lo_x, dMzdx_BC_hi_x, i, j, k, dd); // z component at x nodality

                        amrex::Real dMzdy_BC_lo_y =  1.0/xi_DMI*My(i,j,k); // lower y BC: dMz/dy = -1/xi*My
                        amrex::Real dMzdy_BC_hi_y =  1.0/xi_DMI*My(i,j,k); // higher y BC: dMz/dy = 1/xi*My
                        Hy_DMI(i,j,k) = H_DMI_coeff * DMDy_Mag(Mz, Ms_lo_y, Ms_hi_y, dMzdy_BC_lo_y, dMzdy_BC_hi_y, i, j, k, dd); // z component at x nodality

                        amrex::Real dMxdx_BC_lo_x = -1.0/xi_DMI*Mz(i,j,k);  // lower x BC: dMx/dx = 1/xi*Mz
                        amrex::Real dMxdx_BC_hi_x = -1.0/xi_DMI*Mz(i,j,k); // higher x BC: dMx/dx = -1/xi*Mz
                        amrex::Real dMydy_BC_lo_y = -1.0/xi_DMI*Mz(i,j,k); // lower y BC: dMy/dy = 1/xi*Mz
                        amrex::Real dMydy_BC_hi_y = -1.0/xi_DMI*Mz(i,j,k); // higher y BC: dMy/dy = -1/xi*Mz
                        Hz_DMI(i,j,k) = H_DMI_coeff * (-DMDx_Mag(Mx, Ms_lo_x, Ms_hi_x, dMxdx_BC_lo_x, dMxdx_BC_hi_x, i, j, k, dd) // x component at x nodality
                                                        -DMDy_Mag(My, Ms_lo_y, Ms_hi_y, dMydy_BC_lo_y, dMydy_BC_hi_y, i, j, k, dd)); // y component at x nodality;

                    }
                }
            });
    }
}
