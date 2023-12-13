#include "CartesianAlgorithm_K.H"
#include "MagneX.H"

using namespace amrex;

void CalculateH_exchange(Array< MultiFab, AMREX_SPACEDIM>& Mfield,
                         Array< MultiFab, AMREX_SPACEDIM>& H_exchangefield,
                         MultiFab& Ms,
                         MultiFab& exchange,
                         MultiFab& DMI,
                         const Geometry& geom)
{
    // timer for profiling
    BL_PROFILE_VAR("CalculateH_exchange()",CalculateH_exchange);

    for (MFIter mfi(Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.validbox();

        // extract dd from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dd = geom.CellSizeArray();

        const Array4<Real>& Mx = Mfield[0].array(mfi); 
        const Array4<Real>& My = Mfield[1].array(mfi); 
        const Array4<Real>& Mz = Mfield[2].array(mfi); 
        const Array4<Real>& Ms_arr = Ms.array(mfi);
        const Array4<Real>& DMI_arr = DMI.array(mfi);
        const Array4<Real>& exchange_arr = exchange.array(mfi);
        const Array4<Real>& Hx_exchange = H_exchangefield[0].array(mfi); 
        const Array4<Real>& Hy_exchange = H_exchangefield[1].array(mfi); 
        const Array4<Real>& Hz_exchange = H_exchangefield[2].array(mfi); 

        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // determine if the material is magnetic or not
                if (Ms_arr(i,j,k) > 0.){

                    if (exchange_coupling == 1){
                        if (exchange_arr(i,j,k) == 0.) amrex::Abort("The exchange_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                        // H_exchange - use M^(old_time)
                        amrex::Real const H_exchange_coeff = 2.0 * exchange_arr(i,j,k) / mu0 / Ms_arr(i,j,k) / Ms_arr(i,j,k);
                        // Neumann boundary condition dM/dn = -1/xi (z x n) x M
                        amrex::Real xi_DMI = 0.0; // xi_DMI cannot be zero, this is just initialization
                        
                        amrex::Real Ms_lo_x = Ms_arr(i-1, j, k);
                        amrex::Real Ms_hi_x = Ms_arr(i+1, j, k);
                        amrex::Real Ms_lo_y = Ms_arr(i, j-1, k);
                        amrex::Real Ms_hi_y = Ms_arr(i, j+1, k);
                        amrex::Real Ms_lo_z = Ms_arr(i, j, k-1);
                        amrex::Real Ms_hi_z = Ms_arr(i, j, k+1);
                        
                        // // Neumann boundary condition in scalar form, dMx/dx = -/+ 1/xi*Mz 
                        // at x-faces, dM/dn = dM/dx
                        amrex::Real dMxdx_BC_lo_x = 0.0; // lower x BC: dMx/dx = 1/xi*Mz
                        amrex::Real dMxdx_BC_hi_x = 0.0; // higher x BC: dMx/dx = -1/xi*Mz
                        amrex::Real dMxdy_BC_lo_y = 0.0;
                        amrex::Real dMxdy_BC_hi_y = 0.0;
                        amrex::Real dMxdz_BC_lo_z = 0.0;
                        amrex::Real dMxdz_BC_hi_z = 0.0;

                        amrex::Real dMydx_BC_lo_x = 0.0;
                        amrex::Real dMydx_BC_hi_x = 0.0;
                        amrex::Real dMydy_BC_lo_y = 0.0; // if with DMI, lower y BC: dMy/dy = 1/xi*Mz
                        amrex::Real dMydy_BC_hi_y = 0.0; // if with DMI, higher y BC: dMy/dy = -1/xi*Mz
                        amrex::Real dMydz_BC_lo_z = 0.0;
                        amrex::Real dMydz_BC_hi_z = 0.0;

                        amrex::Real dMzdx_BC_lo_x = 0.0; // if with DMI, lower x BC: dMz/dx = -1/xi*Mx
                        amrex::Real dMzdx_BC_hi_x = 0.0; // if with DMI, higher x BC: dMz/dx = 1/xi*Mx
                        amrex::Real dMzdy_BC_lo_y = 0.0; // if with DMI, lower y BC: dMz/dy = -1/xi*My
                        amrex::Real dMzdy_BC_hi_y = 0.0; // if with DMI, higher y BC: dMz/dy = 1/xi*My
                        amrex::Real dMzdz_BC_lo_z = 0.0; // dMz/dz = 0
                        amrex::Real dMzdz_BC_hi_z = 0.0; // dMz/dz = 0

                        if (DMI_coupling == 1) {
                            if (DMI_arr(i,j,k) == 0.) amrex::Abort("The DMI_arr(i,j,k) is 0.0 while including the DMI coupling");
                            
                            xi_DMI = 2.0*exchange_arr(i,j,k)/DMI_arr(i,j,k);

                            dMxdx_BC_lo_x = -1.0/xi_DMI*Mz(i,j,k) ; // lower x BC: dMx/dx = 1/xi*Mz
                            dMxdx_BC_hi_x = -1.0/xi_DMI*Mz(i,j,k) ; // higher x BC: dMx/dx = -1/xi*Mz

                            dMydy_BC_lo_y = -1.0/xi_DMI*Mz(i,j,k) ; // lower y BC: dMy/dy = 1/xi*Mz
                            dMydy_BC_hi_y = -1.0/xi_DMI*Mz(i,j,k) ; // higher y BC: dMy/dy = -1/xi*Mz

                            dMzdx_BC_lo_x =  1.0/xi_DMI*Mx(i,j,k);  // lower x BC: dMz/dx = -1/xi*Mx
                            dMzdx_BC_hi_x =  1.0/xi_DMI*Mx(i,j,k);  // higher x BC: dMz/dx = 1/xi*Mx
                            dMzdy_BC_lo_y =  1.0/xi_DMI*My(i,j,k);  // lower y BC: dMz/dy = -1/xi*My
                            dMzdy_BC_hi_y =  1.0/xi_DMI*My(i,j,k);  // higher y BC: dMz/dy = 1/xi*My
                        }
                        
                        Hx_exchange(i,j,k) = H_exchange_coeff * Laplacian_Mag(Mx, Ms_lo_x, Ms_hi_x, dMxdx_BC_lo_x, dMxdx_BC_hi_x, 
                                                                                  Ms_lo_y, Ms_hi_y, dMxdy_BC_lo_y, dMxdy_BC_hi_y,
                                                                                  Ms_lo_z, Ms_hi_z, dMxdz_BC_lo_z, dMxdz_BC_hi_z, i, j, k, dd);
                        
                        Hy_exchange(i,j,k) = H_exchange_coeff * Laplacian_Mag(My, Ms_lo_x, Ms_hi_x, dMydx_BC_lo_x, dMydx_BC_hi_x, 
                                                                                  Ms_lo_y, Ms_hi_y, dMydy_BC_lo_y, dMydy_BC_hi_y, 
                                                                                  Ms_lo_z, Ms_hi_z, dMydz_BC_lo_z, dMydz_BC_hi_z, i, j, k, dd);
                        
                        Hz_exchange(i,j,k) = H_exchange_coeff * Laplacian_Mag(Mz, Ms_lo_x, Ms_hi_x, dMzdx_BC_lo_x, dMzdx_BC_hi_x,
                                                                                  Ms_lo_y, Ms_hi_y, dMzdy_BC_lo_y, dMzdy_BC_hi_y,
                                                                                  Ms_lo_z, Ms_hi_z, dMzdz_BC_lo_z, dMzdz_BC_hi_z, i, j, k, dd);

                    }
                } else {
                    Hx_exchange(i,j,k) = Hy_exchange(i,j,k) = Hz_exchange(i,j,k) = 0.;
                }
            });
    }
}
