#include "CartesianAlgorithm.H"
#include "EffectiveDMIField.H"
#include <AMReX_MLMG.H> 
#include <AMReX_MultiFab.H> 
#include <AMReX_VisMF.H>
#include <AMReX_OpenBC.H>

using namespace amrex;

void CalculateH_DMI(
    amrex::Vector<MultiFab>& Mfield,
    std::array< MultiFab, AMREX_SPACEDIM> &   H_DMIfield,
    std::array< MultiFab, AMREX_SPACEDIM >&   Ms,
    std::array< MultiFab, AMREX_SPACEDIM >&   exchange,
    std::array< MultiFab, AMREX_SPACEDIM >&   DMI,
    int exchange_coupling,
    int DMI_coupling,
    Real mu0,
    const Geometry& geom)
{
    // calculate the b_temp_static, a_temp_static
    for (MFIter mfi(Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        // extract dd from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dd = geom.CellSizeArray();

        const Array4<Real>& M_xface = Mfield[0].array(mfi); // note M_xface include x,y,z components at |_x faces
        const Array4<Real>& M_yface = Mfield[1].array(mfi); // note M_yface include x,y,z components at |_y faces
        const Array4<Real>& M_zface = Mfield[2].array(mfi); // note M_zface include x,y,z components at |_z faces
        const Array4<Real>& Ms_xface_arr = Ms[0].array(mfi);
        const Array4<Real>& Ms_yface_arr = Ms[1].array(mfi);
        const Array4<Real>& Ms_zface_arr = Ms[2].array(mfi);
        const Array4<Real>& DMI_xface_arr = DMI[0].array(mfi);
        const Array4<Real>& DMI_yface_arr = DMI[1].array(mfi);
        const Array4<Real>& DMI_zface_arr = DMI[2].array(mfi);
        const Array4<Real>& exchange_xface_arr = exchange[0].array(mfi);
        const Array4<Real>& exchange_yface_arr = exchange[1].array(mfi);
        const Array4<Real>& exchange_zface_arr = exchange[2].array(mfi);
        const Array4<Real>& H_DMI_xface = H_DMIfield[0].array(mfi);   // x,y,z component at |_x faces
        const Array4<Real>& H_DMI_yface = H_DMIfield[1].array(mfi);   // x,y,z component at |_y faces
        const Array4<Real>& H_DMI_zface = H_DMIfield[2].array(mfi);   // x,y,z component at |_z faces

        // extract tileboxes for which to loop
        amrex::IntVect Mxface_stag = Mfield[0].ixType().toIntVect();
        amrex::IntVect Myface_stag = Mfield[1].ixType().toIntVect();
        amrex::IntVect Mzface_stag = Mfield[2].ixType().toIntVect();
        // extract tileboxes for which to loop
        Box const &tbx = mfi.tilebox(Mfield[0].ixType().toIntVect());
        Box const &tby = mfi.tilebox(Mfield[1].ixType().toIntVect());
        Box const &tbz = mfi.tilebox(Mfield[2].ixType().toIntVect());

        amrex::ParallelFor(tbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // determine if the material is magnetic or not
                if (Ms_xface_arr(i,j,k) > 0.){

                    if (DMI_coupling == 1){
                        if (DMI_xface_arr(i,j,k) == 0.) amrex::Abort("The DMI_xface_arr(i,j,k) is 0.0 while including the DMI coupling term H_DMI for H_eff");
                        if (exchange_xface_arr(i,j,k) == 0.) amrex::Abort("The exchange_xface_arr(i,j,k) is 0.0 while DMI is turned on");

                        // H_DMI - use M^(old_time)
                        amrex::Real const H_DMI_coeff = 2.0 * DMI_xface_arr(i,j,k) / mu0 / Ms_xface_arr(i,j,k) / Ms_xface_arr(i,j,k);
                        // Neumann boundary condition dM/dn = -1/xi (z x n) x M
                        amrex::Real const xi_DMI = 2.0*exchange_xface_arr(i,j,k)/DMI_xface_arr(i,j,k);

                        amrex::Real Ms_lo_x = Ms_xface_arr(i-1, j, k);
                        amrex::Real Ms_hi_x = Ms_xface_arr(i+1, j, k);
                        amrex::Real Ms_lo_y = Ms_xface_arr(i, j-1, k);
                        amrex::Real Ms_hi_y = Ms_xface_arr(i, j+1, k);

                        // Neumann boundary condition in scalar form, dMz/dx = +/- 1/xi*Mx 
                        
                        // H_DMI_xface(i,j,k,0) = H_DMI_coeff * dMzdx;
                        amrex::Real dMzdx_BC_lo_x = -1.0/xi_DMI*M_xface(i,j,k,0); // lower x BC: dMz/dx = - 1/xi*Mx 
                        amrex::Real dMzdx_BC_hi_x =  1.0/xi_DMI*M_xface(i,j,k,0); // higher x BC: dMz/dx = 1/xi*Mx 
                        H_DMI_xface(i,j,k,0) = H_DMI_coeff * DMDx_Mag(M_xface, Ms_lo_x, Ms_hi_x, dMzdx_BC_lo_x, dMzdx_BC_hi_x, i, j, k, dd, 2, 0); // z component at x nodality

                        //H_DMI_xface(i,j,k,1) = H_DMI_coeff * dMzdy;
                        amrex::Real dMzdy_BC_lo_y = -1.0/xi_DMI*M_xface(i,j,k,1); // lower y BC: dMz/dy = -1/xi*My
                        amrex::Real dMzdy_BC_hi_y =  1.0/xi_DMI*M_xface(i,j,k,1); // higher y BC: dMz/dy = 1/xi*My
                        H_DMI_xface(i,j,k,1) = H_DMI_coeff * DMDy_Mag(M_xface, Ms_lo_y, Ms_hi_y, dMzdy_BC_lo_y, dMzdy_BC_hi_y, i, j, k, dd, 2, 0); // z component at x nodality

                        // H_DMI_xface(i,j,k,2) = H_DMI_coeff * (-dMxdx - dMydy);
                        amrex::Real dMxdx_BC_lo_x =  1.0/xi_DMI*M_xface(i,j,k,2);  // lower x BC: dMx/dx = 1/xi*Mz
                        amrex::Real dMxdx_BC_hi_x = -1.0/xi_DMI*M_xface(i,j,k,2); // higher x BC: dMx/dx = -1/xi*Mz
                        amrex::Real dMydy_BC_lo_y =  1.0/xi_DMI*M_xface(i,j,k,2); // lower y BC: dMy/dy = 1/xi*Mz
                        amrex::Real dMydy_BC_hi_y = -1.0/xi_DMI*M_xface(i,j,k,2); // higher y BC: dMy/dy = -1/xi*Mz
                        H_DMI_xface(i,j,k,2) = H_DMI_coeff * (-DMDx_Mag(M_xface, Ms_lo_x, Ms_hi_x, dMxdx_BC_lo_x, dMxdx_BC_hi_x, i, j, k, dd, 0, 0) // x component at x nodality
                                                              -DMDy_Mag(M_xface, Ms_lo_y, Ms_hi_y, dMydy_BC_lo_y, dMydy_BC_hi_y, i, j, k, dd, 1, 0)); // y component at x nodality;

                        if (i == 0 && j == 0 && k == 0){
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMzdx_BC_lo_x_xface = " << dMzdx_BC_lo_x << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMzdx_BC_hi_x_xface = " << dMzdx_BC_hi_x << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMzdy_BC_lo_y_xface = " << dMzdy_BC_lo_y << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMzdy_BC_hi_y_xface = " << dMzdy_BC_hi_y << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMxdx_BC_lo_x_xface = " << dMxdx_BC_lo_x << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMxdx_BC_hi_x_xface = " << dMxdx_BC_hi_x << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMydy_BC_lo_y_xface = " << dMydy_BC_lo_y << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMydy_BC_hi_y_xface = " << dMydy_BC_hi_y << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", xi_DMI_xface = " << xi_DMI << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", H_DMI_coeff_xface = " << H_DMI_coeff << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", Hx_DMI_xface = " << H_DMI_xface(i,j,k,0) << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", Hy_DMI_xface = " << H_DMI_xface(i,j,k,1) << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", Hz_DMI_xface = " << H_DMI_xface(i,j,k,2) << "\n";
                        }
                    }
                }
            });

        amrex::ParallelFor(tby,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // determine if the material is magnetic or not
                if (Ms_yface_arr(i,j,k) > 0.){
                    if (DMI_coupling == 1){

                        if (DMI_yface_arr(i,j,k) == 0.) amrex::Abort("The DMI_yface_arr(i,j,k) is 0.0 while including the DMI coupling term H_DMI for H_eff");
                        if (exchange_yface_arr(i,j,k) == 0.) amrex::Abort("The exchange_yface_arr(i,j,k) is 0.0 while DMI is turned on");

                        // H_DMI - use M^(old_time)
                        amrex::Real const H_DMI_coeff = 2.0 * DMI_yface_arr(i,j,k) / mu0 / Ms_yface_arr(i,j,k) / Ms_yface_arr(i,j,k);
                        // Neumann boundary condition dM/dn = -1/xi (z x n) x M
                        amrex::Real const xi_DMI = 2.0*exchange_yface_arr(i,j,k)/DMI_yface_arr(i,j,k);
                        
                        amrex::Real Ms_lo_x = Ms_yface_arr(i-1, j, k);
                        amrex::Real Ms_hi_x = Ms_yface_arr(i+1, j, k);
                        amrex::Real Ms_lo_y = Ms_yface_arr(i, j-1, k);
                        amrex::Real Ms_hi_y = Ms_yface_arr(i, j+1, k);

                        // Neumann boundary condition in scalar form, dMz/dx = +/- 1/xi*Mx 
                        // at y-faces, dM/dx do not overlap the BCs
                        amrex::Real dMzdx_BC_lo_x = -1.0/xi_DMI*M_yface(i,j,k,0); // lower x BC: dMz/dx = - 1/xi*Mx 
                        amrex::Real dMzdx_BC_hi_x =  1.0/xi_DMI*M_yface(i,j,k,0); // higher x BC: dMz/dx = 1/xi*Mx 
                        // H_DMI_yface(i,j,k,0) = H_DMI_coeff * dMzdx; // dMz/dx
                        H_DMI_yface(i,j,k,0) = H_DMI_coeff * DMDx_Mag(M_yface, Ms_lo_x, Ms_hi_x, dMzdx_BC_lo_x, dMzdx_BC_hi_x, i, j, k, dd, 2, 1); // z component at y nodality
                        
                        // at y-faces, dM/dy = dM/dn
                        amrex::Real dMzdy_BC_lo_y = -1.0/xi_DMI*M_yface(i,j,k,1); // lower y BC: dMz/dy = 1/xi*My
                        amrex::Real dMzdy_BC_hi_y =  1.0/xi_DMI*M_yface(i,j,k,1); // higher y BC: dMz/dy = 1/xi*My
                        // H_DMI_yface(i,j,k,1) = H_DMI_coeff * dMzdy; // dMz/dy
                        H_DMI_yface(i,j,k,1) = H_DMI_coeff * DMDy_Mag(M_yface, Ms_lo_y, Ms_hi_y, dMzdy_BC_lo_y, dMzdy_BC_hi_y, i, j, k, dd, 2, 1); // z component at y nodality
                        
                        // Neumann boundary condition in scalar form, dMx/dx = -/+ 1/xi*Mz 
                        // at y-face, dM/dx do not overlap with the BCs
                        amrex::Real dMxdx_BC_lo_x =  1.0/xi_DMI*M_yface(i,j,k,2); // lower x BC : dMx/dx = 1/xi*Mz 
                        amrex::Real dMxdx_BC_hi_x = -1.0/xi_DMI*M_yface(i,j,k,2); // higher x BC : dMx/dx = -1/xi*Mz 
                        amrex::Real dMydy_BC_lo_y =  1.0/xi_DMI*M_yface(i,j,k,2); // lower y BC: dMy/dy =  1/xi*Mz
                        amrex::Real dMydy_BC_hi_y = -1.0/xi_DMI*M_yface(i,j,k,2); // higher y BC: dMy/dy = - 1/xi*Mz
                        // H_DMI_yface(i,j,k,2) = H_DMI_coeff * ( - dMxdx - dMydy); // -dMx/dx-dMy/dy
                        H_DMI_yface(i,j,k,2) = H_DMI_coeff * (-DMDx_Mag(M_yface, Ms_lo_x, Ms_hi_x, dMxdx_BC_lo_x, dMxdx_BC_hi_x, i, j, k, dd, 0, 1) // x component at y nodality
                                                              -DMDy_Mag(M_yface, Ms_lo_y, Ms_hi_y, dMydy_BC_lo_y, dMydy_BC_hi_y, i, j, k, dd, 1, 1)); // y component at ynodality;
                        
                        if (i == 0 && j == 0 && k == 0){
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMzdx_BC_lo_x_yface = " << dMzdx_BC_lo_x << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMzdx_BC_hi_x_yface = " << dMzdx_BC_hi_x << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMzdy_BC_lo_y_yface = " << dMzdy_BC_lo_y << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMzdy_BC_hi_y_yface = " << dMzdy_BC_hi_y << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMxdx_BC_lo_x_yface = " << dMxdx_BC_lo_x << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMxdx_BC_hi_x_yface = " << dMxdx_BC_hi_x << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMydy_BC_lo_y_yface = " << dMydy_BC_lo_y << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMydy_BC_hi_y_yface = " << dMydy_BC_hi_y << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", xi_DMI_yface = " << xi_DMI << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", H_DMI_coeff_yface = " << H_DMI_coeff << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", Hx_DMI_yface = " << H_DMI_yface(i,j,k,0) << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", Hy_DMI_yface = " << H_DMI_yface(i,j,k,1) << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", Hz_DMI_yface = " << H_DMI_yface(i,j,k,2) << "\n";
                        }
                        
                    }
                }
            });

        amrex::ParallelFor(tbz,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // determine if the material is magnetic or not
                if (Ms_zface_arr(i,j,k) > 0.){
                    if (DMI_coupling == 1){

                        if (DMI_zface_arr(i,j,k) == 0.) amrex::Abort("The DMI_zface_arr(i,j,k) is 0.0 while including the DMI coupling term H_DMI for H_eff");
                        if (exchange_zface_arr(i,j,k) == 0.) amrex::Abort("The exchange_zface_arr(i,j,k) is 0.0 while DMI is turned on");

                        // H_DMI - use M^(old_time)
                        amrex::Real const H_DMI_coeff = 2.0 * DMI_zface_arr(i,j,k) / mu0 / Ms_zface_arr(i,j,k) / Ms_zface_arr(i,j,k);
                        // Neumann boundary condition dM/dn = -1/xi (z x n) x M
                        amrex::Real const xi_DMI = 2.0*exchange_zface_arr(i,j,k)/DMI_zface_arr(i,j,k);
                        
                        amrex::Real Ms_lo_x = Ms_zface_arr(i-1, j, k);
                        amrex::Real Ms_hi_x = Ms_zface_arr(i+1, j, k);
                        amrex::Real Ms_lo_y = Ms_zface_arr(i, j-1, k);
                        amrex::Real Ms_hi_y = Ms_zface_arr(i, j+1, k);

                        // // Neumann boundary condition dM/dn = -+ 1/xi*(z x n) x M
                        // at z-faces, dM/dx do not overlap with the BC
                        amrex::Real dMzdx_BC_lo_x = -1.0/xi_DMI*M_zface(i,j,k,0); // lower x BC: dMz/dx = -1/xi*Mx
                        amrex::Real dMzdx_BC_hi_x =  1.0/xi_DMI*M_zface(i,j,k,0); // higher x BC: dMz/dx = 1/xi*Mx
                        // H_DMI_zface(i,j,k,0) = H_DMI_coeff * dMzdx;
                        H_DMI_zface(i,j,k,0) = H_DMI_coeff * DMDx_Mag(M_zface, Ms_lo_x, Ms_hi_x, dMzdx_BC_lo_x, dMzdx_BC_hi_x, i, j, k, dd, 2, 2); // z component at z nodality
                        
                        // at z-faces, dM/dy do not overlap with the BC
                        amrex::Real dMzdy_BC_lo_y = -1.0/xi_DMI*M_zface(i,j,k,1); // lower y BC: dMz/dy = -1/xi*My
                        amrex::Real dMzdy_BC_hi_y =  1.0/xi_DMI*M_zface(i,j,k,1); // higher y BC: dMz/dy = 1/xi*My
                        // H_DMI_zface(i,j,k,1) = H_DMI_coeff * dMzdy;
                        H_DMI_zface(i,j,k,1) = H_DMI_coeff * DMDy_Mag(M_zface, Ms_lo_y, Ms_hi_y, dMzdy_BC_lo_y, dMzdy_BC_hi_y, i, j, k, dd, 2, 2); // z component at z nodality
                        
                        // at z-faces, both dM/dx and dM/dy do not overlap with the BC
                        amrex::Real dMxdx_BC_lo_x =  1.0/xi_DMI*M_zface(i,j,k,2); // lower x BC: dMx/dx = 1/xi*Mz
                        amrex::Real dMxdx_BC_hi_x = -1.0/xi_DMI*M_zface(i,j,k,2); // higher x BC: dMx/dx = - 1/xi*Mz
                        amrex::Real dMydy_BC_lo_y =  1.0/xi_DMI*M_zface(i,j,k,2); // lower y BC: dMy/dy = 1/xi*Mz
                        amrex::Real dMydy_BC_hi_y = -1.0/xi_DMI*M_zface(i,j,k,2); // higher y BC: dMy/dy = -1/xi*Mz
                        // H_DMI_zface(i,j,k,2) = H_DMI_coeff * ( - dMxdx - dMydy);
                        H_DMI_zface(i,j,k,2) = H_DMI_coeff * (-DMDx_Mag(M_zface, Ms_lo_x, Ms_hi_x, dMxdx_BC_lo_x, dMxdx_BC_hi_x, i, j, k, dd, 0, 2) // x component at z nodality
                                                              -DMDy_Mag(M_zface, Ms_lo_y, Ms_hi_y, dMydy_BC_lo_y, dMydy_BC_hi_y, i, j, k, dd, 1, 2)); // y component at z nodality;
                        
                        if (i == 0 && j == 0 && k == 0){
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMzdx_BC_lo_x_zface = " << dMzdx_BC_lo_x << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMzdx_BC_hi_x_zface = " << dMzdx_BC_hi_x << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMzdy_BC_lo_y_zface = " << dMzdy_BC_lo_y << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMzdy_BC_hi_y_zface = " << dMzdy_BC_hi_y << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMxdx_BC_lo_x_zface = " << dMxdx_BC_lo_x << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMxdx_BC_hi_x_zface = " << dMxdx_BC_hi_x << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMydy_BC_lo_y_zface = " << dMydy_BC_lo_y << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", dMydy_BC_hi_y_zface = " << dMydy_BC_hi_y << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", xi_DMI_zface = " << xi_DMI << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", H_DMI_coeff_zface = " << H_DMI_coeff << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", Hx_DMI_zface = " << H_DMI_zface(i,j,k,0) << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", Hy_DMI_zface = " << H_DMI_zface(i,j,k,1) << "\n";
                            amrex::Print() << "i=" << i << ", j=" << j << ", k=" << k << ", Hz_DMI_zface = " << H_DMI_zface(i,j,k,2) << "\n";
                        }
                    }
                }
            });
    }
}