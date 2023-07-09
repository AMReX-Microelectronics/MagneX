#include "CartesianAlgorithm.H"
#include "EffectiveDMIField.H"

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
                        // at x-faces, dM/dn = dM/dx
                        amrex::Real dMzdx = (Ms_hi_x == 0. || Ms_lo_x == 0.) ? 
                                                        ( (Ms_hi_x == 0.) ? 1.0/xi_DMI*M_xface(i,j,k,0) : -1.0/xi_DMI*M_xface(i,j,k,0)) // BC: dMz/dx = +/- 1/xi*Mx 
                                                        : 0.5 *(UpwardDx(M_xface, i, j, k, dd, 2) + DownwardDx(M_xface, i, j, k, dd, 2)); // dMz/dx in bulk

                        // at x-faces, dM/dy do not overlap the BCs
                        amrex::Real dMzdy = (Ms_hi_y == 0. || Ms_lo_y == 0.) ? 
                                                        ( (Ms_hi_y == 0.) ? 0.5 *(1.0/xi_DMI*M_xface(i,j,k,1) + DownwardDy(M_xface, i, j, k, dd, 2)) // BC: dMz/dy = 1/xi*My 
                                                        : 0.5 *(UpwardDy(M_xface, i, j, k, dd, 2) - 1.0/xi_DMI*M_xface(i,j,k,1)))  // BC: dMz/dy = -1/xi*My
                                                        : 0.5 *(UpwardDy(M_xface, i, j, k, dd, 2) + DownwardDy(M_xface, i, j, k, dd, 2)); // dMz/dy in bulk

                        // Neumann boundary condition in scalar form, dMx/dx = -/+ 1/xi*Mz 
                        // at x-faces, dM/dn = dM/dx
                        amrex::Real dMxdx = (Ms_hi_x == 0. || Ms_lo_x == 0.) ? 
                                                        ( (Ms_hi_x == 0.) ? -1.0/xi_DMI*M_xface(i,j,k,2) : 1.0/xi_DMI*M_xface(i,j,k,2)) // BC: dMx/dx = -/+ 1/xi*Mz
                                                        : 0.5 *(UpwardDx(M_xface, i, j, k, dd, 0) + DownwardDx(M_xface, i, j, k, dd, 0)); // dMx/dx in bulk

                        // at x-faces, dM/dy do not overlap the BCs
                        amrex::Real dMydy = (Ms_hi_y == 0. || Ms_lo_y == 0.) ?
                                                        ( (Ms_hi_y == 0.)? 0.5 * (-1.0/xi_DMI*M_xface(i,j,k,2) + DownwardDy(M_xface, i, j, k, dd, 1)) // BC: dMy/dy = -1/xi*Mz
                                                        : 0.5 * (UpwardDy(M_xface, i, j, k, dd, 1) + 1.0/xi_DMI*M_xface(i,j,k,2))) // BC: dMy/dy = 1/xi*Mz
                                                        : 0.5 *(UpwardDy(M_xface, i, j, k, dd, 1) + DownwardDy(M_xface, i, j, k, dd, 1)); // dMy/dy in bulk

                        H_DMI_xface(i,j,k,0) = H_DMI_coeff * dMzdx;
                        H_DMI_xface(i,j,k,1) = H_DMI_coeff * dMzdy;
                        H_DMI_xface(i,j,k,2) = H_DMI_coeff * (-dMxdx - dMydy);

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
                        amrex::Real dMzdx = (Ms_hi_x == 0. || Ms_lo_x == 0.) ? 
                                                        ( (Ms_hi_x == 0.) ? 0.5 *(1.0/xi_DMI*M_yface(i,j,k,0) + DownwardDx(M_yface, i, j, k, dd, 2)) // BC: dMz/dx = 1/xi*Mx
                                                        : 0.5 *(UpwardDx(M_yface, i, j, k, dd, 2) - 1.0/xi_DMI*M_yface(i,j,k,0))) // BC: dMz/dx = -1/xi*Mx
                                                        : 0.5 *(UpwardDx(M_yface, i, j, k, dd, 2) + DownwardDx(M_yface, i, j, k, dd, 2)); // dMz/dx in bulk
                                                            
                        // at y-faces, dM/dy = dM/dn
                        amrex::Real dMzdy = (Ms_hi_y == 0. || Ms_lo_y == 0.) ?
                                                        ((Ms_hi_y == 0.) ? 1.0/xi_DMI*M_yface(i,j,k,1) : -1.0/xi_DMI*M_yface(i,j,k,1)) // BC: dMz/dy = +/- 1/xi*My
                                                        : 0.5 *(UpwardDy(M_yface, i, j, k, dd, 2) + DownwardDy(M_yface, i, j, k, dd, 2)); // dMz/dy in bulk

                        // Neumann boundary condition in scalar form, dMx/dx = -/+ 1/xi*Mz 
                        // at y-face, dM/dx do not overlap with the BCs
                        amrex::Real dMxdx = (Ms_hi_x == 0. || Ms_lo_x == 0.) ? 
                                                        ((Ms_hi_x == 0.) ? 0.5 *(-1.0/xi_DMI*M_yface(i,j,k,2) + DownwardDx(M_yface, i, j, k, dd, 0)) // BC: dMx/dx = -1/xi*Mz
                                                        : 0.5 *(UpwardDx(M_yface, i, j, k, dd, 0) + 1.0/xi_DMI*M_yface(i,j,k,2)) ) // BC: dMx/x = 1/xi*Mz
                                                        : 0.5 *(UpwardDx(M_yface, i, j, k, dd, 0) + DownwardDx(M_yface, i, j, k, dd, 0)); // dMx/dx in bulk;

                        // at y-faces, dM/dy = dM/dn
                        amrex::Real dMydy = (Ms_hi_y == 0. || Ms_lo_y == 0.) ?
                                                        ((Ms_hi_y == 0.) ? -1.0/xi_DMI*M_yface(i,j,k,2) : 1.0/xi_DMI*M_yface(i,j,k,2)) // BC: dMy/dy = -/+ 1/xi*Mz
                                                        : 0.5 *(UpwardDy(M_yface, i, j, k, dd, 1) + DownwardDy(M_yface, i, j, k, dd, 1)); // dMy/dy in bulk

                        H_DMI_yface(i,j,k,0) = H_DMI_coeff * dMzdx; // dMz/dx
                        H_DMI_yface(i,j,k,1) = H_DMI_coeff * dMzdy; // dMz/dy
                        H_DMI_yface(i,j,k,2) = H_DMI_coeff * ( - dMxdx - dMydy); // -dMx/dx-dMy/dy
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

                        // Neumann boundary condition dM/dn = -+ 1/xi*(z x n) x M
                        // at z-faces, dM/dx do not overlap with the BC
                        amrex::Real dMzdx = (Ms_hi_x == 0. || Ms_lo_x == 0.) ? 
                                                        ( (Ms_hi_x == 0.) ? 0.5 *(1.0/xi_DMI*M_zface(i,j,k,0) + DownwardDx(M_zface, i, j, k, dd, 2)) // BC: dMz/dx = 1/xi*Mx
                                                        : 0.5 *(UpwardDx(M_zface, i, j, k, dd, 2) - 1.0/xi_DMI*M_zface(i,j,k,0))) // BC: dMz/dx = -1/xi*Mx
                                                        : 0.5 *(UpwardDx(M_zface, i, j, k, dd, 2) + DownwardDx(M_zface, i, j, k, dd, 2)); // dMz/dx in bulk
                        
                        // at z-faces, dM/dy do not overlap with the BC
                        amrex::Real dMzdy = (Ms_hi_y == 0. || Ms_lo_y == 0.) ?
                                                        (Ms_hi_y == 0. ? 0.5 *(1.0/xi_DMI*M_zface(i,j,k,1) + DownwardDy(M_zface, i, j, k, dd, 2)) // BC: dMz/dy = 1/xi*My
                                                        : 0.5 *(UpwardDy(M_zface, i, j, k, dd, 2) - 1.0/xi_DMI*M_zface(i,j,k,1))) // BC: dMz/dy = -1/xi*My
                                                        : 0.5 *(UpwardDy(M_zface, i, j, k, dd, 2) + DownwardDy(M_zface, i, j, k, dd, 2)); // dMz/dy in bulk

                        // at z-faces, dM/dx do not overlap with the BC
                        amrex::Real dMxdx = (Ms_hi_x == 0. || Ms_lo_x == 0.) ? 
                                                        ( (Ms_hi_x == 0.) ? 0.5 *( -1.0/xi_DMI*M_zface(i,j,k,2) + DownwardDx(M_zface, i, j, k, dd, 0)) // BC: dMx/dx = -1/xi*Mz
                                                        : 0.5 *(UpwardDx(M_zface, i, j, k, dd, 0) + 1.0/xi_DMI*M_zface(i,j,k,2))) // BC: dMx/dx = 1/xi*Mz
                                                        : 0.5 *(UpwardDx(M_zface, i, j, k, dd, 0) + DownwardDx(M_zface, i, j, k, dd, 0)); // dMz/dx in bulk 

                        // at z-faces, dM/dy do not overlap with the BC
                        amrex::Real dMydy = (Ms_hi_y == 0. || Ms_lo_y == 0.) ?
                                                        ( (Ms_hi_y == 0.) ? 0.5 *(-1.0/xi_DMI*M_zface(i,j,k,2) + DownwardDy(M_zface, i, j, k, dd, 1)) // BC: dMy/dy = -1/xi*Mz
                                                        : 0.5 *(UpwardDy(M_zface, i, j, k, dd, 1) + 1.0/xi_DMI*M_zface(i,j,k,2))) // BC: dMy/dy = 1/xi*Mz
                                                        : 0.5 *(UpwardDy(M_zface, i, j, k, dd, 1) + DownwardDy(M_zface, i, j, k, dd, 1)); // dMy/dy in bulk

                        H_DMI_zface(i,j,k,0) = H_DMI_coeff * dMzdx;
                        H_DMI_zface(i,j,k,1) = H_DMI_coeff * dMzdy;
                        H_DMI_zface(i,j,k,2) = H_DMI_coeff * ( - dMxdx - dMydy);
                    }
                }
            });
    }
}