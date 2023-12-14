#include "MagneX.H"

using namespace amrex;

void CalculateH_anisotropy(Array< MultiFab, AMREX_SPACEDIM> &   Mfield,
                           Array< MultiFab, AMREX_SPACEDIM> &   H_anisotropyfield,
                           MultiFab&   Ms,
                           MultiFab&   anisotropy,
                           const Geometry& geom)
{
    // timer for profiling
    BL_PROFILE_VAR("CalculateH_anisotropy()",CalculateH_anisotropy);

    for (MFIter mfi(Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        // extract dd from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dd = geom.CellSizeArray();

        const Box& bx = mfi.validbox();

        const Array4<Real>& Mx = Mfield[0].array(mfi); 
        const Array4<Real>& My = Mfield[1].array(mfi); 
        const Array4<Real>& Mz = Mfield[2].array(mfi); 
        const Array4<Real>& Ms_arr = Ms.array(mfi);
        const Array4<Real>& anisotropy_arr = anisotropy.array(mfi);
        const Array4<Real>& Hx_anisotropy = H_anisotropyfield[0].array(mfi);   
        const Array4<Real>& Hy_anisotropy = H_anisotropyfield[1].array(mfi);   
        const Array4<Real>& Hz_anisotropy = H_anisotropyfield[2].array(mfi);   


        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // determine if the material is magnetic or not
                if (Ms_arr(i,j,k) > 0.){
                    if (anisotropy_coupling == 1){

                        if (anisotropy_arr(i,j,k) == 0.) amrex::Abort("The anisotropy_xface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                        // H_anisotropy 
                        amrex::Real M_dot_anisotropy_axis = 0.0;
                        M_dot_anisotropy_axis = Mx(i, j, k) * anisotropy_axis[0] + My(i, j, k) * anisotropy_axis[1] + Mz(i, j, k) * anisotropy_axis[2];
                        amrex::Real const H_anisotropy_coeff = - 2.0 * anisotropy_arr(i,j,k) / mu0 / Ms_arr(i,j,k) / Ms_arr(i,j,k);
                        Hx_anisotropy(i,j,k) = H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                        Hy_anisotropy(i,j,k) = H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                        Hz_anisotropy(i,j,k) = H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                    }
                }
            });
    }
}
