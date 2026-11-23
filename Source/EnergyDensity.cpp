#include "EnergyDensity.H"
#include <AMReX_MultiFab.H> 
#include <AMReX_VisMF.H>

void CalculateEnergyDensity(
    MultiFab                              &deltaE,
    std::array< MultiFab, AMREX_SPACEDIM> &Mfield,
    std::array< MultiFab, AMREX_SPACEDIM> &H_eff,
    Real                                  mu0
){
    // calculate the b_temp_static, a_temp_static
    for (MFIter mfi(Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        // extract field data   
        const Array4<Real>& Mx = Mfield[0].array(mfi); // Mx is the x component at cell centers
        const Array4<Real>& My = Mfield[1].array(mfi); // My is the y component at cell centers
        const Array4<Real>& Mz = Mfield[2].array(mfi); // Mz is the z component at cell centers
        const Array4<Real>& Hx_eff = H_eff[0].array(mfi); // Hx_eff is the x component at cell centers
        const Array4<Real>& Hy_eff = H_eff[1].array(mfi); // Hy_eff is the y component at cell centers
        const Array4<Real>& Hz_eff = H_eff[2].array(mfi); // Hz_eff is the z component at cell centers
        const Array4<Real>& dE = deltaE.array(mfi);

        // extract tileboxes for which to loop
        Box const &tbx = mfi.tilebox(Mfield[0].ixType().toIntVect());

        amrex::ParallelFor(tbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                dE(i,j,k) = -1.0/2.0*mu0*(Mx(i,j,k) * Hx_eff(i,j,k) + My(i,j,k) * Hy_eff(i,j,k) + Mz(i,j,k) * Hz_eff(i,j,k)) ;
            });
    }
}