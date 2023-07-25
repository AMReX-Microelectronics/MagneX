#include "CartesianAlgorithm.H"
#include "EffectiveExchangeField.H"
#include <AMReX_MLMG.H> 
#include <AMReX_MultiFab.H> 
#include <AMReX_VisMF.H>
#include <AMReX_OpenBC.H>

using namespace amrex;

void CalculateH_anisotropy(
    amrex::Vector<MultiFab>& Mfield,
    std::array< MultiFab, AMREX_SPACEDIM> &   H_anisotropyfield,
    std::array< MultiFab, AMREX_SPACEDIM >&   Ms,
    std::array< MultiFab, AMREX_SPACEDIM >&   anisotropy,
    int anisotropy_coupling,
    amrex::GpuArray<amrex::Real, 3>& anisotropy_axis,
    Real mu0,
    const Geometry& geom
)
{
    for (MFIter mfi(Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        // extract dd from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dd = geom.CellSizeArray();

        const Array4<Real>& M_xface = Mfield[0].array(mfi); // note M_xface include x,y,z components at |_x faces
        const Array4<Real>& M_yface = Mfield[1].array(mfi); // note M_yface include x,y,z components at |_y faces
        const Array4<Real>& M_zface = Mfield[2].array(mfi); // note M_zface include x,y,z components at |_z faces
        const Array4<Real>& Ms_xface_arr = Ms[0].array(mfi);
        const Array4<Real>& Ms_yface_arr = Ms[1].array(mfi);
        const Array4<Real>& Ms_zface_arr = Ms[2].array(mfi);
        const Array4<Real>& anisotropy_xface_arr = anisotropy[0].array(mfi);
        const Array4<Real>& anisotropy_yface_arr = anisotropy[1].array(mfi);
        const Array4<Real>& anisotropy_zface_arr = anisotropy[2].array(mfi);
        const Array4<Real>& H_anisotropy_xface = H_anisotropyfield[0].array(mfi);   // x,y,z component at |_x faces
        const Array4<Real>& H_anisotropy_yface = H_anisotropyfield[1].array(mfi);   // x,y,z component at |_y faces
        const Array4<Real>& H_anisotropy_zface = H_anisotropyfield[2].array(mfi);   // x,y,z component at |_z faces

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
                    if (anisotropy_coupling == 1){

                        if (anisotropy_xface_arr(i,j,k) == 0.) amrex::Abort("The anisotropy_xface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                        // H_anisotropy 
                        amrex::Real M_dot_anisotropy_axis = 0.0;
                        for (int comp=0; comp<3; ++comp) {
                            M_dot_anisotropy_axis += M_xface(i, j, k, comp) * anisotropy_axis[comp];
                        }
                        amrex::Real const H_anisotropy_coeff = - 2.0 * anisotropy_xface_arr(i,j,k) / mu0 / Ms_xface_arr(i,j,k) / Ms_xface_arr(i,j,k);
                        for (int comp=0; comp<3; ++comp) {
                            H_anisotropy_xface(i,j,k,comp) = H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[comp];
                        }
                    }
                }
            });

        amrex::ParallelFor(tby,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // determine if the material is magnetic or not
                if (Ms_yface_arr(i,j,k) > 0.){
                    if (anisotropy_coupling == 1){

                        if (anisotropy_yface_arr(i,j,k) == 0.) amrex::Abort("The anisotropy_yface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                        // H_anisotropy 
                        amrex::Real M_dot_anisotropy_axis = 0.0;
                        for (int comp=0; comp<3; ++comp) {
                            M_dot_anisotropy_axis += M_yface(i, j, k, comp) * anisotropy_axis[comp];
                        }
                        amrex::Real const H_anisotropy_coeff = - 2.0 * anisotropy_yface_arr(i,j,k) / mu0 / Ms_yface_arr(i,j,k) / Ms_yface_arr(i,j,k);
                        for (int comp=0; comp<3; ++comp) {
                            H_anisotropy_yface(i,j,k,comp) = H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[comp];
                        }
                    }
                }
            });

        amrex::ParallelFor(tbz,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // determine if the material is magnetic or not
                if (Ms_zface_arr(i,j,k) > 0.){
                    if (anisotropy_coupling == 1){

                        if (anisotropy_zface_arr(i,j,k) == 0.) amrex::Abort("The anisotropy_zface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                        // H_anisotropy 
                        amrex::Real M_dot_anisotropy_axis = 0.0;
                        for (int comp=0; comp<3; ++comp) {
                            M_dot_anisotropy_axis += M_zface(i, j, k, comp) * anisotropy_axis[comp];
                        }
                        amrex::Real const H_anisotropy_coeff = - 2.0 * anisotropy_zface_arr(i,j,k) / mu0 / Ms_zface_arr(i,j,k) / Ms_zface_arr(i,j,k);
                        for (int comp=0; comp<3; ++comp) {
                            H_anisotropy_zface(i,j,k,comp) = H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[comp];
                        }
                    }
                }
            });
    }
}