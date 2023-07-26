#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;
/*
void EvolveM_2nd(
    //std::array< MultiFab, AMREX_SPACEDIM> &Mfield,
    amrex::Vector<MultiFab>& Mfield,
    std::array< MultiFab, AMREX_SPACEDIM> &H_demagfield,
    std::array< MultiFab, AMREX_SPACEDIM> &H_biasfield, // H bias
    std::array< MultiFab, AMREX_SPACEDIM> &H_exchangefield, // effective exchange field
    std::array< MultiFab, AMREX_SPACEDIM> &H_DMIfield,
    std::array< MultiFab, AMREX_SPACEDIM> &H_anisotropyfield,
    MultiFab                              &PoissonRHS, 
    MultiFab                              &PoissonPhi, 
    std::array< MultiFab, AMREX_SPACEDIM >&   alpha,
    std::array< MultiFab, AMREX_SPACEDIM >&   Ms,
    std::array< MultiFab, AMREX_SPACEDIM >&   gamma,
    std::array< MultiFab, AMREX_SPACEDIM >&   exchange,
    std::array< MultiFab, AMREX_SPACEDIM >&   DMI,
    std::array< MultiFab, AMREX_SPACEDIM >&   anisotropy,
    int demag_coupling,
    int exchange_coupling,
    int DMI_coupling,
    int anisotropy_coupling,
    amrex::GpuArray<amrex::Real, 3>& anisotropy_axis,
    int M_normalization, 
    Real mu0,
    const Geometry& geom,
    amrex::GpuArray<amrex::Real, 3>  prob_lo,
    amrex::GpuArray<amrex::Real, 3>  prob_hi,
    amrex::Real const dt,
    const Real time
);
*/
