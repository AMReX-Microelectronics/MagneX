#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

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
    const Geometry& geom);
