#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void CalculateEnergyDensity(
    MultiFab                              &deltaE,
    std::array< MultiFab, AMREX_SPACEDIM> &Mfield,
    std::array< MultiFab, AMREX_SPACEDIM> &H_eff,
    Real                                  mu0
);