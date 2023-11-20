#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void CalculateH_DMI(Array< MultiFab, AMREX_SPACEDIM> &   Mfield,
                    Array< MultiFab, AMREX_SPACEDIM> &   H_DMIfield,
                    MultiFab&   Ms,
                    MultiFab&   exchange,
                    MultiFab&   DMI,
                    int DMI_coupling,
                    Real mu0,
                    const Geometry& geom);
