#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void CalculateH_anisotropy(
    Array< MultiFab, AMREX_SPACEDIM> &   Mfield,
    Array< MultiFab, AMREX_SPACEDIM> &   H_anisotropyfield,
    MultiFab&   Ms,
    MultiFab&   anisotropy,
    int anisotropy_coupling,
    amrex::GpuArray<amrex::Real, 3>& anisotropy_axis,
    Real mu0,
    const Geometry& geom);
