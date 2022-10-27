#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void InitializeMagneticProperties(std::array< MultiFab, AMREX_SPACEDIM >&  alpha,
                   std::array< MultiFab, AMREX_SPACEDIM >&   Ms, 
                   std::array< MultiFab, AMREX_SPACEDIM >&   gamma,
                   std::array< MultiFab, AMREX_SPACEDIM >&   exchange,
                   std::array< MultiFab, AMREX_SPACEDIM >&   anisotropy,
                   Real        alpha_val,
                   Real        Ms_val,
                   Real        gamma_val,
                   Real        exchange_val,
                   Real        anisotropy_val,
                   amrex::GpuArray<amrex::Real, 3> prob_lo,
                   amrex::GpuArray<amrex::Real, 3> prob_hi,
                   amrex::GpuArray<amrex::Real, 3> mag_lo,
                   amrex::GpuArray<amrex::Real, 3> mag_hi,
                   const       Geometry& geom);


void InitializeFields(std::array< MultiFab, AMREX_SPACEDIM >&  Mfield,
                   std::array< MultiFab, AMREX_SPACEDIM >&   H_biasfield,
                   std::array< MultiFab, AMREX_SPACEDIM >&   Ms,
                   amrex::GpuArray<amrex::Real, 3> prob_lo,
                   amrex::GpuArray<amrex::Real, 3> prob_hi,
                   const       Geometry& geom);
