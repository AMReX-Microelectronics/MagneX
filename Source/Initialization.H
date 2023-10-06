#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void InitializeMagneticProperties(
                   MultiFab&  alpha,
                   MultiFab&   Ms, 
                   MultiFab&   gamma,
                   MultiFab&   exchange,
                   MultiFab&   DMI,
                   MultiFab&   anisotropy,
                   Real        alpha_val,
                   Real        Ms_val,
                   Real        gamma_val,
                   Real        exchange_val,
                   Real        DMI_val,
                   Real        anisotropy_val,
                   amrex::GpuArray<amrex::Real, 3> prob_lo,
                   amrex::GpuArray<amrex::Real, 3> prob_hi,
                   const       Geometry& geom);


void InitializeFields( 
                   Array< MultiFab, AMREX_SPACEDIM >&  Mfield,
                   amrex::GpuArray<amrex::Real, 3> prob_lo,
                   amrex::GpuArray<amrex::Real, 3> prob_hi,
                   const       Geometry& geom);

void ComputeHbias(
                Array< MultiFab, AMREX_SPACEDIM >&   H_biasfield,
                amrex::GpuArray<amrex::Real, 3> prob_lo,
                amrex::GpuArray<amrex::Real, 3> prob_hi,
                const           Real& time,
                const       Geometry& geom);
