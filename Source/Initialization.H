#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void InitializeMagneticProperties(MultiFab&  alpha,
                                  MultiFab&   Ms, 
                                  MultiFab&   gamma,
                                  MultiFab&   exchange,
                                  MultiFab&   DMI,
                                  MultiFab&   anisotropy,
                                  amrex::GpuArray<amrex::Real, 3> prob_lo,
                                  amrex::GpuArray<amrex::Real, 3> prob_hi,
                                  const Geometry& geom,
                                  const Real& time);


void InitializeFields(Array< MultiFab, AMREX_SPACEDIM >&  Mfield,
                      amrex::GpuArray<amrex::Real, 3> prob_lo,
                      amrex::GpuArray<amrex::Real, 3> prob_hi,
                      const       Geometry& geom);

void ComputeHbias(Array< MultiFab, AMREX_SPACEDIM >&   H_biasfield,
                  amrex::GpuArray<amrex::Real, 3> prob_lo,
                  amrex::GpuArray<amrex::Real, 3> prob_hi,
                  const           Real& time,
                  const       Geometry& geom);

void ComputeAlpha(MultiFab&  alpha,
                  amrex::GpuArray<amrex::Real, 3> prob_lo,
                  amrex::GpuArray<amrex::Real, 3> prob_hi,
                  const Geometry& geom,
                  const Real& time);
