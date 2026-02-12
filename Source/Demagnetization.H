#ifndef _DEMAG_H_
#define _DEMAG_H_

#include <AMReX_MultiFab.H>
#include <AMReX_FFT.H>

using namespace amrex;

class Demagnetization {

    Box domain_large;
    BoxArray ba_large;
    DistributionMapping dm_large;
    Geometry geom_large;

    FabArray<BaseFab<GpuComplex<amrex::Real> > > Kxx_fft;
    FabArray<BaseFab<GpuComplex<amrex::Real> > > Kxy_fft;
    FabArray<BaseFab<GpuComplex<amrex::Real> > > Kxz_fft;
    FabArray<BaseFab<GpuComplex<amrex::Real> > > Kyy_fft;
    FabArray<BaseFab<GpuComplex<amrex::Real> > > Kyz_fft;
    FabArray<BaseFab<GpuComplex<amrex::Real> > > Kzz_fft;

    FabArray<BaseFab<GpuComplex<amrex::Real> > > Mx_fft;
    FabArray<BaseFab<GpuComplex<amrex::Real> > > My_fft;
    FabArray<BaseFab<GpuComplex<amrex::Real> > > Mz_fft;
    FabArray<BaseFab<GpuComplex<amrex::Real> > > Hx_fft;
    FabArray<BaseFab<GpuComplex<amrex::Real> > > Hy_fft;
    FabArray<BaseFab<GpuComplex<amrex::Real> > > Hz_fft;

    std::unique_ptr<amrex::FFT::R2C<amrex::Real>> amrex_fft;

public:

    Demagnetization();

    void define();

    void CalculateH_demag(Array<MultiFab, AMREX_SPACEDIM>& Mfield,
                          Array<MultiFab, AMREX_SPACEDIM>& H_demagfield);

};

#endif
