#ifndef _DEMAG_H_
#define _DEMAG_H_

#include <heffte.h>

#include <AMReX_MultiFab.H> 

using namespace amrex;

class Demagnetization {

    BoxArray ba_large;
    DistributionMapping dm_large;
    Geometry geom_large;

    BoxArray ba_fft; 

    MultiFab Kxx_fft_real;
    MultiFab Kxx_fft_imag;
    MultiFab Kxy_fft_real;
    MultiFab Kxy_fft_imag;
    MultiFab Kxz_fft_real;
    MultiFab Kxz_fft_imag;
    MultiFab Kyy_fft_real;
    MultiFab Kyy_fft_imag;
    MultiFab Kyz_fft_real;
    MultiFab Kyz_fft_imag;
    MultiFab Kzz_fft_real;
    MultiFab Kzz_fft_imag;

public:

    Demagnetization();

    void define();

    void CalculateH_demag(Array<MultiFab, AMREX_SPACEDIM>& Mfield,
                          Array<MultiFab, AMREX_SPACEDIM>& H_demagfield);
    
    void ComputeForwardFFT_heffte(const MultiFab&    mf_in,
                                  MultiFab&          mf_dft_real,
                                  MultiFab&          mf_dft_imag);

    void ComputeForwardFFT(const MultiFab&    mf_in,
                           MultiFab&          mf_dft_real,
                           MultiFab&          mf_dft_imag);

    void ComputeInverseFFT_heffte(MultiFab&                        mf_out,
                                  const MultiFab&                  mf_dft_real,
                                  const MultiFab&                  mf_dft_imag);

    void ComputeInverseFFT(MultiFab&                        mf_out,
                           const MultiFab&                  mf_dft_real,
                           const MultiFab&                  mf_dft_imag);

#ifdef AMREX_USE_CUDA
    std::string cufftErrorToString (const cufftResult& err);
#endif

};

#endif
