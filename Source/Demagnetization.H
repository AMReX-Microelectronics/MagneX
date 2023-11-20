#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

#ifdef AMREX_USE_CUDA
#include <cufft.h>
#endif

using namespace amrex;

void ComputePoissonRHS(MultiFab&                        PoissonRHS,
                       Array<MultiFab, AMREX_SPACEDIM>& Mfield,
                       const Geometry&                  geom);

void ComputeHfromPhi(MultiFab&                         PoissonPhi,
                      Array<MultiFab, AMREX_SPACEDIM>& H_demagfield,
                      const Geometry&                  geom);

void ComputeForwardFFT(const MultiFab&    mf,
                       MultiFab&          mf_dft_real,
                       MultiFab&          mf_dft_imag,
                       const Geometry&    geom);

void ComputeInverseFFT(MultiFab&        mf_2,
                       const MultiFab&  mf_dft_real,
                       const MultiFab&  mf_dft_imag,
                       GpuArray<int, 3> n_cell,
                       const Geometry&  geom);

void ComputeDemagTensor(MultiFab&                        Kxx_fft_real,
                        MultiFab&                        Kxx_fft_imag,
                        MultiFab&                        Kxy_fft_real,
                        MultiFab&                        Kxy_fft_imag,
                        MultiFab&                        Kxz_fft_real,
                        MultiFab&                        Kxz_fft_imag,
                        MultiFab&                        Kyy_fft_real,
                        MultiFab&                        Kyy_fft_imag,
                        MultiFab&                        Kyz_fft_real,
                        MultiFab&                        Kyz_fft_imag,
                        MultiFab&                        Kzz_fft_real,
                        MultiFab&                        Kzz_fft_imag,
                        GpuArray<int, 3>                 n_cell_large,
			const Geometry&                  geom_large);

void ComputeHFieldFFT(const Array<MultiFab, AMREX_SPACEDIM>& M_field,
                      Array<MultiFab, AMREX_SPACEDIM>&       H_demagfield,
                      const MultiFab&                        Kxx_fft_real,
                      const MultiFab&                        Kxx_fft_imag,
                      const MultiFab&                        Kxy_fft_real,
                      const MultiFab&                        Kxy_fft_imag,
                      const MultiFab&                        Kxz_fft_real,
                      const MultiFab&                        Kxz_fft_imag,
                      const MultiFab&                        Kyy_fft_real,
                      const MultiFab&                        Kyy_fft_imag,
                      const MultiFab&                        Kyz_fft_real,
                      const MultiFab&                        Kyz_fft_imag,
                      const MultiFab&                        Kzz_fft_real,
                      const MultiFab&                        Kzz_fft_imag,
                      GpuArray<int, 3>                       n_cell_large,
                      const Geometry&                        geom_large);
 
#ifdef AMREX_USE_CUDA                                                                                      
std::string cufftErrorToString (const cufftResult& err);
#endif
