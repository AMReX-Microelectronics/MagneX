#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void ComputePoissonRHS(MultiFab&                        PoissonRHS,
                       Array<MultiFab, AMREX_SPACEDIM>& Mfield,
                       MultiFab&                        Ms,
                       const Geometry&                  geom);

void ComputeHfromPhi(MultiFab&                         PoissonPhi,
                      Array<MultiFab, AMREX_SPACEDIM>& H_demagfield,
                      amrex::GpuArray<amrex::Real, 3>  prob_lo,
                      amrex::GpuArray<amrex::Real, 3>  prob_hi,
                      const Geometry&                  geom);

void ComputeForwardFFT(const MultiFab&    mf,
                       MultiFab&          mf_dft_real,
                       MultiFab&          mf_dft_imag,
                       const Geometry&    geom,
                       long               npts);

void ComputeInverseFFT(MultiFab&        mf_2,
                       const MultiFab&  mf_dft_real,
                       const MultiFab&  mf_dft_imag,
                       GpuArray<int, 3> n_cell,
                       const Geometry&  geom);

void ComputeDemagTensor(MultiFab&        Kxx,
                        MultiFab&        Kxy,
                        MultiFab&        Kxz,
                        MultiFab&        Kyy,
                        MultiFab&        Kyz,
                        MultiFab&        Kzz,
			GpuArray<int, 3> n_cell,
                        const Geometry&  geom);

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
                      GpuArray<Real, 3>                      prob_lo,
                      GpuArray<Real, 3>                      prob_hi,
                      GpuArray<int, 3>                       n_cell,
                      int                                    max_grid_size,
                      const Geometry&                        geom);
 
