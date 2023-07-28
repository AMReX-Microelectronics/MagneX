#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void ComputePoissonRHS(MultiFab&                        PoissonRHS,
                       Array<MultiFab, AMREX_SPACEDIM>& Mfield,
                       MultiFab& Ms,
                       const Geometry&                 geom);

void ComputeHfromPhi(MultiFab&                         PoissonPhi,
                      Array<MultiFab, AMREX_SPACEDIM>& H_demagfield,
                      amrex::GpuArray<amrex::Real, 3> prob_lo,
                      amrex::GpuArray<amrex::Real, 3> prob_hi,
                      const Geometry&                 geom);


// Compute the inverse FFT using the FFTW libraries 
void ComputeInverseFFT(MultiFab&                        mf,
		       const MultiFab&                  mf_dft_real,
		       const MultiFab&                  mf_dft_imag,
                       amrex::GpuArray<amrex::Real, 3>  prob_lo,
                       amrex::GpuArray<amrex::Real, 3>  prob_hi,
		       GpuArray<int, 3>                 n_cell,
                       int                              max_grid_size,
                       const Geometry&                  geom);

// Compute the forward FFT using the FFTW libraries 
void ComputeForwardFFT(const MultiFab&                  mf,
		       MultiFab&                        mf_dft_real,
		       MultiFab&                        mf_dft_imag,
                       amrex::GpuArray<amrex::Real, 3>  prob_lo,
                       amrex::GpuArray<amrex::Real, 3>  prob_hi,
                       amrex::GpuArray<int, 3>          n_cell,
		       int                              max_grid_size,
                       const Geometry&                  geom,
		       long                             npts);

/*
// Function accepts the geometry of the problem and then defines the demagnetization tensor in space.  
void ComputeDemagTensor(MultiFab&                        Kxx,
		                    MultiFab&                        Kxy,
		                    MultiFab&                        Kxz,
		                    MultiFab&                        Kyy,
		                    MultiFab&                        Kyz,
		                    MultiFab&                        Kzz,
                        GpuArray<int, 3>                 n_cell,
                        amrex::GpuArray<amrex::Real, 3>  prob_lo,
                        amrex::GpuArray<amrex::Real, 3>  prob_hi,
                        int                              max_grid_size,
                        const Geometry&                  geom)
*/

/*
// Function takes the magnetization and demagnetization tensor as inputs and then computes inverse FFT of the convolution of their forward FFTs.
// The result is the electric potential and is stored in H_demagfield.
void ComputeHFieldFFT(const Array<MultiFab, AMREX_SPACEDIM>& M_field,
	                    Array<MultiFab, AMREX_SPACEDIM>&       H_demagfield,
                      const MultiFab&                        Kxx,
		                  const MultiFab&                        Kxy,
		                  const MultiFab&                        Kxz,
		                  const MultiFab&                        Kyy,
		                  const MultiFab&                        Kyz,
		                  const MultiFab&                        Kzz,
                      GpuArray<Real, 3>                      prob_lo,
                      GpuArray<Real, 3>                      prob_hi,
                      GpuArray<int, 3>                       n_cell,
		                  int                                    max_grid_size,
                      const Geometry&                        geom)
*/
 
