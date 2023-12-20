#include <AMReX_MultiFab.H>

#ifdef AMREX_USE_CUDA
#include <cufft.h>
#else
#include <fftw3.h>
#include <fftw3-mpi.h>
#endif

#include "MagneX_namespace.H"

using namespace amrex;
using namespace MagneX;

/*
  MagneX.cpp
*/
void InitializeMagneXNamespace();

/*
  Checkpoint.cpp
*/

void WriteCheckPoint(int step,
                     const amrex::Real time,
                     Array< MultiFab, AMREX_SPACEDIM>& Mfield,
                     Array< MultiFab, AMREX_SPACEDIM>& H_biasfield,
		     Array< MultiFab, AMREX_SPACEDIM>& H_demagfield);

void ReadCheckPoint(int& restart,
		    amrex::Real& time,
		    Array< MultiFab, AMREX_SPACEDIM>& Mfield,
		    Array< MultiFab, AMREX_SPACEDIM>& H_biasfield,
		    Array< MultiFab, AMREX_SPACEDIM>& H_demagfield,
		    BoxArray& ba,
		    DistributionMapping& dm);

/*
  ComputeLLGRHS.cpp
*/

void Compute_LLG_RHS(Array< MultiFab, AMREX_SPACEDIM >& LLG_RHS,
                     Array< MultiFab, AMREX_SPACEDIM >& Mfield_old,
                     Array< MultiFab, AMREX_SPACEDIM >& H_demagfield,
                     Array< MultiFab, AMREX_SPACEDIM >& H_biasfield,
                     Array< MultiFab, AMREX_SPACEDIM >& H_exchangefield,
                     Array< MultiFab, AMREX_SPACEDIM >& H_DMIfield,
                     Array< MultiFab, AMREX_SPACEDIM >& H_anisotropyfield,
                     MultiFab&   alpha,
                     MultiFab&   Ms,
                     MultiFab&   gamma);

/*
  Demagnetization.cpp
*/

void ComputeForwardFFT(const MultiFab&    mf,
                       MultiFab&          mf_dft_real,
                       MultiFab&          mf_dft_imag,
                       const Geometry&    geom);

void ComputeInverseFFT(MultiFab&        mf_2,
                       const MultiFab&  mf_dft_real,
                       const MultiFab&  mf_dft_imag,
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

void CalculateH_demag(const Array<MultiFab, AMREX_SPACEDIM>& Mfield,
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

/*
  Diagnostics.cpp
*/

long CountMagneticCells(MultiFab& Ms);

Real SumNormalizedM(MultiFab& Ms,
                    MultiFab& Mfield);

/*
  EffectiveAnisotropyField.cpp
*/

void CalculateH_anisotropy(Array< MultiFab, AMREX_SPACEDIM> &   Mfield,
                           Array< MultiFab, AMREX_SPACEDIM> &   H_anisotropyfield,
                           MultiFab&   Ms,
                           MultiFab&   anisotropy);

/*
  EffectiveDMIField.cpp
*/

void CalculateH_DMI(Array< MultiFab, AMREX_SPACEDIM> &   Mfield,
                    Array< MultiFab, AMREX_SPACEDIM> &   H_DMIfield,
                    MultiFab&   Ms,
                    MultiFab&   exchange,
                    MultiFab&   DMI,
                    const Geometry& geom);

/*
  EffectiveExchangeField.cpp
*/

void CalculateH_exchange(Array< MultiFab, AMREX_SPACEDIM> &   Mfield,
                         Array< MultiFab, AMREX_SPACEDIM> &   H_exchangefield,
                         MultiFab&   Ms,
                         MultiFab&   exchange,
                         MultiFab&   DMI,
                         const Geometry& geom);

/*
  EvolveM_2nd.cpp
*/

void EvolveM_2nd(std::array< MultiFab, AMREX_SPACEDIM> &Mfield,
                 std::array< MultiFab, AMREX_SPACEDIM> &H_demagfield,
                 std::array< MultiFab, AMREX_SPACEDIM> &H_biasfield, // H bias
                 std::array< MultiFab, AMREX_SPACEDIM> &H_exchangefield, // effective exchange field
                 std::array< MultiFab, AMREX_SPACEDIM> &H_DMIfield,
                 std::array< MultiFab, AMREX_SPACEDIM> &H_anisotropyfield,
                 MultiFab                              &alpha,
                 MultiFab                              &Ms,
                 MultiFab                              &gamma,
                 MultiFab                              &exchange,
                 MultiFab                              &DMI,
                 MultiFab                              &anisotropy,
                 MultiFab                              &Kxx_dft_real,
                 MultiFab                              &Kxx_dft_imag,
                 MultiFab                              &Kxy_dft_real,
                 MultiFab                              &Kxy_dft_imag,
                 MultiFab                              &Kxz_dft_real,
                 MultiFab                              &Kxz_dft_imag,
                 MultiFab                              &Kyy_dft_real,
                 MultiFab                              &Kyy_dft_imag,
                 MultiFab                              &Kyz_dft_real,
                 MultiFab                              &Kyz_dft_imag,
                 MultiFab                              &Kzz_dft_real,
                 MultiFab                              &Kzz_dft_imag,
                 GpuArray<int, 3>                      n_cell_large,
                 const Geometry&                       geom_large,
                 const Geometry& geom,
                 const Real& time,
                 const Real& dt);

/*
  Initialization.cpp
*/

void InitializeMagneticProperties(MultiFab& alpha,
                                  MultiFab& Ms, 
                                  MultiFab& gamma,
                                  MultiFab& exchange,
                                  MultiFab& DMI,
                                  MultiFab& anisotropy,
                                  const Geometry& geom,
                                  const Real& time);

void InitializeFields(Array< MultiFab, AMREX_SPACEDIM >&  Mfield,
                      const       Geometry& geom);

void ComputeHbias(Array< MultiFab, AMREX_SPACEDIM >&   H_biasfield,
                  const           Real& time,
                  const       Geometry& geom);

void ComputeAlpha(MultiFab&  alpha,
                  const Geometry& geom,
                  const Real& time);

/*
  NormalizeM.cpp
*/

void NormalizeM(Array< MultiFab, AMREX_SPACEDIM >& Mfield,
                MultiFab& Ms,
                const Geometry& geom);

/*
  Plotfile.cpp
*/

void WritePlotfile(MultiFab& Ms,
                   Array< MultiFab, AMREX_SPACEDIM>& Mfield,
                   Array< MultiFab, AMREX_SPACEDIM>& H_biasfield,
                   Array< MultiFab, AMREX_SPACEDIM>& H_exchangefield,
                   Array< MultiFab, AMREX_SPACEDIM>& H_DMIfield,
                   Array< MultiFab, AMREX_SPACEDIM>& H_anisotropyfield,
                   Array< MultiFab, AMREX_SPACEDIM>& H_demagfield,
                   const Geometry& geom,
                   const Real& time,
                   const int& plt_step);
