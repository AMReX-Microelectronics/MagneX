#include <AMReX_MultiFab.H>

using namespace amrex;

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

void Compute_LLG_RHS(Array< MultiFab, AMREX_SPACEDIM >&  LLG_RHS,
                     const Array< MultiFab, AMREX_SPACEDIM >&   Mfield_old,
                     Array< MultiFab, AMREX_SPACEDIM >&   H_demagfield,
                     Array< MultiFab, AMREX_SPACEDIM >&   H_biasfield,
                     Array< MultiFab, AMREX_SPACEDIM >&   H_exchangefield,
                     Array< MultiFab, AMREX_SPACEDIM >&   H_DMIfield,
                     Array< MultiFab, AMREX_SPACEDIM >&   H_anisotropyfield,
                     MultiFab&   alpha,
                     MultiFab&   Ms,
                     MultiFab&   gamma,
                     int demag_coupling,
                     int exchange_coupling,
                     int DMI_coupling,
                     int anisotropy_coupling,
                     int M_normalization,
                     Real mu0);

/*
  Demagnetization.cpp
*/

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
                           MultiFab&   anisotropy,
                           int anisotropy_coupling,
                           amrex::GpuArray<amrex::Real, 3>& anisotropy_axis,
                           Real mu0,
                           const Geometry& geom);

/*
  EffectiveDMIField.cpp
*/

void CalculateH_DMI(Array< MultiFab, AMREX_SPACEDIM> &   Mfield,
                    Array< MultiFab, AMREX_SPACEDIM> &   H_DMIfield,
                    MultiFab&   Ms,
                    MultiFab&   exchange,
                    MultiFab&   DMI,
                    int DMI_coupling,
                    Real mu0,
                    const Geometry& geom);

/*
  EffectiveExchangeField.cpp
*/

void CalculateH_exchange(Array< MultiFab, AMREX_SPACEDIM> &   Mfield,
                         Array< MultiFab, AMREX_SPACEDIM> &   H_exchangefield,
                         MultiFab&   Ms,
                         MultiFab&   exchange,
                         MultiFab&   DMI,
                         int exchange_coupling,
                         int DMI_coupling,
                         Real mu0,
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
                 MultiFab                              &PoissonRHS, 
                 MultiFab                              &PoissonPhi, 
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
                 std::array< MultiFab, AMREX_SPACEDIM> &Mfield_padded,
                 GpuArray<int, 3>                      n_cell_large,
                 const Geometry&                       geom_large,
                 int demag_coupling,
                 int demag_solver,
                 int exchange_coupling,
                 int DMI_coupling,
                 int anisotropy_coupling,
                 amrex::GpuArray<amrex::Real, 3>& anisotropy_axis,
                 int M_normalization, 
                 Real mu0,
                 const Geometry& geom,
                 const Real& time,
                 const Real& dt,
                 amrex::GpuArray<amrex::Real, 3> prob_lo,
                 amrex::GpuArray<amrex::Real, 3> prob_hi,
                 int timedependent_Hbias,
                 int timedependent_alpha,
                 const Real& iterative_tolerance);

/*
  Initialization.cpp
*/

void InitializeMagneticProperties(MultiFab& alpha,
                                  MultiFab& Ms, 
                                  MultiFab& gamma,
                                  MultiFab& exchange,
                                  MultiFab& DMI,
                                  MultiFab& anisotropy,
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

/*
  NormalizeM.cpp
*/

void NormalizeM(Array< MultiFab, AMREX_SPACEDIM >& Mfield,
                MultiFab& Ms, int M_normalization,
                const Geometry& geom);
