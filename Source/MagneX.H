#include <heffte.h>

#ifdef AMREX_USE_CUDA
#include <cufft.h>
#else
#include <fftw3.h>
#include <fftw3-mpi.h>
#endif

#include <AMReX_MultiFab.H>

#include "MagneX_namespace.H"
#include "Demagnetization.H"

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
                 Demagnetization                       &demag_solver,
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
