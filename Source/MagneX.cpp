#include "MagneX.H"

#include <AMReX_ParmParse.H>

/////////////////////////////////////////////////////
// Default values for these parameters can be seen in
// the InitializeMagneXNamespace() function below
/////////////////////////////////////////////////////

// total number of grid cells in the simulation domain in each direction
AMREX_GPU_MANAGED amrex::GpuArray<int, 3> MagneX::n_cell;

// for spatial domain decomposition
// maximum number of cells in each box in each direction;
int MagneX::max_grid_size_x;
int MagneX::max_grid_size_y;
int MagneX::max_grid_size_z;

// physical lo/hi coordiates of the domain
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 3> MagneX::prob_lo;
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 3> MagneX::prob_hi;

// total steps in simulation
int MagneX::nsteps;

// maximum simulation time (will override nsteps if reached)
amrex::Real MagneX::stop_time;

// time step
amrex::Real MagneX::dt;

// time integration options
// 1 = first order forward Euler
// 2 = iterative predictor-corrector
// 3 = iterative direct solver
// 4 = AMReX and SUNDIALS integrators
int MagneX::TimeIntegratorOption;

// for MRI options, the time scale factor for the fast rhs (with respect to overall coarse time step)
amrex::Real MagneX::fast_dt_ratio;

// for MRI, specify what goes into the fast partition
int MagneX:: fast_H_bias;
int MagneX:: fast_exchange;
int MagneX:: fast_DMI;
int MagneX:: fast_anisotropy;
int MagneX:: fast_demag;
int MagneX:: fast_STT;

// for IMEX, specify what goes into the implicit partition
int MagneX:: implicit_H_bias;
int MagneX:: implicit_exchange;
int MagneX:: implicit_DMI;
int MagneX:: implicit_anisotropy;
int MagneX:: implicit_demag;
int MagneX:: implicit_STT;

// for TimeIntegrationOption 2 and 3 options only
// tolerance threshold (L_inf change between iterations) required for completing time step
// special cases:
// for TimeIntegrationOption=2, iterative_tolerance=0. means force 2 iterations
// for TimeIntegrationOption=3, iterative_tolerance=0. means force 1 iteration
amrex::Real MagneX::iterative_tolerance;

// how often to write a plotfile
int MagneX::plot_int;

// whether or not to include specific variables in plotfile
int MagneX::plot_Ms;
int MagneX::plot_H_bias;
int MagneX::plot_exchange;
int MagneX::plot_DMI;
int MagneX::plot_anisotropy;
int MagneX::plot_demag;
int MagneX::plot_STT_DL;
int MagneX::plot_STT_TL;
int MagneX::plot_theta;

// how often to write a checkpoint
int MagneX::chk_int;

// step to restart from
int MagneX::restart;

// what type of extra diagnostics?
// 4 = standard problem 4
// 3 = standard problem 3
// 2 = standard problem 2
int MagneX::diag_type;

// whether to call the parser each time step, or only at initialization
int MagneX::timedependent_Hbias;
int MagneX::timedependent_alpha;
int MagneX::timedependent_J_STT;

////////////////////////////////
// parameters for Hbias sweeping

// flag for enabling Hbias sweeps for hysteresis calculations
// cannot have Hbias_sweep=1 AND timedependent_Hbias=1
int MagneX::Hbias_sweep;

// tolerance threshold for equilibrium in M in Hbias sweeping
amrex::Real MagneX::equilibrium_tolerance;

// increment size for Hbias  
AMREX_GPU_MANAGED amrex::Real MagneX::increment_size;

// number of increments on Hbias before reversing sign of increment
int MagneX::nsteps_hysteresis;
////////////////////////////////

// permeability
AMREX_GPU_MANAGED amrex::Real MagneX::mu0;

// 0 = unsaturated; 1 = saturated
AMREX_GPU_MANAGED int MagneX::M_normalization;

// 0 = disable precession; 1 = enable precession
AMREX_GPU_MANAGED int MagneX::precession;

// 0 = disable exchange; 1 = enable exchange
AMREX_GPU_MANAGED int MagneX::exchange_coupling;

// 0 = disable DMI; 1 = enable DMI
AMREX_GPU_MANAGED int MagneX::DMI_coupling;

// 0 = disable anisotropy; 1 = enable anisotropy
// anisotropy_axis only required for anistropy_coupling=1
AMREX_GPU_MANAGED int MagneX::anisotropy_coupling;
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 3> MagneX::anisotropy_axis;

// 0 = no demagnetization; 1 = use demanetization
AMREX_GPU_MANAGED int MagneX::demag_coupling;

// demagnetization FFT solver liblrary
// 0 = FFTW (single-MPI), 1 = heFFTe (distributed)
AMREX_GPU_MANAGED int MagneX::FFT_solver;

////////////////////////////////
// parameters for spin-transfer torque (STT); see EffectiveSTTField.cpp for the model

// 0 = disable; 1 = enable the damping-like STT field, H_STT_DL = a_J (m x p)
AMREX_GPU_MANAGED int MagneX::STT_DL_coupling;

// 0 = disable; 1 = enable the field-like STT field, H_STT_TL = b_J p
AMREX_GPU_MANAGED int MagneX::STT_TL_coupling;

// the parameters below are only required if STT_DL_coupling=1 or STT_TL_coupling=1
// (the current density is read from J_STT_parser(x,y,z,t) [A/m^2])

// direction p of the fixed (reference) layer magnetization; normalized after it is read in
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 3> MagneX::STT_polarization_axis;

// thickness d of the free layer [m]
AMREX_GPU_MANAGED amrex::Real MagneX::STT_free_layer_thickness;

// spin polarization P of the current (STT_DL_coupling=1 only)
AMREX_GPU_MANAGED amrex::Real MagneX::STT_polarization;

// Slonczewski asymmetry parameter Lambda (STT_DL_coupling=1 only);
// Lambda = 1 gives a_J = hbar J P / (2 mu0 e Ms d), independent of the angle between m and p
AMREX_GPU_MANAGED amrex::Real MagneX::STT_Lambda;

// secondary spin-torque parameter epsilon' (STT_TL_coupling=1 only);
// b_J = hbar J epsilon' / (mu0 e Ms d), i.e. b_J/a_J = 2 epsilon'/P for Lambda = 1
AMREX_GPU_MANAGED amrex::Real MagneX::STT_epsilon_prime;
////////////////////////////////

void InitializeMagneXNamespace() {

    BL_PROFILE_VAR("InitializeMagneXNamespace()",InitializeMagneXNameSpace);

    /////////////////////////////////////////////////////
    // ParmParse is way of reading inputs from the inputs file
    // pp.get means we require the inputs file to have it
    // pp.query means we optionally need the inputs file to have it - but we must supply a default here
    /////////////////////////////////////////////////////
    ParmParse pp;

    // temporary vectors for reading in parameters from inputs files
    // these get copied into the GpuArray versions of these parameters
    amrex::Vector<int> temp_int(AMREX_SPACEDIM);
    amrex::Vector<amrex::Real> temp(AMREX_SPACEDIM);

    pp.getarr("n_cell",temp_int);
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        n_cell[i] = temp_int[i];
    }

    pp.get("max_grid_size_x",max_grid_size_x);
    pp.get("max_grid_size_y",max_grid_size_y);
    pp.get("max_grid_size_z",max_grid_size_z);

    pp.getarr("prob_lo",temp);
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        prob_lo[i] = temp[i];
    }
    pp.getarr("prob_hi",temp);
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        prob_hi[i] = temp[i];
    }

    nsteps = 1000000000;
    pp.query("nsteps",nsteps);

    stop_time = 1.e9;
    pp.query("stop_time",stop_time);

    pp.get("dt",dt);

    pp.get("TimeIntegratorOption",TimeIntegratorOption);

    fast_dt_ratio = 1.;
    pp.query("fast_dt_ratio",fast_dt_ratio);

    fast_H_bias = 0;
    fast_exchange = 0;
    fast_DMI = 0;
    fast_anisotropy = 0;
    fast_demag = 0;
    pp.query("fast_H_bias",fast_H_bias);
    pp.query("fast_exchange",fast_exchange);
    pp.query("fast_DMI",fast_DMI);
    pp.query("fast_anisotropy",fast_anisotropy);
    pp.query("fast_demag",fast_demag);
    fast_STT = 0;
    pp.query("fast_STT",fast_STT);

    implicit_H_bias = 0;
    implicit_exchange = 0;
    implicit_DMI = 0;
    implicit_anisotropy = 0;
    implicit_demag = 0;
    pp.query("implicit_H_bias",implicit_H_bias);
    pp.query("implicit_exchange",implicit_exchange);
    pp.query("implicit_DMI",implicit_DMI);
    pp.query("implicit_anisotropy",implicit_anisotropy);
    pp.query("implicit_demag",implicit_demag);
    implicit_STT = 0;
    pp.query("implicit_STT",implicit_STT);

    iterative_tolerance = 1.e-9;
    pp.query("iterative_tolerance",iterative_tolerance);

    plot_int = -1;
    pp.query("plot_int",plot_int);

    plot_Ms = 1;
    pp.query("plot_Ms",plot_Ms);
    plot_H_bias = 1;
    pp.query("plot_H_bias",plot_H_bias);
    plot_exchange = 1;
    pp.query("plot_exchange",plot_exchange);
    plot_DMI = 1;
    pp.query("plot_DMI",plot_DMI);
    plot_anisotropy = 1;
    pp.query("plot_anisotropy",plot_anisotropy);
    plot_demag = 1;
    pp.query("plot_demag",plot_demag);
    plot_STT_DL = 1;
    pp.query("plot_STT_DL",plot_STT_DL);
    plot_STT_TL = 1;
    pp.query("plot_STT_TL",plot_STT_TL);
    plot_theta = 1;
    pp.query("plot_theta",plot_theta);

    chk_int= -1;
    pp.query("chk_int",chk_int);

    restart = -1;
    pp.query("restart",restart);

    diag_type = -1;
    pp.query("diag_type",diag_type);

    pp.get("timedependent_Hbias",timedependent_Hbias);
    pp.get("timedependent_alpha",timedependent_alpha);

    Hbias_sweep = 0;
    pp.query("Hbias_sweep", Hbias_sweep);

    if (Hbias_sweep == 1) {
        pp.get("equilibrium_tolerance",equilibrium_tolerance);
        pp.get("increment_size",increment_size);
        pp.get("nsteps_hysteresis",nsteps_hysteresis);
    }

    pp.get("mu0",mu0);

    pp.get("M_normalization", M_normalization);

    precession = 1;
    pp.query("precession",precession);

    pp.get("exchange_coupling", exchange_coupling);
    pp.get("DMI_coupling", DMI_coupling);

    pp.get("anisotropy_coupling", anisotropy_coupling);
    if (anisotropy_coupling) {
        pp.getarr("anisotropy_axis",temp);
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            anisotropy_axis[i] = temp[i];
        }
    }
   
    pp.get("demag_coupling",demag_coupling);
    if (demag_coupling == 1) {
        pp.get("FFT_solver",FFT_solver);
    }

    // spin-transfer torque; both terms are off unless requested
    STT_DL_coupling = 0;
    pp.query("STT_DL_coupling", STT_DL_coupling);
    STT_TL_coupling = 0;
    pp.query("STT_TL_coupling", STT_TL_coupling);

    // defaults, so that these are well defined when STT is off
    timedependent_J_STT = 0;
    STT_polarization_axis[0] = 0.;
    STT_polarization_axis[1] = 0.;
    STT_polarization_axis[2] = 1.;
    STT_free_layer_thickness = 1.;
    STT_polarization = 0.;
    STT_Lambda = 1.;
    STT_epsilon_prime = 0.;

    if (STT_DL_coupling == 1 || STT_TL_coupling == 1) {

        pp.get("timedependent_J_STT",timedependent_J_STT);

        pp.getarr("STT_polarization_axis",temp,0,AMREX_SPACEDIM);
        amrex::Real p_norm = 0.;
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            p_norm += temp[i]*temp[i];
        }
        p_norm = std::sqrt(p_norm);
        if (p_norm <= 0.) {
            amrex::Abort("STT_polarization_axis must be a non-zero vector");
        }
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            STT_polarization_axis[i] = temp[i] / p_norm;
        }

        pp.get("STT_free_layer_thickness",STT_free_layer_thickness);
        if (STT_free_layer_thickness <= 0.) {
            amrex::Abort("STT_free_layer_thickness must be > 0");
        }

        if (STT_DL_coupling == 1) {
            pp.get("STT_polarization",STT_polarization);
            pp.query("STT_Lambda",STT_Lambda);
            if (STT_Lambda <= 0.) {
                amrex::Abort("STT_Lambda must be > 0");
            }
        }

        if (STT_TL_coupling == 1) {
            pp.get("STT_epsilon_prime",STT_epsilon_prime);
        }
    }

    // invalid parameter combination checking
    if (Hbias_sweep == 1 && timedependent_Hbias == 1) {
        amrex::Abort("Cannot have Hbias_sweep == 1 && timedependent_Hbias == 1");
    }

    if (diag_type == 2 && Hbias_sweep == 0) {
        amrex::Abort("Cannot have diag_type == 2 && Hbias_sweep == 0");
    }

    // the STT is a torque that is not derived from an energy, so relaxing
    // without the M x H term (precession == 0) is not defined for it
    if (precession == 0 && (STT_DL_coupling == 1 || STT_TL_coupling == 1)) {
        amrex::Abort("Cannot have precession == 0 with STT_DL_coupling == 1 or STT_TL_coupling == 1");
    }
    
}
