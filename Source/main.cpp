
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H> 
#include <AMReX_OpenBC.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MultiFab.H> 
#include <AMReX_VisMF.H>

#include "Initialization.H"
#include "Demagnetization.H"
#include "EffectiveExchangeField.H"
#include "EffectiveDMIField.H"
#include "EffectiveAnisotropyField.H"
#include "CartesianAlgorithm.H"
#include "ComputeLLGRHS.H"
#include "EvolveM_2nd.H"
#include "NormalizeM.H"
#include "Checkpoint.H"
#include "Diagnostics.H"
#ifdef USE_TIME_INTEGRATOR
#include <AMReX_TimeIntegrator.H>
#endif

using namespace amrex;

void main_main();

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
    return 0;
}

void main_main ()
{

    Real total_step_strt_time = ParallelDescriptor::second();

    // **********************************
    // SIMULATION PARAMETERS
    // **********************************

    // Number of cells in each dimension
    amrex::GpuArray<int, 3> n_cell; 

    // maximum size of each box
    int max_grid_size;

    // physical lo/hi coordiates
    amrex::GpuArray<amrex::Real, 3> prob_lo;
    amrex::GpuArray<amrex::Real, 3> prob_hi;
    
    // total steps in simulation
    int nsteps;

    // 1 = first order forward Euler
    // 2 = iterative predictor-corrector
    // 3 = iterative direct solver
    // 4 = AMReX and SUNDIALS integrators
    int TimeIntegratorOption;

    // time step
    Real dt;

    // how often to write a plotfile
    int plot_int = -1;

    // ho often to write a checkpoint
    int chk_int = -1;

    // step to restart from
    int restart = -1;

    // permeability
    Real mu0;

    // turn on demagnetization
    int demag_coupling;

    // demagnetization solver type
    // -1 = periodic/Neumann MLMG
    // 0 = Open Poisson MLMG
    // 1 = FFT-based
    int demag_solver;

    // 0 = unsaturated; 1 = saturated
    int M_normalization;

    // turn on exchange
    int exchange_coupling;

    // turn on DMI
    int DMI_coupling;

    // turn on anisotropy
    int anisotropy_coupling;
    amrex::GpuArray<amrex::Real, 3> anisotropy_axis; 

    // change alpha during runtime
    int alpha_scale_step = -1;
    Real alpha_scale_factor = 1.;

    // **********************************
    // READ SIMULATION PARAMETERS
    // **********************************
    {
        // ParmParse is way of reading inputs from the inputs file
        // pp.get means we require the inputs file to have it
        // pp.query means we optionally need the inputs file to have it - but we must supply a default here
        ParmParse pp;

        amrex::Vector<int> temp_int(AMREX_SPACEDIM);
        pp.getarr("n_cell",temp_int);
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            n_cell[i] = temp_int[i];
        }

        pp.get("max_grid_size",max_grid_size);

        amrex::Vector<amrex::Real> temp(AMREX_SPACEDIM);
        pp.getarr("prob_lo",temp);
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            prob_lo[i] = temp[i];
        }
        pp.getarr("prob_hi",temp);
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            prob_hi[i] = temp[i];
        }

        nsteps = 10;
        pp.query("nsteps",nsteps);

        pp.get("TimeIntegratorOption",TimeIntegratorOption);
	
        pp.get("dt",dt);

        pp.query("plot_int",plot_int);

        pp.query("chk_int",chk_int);

        pp.query("restart",restart);
	
        pp.get("mu0",mu0);

        pp.get("demag_coupling",demag_coupling);
        if (demag_coupling) {
            pp.get("demag_solver",demag_solver);
        }
        
        pp.get("M_normalization", M_normalization);
        pp.get("exchange_coupling", exchange_coupling);
        pp.get("DMI_coupling", DMI_coupling);

        pp.get("anisotropy_coupling", anisotropy_coupling);
        if (anisotropy_coupling) {
            pp.getarr("anisotropy_axis",temp);
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                anisotropy_axis[i] = temp[i];
            }
        }

        pp.query("alpha_scale_step",alpha_scale_step);
        pp.query("alpha_scale_factor",alpha_scale_factor);
    }

    int start_step = 1;

    // time = starting time in the simulation
    Real time = 0.0;	

    Array<MultiFab, AMREX_SPACEDIM> Mfield;
    Array<MultiFab, AMREX_SPACEDIM> Mfield_old;
    Array<MultiFab, AMREX_SPACEDIM> Mfield_prev_iter;
    Array<MultiFab, AMREX_SPACEDIM> Mfield_error;
    Array<MultiFab, AMREX_SPACEDIM> H_biasfield;
    Array<MultiFab, AMREX_SPACEDIM> H_demagfield;
    Array<MultiFab, AMREX_SPACEDIM> H_exchangefield;
    Array<MultiFab, AMREX_SPACEDIM> H_DMIfield;
    Array<MultiFab, AMREX_SPACEDIM> H_anisotropyfield;

    Array<MultiFab, AMREX_SPACEDIM> LLG_RHS;
    Array<MultiFab, AMREX_SPACEDIM> LLG_RHS_pre;
    Array<MultiFab, AMREX_SPACEDIM> LLG_RHS_avg;

    BoxArray ba;
    DistributionMapping dm;
    
    if (restart > 0) {

        start_step = restart+1;

        // read in Mfield, H_biasfield, and ba
        // create a DistributionMapping dm
        ReadCheckPoint(restart,time,Mfield,H_biasfield,H_demagfield,ba,dm);
      
    }
    
    // **********************************
    // SIMULATION SETUP

    // make BoxArray and Geometry
    // ba will contain a list of boxes that cover the domain
    // geom contains information such as the physical domain size,
    //               number of points in the domain, and periodicity

    // AMREX_D_DECL means "do the first X of these, where X is the dimensionality of the simulation"
    IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
    IntVect dom_hi(AMREX_D_DECL(n_cell[0]-1, n_cell[1]-1, n_cell[2]-1));

    // Make a single box that is the entire domain
    Box domain(dom_lo, dom_hi);
    
    if (restart == -1) {
        // Initialize the boxarray "ba" from the single box "domain"
        ba.define(domain);

        // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
        ba.maxSize(max_grid_size);
    }

    // This defines the physical box in each direction.
    RealBox real_box({AMREX_D_DECL( prob_lo[0], prob_lo[1], prob_lo[2])},
                     {AMREX_D_DECL( prob_hi[0], prob_hi[1], prob_hi[2])});

    // periodic in x and y directions
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)}; // nonperiodic in all directions

    // This defines a Geometry object
    Geometry geom;
    geom.define(domain, real_box, CoordSys::cartesian, is_periodic);

    // How Boxes are distrubuted among MPI processes
    if (restart == -1) {
        dm.define(ba);
    }
   
    // Allocate multifabs
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
        //Cell-centered fields
        Mfield_old[dir].define(ba, dm, 1, 1);
        Mfield_prev_iter[dir].define(ba, dm, 1, 1);
        Mfield_error[dir].define(ba, dm, 1, 1);

        H_exchangefield[dir].define(ba, dm, 1, 0);
        H_DMIfield[dir].define(ba, dm, 1, 0);
        H_anisotropyfield[dir].define(ba, dm, 1, 0);

        // set to zero in case we don't include
        H_exchangefield[dir].setVal(0.);
        H_DMIfield[dir].setVal(0.);
        H_anisotropyfield[dir].setVal(0.);

        LLG_RHS[dir].define(ba, dm, 1, 0);
        LLG_RHS_pre[dir].define(ba, dm, 1, 0);
        LLG_RHS_avg[dir].define(ba, dm, 1, 0);
    }

    if (restart == -1) {
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
            //Cell-centered fields
            Mfield[dir].define(ba, dm, 1, 1);
            H_biasfield[dir].define(ba, dm, 1, 1);
            H_demagfield[dir].define(ba, dm, 1, 1);
        }
    }

    MultiFab alpha(ba, dm, 1, 0);
    MultiFab gamma(ba, dm, 1, 0);
    MultiFab Ms(ba, dm, 1, 1);
    MultiFab exchange(ba, dm, 1, 0);
    MultiFab DMI(ba, dm, 1, 0);
    MultiFab anisotropy(ba, dm, 1, 0);

    amrex::Print() << "==================== Initial Setup ====================\n";
    amrex::Print() << " demag_coupling       = " << demag_coupling      << "\n";
    amrex::Print() << " M_normalization      = " << M_normalization     << "\n";
    amrex::Print() << " exchange_coupling    = " << exchange_coupling   << "\n";
    amrex::Print() << " DMI_coupling         = " << DMI_coupling        << "\n";
    amrex::Print() << " anisotropy_coupling  = " << anisotropy_coupling << "\n";
    amrex::Print() << " TimeIntegratorOption = " << TimeIntegratorOption << "\n";
    amrex::Print() << "=======================================================\n";

    MultiFab PoissonRHS(ba, dm, 1, 0);
    MultiFab PoissonPhi(ba, dm, 1, 1); // one ghost cell

    // initialize to zero; for demag_coupling==0 these aren't used but are still included in plotfile
    PoissonPhi.setVal(0.);
    PoissonRHS.setVal(0.);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        LLG_RHS[idim].setVal(0.);
        LLG_RHS_pre[idim].setVal(0.);
        LLG_RHS_avg[idim].setVal(0.);
        H_demagfield[idim].setVal(0.);
    }

    MultiFab Plt(ba, dm, 21, 0);

    // needed for FFT-based demag solver
    BoxArray ba_large;
    DistributionMapping dm_large;
    Geometry geom_large;

    // Create a zero-padded Magnetization field for the convolution method
    Array<MultiFab, AMREX_SPACEDIM> Mfield_padded;
    MultiFab Kxx_dft_real;
    MultiFab Kxx_dft_imag;
    MultiFab Kxy_dft_real;
    MultiFab Kxy_dft_imag;
    MultiFab Kxz_dft_real;
    MultiFab Kxz_dft_imag;
    MultiFab Kyy_dft_real;
    MultiFab Kyy_dft_imag;
    MultiFab Kyz_dft_real;
    MultiFab Kyz_dft_imag;
    MultiFab Kzz_dft_real;
    MultiFab Kzz_dft_imag;

    // Create a double-sized n_cell array
    amrex::GpuArray<int, 3> n_cell_large; // Number of cells in each dimension
    n_cell_large[0] = 2*n_cell[0];
    n_cell_large[1] = 2*n_cell[1];
    n_cell_large[2] = 2*n_cell[2];

    // Solver for Poisson equation
    LPInfo info;

    ///////////////////////////////
    // periodic / neumann multigrid solver
    ///////////////////////////////

    std::unique_ptr<MLMG> mlmg;
    MLABecLaplacian mlabec;

    if (demag_coupling == 1 && demag_solver == -1) {
    
        mlabec.define({geom}, {ba}, {dm}, info);
        mlabec.setEnforceSingularSolvable(false);

        // order of stencil
        int linop_maxorder = 2;
        mlabec.setMaxOrder(linop_maxorder);

        // build array of boundary conditions needed by MLABecLaplacian
        std::array<LinOpBCType, AMREX_SPACEDIM> lo_mlmg_bc;
        std::array<LinOpBCType, AMREX_SPACEDIM> hi_mlmg_bc;

        // boundary conditions - FIXME allow for user to control periodicity
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            if (is_periodic[idim]) {
                lo_mlmg_bc[idim] = hi_mlmg_bc[idim] = LinOpBCType::Periodic;
            } else {
                lo_mlmg_bc[idim] = hi_mlmg_bc[idim] = LinOpBCType::Neumann;
            }
        }

        mlabec.setDomainBC(lo_mlmg_bc,hi_mlmg_bc);

        MultiFab alpha_cc(ba, dm, 1, 0);
        std::array< MultiFab, AMREX_SPACEDIM > beta_face;
        AMREX_D_TERM(beta_face[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 1, 0);,
                     beta_face[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 1, 0);,
                     beta_face[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 1, 0););

        alpha_cc.setVal(0.);
        beta_face[0].setVal(1.);
        beta_face[1].setVal(1.);
        beta_face[2].setVal(1.);

        // (A*alpha_cc - B * div beta grad) phi = rhs
        mlabec.setScalars(0.0, 1.0); // A = 0.0, B = 1.0
        mlabec.setACoeffs(0, alpha_cc); //First argument 0 is lev
        mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(beta_face));

        // set boundary conditions to homogeneous Neumann
        mlabec.setLevelBC(0, &PoissonPhi);

        // declare MLMG object
        mlmg = std::make_unique<MLMG>(mlabec);
        mlmg->setVerbose(2);
    }
        

    ///////////////////////////////
    // openBC multigrid solver
    ///////////////////////////////

    OpenBCSolver openbc;
    
    if (demag_coupling == 1 && demag_solver == 0) {
        openbc.define({geom}, {ba}, {dm}, info);
        openbc.setVerbose(2);
    }

    if (demag_coupling == 1 && demag_solver == 1) {

        // **********************************
        // SIMULATION SETUP
        // make BoxArray and Geometry
        // ba will contain a list of boxes that cover the domain
        // geom contains information such as the physical domain size,
        // number of points in the domain, and periodicity

        // AMREX_D_DECL means "do the first X of these, where X is the dimensionality of the simulation"
        IntVect dom_lo_large(AMREX_D_DECL(                0,                 0,                 0));
        IntVect dom_hi_large(AMREX_D_DECL(n_cell_large[0]-1, n_cell_large[1]-1, n_cell_large[2]-1));

        // Make a single box that is the entire domain
        Box domain_large(dom_lo_large, dom_hi_large);

        // Initialize the boxarray "ba" from the single box "domain"
        ba_large.define(domain_large);

        // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
        ba_large.maxSize(max_grid_size);

        // How Boxes are distrubuted among MPI processes
        dm_large.define(ba_large);
	     
        // This defines a Geometry object
        geom_large.define(domain_large, real_box, CoordSys::cartesian, is_periodic);
	    
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
            // Cell-centered fields
            Mfield_padded[dir].define(ba_large, dm_large, 1, 1);
        }

        // Allocate the demag tensor fft multifabs
        Kxx_dft_real.define(ba_large, dm_large, 1, 0);
        Kxx_dft_imag.define(ba_large, dm_large, 1, 0);
        Kxy_dft_real.define(ba_large, dm_large, 1, 0);
        Kxy_dft_imag.define(ba_large, dm_large, 1, 0);
        Kxz_dft_real.define(ba_large, dm_large, 1, 0);
        Kxz_dft_imag.define(ba_large, dm_large, 1, 0);
        Kyy_dft_real.define(ba_large, dm_large, 1, 0);
        Kyy_dft_imag.define(ba_large, dm_large, 1, 0);
        Kyz_dft_real.define(ba_large, dm_large, 1, 0);
        Kyz_dft_imag.define(ba_large, dm_large, 1, 0);
        Kzz_dft_real.define(ba_large, dm_large, 1, 0);
        Kzz_dft_imag.define(ba_large, dm_large, 1, 0);

        // Call function that computes the demag tensors and then takes the forward FFT of each of them, which are returns
        ComputeDemagTensor(Kxx_dft_real, Kxx_dft_imag, Kxy_dft_real, Kxy_dft_imag, Kxz_dft_real, Kxz_dft_imag,
                           Kyy_dft_real, Kyy_dft_imag, Kyz_dft_real, Kyz_dft_imag, Kzz_dft_real, Kzz_dft_imag,
                           n_cell_large, geom_large);
        

    }

    InitializeMagneticProperties(alpha, Ms, gamma, exchange, DMI, anisotropy, prob_lo, prob_hi, geom);


    // count how many magnetic cells are in the domain
    long num_mag = CountMagneticCells(Ms);
    
    if (restart == -1) {      
        //Initialize fields
        InitializeFields(Mfield, prob_lo, prob_hi, geom);
        ComputeHbias(H_biasfield, prob_lo, prob_hi, time, geom);

        if (demag_coupling == 1) {
            
	    if (demag_solver == -1 || demag_solver == 0) {

                // Solve Poisson's equation laplacian(Phi) = div(M) and get H_demagfield = -grad(Phi)
                // Compute RHS of Poisson equation
                ComputePoissonRHS(PoissonRHS, Mfield, geom);

		//Initial guess for phi
                PoissonPhi.setVal(0.);

                if (demag_solver == -1) {                    
                    mlmg->solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
                } else if (demag_solver == 0) {
                    
                    openbc.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
                }

	        // Calculate H from Phi
	        ComputeHfromPhi(PoissonPhi, H_demagfield, geom);	    

            } else {

                // copy Mfield used for the RHS calculation in the Poisson option into Mfield_padded
                for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
                    Mfield_padded[dir].setVal(0.);
                    Mfield_padded[dir].ParallelCopy(Mfield[dir], 0, 0, 1);
                }

                ComputeHFieldFFT(Mfield_padded, H_demagfield,
                                 Kxx_dft_real, Kxx_dft_imag, Kxy_dft_real, Kxy_dft_imag, Kxz_dft_real, Kxz_dft_imag,
                                 Kyy_dft_real, Kyy_dft_imag, Kyz_dft_real, Kyz_dft_imag, Kzz_dft_real, Kzz_dft_imag,
                                 n_cell_large, geom_large);
	    }
	}
        
        if (exchange_coupling == 1) {
            CalculateH_exchange(Mfield, H_exchangefield, Ms, exchange, DMI, exchange_coupling, DMI_coupling, mu0, geom);
        }

        if (DMI_coupling == 1) {
            CalculateH_DMI(Mfield, H_DMIfield, Ms, exchange, DMI, DMI_coupling, mu0, geom);
        }

        if (anisotropy_coupling == 1) {
            CalculateH_anisotropy(Mfield, H_anisotropyfield, Ms, anisotropy, anisotropy_coupling, anisotropy_axis, mu0, geom);
        }
    }

    // Write a plotfile of the initial data if plot_int > 0
    if (plot_int > 0)
    {
        int plt_step = 0;
        if (restart > 0) {
            plt_step = restart;
        }
        const std::string& pltfile = amrex::Concatenate("plt",plt_step,8);

        MultiFab::Copy(Plt, Ms, 0, 0, 1, 0);
        MultiFab::Copy(Plt, Mfield[0], 0, 1, 1, 0);
        MultiFab::Copy(Plt, Mfield[1], 0, 2, 1, 0);
        MultiFab::Copy(Plt, Mfield[2], 0, 3, 1, 0);
        MultiFab::Copy(Plt, H_biasfield[0], 0, 4, 1, 0);
        MultiFab::Copy(Plt, H_biasfield[1], 0, 5, 1, 0);
        MultiFab::Copy(Plt, H_biasfield[2], 0, 6, 1, 0);
        MultiFab::Copy(Plt, H_exchangefield[0], 0, 7, 1, 0);
        MultiFab::Copy(Plt, H_exchangefield[1], 0, 8, 1, 0);
        MultiFab::Copy(Plt, H_exchangefield[2], 0, 9, 1, 0);
        MultiFab::Copy(Plt, H_DMIfield[0], 0, 10, 1, 0);
        MultiFab::Copy(Plt, H_DMIfield[1], 0, 11, 1, 0);
        MultiFab::Copy(Plt, H_DMIfield[2], 0, 12, 1, 0);
        MultiFab::Copy(Plt, H_anisotropyfield[0], 0, 13, 1, 0);
        MultiFab::Copy(Plt, H_anisotropyfield[1], 0, 14, 1, 0);
        MultiFab::Copy(Plt, H_anisotropyfield[2], 0, 15, 1, 0);
        MultiFab::Copy(Plt, H_demagfield[0], 0, 16, 1, 0);
        MultiFab::Copy(Plt, H_demagfield[1], 0, 17, 1, 0);
        MultiFab::Copy(Plt, H_demagfield[2], 0, 18, 1, 0);
        MultiFab::Copy(Plt, PoissonRHS, 0, 19, 1, 0);
        MultiFab::Copy(Plt, PoissonPhi, 0, 20, 1, 0);

        WriteSingleLevelPlotfile(pltfile, Plt, {"Ms",
                                                "Mx",
                                                "My",
                                                "Mz",
                                                "Hx_bias",
                                                "Hy_bias",
                                                "Hz_bias",
                                                "Hx_exchange",
                                                "Hy_exchange",
                                                "Hz_exchange",
                                                "Hx_DMI",
                                                "Hy_DMI",
                                                "Hz_DMI",
                                                "Hx_anisotropy",
                                                "Hy_anisotropy",
                                                "Hz_anisotropy",
                                                "Hx_demagfield","Hy_demagfield","Hz_demagfield",
                                                "PoissonRHS","PoissonPhi"},
                                                 geom, time, plt_step);
    }

    // copy new solution into old solution
    for (int comp = 0; comp < 3; comp++) {
        MultiFab::Copy(Mfield_old[comp], Mfield[comp], 0, 0, 1, 1);
    }

#ifdef USE_TIME_INTEGRATOR
    TimeIntegrator<Vector<MultiFab> > integrator(Mfield_old);
#endif 

    for (int step = start_step; step <= nsteps; ++step) {
        
        Real step_strt_time = ParallelDescriptor::second();

        // Fow now always assume H_bias is constant over the time step
        // This need to be fixed for second-order options
        ComputeHbias(H_biasfield, prob_lo, prob_hi, time, geom);

        // scale alpha
        if (step == alpha_scale_step) {
            alpha.mult(alpha_scale_factor);
        }

        // compute old-time LLG_RHS
        if (TimeIntegratorOption == 1 ||
            TimeIntegratorOption == 2 ||
            TimeIntegratorOption == 3) {
            
    	    // Evolve H_demag
            if (demag_coupling == 1) {
            
                if (demag_solver == -1 || demag_solver == 0) {

                    //Solve Poisson's equation laplacian(Phi) = div(M) and get H_demagfield = -grad(Phi)
                    //Compute RHS of Poisson equation
                    ComputePoissonRHS(PoissonRHS, Mfield_old, geom);
                
                    //Initial guess for phi
                    PoissonPhi.setVal(0.);

                    if (demag_solver == -1) {
                        mlmg->solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
                    } else if (demag_solver == 0) {
                        openbc.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
                    }

                    // Calculate H from Phi
                    ComputeHfromPhi(PoissonPhi, H_demagfield, geom);
                } else {

                    // copy Mfield used for the RHS calculation in the Poisson option into Mfield_padded
                    for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
                        Mfield_padded[dir].setVal(0.);
                        Mfield_padded[dir].ParallelCopy(Mfield_old[dir], 0, 0, 1);
                    }

                    ComputeHFieldFFT(Mfield_padded, H_demagfield,
                                     Kxx_dft_real, Kxx_dft_imag, Kxy_dft_real, Kxy_dft_imag, Kxz_dft_real, Kxz_dft_imag,
                                     Kyy_dft_real, Kyy_dft_imag, Kyz_dft_real, Kyz_dft_imag, Kzz_dft_real, Kzz_dft_imag,
                                     n_cell_large, geom_large);
                }
            }

            if (exchange_coupling == 1) {
                CalculateH_exchange(Mfield_old, H_exchangefield, Ms, exchange, DMI, exchange_coupling, DMI_coupling, mu0, geom);
            }

            if (DMI_coupling == 1) {
                CalculateH_DMI(Mfield_old, H_DMIfield, Ms, exchange, DMI, DMI_coupling, mu0, geom);
            }

            if (anisotropy_coupling == 1) {
                CalculateH_anisotropy(Mfield_old, H_anisotropyfield, Ms, anisotropy, anisotropy_coupling, anisotropy_axis, mu0, geom);
            }
        }
        
        if (TimeIntegratorOption == 1) { // first order forward Euler

            // Evolve M
            // Compute f^n = f(M^n, H^n)
            Compute_LLG_RHS(LLG_RHS, Mfield_old, H_demagfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield, alpha,
                            Ms, gamma, demag_coupling, exchange_coupling, DMI_coupling, anisotropy_coupling, M_normalization, mu0);

            // M^{n+1} = M^n + dt * f^n
            for (int i = 0; i < 3; i++) {
                MultiFab::LinComb(Mfield[i], 1.0, Mfield_old[i], 0, dt, LLG_RHS[i], 0, 0, 1, 0);
            }

            NormalizeM(Mfield, Ms, M_normalization);
 
            for (int comp = 0; comp < 3; comp++)
            {
                // fill periodic ghost cells
                Mfield[comp].FillBoundary(geom.periodicity());
            }
 
            // copy new solution into old solution
            for (int comp = 0; comp < 3; comp++)
            {
                MultiFab::Copy(Mfield_old[comp], Mfield[comp], 0, 0, 1, 1);
    
                // fill periodic ghost cells
                Mfield_old[comp].FillBoundary(geom.periodicity());
 
            }
        } else if (TimeIntegratorOption == 2) { // iterative predictor-corrector
        
            Real M_tolerance = 1.e-6;
            int iter = 0;
    
            // Compute f^{n} = f(M^{n}, H^{n})
            Compute_LLG_RHS(LLG_RHS, Mfield_old, H_demagfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield, alpha, Ms,
                            gamma, demag_coupling, exchange_coupling, DMI_coupling, anisotropy_coupling, M_normalization, mu0);

            // copy old RHS into predicted RHS so first pass through is forward Euler
            for (int comp = 0; comp < 3; comp++) {
                MultiFab::Copy(LLG_RHS_pre[comp], LLG_RHS[comp], 0, 0, 1, 0);
                MultiFab::Copy(Mfield_prev_iter[comp], Mfield[comp], 0, 0, 1, 1);
            }

            while(1) { 
    
		// Corrector step update M
                // M^{n+1, *} = M^n + 0.5 * dt * (f^n + f^{n+1, *})
                for (int i = 0; i < 3; i++) {
                    MultiFab::LinComb(LLG_RHS_avg[i], 0.5, LLG_RHS[i], 0, 0.5, LLG_RHS_pre[i], 0, 0, 1, 0);
                    MultiFab::LinComb(Mfield[i], 1.0, Mfield_old[i], 0, dt, LLG_RHS_avg[i], 0, 0, 1, 0);
                }

                // Normalize M
                NormalizeM(Mfield, Ms, M_normalization);

                for (MFIter mfi(Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
     
                    const Box& bx = mfi.validbox();
    
                    Array4<Real> const& Ms_arr = Ms.array(mfi);
    
                    Array4<Real> const& Mx_error = Mfield_error[0].array(mfi);
                    Array4<Real> const& My_error = Mfield_error[1].array(mfi);
                    Array4<Real> const& Mz_error = Mfield_error[2].array(mfi);
                    Array4<Real> const& Mx = Mfield[0].array(mfi);
                    Array4<Real> const& My = Mfield[1].array(mfi);
                    Array4<Real> const& Mz = Mfield[2].array(mfi);
                    Array4<Real> const& Mx_prev_iter = Mfield_prev_iter[0].array(mfi);
                    Array4<Real> const& My_prev_iter = Mfield_prev_iter[1].array(mfi);
                    Array4<Real> const& Mz_prev_iter = Mfield_prev_iter[2].array(mfi);
    
                    amrex::ParallelFor (bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        if (Ms_arr(i,j,k) > 0) {
                            Mx_error(i,j,k) = amrex::Math::abs(Mx(i,j,k) - Mx_prev_iter(i,j,k)) / Ms_arr(i,j,k);
                            My_error(i,j,k) = amrex::Math::abs(My(i,j,k) - My_prev_iter(i,j,k)) / Ms_arr(i,j,k);
                            Mz_error(i,j,k) = amrex::Math::abs(Mz(i,j,k) - Mz_prev_iter(i,j,k)) / Ms_arr(i,j,k);
                        } else {
                            Mx_error(i,j,k) = 0.;
                            My_error(i,j,k) = 0.;
                            Mz_error(i,j,k) = 0.;
                        }
                    });
                }
    
                amrex::Real M_mag_error_max = -1.;
                M_mag_error_max = std::max(Mfield_error[0].norm0(), Mfield_error[1].norm0());
                M_mag_error_max = std::max(M_mag_error_max, Mfield_error[2].norm0());

                // copy new solution into Mfield_pre_iter
                for (int comp = 0; comp < 3; comp++) {
                    MultiFab::Copy(Mfield_prev_iter[comp], Mfield[comp], 0, 0, 1, 1);
                    // fill periodic ghost cells 
                    Mfield_prev_iter[comp].FillBoundary(geom.periodicity());
                }
    
                iter = iter + 1;

                // terminate while loop of error threshold is small enough
                amrex::Print() << "iter = " << iter << ", M_mag_error_max = " << M_mag_error_max << "\n";
                if (M_mag_error_max <= M_tolerance) break;

                // Poisson solve and H_demag computation with M_field_pre
                if (demag_coupling == 1) { 
            
                    if (demag_solver == -1 || demag_solver == 0) {
                        //Solve Poisson's equation laplacian(Phi) = div(M) and get H_demagfield = -grad(Phi)
                        //Compute RHS of Poisson equation
                        ComputePoissonRHS(PoissonRHS, Mfield_prev_iter, geom);
    
                        //Initial guess for phi
                        PoissonPhi.setVal(0.);

                        if (demag_solver == -1) {
                            mlmg->solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
                        } else if (demag_solver == 0) {
                            openbc.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
                        }

                        // Calculate H from Phi
                        ComputeHfromPhi(PoissonPhi, H_demagfield, geom);

                    } else {

                        // copy Mfield used for the RHS calculation in the Poisson option into Mfield_padded
                        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
                            Mfield_padded[dir].setVal(0.);
                            Mfield_padded[dir].ParallelCopy(Mfield_prev_iter[dir], 0, 0, 1);
                        }

                        ComputeHFieldFFT(Mfield_padded, H_demagfield,
                                         Kxx_dft_real, Kxx_dft_imag, Kxy_dft_real, Kxy_dft_imag, Kxz_dft_real, Kxz_dft_imag,
                                         Kyy_dft_real, Kyy_dft_imag, Kyz_dft_real, Kyz_dft_imag, Kzz_dft_real, Kzz_dft_imag,
                                         n_cell_large, geom_large);
                    }
                }
    
                if (exchange_coupling == 1) {
                    CalculateH_exchange(Mfield_prev_iter, H_exchangefield, Ms, exchange, DMI, exchange_coupling, DMI_coupling, mu0, geom);
                }
        
                if (DMI_coupling == 1) {
                    CalculateH_DMI(Mfield_prev_iter, H_DMIfield, Ms, exchange, DMI, DMI_coupling, mu0, geom);
                }
    
                if (anisotropy_coupling == 1) {
                    CalculateH_anisotropy(Mfield_prev_iter, H_anisotropyfield, Ms, anisotropy, anisotropy_coupling, anisotropy_axis, mu0, geom);
                }
    
                // LLG RHS with new H_demag and M_field_pre
                // Compute f^{n+1, *} = f(M^{n+1, *}, H^{n+1, *})
                Compute_LLG_RHS(LLG_RHS_pre, Mfield_prev_iter, H_demagfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield, alpha, Ms,
                                gamma, demag_coupling, exchange_coupling, DMI_coupling, anisotropy_coupling, M_normalization, mu0);

            }

            // copy new solution into old solution
            for (int comp = 0; comp < 3; comp++) {
                MultiFab::Copy(Mfield_old[comp], Mfield[comp], 0, 0, 1, 1);
                // fill periodic ghost cells
                Mfield_old[comp].FillBoundary(geom.periodicity());
            }
    
        } else if (TimeIntegratorOption == 3) { // iterative direct solver (ARTEMIS way)
        
            EvolveM_2nd(Mfield, H_demagfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield, PoissonRHS, PoissonPhi, alpha, Ms,
                        gamma, exchange, DMI, anisotropy,
                        Kxx_dft_real, Kxx_dft_imag, Kxy_dft_real, Kxy_dft_imag, Kxz_dft_real, Kxz_dft_imag,
                        Kyy_dft_real, Kyy_dft_imag, Kyz_dft_real, Kyz_dft_imag, Kzz_dft_real, Kzz_dft_imag, Mfield_padded,
                        n_cell_large, geom_large,
                        demag_coupling, demag_solver, exchange_coupling, DMI_coupling, anisotropy_coupling, anisotropy_axis,
                        M_normalization, mu0, geom, dt);

        }  else if (TimeIntegratorOption == 4) { // AMReX and SUNDIALS integrators

#ifdef USE_TIME_INTEGRATOR

            // Create a RHS source function we will integrate
            auto source_fun = [&](Vector<MultiFab>& rhs, const Vector<MultiFab>& old_state, const Real ) {

                // User function to calculate the rhs MultiFab given the state MultiFab
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                    rhs[idim].setVal(0.);
                } 
 
    	        // Evolve H_demag
                if (demag_coupling == 1) {
            
                    if (demag_solver == -1 || demag_solver == 0) {

                        //Solve Poisson's equation laplacian(Phi) = div(M) and get H_demagfield = -grad(Phi)
                        //Compute RHS of Poisson equation
                        ComputePoissonRHS(PoissonRHS, old_state, geom);
                     
                        //Initial guess for phi
                        PoissonPhi.setVal(0.);

                        if (demag_solver == -1) {
                            mlmg->solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
                        } else if (demag_solver == 0) {
                            openbc.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
                        }

                        // Calculate H from Phi
                        ComputeHfromPhi(PoissonPhi, H_demagfield, geom);
                    } else {

                        // copy Mfield used for the RHS calculation in the Poisson option into Mfield_padded
                        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
                            Mfield_padded[dir].setVal(0.);
                            Mfield_padded[dir].ParallelCopy(old_state[dir], 0, 0, 1);
                        }

                        ComputeHFieldFFT(Mfield_padded, H_demagfield,
                                         Kxx_dft_real, Kxx_dft_imag, Kxy_dft_real, Kxy_dft_imag, Kxz_dft_real, Kxz_dft_imag,
                                         Kyy_dft_real, Kyy_dft_imag, Kyz_dft_real, Kyz_dft_imag, Kzz_dft_real, Kzz_dft_imag,
                                         n_cell_large, geom_large);
                    }
                }

                if (exchange_coupling == 1) {
                    CalculateH_exchange(Mfield, H_exchangefield, Ms, exchange, DMI, exchange_coupling, DMI_coupling, mu0, geom);
                }

                if (DMI_coupling == 1) {
                    CalculateH_DMI(Mfield, H_DMIfield, Ms, exchange, DMI, DMI_coupling, mu0, geom);
                }

                if (anisotropy_coupling == 1) {
                    CalculateH_anisotropy(Mfield, H_anisotropyfield, Ms, anisotropy, anisotropy_coupling, anisotropy_axis, mu0, geom);
                }

                // Compute f^n = f(M^n, H^n) 
                Compute_LLG_RHS(rhs, old_state, H_demagfield, H_biasfield, alpha, Ms, gamma, exchange, anisotropy, demag_coupling,
                                exchange_coupling, anisotropy_coupling, anisotropy_axis, M_normalization, mu0);
            };

            // Create a function to call after updating a state
            auto post_update_fun = [&](Vector<MultiFab>& state, const Real ) {
                // Call user function to update state MultiFab, e.g. fill BCs
                NormalizeM(state, Ms, M_normalization);
                
                for (int comp = 0; comp < 3; comp++) {
                    // fill periodic ghost cells
                    state[comp].FillBoundary(geom.periodicity());
                }

                //post_update(Mfield_old, time, geom);
            };

            // Attach the right hand side and post-update functions
            // to the integrator
            integrator.set_rhs(source_fun);
            integrator.set_post_update(post_update_fun);
                
            // integrate forward one step from `time` by `dt` to fill S_new
            integrator.advance(Mfield_old, Mfield, time, dt);
#endif

        } else {
            amrex::Abort("Time integrator order not recognized");
        }

        // copy new solution into old solution
        for (int comp = 0; comp < 3; comp++) {
            MultiFab::Copy(Mfield_old[comp], Mfield[comp], 0, 0, 1, 1);
            // fill periodic ghost cells
            Mfield_old[comp].FillBoundary(geom.periodicity());
        }

        Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);

        amrex::Print() << "Advanced step " << step << " in " << step_stop_time << " seconds\n";

        // update time
        time = time + dt;

        Print() << "time = " << time << " "
                << "Sum_normalized_M: "
                << SumNormalizedM(Ms,Mfield_old[0])/num_mag << " "
                << SumNormalizedM(Ms,Mfield_old[1])/num_mag << " "
                << SumNormalizedM(Ms,Mfield_old[2])/num_mag << std::endl;

        // Write a plotfile of the data if plot_int > 0
        if (plot_int > 0 && step%plot_int == 0) {
            const std::string& pltfile = amrex::Concatenate("plt",step,8);

            MultiFab::Copy(Plt, Ms, 0, 0, 1, 0);
            MultiFab::Copy(Plt, Mfield[0], 0, 1, 1, 0);
            MultiFab::Copy(Plt, Mfield[1], 0, 2, 1, 0);
            MultiFab::Copy(Plt, Mfield[2], 0, 3, 1, 0);
            MultiFab::Copy(Plt, H_biasfield[0], 0, 4, 1, 0);
            MultiFab::Copy(Plt, H_biasfield[1], 0, 5, 1, 0);
            MultiFab::Copy(Plt, H_biasfield[2], 0, 6, 1, 0);
            MultiFab::Copy(Plt, H_exchangefield[0], 0, 7, 1, 0);
            MultiFab::Copy(Plt, H_exchangefield[1], 0, 8, 1, 0);
            MultiFab::Copy(Plt, H_exchangefield[2], 0, 9, 1, 0);
            MultiFab::Copy(Plt, H_DMIfield[0], 0, 10, 1, 0);
            MultiFab::Copy(Plt, H_DMIfield[1], 0, 11, 1, 0);
            MultiFab::Copy(Plt, H_DMIfield[2], 0, 12, 1, 0);
            MultiFab::Copy(Plt, H_anisotropyfield[0], 0, 13, 1, 0);
            MultiFab::Copy(Plt, H_anisotropyfield[1], 0, 14, 1, 0);
            MultiFab::Copy(Plt, H_anisotropyfield[2], 0, 15, 1, 0);
            MultiFab::Copy(Plt, H_demagfield[0], 0, 16, 1, 0);
            MultiFab::Copy(Plt, H_demagfield[1], 0, 17, 1, 0);
            MultiFab::Copy(Plt, H_demagfield[2], 0, 18, 1, 0);
            MultiFab::Copy(Plt, PoissonRHS, 0, 19, 1, 0);
            MultiFab::Copy(Plt, PoissonPhi, 0, 20, 1, 0);

            WriteSingleLevelPlotfile(pltfile, Plt, {"Ms",
                                                    "Mx",
                                                    "My",
                                                    "Mz",
                                                    "Hx_bias",
                                                    "Hy_bias",
                                                    "Hz_bias",
                                                    "Hx_exchange",
                                                    "Hy_exchange",
                                                    "Hz_exchange",
                                                    "Hx_DMI",
                                                    "Hy_DMI",
                                                    "Hz_DMI",
                                                    "Hx_anisotropy",
                                                    "Hy_anisotropy",
                                                    "Hz_anisotropy",
                                                    "Hx_demagfield","Hy_demagfield","Hz_demagfield",
                                                    "PoissonRHS","PoissonPhi"},
                                                     geom, time, step);

        }

        // Write a checkpoint file if chk_int > 0
        if (chk_int > 0 && step%chk_int == 0) {
            WriteCheckPoint(step,time,Mfield,H_biasfield,H_demagfield);
        }

        // MultiFab memory usage
        const int IOProc = ParallelDescriptor::IOProcessorNumber();

        amrex::Long min_fab_megabytes  = amrex::TotalBytesAllocatedInFabsHWM()/1048576;
        amrex::Long max_fab_megabytes  = min_fab_megabytes;

        ParallelDescriptor::ReduceLongMin(min_fab_megabytes, IOProc);
        ParallelDescriptor::ReduceLongMax(max_fab_megabytes, IOProc);

        amrex::Print() << "High-water FAB megabyte spread across MPI nodes: ["
                       << min_fab_megabytes << " ... " << max_fab_megabytes << "]\n";

        min_fab_megabytes  = amrex::TotalBytesAllocatedInFabs()/1048576;
        max_fab_megabytes  = min_fab_megabytes;

        ParallelDescriptor::ReduceLongMin(min_fab_megabytes, IOProc);
        ParallelDescriptor::ReduceLongMax(max_fab_megabytes, IOProc);

        amrex::Print() << "Curent     FAB megabyte spread across MPI nodes: ["
                       << min_fab_megabytes << " ... " << max_fab_megabytes << "]\n";

    }
    
    Real total_step_stop_time = ParallelDescriptor::second() - total_step_strt_time;
    ParallelDescriptor::ReduceRealMax(total_step_stop_time);

    amrex::Print() << "Total run time " << total_step_stop_time << " seconds\n";
}
