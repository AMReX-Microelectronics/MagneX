
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H> 
#include <AMReX_OpenBC.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MultiFab.H> 
#include <AMReX_VisMF.H>
#include "myfunc.H"
#include "Initialization.H"
#include "MagnetostaticSolver.H"
#include "EffectiveExchangeField.H"
#include "EffectiveDMIField.H"
#include "EffectiveAnisotropyField.H"
#include "CartesianAlgorithm.H"
#include "Diagnostics.H"
#include "EvolveM.H"
#include "EvolveM_2nd.H"
#include "Checkpoint.H"
#ifdef USE_TIME_INTEGRATOR
#include <AMReX_TimeIntegrator.H>
#endif

using namespace amrex;

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

    amrex::GpuArray<int, 3> n_cell; // Number of cells in each dimension

    // size of each box (or grid)
    int max_grid_size;

    // total steps in simulation
    int nsteps;

    // how often to write a plotfile
    int plot_int;

    // ho often to write a checkpoint
    int chk_int;

    // step to restart from
    int restart;

    // 0 = Open Poisson MLMG
    // 1 = FFT-based
    int demag_solver;

    // time step
    Real dt;
    
    amrex::GpuArray<amrex::Real, 3> prob_lo; // physical lo coordinate
    amrex::GpuArray<amrex::Real, 3> prob_hi; // physical hi coordinate

    amrex::GpuArray<amrex::Real, 3> mag_lo; // physical lo coordinate of magnetic region
    amrex::GpuArray<amrex::Real, 3> mag_hi; // physical hi coordinate of magnetic region

    int TimeIntegratorOption;

    // Magnetic Properties
    Real alpha_val, gamma_val, Ms_val, exchange_val, DMI_val, anisotropy_val;
    Real mu0;
    amrex::GpuArray<amrex::Real, 3> anisotropy_axis; 

    int demag_coupling;
    int M_normalization;
    int exchange_coupling;
    int DMI_coupling;
    int anisotropy_coupling;

    // inputs parameters
    {
        // ParmParse is way of reading inputs from the inputs file
        // pp.get means we require the inputs file to have it
        // pp.query means we optionally need the inputs file to have it - but we must supply a default here
        ParmParse pp;

        // We need to get n_cell from the inputs file - this is the number of cells on each side of
        amrex::Vector<int> temp_int(AMREX_SPACEDIM);
        if (pp.queryarr("n_cell",temp_int)) {
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                n_cell[i] = temp_int[i];
            }
        }

        // The domain is broken into boxes of size max_grid_size
        pp.get("max_grid_size",max_grid_size);

        pp.get("TimeIntegratorOption",TimeIntegratorOption);

        // Material Properties
	
        pp.get("mu0",mu0);
        pp.get("alpha_val",alpha_val);
        pp.get("gamma_val",gamma_val);
        pp.get("Ms_val",Ms_val);
        pp.get("exchange_val",exchange_val);
        pp.get("DMI_val",DMI_val);
        pp.get("anisotropy_val",anisotropy_val);

        demag_solver = 0;
	pp.query("demag_solver",demag_solver);
        pp.get("demag_coupling",demag_coupling);
        pp.get("M_normalization", M_normalization);
        pp.get("exchange_coupling", exchange_coupling);
        pp.get("DMI_coupling", DMI_coupling);
        pp.get("anisotropy_coupling", anisotropy_coupling);

        // Default nsteps to 10, allow us to set it to something else in the inputs file
        nsteps = 10;
        pp.query("nsteps",nsteps);

        // Default plot_int to -1, allow us to set it to something else in the inputs file
        //  If plot_int < 0 then no plot files will be written
        plot_int = -1;
        pp.query("plot_int",plot_int);

	    // Default chk_int to -1, allow us to set it to something else in the inputs file
        //  If chk_int < 0 then no chk files will be written
        chk_int = -1;
        pp.query("chk_int",chk_int);

	    // query restart
	    restart = -1;
	    pp.query("restart",restart);
	
        // time step
        pp.get("dt",dt);

        amrex::Vector<amrex::Real> temp(AMREX_SPACEDIM);
        if (pp.queryarr("prob_lo",temp)) {
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                prob_lo[i] = temp[i];
            }
        }
        if (pp.queryarr("prob_hi",temp)) {
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                prob_hi[i] = temp[i];
            }
        }

        if (pp.queryarr("mag_lo",temp)) {
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                mag_lo[i] = temp[i];
            }
        }
        if (pp.queryarr("mag_hi",temp)) {
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                mag_hi[i] = temp[i];
            }
        }
        if (pp.queryarr("anisotropy_axis",temp)) {
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                anisotropy_axis[i] = temp[i];
            }
        }
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

    // Nghost = number of ghost cells for each array
    int Nghost = 1;
    int Ncomp = 1;

    // How Boxes are distrubuted among MPI processes
    if (restart == -1) {
        dm.define(ba);
    }
   
    // Allocate multifabs
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
        //Cell-centered fields
        Mfield_old[dir].define(ba, dm, Ncomp, Nghost);
        Mfield_prev_iter[dir].define(ba, dm, Ncomp, Nghost);
        Mfield_error[dir].define(ba, dm, Ncomp, Nghost);

        H_exchangefield[dir].define(ba, dm, Ncomp, 0);
        H_DMIfield[dir].define(ba, dm, Ncomp, 0);
        H_anisotropyfield[dir].define(ba, dm, Ncomp, 0);

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
            Mfield[dir].define(ba, dm, Ncomp, Nghost);
            H_biasfield[dir].define(ba, dm, Ncomp, Nghost);
            H_demagfield[dir].define(ba, dm, 1, Nghost);
        }
    }

    MultiFab alpha(ba, dm, Ncomp, 0);
    MultiFab gamma(ba, dm, Ncomp, 0);
    MultiFab Ms(ba, dm, Ncomp, Nghost);
    MultiFab exchange(ba, dm, Ncomp, 0);
    MultiFab DMI(ba, dm, Ncomp, 0);
    MultiFab anisotropy(ba, dm, Ncomp, 0);

    amrex::Print() << "==================== Initial Setup ====================\n";
    amrex::Print() << " demag_coupling      = " << demag_coupling      << "\n";
    amrex::Print() << " M_normalization     = " << M_normalization     << "\n";
    amrex::Print() << " exchange_coupling   = " << exchange_coupling   << "\n";
    amrex::Print() << " DMI_coupling        = " << DMI_coupling        << "\n";
    amrex::Print() << " anisotropy_coupling = " << anisotropy_coupling << "\n";
    amrex::Print() << " Ms                  = " << Ms_val              << "\n";
    amrex::Print() << " alpha               = " << alpha_val           << "\n";
    amrex::Print() << " gamma               = " << gamma_val           << "\n";
    amrex::Print() << " exchange_value      = " << exchange_val        << "\n";
    amrex::Print() << " DMI_value           = " << DMI_val             << "\n";
    amrex::Print() << " anisotropy_value    = " << anisotropy_val      << "\n";
    amrex::Print() << "=======================================================\n";

    MultiFab PoissonRHS(ba, dm, 1, 0);
    MultiFab PoissonPhi(ba, dm, 1, 1); // one ghost cell

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

    long npts_large;
    
    // Solver for Poisson equation
    LPInfo info;
#ifdef NEUMANN

    MLABecLaplacian mlabec;
    mlabec.define({geom}, {ba}, {dm}, info);

    mlabec.setEnforceSingularSolvable(false);

    // order of stencil
    int linop_maxorder = 2;
    mlabec.setMaxOrder(linop_maxorder);

    // build array of boundary conditions needed by MLABecLaplacian
    std::array<LinOpBCType, AMREX_SPACEDIM> lo_mlmg_bc;
    std::array<LinOpBCType, AMREX_SPACEDIM> hi_mlmg_bc;

    //Periodic
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if(is_periodic[idim]){
            lo_mlmg_bc[idim] = hi_mlmg_bc[idim] = LinOpBCType::Periodic;
        } else {
            lo_mlmg_bc[idim] = hi_mlmg_bc[idim] = LinOpBCType::Neumann;
        }
    }

    mlabec.setDomainBC(lo_mlmg_bc,hi_mlmg_bc);

    { // add this brace so alpha_cc and beta_face go out of scope afterwards (save memory)
        // coefficients for solver
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
    }

    //Declare MLMG object
    MLMG mlmg(mlabec);
    mlmg.setVerbose(2);
#else
    OpenBCSolver openbc({geom}, {ba}, {dm}, info);
    openbc.setVerbose(2);
#endif
    if (demag_solver == 1) {

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

        npts_large = domain_large.numPts();

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
            Mfield_padded[dir].define(ba_large, dm_large, Ncomp, Nghost);
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
                           n_cell_large, geom_large, npts_large);
        

    }

    InitializeMagneticProperties(alpha, Ms, gamma, exchange, DMI, anisotropy,
                                 alpha_val, Ms_val, gamma_val, exchange_val, DMI_val, anisotropy_val,
                                 prob_lo, prob_hi, mag_lo, mag_hi, geom);

    // initialize to zero; for demag_coupling==0 these aren't used but are still included in plotfile
    PoissonPhi.setVal(0.);
    PoissonRHS.setVal(0.);

    if (restart == -1) {      
        //Initialize fields
        InitializeFields(Mfield, H_biasfield, Ms, prob_lo, prob_hi, geom);

        if(demag_coupling == 1){ 
            
	    if (demag_solver == 0) {

                // Solve Poisson's equation laplacian(Phi) = div(M) and get H_demagfield = -grad(Phi)
                // Compute RHS of Poisson equation
                ComputePoissonRHS(PoissonRHS, Mfield, Ms, geom);

		//Initial guess for phi
                PoissonPhi.setVal(0.);
#ifdef NEUMANN
                // set boundary conditions to homogeneous Neumann
                mlabec.setLevelBC(0, &PoissonPhi);

                mlmg.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
#else
                openbc.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
#endif
	        // Calculate H from Phi
	        ComputeHfromPhi(PoissonPhi, H_demagfield, prob_lo, prob_hi, geom);	    

            } else {

                // copy Mfield used for the RHS calculation in the Poisson option into Mfield_padded
                for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
                    Mfield_padded[dir].setVal(0.);
                    Mfield_padded[dir].ParallelCopy(Mfield[dir], 0, 0, 1);
                }

                ComputeHFieldFFT(Mfield_padded, H_demagfield,
                                 Kxx_dft_real, Kxx_dft_imag, Kxy_dft_real, Kxy_dft_imag, Kxz_dft_real, Kxz_dft_imag,
                                 Kyy_dft_real, Kyy_dft_imag, Kyz_dft_real, Kyz_dft_imag, Kzz_dft_real, Kzz_dft_imag,
                                 n_cell_large, geom_large, npts_large);
	    }
	}
        
        if (exchange_coupling == 1){
            CalculateH_exchange(Mfield, H_exchangefield, Ms, exchange, DMI, exchange_coupling, DMI_coupling, mu0, geom);
        }

        if(DMI_coupling == 1){
            CalculateH_DMI(Mfield, H_DMIfield, Ms, exchange, DMI, exchange_coupling, DMI_coupling, mu0, geom);
        }

        if(anisotropy_coupling == 1){
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

        //Averaging face-centerd Multifabs to cell-centers for plotting 
        //mf_avg_fc_to_cc(Plt, Mfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield, Ms);
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
    for(int comp = 0; comp < 3; comp++) {
        MultiFab::Copy(Mfield_old[comp], Mfield[comp], 0, 0, 1, Nghost);
        MultiFab::Copy(Mfield_prev_iter[comp], Mfield[comp], 0, 0, 1, Nghost);
        MultiFab::Copy(Mfield_error[comp], Mfield[comp], 0, 0, 1, Nghost);

        // fill periodic ghost cells
        Mfield_old[comp].FillBoundary(geom.periodicity());
        Mfield_prev_iter[comp].FillBoundary(geom.periodicity());
        Mfield_error[comp].FillBoundary(geom.periodicity());
    }

#ifdef USE_TIME_INTEGRATOR
    TimeIntegrator<Vector<MultiFab> > integrator(Mfield_old);
#endif 

    for (int step = start_step; step <= nsteps; ++step) {
        
        Real step_strt_time = ParallelDescriptor::second();
        
        if (TimeIntegratorOption == 1){ // first order forward Euler
            
            amrex::Print() << "TimeIntegratorOption = " << TimeIntegratorOption << "\n";

    	    // Evolve H_demag
            if(demag_coupling == 1) {
            
                if (demag_solver == 0) {

                    //Solve Poisson's equation laplacian(Phi) = div(M) and get H_demagfield = -grad(Phi)
                    //Compute RHS of Poisson equation
                    ComputePoissonRHS(PoissonRHS, Mfield_old, Ms, geom);
                
                    //Initial guess for phi
                    PoissonPhi.setVal(0.);
#ifdef NEUMANN
                    mlmg.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
#else
                    openbc.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
#endif

                    // Calculate H from Phi
                    ComputeHfromPhi(PoissonPhi, H_demagfield, prob_lo, prob_hi, geom);
                } else {

                    // copy Mfield used for the RHS calculation in the Poisson option into Mfield_padded
                    for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
                        Mfield_padded[dir].setVal(0.);
                        Mfield_padded[dir].ParallelCopy(Mfield_old[dir], 0, 0, 1);
                    }

                    ComputeHFieldFFT(Mfield_padded, H_demagfield,
                                     Kxx_dft_real, Kxx_dft_imag, Kxy_dft_real, Kxy_dft_imag, Kxz_dft_real, Kxz_dft_imag,
                                     Kyy_dft_real, Kyy_dft_imag, Kyz_dft_real, Kyz_dft_imag, Kzz_dft_real, Kzz_dft_imag,
                                     n_cell_large, geom_large, npts_large);
                }
            }

            if (exchange_coupling == 1){
                CalculateH_exchange(Mfield_old, H_exchangefield, Ms, exchange, DMI, exchange_coupling, DMI_coupling, mu0, geom);
            }

            if(DMI_coupling == 1){
                CalculateH_DMI(Mfield_old, H_DMIfield, Ms, exchange, DMI, exchange_coupling, DMI_coupling, mu0, geom);
            }

            if(anisotropy_coupling == 1){
                CalculateH_anisotropy(Mfield_old, H_anisotropyfield, Ms, anisotropy, anisotropy_coupling, anisotropy_axis, mu0, geom);
            }

            //Evolve M
            // Compute f^n = f(M^n, H^n)
            Compute_LLG_RHS(LLG_RHS, Mfield_old, H_demagfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield, alpha, Ms, gamma, demag_coupling, exchange_coupling, DMI_coupling, anisotropy_coupling, M_normalization, mu0, geom, time);

            // M^{n+1} = M^n + dt * f^n
	        for(int i = 0; i < 3; i++){
                MultiFab::LinComb(Mfield[i], 1.0, Mfield_old[i], 0, dt, LLG_RHS[i], 0, 0, 1, 0);
	        }

            NormalizeM(Mfield, Ms, M_normalization);
 
            for(int comp = 0; comp < 3; comp++)
            {
                // fill periodic ghost cells
                Mfield[comp].FillBoundary(geom.periodicity());
            }
 
            // copy new solution into old solution
            for(int comp = 0; comp < 3; comp++)
            {
                MultiFab::Copy(Mfield_old[comp], Mfield[comp], 0, 0, 1, Nghost);
    
                // fill periodic ghost cells
                Mfield_old[comp].FillBoundary(geom.periodicity());
 
            }
        } else if (TimeIntegratorOption == 2){ //2nd Order predictor-corrector
        
            amrex::Print() << "TimeIntegratorOption = " << TimeIntegratorOption << "\n";

            Real M_tolerance = 1.e-6;
            int iter = 0;
            int stop_iter = 0;

	        // Evolve H_demag (H^{n})
	        if(demag_coupling == 1){
            
                    if (demag_solver == 0) {
                        //Solve Poisson's equation laplacian(Phi) = div(M) and get H_demagfield = -grad(Phi)
                        //Compute RHS of Poisson equation
                        ComputePoissonRHS(PoissonRHS, Mfield_prev_iter, Ms, geom);
                
                        //Initial guess for phi
                        PoissonPhi.setVal(0.);
#ifdef NEUMANN
                        mlmg.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
#else
                        openbc.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
#endif

                        // Calculate H from Phi
                        ComputeHfromPhi(PoissonPhi, H_demagfield, prob_lo, prob_hi, geom);
                    } else {

                        // copy Mfield used for the RHS calculation in the Poisson option into Mfield_padded
                        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
                            Mfield_padded[dir].setVal(0.);
                            Mfield_padded[dir].ParallelCopy(Mfield_prev_iter[dir], 0, 0, 1);
                        }

                        ComputeHFieldFFT(Mfield_padded, H_demagfield,
                                         Kxx_dft_real, Kxx_dft_imag, Kxy_dft_real, Kxy_dft_imag, Kxz_dft_real, Kxz_dft_imag,
                                         Kyy_dft_real, Kyy_dft_imag, Kyz_dft_real, Kyz_dft_imag, Kzz_dft_real, Kzz_dft_imag,
                                         n_cell_large, geom_large, npts_large);

                    }
	        }

            if (exchange_coupling == 1){
                CalculateH_exchange(Mfield_old, H_exchangefield, Ms, exchange, DMI, exchange_coupling, DMI_coupling, mu0, geom);
            }
    
            if(DMI_coupling == 1){
                CalculateH_DMI(Mfield_old, H_DMIfield, Ms, exchange, DMI, exchange_coupling, DMI_coupling, mu0, geom);
            }
    
            if(anisotropy_coupling == 1){
                CalculateH_anisotropy(Mfield_old, H_anisotropyfield, Ms, anisotropy, anisotropy_coupling, anisotropy_axis, mu0, geom);
            }
    
	        // Compute f^{n} = f(M^{n}, H^{n})
            Compute_LLG_RHS(LLG_RHS, Mfield_old, H_demagfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield, alpha, Ms, gamma, demag_coupling, exchange_coupling, DMI_coupling, anisotropy_coupling, M_normalization, mu0, geom, time);

            while(!stop_iter){
                
                // Poisson solve and H_demag computation with M_field_pre
                if(demag_coupling == 1) { 
            
                    if (demag_solver == 0) {
                        //Solve Poisson's equation laplacian(Phi) = div(M) and get H_demagfield = -grad(Phi)
                        //Compute RHS of Poisson equation
                        ComputePoissonRHS(PoissonRHS, Mfield_prev_iter, Ms, geom);
    
                        //Initial guess for phi
                        PoissonPhi.setVal(0.);
#ifdef NEUMANN
                        mlmg.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
#else
                        openbc.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
#endif

                        // Calculate H from Phi
                        ComputeHfromPhi(PoissonPhi, H_demagfield, prob_lo, prob_hi, geom);

                    } else {

                        // copy Mfield used for the RHS calculation in the Poisson option into Mfield_padded
                        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
                            Mfield_padded[dir].setVal(0.);
                            Mfield_padded[dir].ParallelCopy(Mfield_prev_iter[dir], 0, 0, 1);
                        }

                        ComputeHFieldFFT(Mfield_padded, H_demagfield,
                                         Kxx_dft_real, Kxx_dft_imag, Kxy_dft_real, Kxy_dft_imag, Kxz_dft_real, Kxz_dft_imag,
                                         Kyy_dft_real, Kyy_dft_imag, Kyz_dft_real, Kyz_dft_imag, Kzz_dft_real, Kzz_dft_imag,
                                         n_cell_large, geom_large, npts_large);
                    }
                }
    
                if (exchange_coupling == 1){
                    CalculateH_exchange(Mfield_prev_iter, H_exchangefield, Ms, exchange, DMI, exchange_coupling, DMI_coupling, mu0, geom);
                }
        
                if(DMI_coupling == 1){
                    CalculateH_DMI(Mfield_prev_iter, H_DMIfield, Ms, exchange, DMI, exchange_coupling, DMI_coupling, mu0, geom);
                }
    
                if(anisotropy_coupling == 1){
                    CalculateH_anisotropy(Mfield_prev_iter, H_anisotropyfield, Ms, anisotropy, anisotropy_coupling, anisotropy_axis, mu0, geom);
                }
    
	            // LLG RHS with new H_demag and M_field_pre
	            // Compute f^{n+1, *} = f(M^{n+1, *}, H^{n+1, *})
                Compute_LLG_RHS(LLG_RHS_pre, Mfield_prev_iter, H_demagfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield, alpha, Ms, gamma, demag_coupling, exchange_coupling, DMI_coupling, anisotropy_coupling, M_normalization, mu0, geom, time);
    
	            // Corrector step update M
	            // M^{n+1, *} = M^n + 0.5 * dt * (f^n + f^{n+1, *})
	            for(int i = 0; i < 3; i++){
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
    
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
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
	            for(int comp = 0; comp < 3; comp++) {
                    MultiFab::Copy(Mfield_prev_iter[comp], Mfield[comp], 0, 0, 1, Nghost);
                    // fill periodic ghost cells 
                    Mfield_prev_iter[comp].FillBoundary(geom.periodicity());
	            }
    
	            iter = iter + 1;
    
	            amrex::Print() << "iter = " << iter << ", M_mag_error_max = " << M_mag_error_max << "\n";
	            if (M_mag_error_max <= M_tolerance) stop_iter = 1;

	        } // while stop_iter

            // copy new solution into old solution
            for (int comp = 0; comp < 3; comp++){
                MultiFab::Copy(Mfield_old[comp], Mfield[comp], 0, 0, 1, Nghost);
                // fill periodic ghost cells
                Mfield_old[comp].FillBoundary(geom.periodicity());
            }
    
        } else if (TimeIntegratorOption == 3) { // artemis way
        
            amrex::Print() << "TimeIntegratorOption = " << TimeIntegratorOption << "\n";

            EvolveM_2nd(Mfield, H_demagfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield, PoissonRHS, PoissonPhi, alpha, Ms, gamma, exchange, DMI, anisotropy, demag_coupling, exchange_coupling, DMI_coupling, anisotropy_coupling, anisotropy_axis, M_normalization, mu0, geom, prob_lo, prob_hi, dt, time);

        }  else if (TimeIntegratorOption == 4) { // amrex and sundials integrators

#ifdef USE_TIME_INTEGRATOR
            amrex::Print() << "TimeIntegratorOption = SUNDIALS" << "\n";

	        // Create a RHS source function we will integrate
            auto source_fun = [&](Vector<MultiFab>& rhs, const Vector<MultiFab>& old_state, const Real ){

                // User function to calculate the rhs MultiFab given the state MultiFab
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                    rhs[idim].setVal(0.);
                } 
 
    	        // Evolve H_demag
                if(demag_coupling == 1) {
            
                    if (demag_solver == 0) {

                        //Solve Poisson's equation laplacian(Phi) = div(M) and get H_demagfield = -grad(Phi)
                        //Compute RHS of Poisson equation
                        ComputePoissonRHS(PoissonRHS, old_state, Ms, geom);
                     
                        //Initial guess for phi
                        PoissonPhi.setVal(0.);
#ifdef NEUMANN
                        mlmg.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
#else
                        openbc.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
#endif
                        // Calculate H from Phi
                        ComputeHfromPhi(PoissonPhi, H_demagfield, prob_lo, prob_hi, geom);
                    } else {

                        // copy Mfield used for the RHS calculation in the Poisson option into Mfield_padded
                        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
                            Mfield_padded[dir].setVal(0.);
                            Mfield_padded[dir].ParallelCopy(old_state[dir], 0, 0, 1);
                        }

                        ComputeHFieldFFT(Mfield_padded, H_demagfield,
                                         Kxx_dft_real, Kxx_dft_imag, Kxy_dft_real, Kxy_dft_imag, Kxz_dft_real, Kxz_dft_imag,
                                         Kyy_dft_real, Kyy_dft_imag, Kyz_dft_real, Kyz_dft_imag, Kzz_dft_real, Kzz_dft_imag,
                                         n_cell_large, geom_large, npts_large);
                    }
                }

                if (exchange_coupling == 1){
                    CalculateH_exchange(Mfield, H_exchangefield, Ms, exchange, DMI, exchange_coupling, DMI_coupling, mu0, geom);
                }

                if(DMI_coupling == 1){
                    CalculateH_DMI(Mfield, H_DMIfield, Ms, exchange, DMI, exchange_coupling, DMI_coupling, mu0, geom);
                }

                if(anisotropy_coupling == 1){
                    CalculateH_anisotropy(Mfield, H_anisotropyfield, Ms, anisotropy, anisotropy_coupling, anisotropy_axis, mu0, geom);
                }

                // Compute f^n = f(M^n, H^n) 
                Compute_LLG_RHS(rhs, old_state, H_demagfield, H_biasfield, alpha, Ms, gamma, exchange, anisotropy, demag_coupling, exchange_coupling, anisotropy_coupling, anisotropy_axis, M_normalization, mu0, geom, time);
            };

	        // Create a function to call after updating a state
            auto post_update_fun = [&](Vector<MultiFab>& state, const Real ) {
                // Call user function to update state MultiFab, e.g. fill BCs
                NormalizeM(state, Ms, M_normalization);

                for(int comp = 0; comp < 3; comp++){
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
        for(int comp = 0; comp < 3; comp++) {
            MultiFab::Copy(Mfield_old[comp], Mfield[comp], 0, 0, 1, Nghost);

            // fill periodic ghost cells
            Mfield_old[comp].FillBoundary(geom.periodicity());
        }

	    Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);

        amrex::Print() << "Advanced step " << step << " in " << step_stop_time << " seconds\n";

        // update time
        time = time + dt;

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
