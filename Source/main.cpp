
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H> 
#include <AMReX_MultiFab.H> 
#include <AMReX_VisMF.H>
#include "myfunc.H"
#include "MicroMag.H"

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

    // time step
    Real dt;
    
    amrex::GpuArray<amrex::Real, 3> prob_lo; // physical lo coordinate
    amrex::GpuArray<amrex::Real, 3> prob_hi; // physical hi coordinate

    amrex::GpuArray<amrex::Real, 3> mag_lo; // physical lo coordinate of magnetic region
    amrex::GpuArray<amrex::Real, 3> mag_hi; // physical hi coordinate of magnetic region

    Real Phi_Bc_hi;
    Real Phi_Bc_lo;

    int TimeIntegratorOrder;

    // Magnetic Properties
    Real alpha_val, gamma_val, Ms_val, exchange_val, anisotropy_val;

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

        pp.get("Phi_Bc_hi",Phi_Bc_hi);
        pp.get("Phi_Bc_lo",Phi_Bc_lo);

	pp.get("TimeIntegratorOrder",TimeIntegratorOrder);

        // Material Properties
	
        pp.get("alpha_val",alpha_val);
        pp.get("gamma_val",gamma_val);
        pp.get("Ms_val",Ms_val);
        pp.get("exchange_val",exchange_val);
        pp.get("anisotropy_val",anisotropy_val);

        // Default nsteps to 10, allow us to set it to something else in the inputs file
        nsteps = 10;
        pp.query("nsteps",nsteps);

        // Default plot_int to -1, allow us to set it to something else in the inputs file
        //  If plot_int < 0 then no plot files will be written
        plot_int = -1;
        pp.query("plot_int",plot_int);

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
    }


    // **********************************
    // SIMULATION SETUP

    // make BoxArray and Geometry
    // ba will contain a list of boxes that cover the domain
    // geom contains information such as the physical domain size,
    //               number of points in the domain, and periodicity
    BoxArray ba;
    Geometry geom;

    // AMREX_D_DECL means "do the first X of these, where X is the dimensionality of the simulation"
    IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
    IntVect dom_hi(AMREX_D_DECL(n_cell[0]-1, n_cell[1]-1, n_cell[2]-1));

    // Make a single box that is the entire domain
    Box domain(dom_lo, dom_hi);

    // Initialize the boxarray "ba" from the single box "domain"
    ba.define(domain);

    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba.maxSize(max_grid_size);

    // This defines the physical box in each direction.
    RealBox real_box({AMREX_D_DECL( prob_lo[0], prob_lo[1], prob_lo[2])},
                     {AMREX_D_DECL( prob_hi[0], prob_hi[1], prob_hi[2])});

    // periodic in x and y directions
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,0)};

    // This defines a Geometry object
    geom.define(domain, real_box, CoordSys::cartesian, is_periodic);

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // Nghost = number of ghost cells for each array
    int Nghost = 1;

    // Ncomp = number of components for each array
    int Ncomp = 1;

    // How Boxes are distrubuted among MPI processes
    DistributionMapping dm(ba);

    // Allocate multifabs

    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Mfield;
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Mfield_old;
    std::array<std::unique_ptr<amrex::MultiFab>, 3> &Hfield; // H Demag
    std::array<std::unique_ptr<amrex::MultiFab>, 3> const &H_biasfield; // H bias

    MultiFab alpha(ba, dm, Ncomp, Nghost);
    MultiFab gamma(ba, dm, Ncomp, Nghost);
    MultiFab Ms(ba, dm, Ncomp, Nghost);
    MultiFab exchange(ba, dm, Ncomp, Nghost);
    MultiFab anisotropy(ba, dm, Ncomp, Nghost);

    int demag_coupling = 0;
    int M_normalization = 1;
    int exchange_coupling = 0;
    int anisotropy_coupling = 1;

    MultiFab PoissonRHS(ba, dm, 1, 0);
    MultiFab PoissonPhi(ba, dm, 1, 1);


    MultiFab Plt(ba, dm, 5, 0);

    //Solver for Poisson equation
    LPInfo info;
    MLABecLaplacian mlabec({geom}, {ba}, {dm}, info);

    //Force singular system to be solvable
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
          lo_mlmg_bc[idim] = hi_mlmg_bc[idim] = LinOpBCType::Dirichlet;
        }
    } 

    mlabec.setDomainBC(lo_mlmg_bc,hi_mlmg_bc);

    // coefficients for solver
    MultiFab alpha_cc(ba, dm, 1, 0);
    std::array< MultiFab, AMREX_SPACEDIM > beta_face;
    AMREX_D_TERM(beta_face[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 1, 0);,
                 beta_face[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 1, 0);,
                 beta_face[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 1, 0););
    
    // set cell-centered alpha coefficient to zero
    alpha_cc.setVal(0.);
    beta_face[0].setVal(1.);
    beta_face[1].setVal(1.);
    beta_face[2].setVal(1.);

    // Set Dirichlet BC for Phi in z
    SetPhiBC_z(PoissonPhi, n_cell, Phi_Bc_lo, Phi_Bc_hi); 
    
    // set Dirichlet BC by reading in the ghost cell values
    mlabec.setLevelBC(0, &PoissonPhi);
    
    // (A*alpha_cc - B * div beta grad) phi = rhs
    mlabec.setScalars(0.0, 1.0); // A = 0.0, B = 1.0
    mlabec.setACoeffs(0, alpha_cc); //First argument 0 is lev
    mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(beta_face));  

    //Declare MLMG object
    MLMG mlmg(mlabec);
    mlmg.setVerbose(2);

    // time = starting time in the simulation
    Real time = 0.0;	
   
    //Next steps (06/16/2022: Initialze M, slve Poisson's equation for Phi, Compute H from Phi, H_exchane and H_anisotropy from M)
    //Modify the commented out block below to output initial states

    InitializeMagneticProperties(alpha, Ms, gamma, exchange, anisotropy,
                                 alpha_val, Ms_val, gamma_val, exchange_val, anisotropy_val, 
                                 prob_lo, prob_hi, mag_lo, mag_hi, geom);

    // Write a plotfile of the initial data if plot_int > 0
    if (plot_int > 0)
    {
        int step = 0;
        const std::string& pltfile = amrex::Concatenate("plt",step,8);
        MultiFab::Copy(Plt, alpha, 0, 0, 1, 0);  
        MultiFab::Copy(Plt, Ms, 0, 1, 1, 0);
        MultiFab::Copy(Plt, gamma, 0, 2, 1, 0);
        MultiFab::Copy(Plt, exchange, 0, 3, 1, 0);
        MultiFab::Copy(Plt, anisotropy, 0, 4, 1, 0);
        WriteSingleLevelPlotfile(pltfile, Plt, {"alpha","Ms","gamma","exchange","anisotropy"}, geom, time, 0);
    }

    for (int step = 1; step <= nsteps; ++step)
    {
        Real step_strt_time = ParallelDescriptor::second();

    	    // Evolve M


        for (MFIter mfi(*Mfield[0]); mfi.isValid(); ++mfi)
        {
        
        // extract field data
              Array4<Real> const &Hx = Hfield[0]->array(mfi);
              Array4<Real> const &Hy = Hfield[1]->array(mfi);
              Array4<Real> const &Hz = Hfield[2]->array(mfi);
              Array4<Real> const &Mx = Mfield[0]->array(mfi);         
              Array4<Real> const &My = Mfield[1]->array(mfi);         
              Array4<Real> const &Mz = Mfield[2]->array(mfi);         
              Array4<Real> const &Mx_old = Mfield_old[0]->array(mfi); 
              Array4<Real> const &My_old = Mfield_old[1]->array(mfi); 
              Array4<Real> const &Mz_old = Mfield_old[2]->array(mfi); 
              Array4<Real> const &Hx_bias = H_biasfield[0]->array(mfi);
              Array4<Real> const &Hy_bias = H_biasfield[1]->array(mfi);
              Array4<Real> const &Hz_bias = H_biasfield[2]->array(mfi);
          
              const Array4<Real>& alpha_arr = alpha.array(mfi);
              const Array4<Real>& gamma_arr = gamma.array(mfi);
              const Array4<Real>& Ms_arr = Ms.array(mfi);
              const Array4<Real>& exchange_arr = exchange.array(mfi);
              const Array4<Real>& anisotropy_arr = anisotropy.array(mfi);
 
//              amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
//              {
//                 if (Ms_arr(i,j,k) > 0._rt)
//                 {
//                    amrex::real Hx_eff = Hx_bias(i,j,k);
//                    amrex::real Hy_eff = Hy_bias(i,j,k);
//                    amrex::real Hz_eff = Hz_bias(i,j,k);
//                 
//                    if(demag_coupling == 1)
//                    {
//                    }
//                 }
//              }

        }  
	
//        // copy new solution into old solution
//        MultiFab::Copy(P_old, P_new, 0, 0, 1, 0);
//
//        // fill periodic ghost cells
//        P_old.FillBoundary(geom.periodicity());
//
//	//Compute RHS of Poisson equation
//	ComputePoissonRHS(PoissonRHS, P_old, charge_den, 
//			FE_lo, FE_hi, DE_lo, DE_hi, SC_lo, SC_hi, 
//			P_BC_flag_lo, P_BC_flag_hi, lambda, 
//			prob_lo, prob_hi, 
//			geom);
//
//        //Initial guess for phi
//        PoissonPhi.setVal(0.);
//        mlmg.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
//
//        // Calculate H from Phi
//
//	ComputeEfromPhi(PoissonPhi, Ex, Ey, Ez, prob_lo, prob_hi, geom);
//
	Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);

        amrex::Print() << "Advanced step " << step << " in " << step_stop_time << " seconds\n";

        // update time
        time = time + dt;

//        // Write a plotfile of the current data (plot_int was defined in the inputs file)
//        if (plot_int > 0 && step%plot_int == 0)
//        {
//            const std::string& pltfile = amrex::Concatenate("plt",step,8);
//            MultiFab::Copy(Plt, P_old, 0, 0, 1, 0);  
//            MultiFab::Copy(Plt, PoissonPhi, 0, 1, 1, 0);
//            MultiFab::Copy(Plt, PoissonRHS, 0, 2, 1, 0);
//            MultiFab::Copy(Plt, Ex, 0, 3, 1, 0);
//            MultiFab::Copy(Plt, Ey, 0, 4, 1, 0);
//            MultiFab::Copy(Plt, Ez, 0, 5, 1, 0);
//            MultiFab::Copy(Plt, hole_den, 0, 6, 1, 0);
//            MultiFab::Copy(Plt, e_den, 0, 7, 1, 0);
//            MultiFab::Copy(Plt, charge_den, 0, 8, 1, 0);
//            WriteSingleLevelPlotfile(pltfile, Plt, {"P","Phi","PoissonRHS","Ex","Ey","Ez","holes","electrons","charge"}, geom, time, step);
//        }
//
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
