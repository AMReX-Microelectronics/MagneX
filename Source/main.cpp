#include "MagneX.H"
#include "Demagnetization.H"
#include <AMReX_MultiFab.H> 
#include <AMReX_VisMF.H>
#include <AMReX_ParmParse.H>
#ifdef AMREX_USE_SUNDIALS
#include <AMReX_TimeIntegrator.H>
#endif
<<<<<<< HEAD
#include <torch/script.h>
=======

#include <cmath>

>>>>>>> development
using namespace amrex;
using namespace MagneX;

void main_main();

int main (int argc, char* argv[])
{
    // timer for profiling
    BL_PROFILE_VAR("main()",main);

    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
    return 0;
}

void main_main ()
{
  
    // timer for profiling
    BL_PROFILE_VAR("main_main()",main_main);

    Real total_step_strt_time = ParallelDescriptor::second();
  
    std::ofstream outputFile("Diagnostics.txt", std::ofstream::trunc);

    // **********************************
    // READ SIMULATION PARAMETERS
    // **********************************
    InitializeMagneXNamespace();

    int start_step = 1;

    // for std4 diagnostic
    Real normalized_Mx_prev = 0.;
    Real dot_product_prev = 0.;
    
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
    Array<MultiFab, AMREX_SPACEDIM> Heff;

    Array<MultiFab, AMREX_SPACEDIM> LLG_RHS;
    Array<MultiFab, AMREX_SPACEDIM> LLG_RHS_pre;
    Array<MultiFab, AMREX_SPACEDIM> LLG_RHS_avg;
    torch::jit::script::Module ml_module;
    torch::jit::script::Module x_norm_module;
    torch::jit::script::Module y_norm_module;

    // Declare variables for hysteresis
    Real normalized_Mx;
    Real normalized_My;
    Real normalized_Mz;

    Real normalized_Mx_old = 0.;
    Real normalized_My_old = 0.;
    Real normalized_Mz_old = 0.;

    // Initialize err to some arbitrary value greater then tol
    Real err = equilibrium_tolerance + 1.0;
    Real err_x = equilibrium_tolerance + 1.0;
    Real err_y = equilibrium_tolerance + 1.0;
    Real err_z = equilibrium_tolerance + 1.0;

    int increment_Hbias = 0;

    Real Hbias_magn;
    Real M;
    Real M_old = 0.;
    
    // Changes to +1 when we want to reverse Hbias trend
    int sign = -1;

    // Count how many times we have incremented Hbias
    int increment_count = 0;

    Real total_energy;
    Real demag_energy;
    Real exchange_energy;
    Real anis_energy;

    int steady = 0;

    BoxArray ba;
    DistributionMapping dm;
    
    if (restart > 0) {

        start_step = restart+1;

        // read in Mfield, H_biasfield, and ba
        // create a DistributionMapping dm
        ReadCheckPoint(restart,time,Mfield,H_biasfield,H_demagfield,ba,dm);
      
    }
    
    // **********************************
    // // LOAD PYTORCH MODEL

    BL_PROFILE_VAR("LoadPytorch",LoadPytorch);

    // Load pytorch module via torch script


    std::string ml_model_name;
    std::string x_normalizer_name;
    std::string y_normalizer_name;

    ParmParse pp_ml;
    pp_ml.query("ml_model_name", ml_model_name);
    pp_ml.query("x_normalizer_name", x_normalizer_name);
    pp_ml.query("y_normalizer_name", y_normalizer_name);

    amrex::Print()<<"\n"<<ml_model_name<<"\n";
    amrex::Print()<<x_normalizer_name<<"\n";
    amrex::Print()<<y_normalizer_name<<"\n";


    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        ml_module = torch::jit::load(ml_model_name, torch::kCUDA);
        x_norm_module = torch::jit::load(x_normalizer_name, torch::kCUDA);
        y_norm_module = torch::jit::load(y_normalizer_name, torch::kCUDA);
    }
    catch (const c10::Error& e) {
        amrex::Abort("Error loading the model\n");
    }

    Print() << "Model loaded.\n";





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
        // create IntVect of max_grid_size
        IntVect max_grid_size(AMREX_D_DECL(max_grid_size_x,max_grid_size_y,max_grid_size_z));
        ba.maxSize(max_grid_size);

        // How Boxes are distrubuted among MPI processes
        dm.define(ba);
    }

    // This defines the physical box in each direction.
    RealBox real_box({AMREX_D_DECL( prob_lo[0], prob_lo[1], prob_lo[2])}, 
                     {AMREX_D_DECL( prob_hi[0], prob_hi[1], prob_hi[2])});

    // periodic in x and y directions
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)}; // nonperiodic in all directions

    // This defines a Geometry object
    Geometry geom;
    geom.define(domain, real_box, CoordSys::cartesian, is_periodic);

    // Allocate multifabs
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
        //Cell-centered fields
        Mfield_old[dir].define(ba, dm, 1, 1);
        Mfield_prev_iter[dir].define(ba, dm, 1, 1);
        Mfield_error[dir].define(ba, dm, 1, 0);

        H_exchangefield[dir].define(ba, dm, 1, 0);
        H_DMIfield[dir].define(ba, dm, 1, 0);
        H_anisotropyfield[dir].define(ba, dm, 1, 0);
         
	Heff[dir].define(ba, dm, 1, 1);

        // set to zero in case we don't include
        H_exchangefield[dir].setVal(0.);
        H_DMIfield[dir].setVal(0.);
        H_anisotropyfield[dir].setVal(0.);
        Heff[dir].setVal(0.);

        LLG_RHS[dir].define(ba, dm, 1, 0);
        LLG_RHS_pre[dir].define(ba, dm, 1, 0);
        LLG_RHS_avg[dir].define(ba, dm, 1, 0);
    }

    if (restart == -1) {
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
            //Cell-centered fields
            Mfield[dir].define(ba, dm, 1, 1);
            H_biasfield[dir].define(ba, dm, 1, 0);
            H_demagfield[dir].define(ba, dm, 1, 0);
        }
    }

    // one ghost cell
    MultiFab Ms(ba, dm, 1, 1);

    // no ghost cells
    MultiFab alpha(ba, dm, 1, 0);
    MultiFab gamma(ba, dm, 1, 0);
    MultiFab exchange(ba, dm, 1, 0);
    MultiFab DMI(ba, dm, 1, 0);
    MultiFab anisotropy(ba, dm, 1, 0);
 
    amrex::Print() << "==================== Initial Setup ====================\n";
    amrex::Print() << " ba           = " << ba          << "\n";
    amrex::Print() << " dm           = " << dm          << "\n";
    amrex::Print() << " precession           = " << precession          << "\n";
    amrex::Print() << " demag_coupling       = " << demag_coupling      << "\n";
    if (demag_coupling == 1) amrex::Print() << " FFT_solver           = " << FFT_solver << "\n";
    amrex::Print() << " M_normalization      = " << M_normalization     << "\n";
    amrex::Print() << " exchange_coupling    = " << exchange_coupling   << "\n";
    amrex::Print() << " DMI_coupling         = " << DMI_coupling        << "\n";
    amrex::Print() << " anisotropy_coupling  = " << anisotropy_coupling << "\n";
    amrex::Print() << " TimeIntegratorOption = " << TimeIntegratorOption << "\n";
    amrex::Print() << "=======================================================\n";

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        LLG_RHS[idim].setVal(0.);
        LLG_RHS_pre[idim].setVal(0.);
        LLG_RHS_avg[idim].setVal(0.);
        H_demagfield[idim].setVal(0.);
    }

    // Create a zero-padded Magnetization field for the convolution method
    Demagnetization demag_solver;
    
    if (demag_coupling == 1) {
        demag_solver.define();       
    }

    InitializeMagneticProperties(alpha, Ms, gamma, exchange, DMI, anisotropy, geom, time);
    ComputeHbias(H_biasfield, time, geom);

    // Extract anisotropy and exchange constants
    int comp=0;
    Real ani = anisotropy.max(comp);
    comp=0;
    Real exch = exchange.max(comp);

    // Magnetostatic energy density in SI
    Real Km =.5*mu0*(std::pow(8.e5,2));

    // count how many magnetic cells are in the domain
    long num_mag = CountMagneticCells(Ms);
    
    if (restart == -1) {      
        //Initialize fields
        InitializeFields(Mfield, geom);

        if (demag_coupling == 1) {

            demag_solver.CalculateH_demag(Mfield, H_demagfield);
            // CalculateH_demag_ML(Mfield, x_norm_module, ml_module, y_norm_module, tensoropt, H_demagfield);
        
        }       
        
        if (exchange_coupling == 1) {
            CalculateH_exchange(Mfield, H_exchangefield, Ms, exchange, DMI, geom);
        }

        if (DMI_coupling == 1) {
            CalculateH_DMI(Mfield, H_DMIfield, Ms, exchange, DMI, geom);
        }

        if (anisotropy_coupling == 1) {
            CalculateH_anisotropy(Mfield, H_anisotropyfield, Ms, anisotropy);
        }
    }

    // Write a plotfile of the initial data if plot_int > 0
    if (plot_int > 0)
    {
        int plt_step = 0;
        if (restart > 0) {
            plt_step = restart;
        }

        WritePlotfile(Ms, Mfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield,
                      H_demagfield, geom, time, plt_step);
    }

    // copy new solution into old solution
    for (int comp = 0; comp < 3; comp++) {
        MultiFab::Copy(Mfield_old[comp], Mfield[comp], 0, 0, 1, 1);
    }

#ifdef AMREX_USE_SUNDIALS

    std::string theStrategy;
    amrex::ParmParse pp("integration.sundials");
    pp.get("strategy", theStrategy);
    int using_MRI = theStrategy == "MRI" ? 1 : 0;
    
    //alias Mfield and Mfield_old from Array<MultiFab, AMREX_SPACEDIM> into a vector of MultiFabs amrex::Vector<MultiFab>
    //This is needed for sundials inetgrator ==> integrator.advance(vMfield_old, vMfield, time, dt)
    amrex::Vector<MultiFab> vMfield_old(AMREX_SPACEDIM);
    amrex::Vector<MultiFab> vMfield(AMREX_SPACEDIM);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        vMfield_old[idim] = MultiFab(Mfield_old[idim],amrex::make_alias,0,Mfield_old[idim].nComp());
        vMfield[idim] = MultiFab(Mfield[idim],amrex::make_alias,0,Mfield_old[idim].nComp());
    }
    TimeIntegrator<Vector<MultiFab> > integrator(vMfield_old);
#endif 

    for (int step = start_step; step <= nsteps; ++step) {
        
        Real step_strt_time = ParallelDescriptor::second();

        if (timedependent_Hbias) {
            ComputeHbias(H_biasfield, time, geom);
        }

        // Check to increment Hbias for hysteresis
	if ((Hbias_sweep == 1) && (increment_Hbias == 1)) {
           
	   if (increment_count == nsteps_hysteresis) {
	       sign *= -1;
               outputFile << "time = " << time << " "
                    << "Reverse_Hbias_evolution "
                    << normalized_Mx/num_mag << " "
                    << normalized_My/num_mag << " "
                    << normalized_Mz/num_mag << std::endl;
	   }

           increment_count += 1;

	   for (MFIter mfi(H_biasfield[0]); mfi.isValid(); ++mfi)
           {
               const Box& bx = mfi.tilebox();

	       // extract dx from the geometry object
	       GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

	       // extract field data
	       const Array4<Real>& Hx_bias = H_biasfield[0].array(mfi);
	       const Array4<Real>& Hy_bias = H_biasfield[1].array(mfi);
	       const Array4<Real>& Hz_bias = H_biasfield[2].array(mfi);

	       amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
	       {
	           Hx_bias(i,j,k) += sign*increment_size;
	           Hy_bias(i,j,k) += sign*increment_size;
	           Hz_bias(i,j,k) += sign*increment_size;
		});
	    }

            normalized_Mx = SumNormalizedM(Ms,Mfield[0]);
            normalized_My = SumNormalizedM(Ms,Mfield[1]);
            normalized_Mz = SumNormalizedM(Ms,Mfield[2]);
	    
	    outputFile << "time = " << time << " "
                    << "Hbias_increment: "
                    << normalized_Mx/num_mag << " "
                    << normalized_My/num_mag << " "
                    << normalized_Mz/num_mag << std::endl;

	    increment_Hbias = 0;
	}

        if (timedependent_alpha) {
            ComputeAlpha(alpha,geom,time);
        }

        // compute old-time LLG_RHS
        if (TimeIntegratorOption == 1 ||
            TimeIntegratorOption == 2 ||
            TimeIntegratorOption == 3) {
            
    	    // Evolve H_demag
            if (demag_coupling == 1) {
                std::string fname_before = "M_old_before" + std::to_string(step) + "before";
                std::string fname_after = "M_old_after" + std::to_string(step) + "after";

                if (step < 2) {
                    outputMFAscii(Mfield_old, fname_before);
                    demag_solver.CalculateH_demag(Mfield_old, H_demagfield);
                    outputMFAscii(Mfield_old, fname_after);
                } else {
                    amrex::Print()<<"\n"<<ml_model_name<<"\n";
                    // CalculateH_demag_ML(Mfield_old, x_norm_module, ml_module, y_norm_module, H_demagfield);
                    demag_solver.CalculateH_demag(Mfield_old, H_demagfield);
                }
                if ( (plot_int > 0 && step%plot_int == 0 && step >= 200000)) {
                    WritePlotfile(Ms, Mfield_old, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield,
                          H_demagfield, geom, time, step);
                }
                // amrex::Print()<<"\n"<<ml_model_name<<"\n";
                // // demag_solver.CalculateH_demag(Mfield_old, H_demagfield);
                // CalculateH_demag_ML(Mfield_old, x_norm_module, ml_module, y_norm_module, tensoropt, H_demagfield);
            }

            if (exchange_coupling == 1) {
                CalculateH_exchange(Mfield_old, H_exchangefield, Ms, exchange, DMI, geom);
            }

            if (DMI_coupling == 1) {
                CalculateH_DMI(Mfield_old, H_DMIfield, Ms, exchange, DMI, geom);
            }

            if (anisotropy_coupling == 1) {
                CalculateH_anisotropy(Mfield_old, H_anisotropyfield, Ms, anisotropy);
            }
        }

        if (TimeIntegratorOption == 1) { // first order forward Euler

            // Evolve M
            // Compute f^n = f(M^n, H^n)
            Compute_LLG_RHS(LLG_RHS, Mfield_old, H_demagfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield, alpha,
                            Ms, gamma);

            // M^{n+1} = M^n + dt * f^n
            for (int i = 0; i < 3; i++) {
                MultiFab::LinComb(Mfield[i], 1.0, Mfield_old[i], 0, dt, LLG_RHS[i], 0, 0, 1, 0);
            }

            // Normalize M and fill ghost cells
            NormalizeM(Mfield, Ms, geom);
            
        } else if (TimeIntegratorOption == 2) { // iterative predictor-corrector
    
            // Compute f^{n} = f(M^{n}, H^{n})
            Compute_LLG_RHS(LLG_RHS, Mfield_old, H_demagfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield, alpha,
                            Ms, gamma);

            for (int comp = 0; comp < 3; comp++) {
                // copy old RHS into predicted RHS so first pass through is forward Euler
                MultiFab::Copy(LLG_RHS_pre[comp], LLG_RHS[comp], 0, 0, 1, 0);
                // copy Mfield old into Mfield_prev_iter so we can track the change in the predictor
                MultiFab::Copy(Mfield_prev_iter[comp], Mfield_old[comp], 0, 0, 1, 1);
            }

            // compute new-time Hbias
            if (timedependent_Hbias) {
                ComputeHbias(H_biasfield, time+dt, geom);
            }

            // compute new-time alpha
            if (timedependent_alpha) {
                ComputeAlpha(alpha,geom,time+dt);
            }

            int iter = 1;

            while(1) { 
    
		// Corrector step update M
                // M^{n+1, *} = M^n + 0.5 * dt * (f^n + f^{n+1, *})
                for (int i = 0; i < 3; i++) {
                    MultiFab::LinComb(LLG_RHS_avg[i], 0.5, LLG_RHS[i], 0, 0.5, LLG_RHS_pre[i], 0, 0, 1, 0);
                    MultiFab::LinComb(Mfield[i], 1.0, Mfield_old[i], 0, dt, LLG_RHS_avg[i], 0, 0, 1, 0);
                }

                // Normalize M and fill ghost cells
                NormalizeM(Mfield, Ms, geom);
                
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

                if (iter == 1) {
                    amrex::Print() << "iter = " << iter << ", relative change from old to new = " << M_mag_error_max << "\n";
                } else {
                    // terminate while loop of error threshold is small enough
                    amrex::Print() << "iter = " << iter << ", relative change from prev_new to new = " << M_mag_error_max << "\n";
                    if (M_mag_error_max <= iterative_tolerance || iterative_tolerance == 0.) break;
                }

                // copy new solution into Mfield_prev_iter
                for (int comp = 0; comp < 3; comp++) {
                    MultiFab::Copy(Mfield_prev_iter[comp], Mfield[comp], 0, 0, 1, 1);
                }
    
                iter++;
        
                // Poisson solve and H_demag computation with Mfield
                if (demag_coupling == 1) { 
                    amrex::Print()<<"\n"<<"function visited" << std::endl;
                    demag_solver.CalculateH_demag(Mfield, H_demagfield);
                    // CalculateH_demag_ML(Mfield, x_norm_module, ml_module, y_norm_module, tensoropt, H_demagfield);
                }
    
                if (exchange_coupling == 1) {
                    CalculateH_exchange(Mfield, H_exchangefield, Ms, exchange, DMI, geom);
                }
        
                if (DMI_coupling == 1) {
                    CalculateH_DMI(Mfield, H_DMIfield, Ms, exchange, DMI, geom);
                }
    
                if (anisotropy_coupling == 1) {
                    CalculateH_anisotropy(Mfield, H_anisotropyfield, Ms, anisotropy);
                }
    
                // LLG RHS with new H_demag and Mfield_pre
                // Compute f^{n+1, *} = f(M^{n+1, *}, H^{n+1, *})
                Compute_LLG_RHS(LLG_RHS_pre, Mfield, H_demagfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield, alpha,
                                Ms, gamma);

            }
    
        } else if (TimeIntegratorOption == 3) { // iterative direct solver (ARTEMIS way)
        
            EvolveM_2nd(Mfield, H_demagfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield, alpha, Ms,
                        gamma, exchange, DMI, anisotropy, demag_solver,
                        geom, time, dt);

        }  else if (TimeIntegratorOption == 4) { // AMReX and SUNDIALS integrators

#ifdef AMREX_USE_SUNDIALS
	    // Create a RHS source function we will integrate
            // for MRI this represents the slow processes
            auto rhs_fun = [&](Vector<MultiFab>& rhs, const Vector<MultiFab>& state, const Real ) {
                
                // User function to calculate the rhs MultiFab given the state MultiFab
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                    rhs[idim].setVal(0.);
                } 

	        //alias rhs and state from vector of MultiFabs amrex::Vector<MultiFab> into Array<MultiFab, AMREX_SPACEDIM>
		//This is needed since CalculateH_* and Compute_LLG_RHS function take Array<MultiFab, AMREX_SPACEDIM> as input param

                Array<MultiFab, AMREX_SPACEDIM> ar_rhs{AMREX_D_DECL(MultiFab(rhs[0],amrex::make_alias,0,rhs[0].nComp()),
		                                                    MultiFab(rhs[1],amrex::make_alias,0,rhs[1].nComp()),
			       			                    MultiFab(rhs[2],amrex::make_alias,0,rhs[2].nComp()))};

                Array<MultiFab, AMREX_SPACEDIM> ar_state{AMREX_D_DECL(MultiFab(state[0],amrex::make_alias,0,state[0].nComp()),
                                                                      MultiFab(state[1],amrex::make_alias,0,state[1].nComp()),
                                                                      MultiFab(state[2],amrex::make_alias,0,state[2].nComp()))};

    	        // Evolve H_demag
                if (demag_coupling == 1) {
<<<<<<< HEAD
                    amrex::Print() << "this ar_old_state is used" << "\n";
                    demag_solver.CalculateH_demag(ar_old_state, H_demagfield);
                    // CalculateH_demag_ML(ar_old_state, x_norm_module, ml_module, y_norm_module, tensoropt, H_demagfield);
=======
                    demag_solver.CalculateH_demag(ar_state, H_demagfield);
                }

                if (using_MRI) {

                    // using MRI, set these processes to zero
                    for (int d=0; d<AMREX_SPACEDIM; ++d) {
                        H_exchangefield[d].setVal(0.);
                        H_DMIfield[d].setVal(0.);
                        H_anisotropyfield[d].setVal(0.);
                    }
                } else {

                    if (exchange_coupling == 1) {
                        CalculateH_exchange(ar_state, H_exchangefield, Ms, exchange, DMI, geom);
                    }

                    if (DMI_coupling == 1) {
                        CalculateH_DMI(ar_state, H_DMIfield, Ms, exchange, DMI, geom);
                    }

                    if (anisotropy_coupling == 1) {
                        CalculateH_anisotropy(ar_state, H_anisotropyfield, Ms, anisotropy);
                    }
                }

                // Compute f^n = f(M^n, H^n) 
                Compute_LLG_RHS(ar_rhs, ar_state, H_demagfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield, alpha, Ms, gamma);
            };

            // Create a fast RHS source function we will integrate
            auto rhs_fast_fun = [&](Vector<MultiFab>& rhs, const Vector<MultiFab>& stage_data, const Vector<MultiFab>& state, const Real ) {
                
                // User function to calculate the rhs MultiFab given the state MultiFab
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                    rhs[idim].setVal(0.);
                } 

	        //alias rhs and state from vector of MultiFabs amrex::Vector<MultiFab> into Array<MultiFab, AMREX_SPACEDIM>
		//This is needed since CalculateH_* and Compute_LLG_RHS function take Array<MultiFab, AMREX_SPACEDIM> as input param

                Array<MultiFab, AMREX_SPACEDIM> ar_rhs{AMREX_D_DECL(MultiFab(rhs[0],amrex::make_alias,0,rhs[0].nComp()),
		                                                    MultiFab(rhs[1],amrex::make_alias,0,rhs[1].nComp()),
			       			                    MultiFab(rhs[2],amrex::make_alias,0,rhs[2].nComp()))};

                Array<MultiFab, AMREX_SPACEDIM> ar_state{AMREX_D_DECL(MultiFab(state[0],amrex::make_alias,0,state[0].nComp()),
                                                                      MultiFab(state[1],amrex::make_alias,0,state[1].nComp()),
                                                                      MultiFab(state[2],amrex::make_alias,0,state[2].nComp()))};

    	        // fast RHS does not have demag
                if (demag_coupling == 1) {
                    for (int d=0; d<AMREX_SPACEDIM; ++d) {
                        H_demagfield[d].setVal(0.);
                    }
>>>>>>> development
                }

                if (exchange_coupling == 1) {
                    CalculateH_exchange(ar_state, H_exchangefield, Ms, exchange, DMI, geom);
                }

                if (DMI_coupling == 1) {
                    CalculateH_DMI(ar_state, H_DMIfield, Ms, exchange, DMI, geom);
                }

                if (anisotropy_coupling == 1) {
                    CalculateH_anisotropy(ar_state, H_anisotropyfield, Ms, anisotropy);
                }
		
                // Compute f^n = f(M^n, H^n) 
                Compute_LLG_RHS(ar_rhs, ar_state, H_demagfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield, alpha, Ms, gamma);
            };

            // Create a function to call after updating a state
            auto post_update_fun = [&](Vector<MultiFab>& state, const Real ) {
               
                Array<MultiFab, AMREX_SPACEDIM> ar_state{AMREX_D_DECL(MultiFab(state[0],amrex::make_alias,0,state[0].nComp()),
		                                                      MultiFab(state[1],amrex::make_alias,0,state[1].nComp()),
			       			                      MultiFab(state[2],amrex::make_alias,0,state[2].nComp()))};

                // Normalize M and fill ghost cells
                NormalizeM(ar_state, Ms, geom);
            };

            // Attach the right hand side and post-update functions
            // to the integrator
            integrator.set_rhs(rhs_fun);
            integrator.set_fast_rhs(rhs_fast_fun);
            integrator.set_post_update(post_update_fun);

            // This sets the ratio of slow timestep size to fast timestep size as an integer,
            // or equivalently, the number of fast timesteps per slow timestep.
            integrator.set_slow_fast_timestep_ratio(10);
                
            // integrate forward one step from `time` by `dt` to fill S_new
            integrator.advance(vMfield_old, vMfield, time, dt);
#else
            amrex::Abort("Trying to use TimeIntegratorOption == 4 but complied with USE_SUNDIALS=FALSE; make realclean and then recompile with USE_SUNDIALS=TRUE");
#endif

        } else {
            amrex::Abort("Time integrator order not recognized");
        }

        // standard problem diagnostics
        bool diag_std4_plot = false;
        if (diag_type == 4 || diag_type == 2 || diag_type == 3) {
            
            normalized_Mx = SumNormalizedM(Ms,Mfield[0]);
            normalized_My = SumNormalizedM(Ms,Mfield[1]);
            normalized_Mz = SumNormalizedM(Ms,Mfield[2]);

            if (diag_type == 4) {
                if (normalized_Mx_prev > 0 && normalized_Mx <= 0.) {
                    diag_std4_plot = true;
                }
            }

            normalized_Mx_prev = normalized_Mx;
  	
	    outputFile << "time = " << time << " "
                    << "Sum_normalized_M: "
                    << normalized_Mx/num_mag << " "
                    << normalized_My/num_mag << " "
                    << normalized_Mz/num_mag << std::endl;

            // Check if field is equilibirated
	    // If so, we will increment Hbias 
            if ((Hbias_sweep == 1 || steady_stop == 1) && (step > 1)) {
	    
	        err_x = amrex::Math::abs((normalized_Mx/num_mag) - normalized_Mx_old);
	        err_y = amrex::Math::abs((normalized_My/num_mag) - normalized_My_old);
	        err_z = amrex::Math::abs((normalized_Mz/num_mag) - normalized_Mz_old);
                 
		outputFile << "time = " << time << " "
                    << "error: "
                    << err_x << " "
                    << err_y << " "
                    << err_z << std::endl;


 	        normalized_Mx_old = normalized_Mx/num_mag;
	        normalized_My_old = normalized_My/num_mag;
	        normalized_Mz_old = normalized_Mz/num_mag;

	        if ((err_x < equilibrium_tolerance) && (err_y < equilibrium_tolerance) && (err_z < equilibrium_tolerance)) {
	            if (Hbias_sweep == 1) {
		        increment_Hbias = 1;
		    }
		    if (steady_stop == 1 && step >= 1000) {
		        steady = 1;
		    }
                    // Reset the error
	            normalized_Mx_old = 0.;
		    normalized_My_old = 0.;
                    normalized_Mz_old = 0.;
		}

	    }

            // standard problem 2 diagnostics
	    if (diag_type == 2) {
                Real Hbias_x = SumHbias(H_biasfield[0],Ms)/num_mag;
                Real Hbias_y = SumHbias(H_biasfield[1],Ms)/num_mag;
                Real Hbias_z = SumHbias(H_biasfield[2],Ms)/num_mag;
                Hbias_magn = sqrt(Hbias_x*Hbias_x + Hbias_y*Hbias_y + Hbias_z*Hbias_z);
                if (Hbias_x < 0) Hbias_magn *= -1.;

                M = (normalized_Mx/num_mag) + (normalized_My/num_mag) + (normalized_Mz/num_mag);

                if ( (M_old > 0 && M <= 0.) || (M_old < 0 && M >= 0.) ) {
	            outputFile << "time = " << time << " "
                               << "Coercivity = " << Hbias_magn <<  std::endl;
                }

                M_old = M;

                if (increment_Hbias == 1 && (increment_count == nsteps_hysteresis/2 || increment_count == 3*nsteps_hysteresis/2) ) {
	            outputFile << "time = " << time << " "
                               << "Hbias_magn = " << Hbias_magn << " "
                               << "Remanance = " << normalized_Mx/num_mag << " " << normalized_My/num_mag << " " << normalized_Mz/num_mag << std::endl;
                }

	        if (increment_Hbias == 1) {
	            outputFile << "time = " << time << " "
                               << "Hbias = " << Hbias_magn << " "
                               << "M/Ms = " << M <<  std::endl;
	        }
            }		

	    // standard problem 3 diagnostics
            if (diag_type == 3) {

    		demag_energy = DemagEnergy(Ms, Mfield[0], Mfield[1], Mfield[2], H_demagfield[0], H_demagfield[1], H_demagfield[2]);
                exchange_energy = ExchangeEnergy(Mfield, Ms, geom, exch);		
		anis_energy = AnisotropyEnergy(Ms, Mfield[0], Mfield[1], Mfield[2], ani);

                demag_energy /= Km*num_mag;
		exchange_energy /= Km*num_mag;
		anis_energy /= Km*num_mag;
		
		total_energy = anis_energy + exchange_energy + demag_energy;
	    
	        outputFile << "time = " << time << " "
                           << "demag_energy = "<< demag_energy << " " 
		 	   << "exchange_energy = "<< exchange_energy << " "
		 	   << "anis_energy = "<< anis_energy << " "
			   << "total_energy = "<< total_energy <<  std::endl;

	     }

        }
        // Write a plotfile of the data if plot_int > 0
        // if ( (plot_int > 0 && step%plot_int == 0) || diag_std4_plot) {
        //     WritePlotfile(Ms, Mfield_old, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield,
        //                   H_demagfield, geom, time, step);
        // }

        // copy new solution into old solution
        for (int comp = 0; comp < 3; comp++) {
            MultiFab::Copy(Mfield_old[comp], Mfield[comp], 0, 0, 1, 1);
        }

        // update time
        time = time + dt;

        Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);

        amrex::Print() << "Advanced step " << step << " in " << step_stop_time << " seconds; time = " << time << "\n";

<<<<<<< HEAD
        // // Write a plotfile of the data if plot_int > 0
        // if ( (plot_int > 0 && step%plot_int == 0) || diag_std4_plot) {
        //     WritePlotfile(Ms, Mfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield,
        //                   H_demagfield, geom, time, step);
        // }

        /*
	
	If we want to reach equilibrium in M before advancing Hbias, we enter this inner loop.   
	Different from ferrox in that we need to evolve the whole field to sweep through the Mfield until equilibrium.
	
	*/

        if(Hbias_sweep == 1)
        {

	    if (demag_coupling == 1) {
        amrex::Print() << "this Mfield_old 3 is used" << "\n";    
		demag_solver.CalculateH_demag(Mfield_old, H_demagfield);
        // CalculateH_demag_ML(Mfield_old, x_norm_module, ml_module, y_norm_module, tensoropt, H_demagfield);
	    }

	    if (exchange_coupling == 1) {
		CalculateH_exchange(Mfield_old, H_exchangefield, Ms, exchange, DMI, geom);
	    }

	    if (DMI_coupling == 1) {
		CalculateH_DMI(Mfield_old, H_DMIfield, Ms, exchange, DMI, geom);
	    }

	    if (anisotropy_coupling == 1) {
		CalculateH_anisotropy(Mfield_old, H_anisotropyfield, Ms, anisotropy);
	    }

           // iterate to compute M^{n+1} until equilibirum reached.
           while(err > equilibrium_tolerance){
	
		    // Evolve M with simple Forward Euler
		    // Compute f^n = f(M^n, H^n)
		    Compute_LLG_RHS(LLG_RHS, Mfield_old, H_demagfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield, alpha,
				    Ms, gamma);

		    // M^{n+1} = M^n + dt * f^n
		    for (int i = 0; i < 3; i++) {
			MultiFab::LinComb(Mfield[i], 1.0, Mfield_old[i], 0, dt, LLG_RHS[i], 0, 0, 1, 0);
		    }

		    NormalizeM(Mfield, Ms, geom);

		    normalized_Mx = SumNormalizedM(Ms,Mfield[0])/num_mag;
                    normalized_My = SumNormalizedM(Ms,Mfield[1])/num_mag;
                    normalized_Mz = SumNormalizedM(Ms,Mfield[2])/num_mag;

		    outputFile << "time = " << time << " "
		               << "equilibirate_Sum_normalized_M: "
		               << normalized_Mx << " "
		               << normalized_My << " "
		               << normalized_Mz << std::endl;

                    M_magnitude = sqrt(normalized_Mx*normalized_Mx + normalized_My*normalized_My + normalized_Mz*normalized_Mz);
                   
                    err = amrex::Math::abs(M_magnitude_old - M_magnitude);

                    M_magnitude_old = M_magnitude;

		    if (timedependent_alpha) {
	                ComputeAlpha(alpha,geom,time);
		    }

		    if (demag_coupling == 1) {
                amrex::Print() << "this Mfield_old 4 is used" << "\n";  
			demag_solver.CalculateH_demag(Mfield_old, H_demagfield);
            // CalculateH_demag_ML(Mfield_old, x_norm_module, ml_module, y_norm_module, tensoropt, H_demagfield);
		    }

		    if (exchange_coupling == 1) {
			CalculateH_exchange(Mfield_old, H_exchangefield, Ms, exchange, DMI, geom);
		    }

		    if (DMI_coupling == 1) {
			CalculateH_DMI(Mfield_old, H_DMIfield, Ms, exchange, DMI, geom);
		    }

		    if (anisotropy_coupling == 1) {
			CalculateH_anisotropy(Mfield_old, H_anisotropyfield, Ms, anisotropy);
		    }
            }
        }

        err = equilibrium_tolerance + 1.;

=======
        // Write a plotfile of the data if plot_int > 0
        if ( (plot_int > 0 && step%plot_int == 0) || (plot_int > 0 && time > stop_time) || diag_std4_plot) {
            WritePlotfile(Ms, Mfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield,
                          H_demagfield, geom, time, step);
        }

>>>>>>> development
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

        // If we have completed the hysteresis loop, we end the simulation
        if (increment_Hbias == 1 && increment_count == 2*nsteps_hysteresis) {
            break;
        }

	// If we have reached a steady-state and have steady_stop == 1, we end simulation 
	if (steady_stop == 1 && steady == 1) {
	    break;
	}

        if (time > stop_time) {
            amrex::Print() << "Stop time reached\n";
            break;
        }

    }
    
    Real total_step_stop_time = ParallelDescriptor::second() - total_step_strt_time;
    ParallelDescriptor::ReduceRealMax(total_step_stop_time);

    amrex::Print() << "Total run time " << total_step_stop_time << " seconds\n";

    outputFile.close();
}
