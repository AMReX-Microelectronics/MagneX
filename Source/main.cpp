#include "MagneX.H"
#include "Demagnetization.H"

#include <AMReX_MultiFab.H> 
#include <AMReX_VisMF.H>

#ifdef USE_TIME_INTEGRATOR
#include <AMReX_TimeIntegrator.H>
#endif

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
  
    // **********************************
    // READ SIMULATION PARAMETERS
    // **********************************
    InitializeMagneXNamespace();

    int start_step = 1;

    // for std4 diagnostic
    Real normalized_Mx_prev = 0.;
    
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

    MultiFab alpha(ba, dm, 1, 0);
    MultiFab gamma(ba, dm, 1, 0);
    MultiFab Ms(ba, dm, 1, 1);
    MultiFab exchange(ba, dm, 1, 0);
    MultiFab DMI(ba, dm, 1, 0);
    MultiFab anisotropy(ba, dm, 1, 0);

    amrex::Print() << "==================== Initial Setup ====================\n";
    amrex::Print() << " precession           = " << precession          << "\n";
    amrex::Print() << " demag_coupling       = " << demag_coupling      << "\n";
    if (demag_coupling == 1) amrex::Print() << " FFT_solver           = " << FFT_solver << "\n";
    amrex::Print() << " anisotropy_coupling  = " << anisotropy_coupling << "\n";
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

    // count how many magnetic cells are in the domain
    long num_mag = CountMagneticCells(Ms);
    
    if (restart == -1) {      
        //Initialize fields
        InitializeFields(Mfield, geom);

        if (demag_coupling == 1) {
            demag_solver.CalculateH_demag(Mfield, H_demagfield);
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

#ifdef USE_TIME_INTEGRATOR
    TimeIntegrator<Vector<MultiFab> > integrator(Mfield_old);
#endif 

    for (int step = start_step; step <= nsteps; ++step) {
        
        Real step_strt_time = ParallelDescriptor::second();

        if (timedependent_Hbias) {
            ComputeHbias(H_biasfield, time, geom);
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
                demag_solver.CalculateH_demag(Mfield_old, H_demagfield);
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

                // Normalize M
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
                    demag_solver.CalculateH_demag(Mfield, H_demagfield);
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

#ifdef USE_TIME_INTEGRATOR

            // Create a RHS source function we will integrate
            auto source_fun = [&](Vector<MultiFab>& rhs, const Vector<MultiFab>& old_state, const Real ) {

                // User function to calculate the rhs MultiFab given the state MultiFab
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                    rhs[idim].setVal(0.);
                } 
 
    	        // Evolve H_demag
                if (demag_coupling == 1) {
                    demag_solver.CalculateH_demag(old_state, H_demagfield);
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

                // Compute f^n = f(M^n, H^n) 
                Compute_LLG_RHS(rhs, old_state, H_demagfield, H_biasfield, alpha, Ms, gamma, exchange, anisotropy, demag_coupling,
                                exchange_coupling, anisotropy_coupling, anisotropy_axis, M_normalization, mu0);
            };

            // Create a function to call after updating a state
            auto post_update_fun = [&](Vector<MultiFab>& state, const Real ) {
                // Call user function to update state MultiFab, e.g. fill BCs
                NormalizeM(state, Ms, geom);
                
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

        // standard problem 4 diagnostics
        bool diag_std4_plot = false;
        if (diag_type == 4) {
            Real normalized_Mx = SumNormalizedM(Ms,Mfield[0]);
            Real normalized_My = SumNormalizedM(Ms,Mfield[1]);
            Real normalized_Mz = SumNormalizedM(Ms,Mfield[2]);
        
            Print() << "time = " << time << " "
                    << "Sum_normalized_M: "
                    << normalized_Mx/num_mag << " "
                    << normalized_My/num_mag << " "
                    << normalized_Mz/num_mag << std::endl;

            if (normalized_Mx_prev > 0 && normalized_Mx <= 0.) {
                diag_std4_plot = true;
            }
            
            normalized_Mx_prev = normalized_Mx;
        }
        
	/*
        // standard problem 3 diagnostics	
        if (diag_type == 3) {

	    Calculate_Heff(Heff, H_demagfield, H_anisotropyfield, H_exchangefield, H_biasfield);

	    IntVect location(AMREX_D_DECL(0,0,0));

	    Real magnitude = Coercivity(Mfield, Heff, location);

	        Print() << "Coercivity retrieved at: " << location << "  "
			<< "... with magnitude = " << magnitude << std::endl;

	}
        */

	if (diag_type == 2) {
	
            Real Heff_x = SumHeff(H_demagfield[0], H_exchangefield[0], H_biasfield[0]);
            Real Heff_y = SumHeff(H_demagfield[1], H_exchangefield[1], H_biasfield[1]);
            Real Heff_z = SumHeff(H_demagfield[2], H_exchangefield[2], H_biasfield[2]);

	    Print() << "time = " << time << " "
                    << "Heff: "
                    << Heff_x/num_mag << " "
                    << Heff_y/num_mag << " "
                    << Heff_z/num_mag << std::endl;

	}

        // copy new solution into old solution
        for (int comp = 0; comp < 3; comp++) {
            MultiFab::Copy(Mfield_old[comp], Mfield[comp], 0, 0, 1, 1);
        }

        // update time
        time = time + dt;

        Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);

        amrex::Print() << "Advanced step " << step << " in " << step_stop_time << " seconds\n";

        // Write a plotfile of the data if plot_int > 0
        if ( (plot_int > 0 && step%plot_int == 0) || diag_std4_plot) {
            WritePlotfile(Ms, Mfield, H_biasfield, H_exchangefield, H_DMIfield, H_anisotropyfield,
                          H_demagfield, geom, time, step);
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
