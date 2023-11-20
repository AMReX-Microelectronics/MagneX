#include "EvolveM_2nd.H"
#include "CartesianAlgorithm.H"
#include <AMReX_OpenBC.H>
#include "Demagnetization.H"
#include "EffectiveExchangeField.H"
#include "EffectiveDMIField.H"
#include "EffectiveAnisotropyField.H"
#include "NormalizeM.H"
#include <AMReX_MLMG.H> 
#include <AMReX_MultiFab.H> 
#include <AMReX_VisMF.H>

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
                 amrex::Real const dt)
{

    // build temporary vector<multifab,3> Mfield_prev, Mfield_error, a_temp, a_temp_static, b_temp_static
    std::array<MultiFab, 3> H_demagfield_prev;    // H^(new_time) of the (r-1)th iteration
    std::array<MultiFab, 3> H_exchangefield_prev;    // H^(new_time) of the (r-1)th iteration
    std::array<MultiFab, 3> H_DMIfield_prev;    // H^(new_time) of the (r-1)th iteration
    std::array<MultiFab, 3> H_anisotropyfield_prev;    // H^(new_time) of the (r-1)th iteration
    std::array<MultiFab, 3> Mfield_old;    // M^(old_time) before the current time step
    std::array<MultiFab, 3> Mfield_prev;   // M^(new_time) of the (r-1)th iteration
    std::array<MultiFab, 3> Mfield_error;  // The error of the M field between the two consecutive iterations
    std::array<MultiFab, 3> a_temp;        // right-hand side of vector a, see the documentation
    std::array<MultiFab, 3> a_temp_static; // α M^(old_time)/|M| in the right-hand side of vector a, see the documentation
    std::array<MultiFab, 3> b_temp_static; // right-hand side of vector b, see the documentation

    BoxArray ba = Mfield[0].boxArray();
    DistributionMapping dm = Mfield[0].DistributionMap();

    for (int i = 0; i < 3; i++){
        H_demagfield_prev[i].define(ba, dm, 1, 0);
        H_exchangefield_prev[i].define(ba, dm, 1, 0);
        H_DMIfield_prev[i].define(ba, dm, 1, 0);
        H_anisotropyfield_prev[i].define(ba, dm, 1, 0);
        Mfield_old[i].define(ba, dm, 1, 1);
        Mfield_prev[i].define(ba, dm, 1, 1);
        
        MultiFab::Copy(H_demagfield_prev[i], H_demagfield[i], 0, 0, 1, 0);
        MultiFab::Copy(H_exchangefield_prev[i], H_exchangefield[i], 0, 0, 1, 0);
        MultiFab::Copy(H_DMIfield_prev[i], H_DMIfield[i], 0, 0, 1, 0);
        MultiFab::Copy(H_anisotropyfield_prev[i], H_anisotropyfield[i], 0, 0, 1, 0);
        MultiFab::Copy(Mfield_old[i], Mfield[i], 0, 0, 1, 1);
        MultiFab::Copy(Mfield_prev[i], Mfield[i], 0, 0, 1, 1);
        
        // fill periodic ghost cells
        Mfield_old[i].FillBoundary(geom.periodicity());
        Mfield_prev[i].FillBoundary(geom.periodicity());

        Mfield_error[i].define(ba, dm, 1, 0);
        Mfield_error[i].setVal(0.); // reset Mfield_error to zero

        // initialize a_temp, a_temp_static, b_temp_static
        a_temp[i].define(ba, dm, 1, 0);
        a_temp_static[i].define(ba, dm, 1, 0);
        b_temp_static[i].define(ba, dm, 1, 0);

    }    
    
    // calculate the b_temp_static, a_temp_static
    for (MFIter mfi(a_temp_static[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Array4<Real>& alpha_arr = alpha.array(mfi);
        const Array4<Real>& gamma_arr = gamma.array(mfi);
        const Array4<Real>& Ms_arr = Ms.array(mfi);

        // extract field data   
        const Array4<Real>& Hx_bias = H_biasfield[0].array(mfi);
        const Array4<Real>& Hy_bias = H_biasfield[1].array(mfi);
        const Array4<Real>& Hz_bias = H_biasfield[2].array(mfi);
        const Array4<Real>& Hx_demag = H_demagfield[0].array(mfi);
        const Array4<Real>& Hy_demag = H_demagfield[1].array(mfi);
        const Array4<Real>& Hz_demag = H_demagfield[2].array(mfi);
        const Array4<Real>& Hx_exchange = H_exchangefield[0].array(mfi);
        const Array4<Real>& Hy_exchange = H_exchangefield[1].array(mfi);
        const Array4<Real>& Hz_exchange = H_exchangefield[2].array(mfi);
        const Array4<Real>& Hx_DMI = H_DMIfield[0].array(mfi);
        const Array4<Real>& Hy_DMI = H_DMIfield[1].array(mfi);
        const Array4<Real>& Hz_DMI = H_DMIfield[2].array(mfi);
        const Array4<Real>& Hx_anisotropy = H_anisotropyfield[0].array(mfi);
        const Array4<Real>& Hy_anisotropy = H_anisotropyfield[1].array(mfi);
        const Array4<Real>& Hz_anisotropy = H_anisotropyfield[2].array(mfi);
        const Array4<Real>& Mx_old = Mfield_old[0].array(mfi);
        const Array4<Real>& My_old = Mfield_old[1].array(mfi);
        const Array4<Real>& Mz_old = Mfield_old[2].array(mfi);

        // extract field data of a_temp_static and b_temp_static
        const Array4<Real>& ax_temp_static = a_temp_static[0].array(mfi);
        const Array4<Real>& ay_temp_static = a_temp_static[1].array(mfi);
        const Array4<Real>& az_temp_static = a_temp_static[2].array(mfi);
        const Array4<Real>& bx_temp_static = b_temp_static[0].array(mfi);
        const Array4<Real>& by_temp_static = b_temp_static[1].array(mfi);
        const Array4<Real>& bz_temp_static = b_temp_static[2].array(mfi);

        // extract tileboxes for which to loop
        const Box& tbx = mfi.tilebox();

        // loop over cells and update fields
        amrex::ParallelFor(tbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                // determine if the material is nonmagnetic or not
                if (Ms_arr(i,j,k) > 0._rt){

                    // when working on M(i,j,k, 0:2) we have direct access to M(i,j,k,0:2) and Hx(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    amrex::Real Hx_eff_old = Hx_bias(i,j,k);
                    amrex::Real Hy_eff_old = Hy_bias(i,j,k);
                    amrex::Real Hz_eff_old = Hz_bias(i,j,k);

                    if (demag_coupling == 1){
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                        // H_maxwell - use H^(old_time)
                        Hx_eff_old += Hx_demag(i,j,k);
                        Hy_eff_old += Hy_demag(i,j,k);
                        Hz_eff_old += Hz_demag(i,j,k);
                    }

                    if (exchange_coupling == 1){

                        // H_exchange - use M^(old_time)

                        Hx_eff_old += Hx_exchange(i, j, k);
                        Hy_eff_old += Hy_exchange(i, j, k);
                        Hz_eff_old += Hz_exchange(i, j, k);

                    }

                    if (DMI_coupling == 1){

                        // H_DMI - use M^(old_time)

                        Hx_eff_old += Hx_DMI(i, j, k);
                        Hy_eff_old += Hy_DMI(i, j, k);
                        Hz_eff_old += Hz_DMI(i, j, k);

                    }

                    if (anisotropy_coupling == 1){

                        // H_anisotropy - use M^(old_time)
                        Hx_eff_old += Hx_anisotropy(i, j, k);
                        Hy_eff_old += Hy_anisotropy(i, j, k);
                        Hz_eff_old += Hz_anisotropy(i, j, k);
                    }

                    // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                    amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(Mx_old(i, j, k), 2._rt) + std::pow(My_old(i, j, k), 2._rt) + std::pow(Mz_old(i, j, k), 2._rt))
                                                              : Ms_arr(i,j,k);
                    // a_temp_static_coeff does not change in the current step for SATURATED materials; but it does change for UNSATURATED ones
                    amrex::Real a_temp_static_coeff = alpha_arr(i,j,k) / M_magnitude;

                    // calculate the b_temp_static_coeff (it is divided by 2.0 because the derivation is based on an interger dt)
                    amrex::Real b_temp_static_coeff = - mu0 * amrex::Math::abs(gamma_arr(i,j,k)) / 2.;

                    // calculate a_temp_static
                    ax_temp_static(i, j, k) = a_temp_static_coeff * Mx_old(i, j, k);
                    ay_temp_static(i, j, k) = a_temp_static_coeff * My_old(i, j, k);
                    az_temp_static(i, j, k) = a_temp_static_coeff * Mz_old(i, j, k);

                    // calculate b_temp_static
                    bx_temp_static(i, j, k) = Mx_old(i, j, k) + dt * b_temp_static_coeff * (My_old(i, j, k) * Hz_eff_old - Mz_old(i, j, k) * Hy_eff_old);

                    // y component on x-faces of grid
                    by_temp_static(i, j, k) = My_old(i, j, k) + dt * b_temp_static_coeff * (Mz_old(i, j, k) * Hx_eff_old - Mx_old(i, j, k) * Hz_eff_old);

                    // z component on x-faces of grid
                    bz_temp_static(i, j, k) = Mz_old(i, j, k) + dt * b_temp_static_coeff * (Mx_old(i, j, k) * Hy_eff_old - My_old(i, j, k) * Hx_eff_old);
                }
            });
    }

    // initialize M_max_iter, M_iter, M_tol, M_iter_error
    // maximum number of iterations allowed
    int M_max_iter = 100;
    int M_iter = 0;
    // relative tolerance stopping criteria for 2nd-order iterative algorithm
    amrex::Real M_tol = 1.e-6;

    // begin the iteration
    while (1) {

        for (MFIter mfi(Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi){

            const Array4<Real>& alpha_arr = alpha.array(mfi);
            const Array4<Real>& gamma_arr = gamma.array(mfi);
            const Array4<Real>& Ms_arr = Ms.array(mfi);

            // extract field data
            const Array4<Real>& Mx = Mfield[0].array(mfi);
            const Array4<Real>& My = Mfield[1].array(mfi);
            const Array4<Real>& Mz = Mfield[2].array(mfi);
            const Array4<Real>& Hx_bias = H_biasfield[0].array(mfi);
            const Array4<Real>& Hy_bias = H_biasfield[1].array(mfi);
            const Array4<Real>& Hz_bias = H_biasfield[2].array(mfi);
            const Array4<Real>& Hx_demag_prev = H_demagfield_prev[0].array(mfi);
            const Array4<Real>& Hy_demag_prev = H_demagfield_prev[1].array(mfi);
            const Array4<Real>& Hz_demag_prev = H_demagfield_prev[2].array(mfi);
            const Array4<Real>& Hx_exchange_prev = H_exchangefield_prev[0].array(mfi);
            const Array4<Real>& Hy_exchange_prev = H_exchangefield_prev[1].array(mfi);
            const Array4<Real>& Hz_exchange_prev = H_exchangefield_prev[2].array(mfi);
            const Array4<Real>& Hx_DMI_prev = H_DMIfield_prev[0].array(mfi);
            const Array4<Real>& Hy_DMI_prev = H_DMIfield_prev[1].array(mfi);
            const Array4<Real>& Hz_DMI_prev = H_DMIfield_prev[2].array(mfi);
            const Array4<Real>& Hx_anisotropy_prev = H_anisotropyfield_prev[0].array(mfi);
            const Array4<Real>& Hy_anisotropy_prev = H_anisotropyfield_prev[1].array(mfi);
            const Array4<Real>& Hz_anisotropy_prev = H_anisotropyfield_prev[2].array(mfi);

            // extract field data of Mfield_prev, Mfield_error, a_temp, a_temp_static, and b_temp_static
            const Array4<Real>& Mx_prev = Mfield_prev[0].array(mfi);
            const Array4<Real>& My_prev = Mfield_prev[1].array(mfi);
            const Array4<Real>& Mz_prev = Mfield_prev[2].array(mfi);
            const Array4<Real>& Mx_old = Mfield_old[0].array(mfi);
            const Array4<Real>& My_old = Mfield_old[1].array(mfi);
            const Array4<Real>& Mz_old = Mfield_old[2].array(mfi);
            const Array4<Real>& Mx_error = Mfield_error[0].array(mfi);
            const Array4<Real>& My_error = Mfield_error[1].array(mfi);
            const Array4<Real>& Mz_error = Mfield_error[2].array(mfi);
            const Array4<Real>& ax_temp = a_temp[0].array(mfi);
            const Array4<Real>& ay_temp = a_temp[1].array(mfi);
            const Array4<Real>& az_temp = a_temp[2].array(mfi);
            const Array4<Real>& ax_temp_static = a_temp_static[0].array(mfi);
            const Array4<Real>& ay_temp_static = a_temp_static[1].array(mfi);
            const Array4<Real>& az_temp_static = a_temp_static[2].array(mfi);
            const Array4<Real>& bx_temp_static = b_temp_static[0].array(mfi);
            const Array4<Real>& by_temp_static = b_temp_static[1].array(mfi);
            const Array4<Real>& bz_temp_static = b_temp_static[2].array(mfi);

            // extract tileboxes for which to loop
            const Box& tbx = mfi.tilebox();

            // loop over cells and update fields
            amrex::ParallelFor(tbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                    // determine if the material is nonmagnetic or not
                    if (Ms_arr(i,j,k) > 0.){

                        // H_bias
                        amrex::Real Hx_eff_prev = Hx_bias(i,j,k);
                        amrex::Real Hy_eff_prev = Hy_bias(i,j,k);
                        amrex::Real Hz_eff_prev = Hz_bias(i,j,k);

                        if (demag_coupling == 1){
                            // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                            // H_maxwell - use H^[(new_time),r-1]
                            Hx_eff_prev += Hx_demag_prev(i,j,k);
                            Hy_eff_prev += Hy_demag_prev(i,j,k);
                            Hz_eff_prev += Hz_demag_prev(i,j,k);
                        }

                        if (exchange_coupling == 1){

                            // H_exchange - use M^[(new_time),r-1]
                            
                            Hx_eff_prev += Hx_exchange_prev(i, j, k);
                            Hy_eff_prev += Hy_exchange_prev(i, j, k);
                            Hz_eff_prev += Hz_exchange_prev(i, j, k);
                        }

                        if (DMI_coupling == 1){

                            // H_DMI - use M^[(new_time),r-1]

                            Hx_eff_prev += Hx_DMI_prev(i,j,k);
                            Hy_eff_prev += Hy_DMI_prev(i,j,k);
                            Hz_eff_prev += Hz_DMI_prev(i,j,k);
                        }

                        if (anisotropy_coupling == 1){

                            // H_anisotropy - use M^[(new_time),r-1]
                            Hx_eff_prev += Hx_anisotropy_prev(i,j,k);
                            Hy_eff_prev += Hy_anisotropy_prev(i,j,k);
                            Hz_eff_prev += Hz_anisotropy_prev(i,j,k);
                        }

                        // calculate the a_temp_dynamic_coeff (it is divided by 2.0 because the derivation is based on an interger dt,
                        // while in real simulations, the input dt is actually dt/2.0)
                        amrex::Real a_temp_dynamic_coeff = mu0 * amrex::Math::abs(gamma_arr(i,j,k)) / 2.;

                        amrex::GpuArray<amrex::Real,3> H_eff_prev;
                        H_eff_prev[0] = Hx_eff_prev;
                        H_eff_prev[1] = Hy_eff_prev;
                        H_eff_prev[2] = Hz_eff_prev;

                        ax_temp(i, j, k) = (M_normalization != 0) ? -(dt * a_temp_dynamic_coeff * Hx_eff_prev + ax_temp_static(i, j, k))
                                                                       : -(dt * a_temp_dynamic_coeff * Hx_eff_prev + 0.5 * ax_temp_static(i, j, k)
                                                                         + 0.5 * alpha_arr(i,j,k) * 1. / std::sqrt(std::pow(Mx(i, j, k), 2.) + std::pow(My(i, j, k), 2.) + std::pow(Mz(i, j, k), 2.)) * Mx_old(i, j, k));

                        ay_temp(i, j, k) = (M_normalization != 0) ? -(dt * a_temp_dynamic_coeff * Hy_eff_prev + ay_temp_static(i, j, k))
                                                                       : -(dt * a_temp_dynamic_coeff * Hy_eff_prev + 0.5 * ay_temp_static(i, j, k)
                                                                         + 0.5 * alpha_arr(i,j,k) * 1. / std::sqrt(std::pow(Mx(i, j, k), 2.) + std::pow(My(i, j, k), 2.) + std::pow(Mz(i, j, k), 2.)) * My_old(i, j, k));

                        az_temp(i, j, k) = (M_normalization != 0) ? -(dt * a_temp_dynamic_coeff * Hz_eff_prev + az_temp_static(i, j, k))
                                                                       : -(dt * a_temp_dynamic_coeff * Hz_eff_prev + 0.5 * az_temp_static(i, j, k)
                                                                         + 0.5 * alpha_arr(i,j,k) * 1. / std::sqrt(std::pow(Mx(i, j, k), 2.) + std::pow(My(i, j, k), 2.) + std::pow(Mz(i, j, k), 2.)) * Mz_old(i, j, k));
                        
                        amrex::Real a_square = pow(ax_temp(i, j, k), 2.0) + pow(ay_temp(i, j, k), 2.0) + pow(az_temp(i, j, k), 2.0);
                        amrex::Real a_dot_b =  ax_temp(i, j, k) * bx_temp_static(i, j, k) + ay_temp(i, j, k) * by_temp_static(i, j, k) + az_temp(i, j, k) * bz_temp_static(i, j, k);
                        
                        amrex::Real a_cross_b_x = ay_temp(i, j, k) * bz_temp_static(i, j, k) - az_temp(i, j, k) * by_temp_static(i, j, k);
                        Mx(i,j,k) = ( bx_temp_static(i, j, k) + a_dot_b * ax_temp(i, j, k) - a_cross_b_x ) / ( 1.0 + a_square);

                        amrex::Real a_cross_b_y = az_temp(i, j, k) * bx_temp_static(i, j, k) - ax_temp(i, j, k) * bz_temp_static(i, j, k);
                        My(i,j,k) = ( by_temp_static(i, j, k) + a_dot_b * ay_temp(i, j, k) - a_cross_b_y ) / ( 1.0 + a_square);

                        amrex::Real a_cross_b_z = ax_temp(i, j, k) * by_temp_static(i, j, k) - ay_temp(i, j, k) * bx_temp_static(i, j, k);
                        Mz(i,j,k) = ( bz_temp_static(i, j, k) + a_dot_b * az_temp(i, j, k) - a_cross_b_z ) / ( 1.0 + a_square);

                        // temporary normalized magnitude of M field at the fixed point
                        // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                        amrex::Real M_magnitude_normalized = std::sqrt(std::pow(Mx(i, j, k), 2.) + std::pow(My(i, j, k), 2.) + std::pow(Mz(i, j, k), 2.)) / Ms_arr(i,j,k);
                        amrex::Real normalized_error = 0.1;

                        if (M_normalization == 1){
                            // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                            // check the normalized error
                            if (amrex::Math::abs(1. - M_magnitude_normalized) > normalized_error){
                                amrex::Abort("Exceed the normalized error of the M field");
                            }
                            // normalize the M field
                            Mx(i, j, k) /= M_magnitude_normalized;
                            My(i, j, k) /= M_magnitude_normalized;
                            Mz(i, j, k) /= M_magnitude_normalized;
                        }
                        else if (M_normalization == 0){
                            // check the normalized error
                            if (M_magnitude_normalized > (1. + normalized_error)){
                                amrex::Abort("Caution: Unsaturated material has M exceeding the saturation magnetization");
                            }
                            else if (M_magnitude_normalized > 1. && M_magnitude_normalized <= 1. + normalized_error){
                                // normalize the M field
                                Mx(i, j, k) /= M_magnitude_normalized;
                                My(i, j, k) /= M_magnitude_normalized;
                                Mz(i, j, k) /= M_magnitude_normalized;
                            }
                        }

                        // calculate M_error
                        Mx_error(i,j,k) = amrex::Math::abs(Mx(i,j,k) - Mx_prev(i,j,k)) / Ms_arr(i,j,k);
                        My_error(i,j,k) = amrex::Math::abs(My(i,j,k) - My_prev(i,j,k)) / Ms_arr(i,j,k);
                        Mz_error(i,j,k) = amrex::Math::abs(Mz(i,j,k) - Mz_prev(i,j,k)) / Ms_arr(i,j,k);
                    }
                });
        }

        // update H_demag
        if(demag_coupling == 1) {
            
            if (demag_solver == -1 || demag_solver == 0) {

                amrex::Abort("FIXME: MLMG solvers in EvolveM_2nd");
                /*
                //Solve Poisson's equation laplacian(Phi) = div(M) and get H_demagfield = -grad(Phi)
                //Compute RHS of Poisson equation
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
                */
                
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

       if (exchange_coupling == 1){
          CalculateH_exchange(Mfield, H_exchangefield, Ms, exchange, DMI, exchange_coupling, DMI_coupling, mu0, geom);
       }

       if (DMI_coupling == 1){
          CalculateH_DMI(Mfield, H_DMIfield, Ms, exchange, DMI, DMI_coupling, mu0, geom);
       }

       if(anisotropy_coupling == 1){
         CalculateH_anisotropy(Mfield, H_anisotropyfield, Ms, anisotropy, anisotropy_coupling, anisotropy_axis, mu0, geom);
       }

        // Check the error between Mfield and Mfield_prev and decide whether another iteration is needed
        amrex::Real M_iter_maxerror = -1.;
        M_iter_maxerror = std::max(Mfield_error[0].norm0(), Mfield_error[1].norm0());
        M_iter_maxerror = std::max(M_iter_maxerror, Mfield_error[2].norm0());

        if (M_iter_maxerror <= M_tol){

            // normalize M
	    NormalizeM(Mfield,Ms,M_normalization);

            break;
        }
        else{

            // Copy Mfield to Mfield_previous and fill periodic/interior ghost cells
            for (int i = 0; i < 3; i++){
                MultiFab::Copy(Mfield_prev[i], Mfield[i], 0, 0, 1, Mfield[i].nGrow());
                MultiFab::Copy(H_demagfield_prev[i], H_demagfield[i], 0, 0, 1, H_demagfield[i].nGrow());
                MultiFab::Copy(H_exchangefield_prev[i], H_exchangefield[i], 0, 0, 1, H_exchangefield[i].nGrow());
                MultiFab::Copy(H_DMIfield_prev[i], H_DMIfield[i], 0, 0, 1, H_DMIfield[i].nGrow());
                MultiFab::Copy(H_anisotropyfield_prev[i], H_anisotropyfield[i], 0, 0, 1, H_anisotropyfield[i].nGrow());
                Mfield_prev[i].FillBoundary(geom.periodicity());
                H_demagfield_prev[i].FillBoundary(geom.periodicity());
                H_exchangefield_prev[i].FillBoundary(geom.periodicity());
                H_DMIfield_prev[i].FillBoundary(geom.periodicity());
                H_anisotropyfield_prev[i].FillBoundary(geom.periodicity());
            }
        }

        if (M_iter >= M_max_iter){
            amrex::Abort("The M_iter exceeds the M_max_iter");
            amrex::Print() << "The M_iter = " << M_iter << " exceeds the M_max_iter = " << M_max_iter << std::endl;
        }
        else{
            M_iter++;
            amrex::Print() << "Finish " << M_iter << " times iteration with M_iter_maxerror = " << M_iter_maxerror << " and M_tol = " << M_tol << std::endl;
        }

    } // end the iteration

}
