#include "EvolveM_2nd.H"
#include "CartesianAlgorithm.H"
#include <AMReX_OpenBC.H>
#include "MagnetostaticSolver.H"
#include "EffectiveExchangeField.H"
#include "EffectiveDMIField.H"
#include <AMReX_MLMG.H> 
#include <AMReX_MultiFab.H> 
#include <AMReX_VisMF.H>

void EvolveM_2nd(
    //std::array< MultiFab, AMREX_SPACEDIM> &Mfield,
    amrex::Vector<MultiFab>& Mfield,
    std::array< MultiFab, AMREX_SPACEDIM> &H_demagfield,
    std::array< MultiFab, AMREX_SPACEDIM> &H_biasfield, // H bias
    std::array< MultiFab, AMREX_SPACEDIM> &H_exchangefield, // effective exchange field
    std::array< MultiFab, AMREX_SPACEDIM> &H_DMIfield,
    MultiFab                              &PoissonRHS, 
    MultiFab                              &PoissonPhi, 
    std::array< MultiFab, AMREX_SPACEDIM >&   alpha,
    std::array< MultiFab, AMREX_SPACEDIM >&   Ms,
    std::array< MultiFab, AMREX_SPACEDIM >&   gamma,
    std::array< MultiFab, AMREX_SPACEDIM >&   exchange,
    std::array< MultiFab, AMREX_SPACEDIM >&   DMI,
    std::array< MultiFab, AMREX_SPACEDIM >&   anisotropy,
    int demag_coupling,
    int exchange_coupling,
    int DMI_coupling,
    int anisotropy_coupling,
    amrex::GpuArray<amrex::Real, 3>& anisotropy_axis,
    int M_normalization, 
    Real mu0,
    const Geometry& geom,
    amrex::GpuArray<amrex::Real, 3>  prob_lo,
    amrex::GpuArray<amrex::Real, 3>  prob_hi,
    amrex::Real const dt,
    const Real time
){

    // build temporary vector<multifab,3> Mfield_prev, Mfield_error, a_temp, a_temp_static, b_temp_static
    std::array<MultiFab, 3> H_demagfield_old;    // H^(old_time) before the current time step
    std::array<MultiFab, 3> H_demagfield_prev;    // H^(new_time) of the (r-1)th iteration
    std::array<MultiFab, 3> H_exchangefield_old;    // H^(old_time) before the current time step
    std::array<MultiFab, 3> H_exchangefield_prev;    // H^(new_time) of the (r-1)th iteration
    std::array<MultiFab, 3> H_DMIfield_old;    // H^(old_time) before the current time step
    std::array<MultiFab, 3> H_DMIfield_prev;    // H^(new_time) of the (r-1)th iteration
    std::array<MultiFab, 3> Mfield_old;    // M^(old_time) before the current time step
    std::array<MultiFab, 3> Mfield_prev;   // M^(new_time) of the (r-1)th iteration
    std::array<MultiFab, 3> Mfield_error;  // The error of the M field between the two consecutive iterations
    std::array<MultiFab, 3> a_temp;        // right-hand side of vector a, see the documentation
    std::array<MultiFab, 3> a_temp_static; // α M^(old_time)/|M| in the right-hand side of vector a, see the documentation
    std::array<MultiFab, 3> b_temp_static; // right-hand side of vector b, see the documentation

    BoxArray ba = H_demagfield[0].boxArray(); // H_demagfield is cell centered

    DistributionMapping dm = Mfield[0].DistributionMap();
    LPInfo info;
    OpenBCSolver openbc({geom}, {ba}, {dm}, info);

    for (int i = 0; i < 3; i++){
        H_demagfield_old[i].define(ba, dm, 1, H_demagfield[i].nGrow()); // only demag fields are cell centered
        H_demagfield_prev[i].define(ba, dm, 1, H_demagfield[i].nGrow());
        H_exchangefield_old[i].define(convert(ba, IntVect::TheDimensionVector(i)), dm, 3, H_exchangefield[i].nGrow()); // match ghost cell number with main function
        H_exchangefield_prev[i].define(convert(ba, IntVect::TheDimensionVector(i)), dm, 3, H_exchangefield[i].nGrow());
        H_DMIfield_old[i].define(convert(ba, IntVect::TheDimensionVector(i)), dm, 3, H_DMIfield[i].nGrow()); // match ghost cell number with main function
        H_DMIfield_prev[i].define(convert(ba, IntVect::TheDimensionVector(i)), dm, 3, H_DMIfield[i].nGrow());
        Mfield_old[i].define(convert(ba, IntVect::TheDimensionVector(i)), dm, 3, Mfield[i].nGrow()); // match ghost cell number with main function
        Mfield_prev[i].define(convert(ba, IntVect::TheDimensionVector(i)), dm, 3, Mfield[i].nGrow());
        
        MultiFab::Copy(H_demagfield_prev[i], H_demagfield[i], 0, 0, 1, H_demagfield[i].nGrow());
        MultiFab::Copy(H_demagfield_old[i], H_demagfield[i], 0, 0, 1, H_demagfield[i].nGrow());
        MultiFab::Copy(H_exchangefield_old[i], H_exchangefield[i], 0, 0, 3, H_exchangefield[i].nGrow());
        MultiFab::Copy(H_exchangefield_prev[i], H_exchangefield[i], 0, 0, 3, H_exchangefield[i].nGrow());
        MultiFab::Copy(H_DMIfield_old[i], H_DMIfield[i], 0, 0, 3, H_DMIfield[i].nGrow());
        MultiFab::Copy(H_DMIfield_prev[i], H_DMIfield[i], 0, 0, 3, H_DMIfield[i].nGrow());
        MultiFab::Copy(Mfield_old[i], Mfield[i], 0, 0, 3, Mfield[i].nGrow());
        MultiFab::Copy(Mfield_prev[i], Mfield[i], 0, 0, 3, Mfield[i].nGrow());
        
        // fill periodic ghost cells
        H_demagfield_old[i].FillBoundary(geom.periodicity());
        H_demagfield_prev[i].FillBoundary(geom.periodicity());
        H_exchangefield_old[i].FillBoundary(geom.periodicity());
        H_exchangefield_prev[i].FillBoundary(geom.periodicity());
        H_DMIfield_old[i].FillBoundary(geom.periodicity());
        H_DMIfield_prev[i].FillBoundary(geom.periodicity());
        Mfield_old[i].FillBoundary(geom.periodicity());
        Mfield_prev[i].FillBoundary(geom.periodicity());

        Mfield_error[i].define(convert(ba, IntVect::TheDimensionVector(i)), dm, 3, Mfield[i].nGrow());
        Mfield_error[i].setVal(0.); // reset Mfield_error to zero
        Mfield_error[i].FillBoundary(geom.periodicity());

        // initialize a_temp, a_temp_static, b_temp_static
        a_temp[i].define(convert(ba, IntVect::TheDimensionVector(i)), dm, 3, Mfield[i].nGrow());
        a_temp_static[i].define(convert(ba, IntVect::TheDimensionVector(i)), dm, 3, Mfield[i].nGrow());
        b_temp_static[i].define(convert(ba, IntVect::TheDimensionVector(i)), dm, 3, Mfield[i].nGrow());

    }
    
    // calculate the b_temp_static, a_temp_static
    for (MFIter mfi(a_temp_static[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.growntilebox(1); 

        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        const Array4<Real>& alpha_xface_arr = alpha[0].array(mfi);
        const Array4<Real>& alpha_yface_arr = alpha[1].array(mfi);
        const Array4<Real>& alpha_zface_arr = alpha[2].array(mfi);

        const Array4<Real>& gamma_xface_arr = gamma[0].array(mfi);
        const Array4<Real>& gamma_yface_arr = gamma[1].array(mfi);
        const Array4<Real>& gamma_zface_arr = gamma[2].array(mfi);

        const Array4<Real>& Ms_xface_arr = Ms[0].array(mfi);
        const Array4<Real>& Ms_yface_arr = Ms[1].array(mfi);
        const Array4<Real>& Ms_zface_arr = Ms[2].array(mfi);

        const Array4<Real>& exchange_xface_arr = exchange[0].array(mfi);
        const Array4<Real>& exchange_yface_arr = exchange[1].array(mfi);
        const Array4<Real>& exchange_zface_arr = exchange[2].array(mfi);

        const Array4<Real>& DMI_xface_arr = DMI[0].array(mfi);
        const Array4<Real>& DMI_yface_arr = DMI[1].array(mfi);
        const Array4<Real>& DMI_zface_arr = DMI[2].array(mfi);

        const Array4<Real>& anisotropy_xface_arr = anisotropy[0].array(mfi);
        const Array4<Real>& anisotropy_yface_arr = anisotropy[1].array(mfi);
        const Array4<Real>& anisotropy_zface_arr = anisotropy[2].array(mfi);

        // extract field data   
        const Array4<Real>& H_bias_xface = H_biasfield[0].array(mfi); // Hx_bias is the x component at |_x faces
        const Array4<Real>& H_bias_yface = H_biasfield[1].array(mfi); // Hy_bias is the y component at |_y faces
        const Array4<Real>& H_bias_zface = H_biasfield[2].array(mfi); // Hz_bias is the z component at |_z faces
        const Array4<Real>& Hx_demag = H_demagfield[0].array(mfi);   // Hx_old is the x component at |_x faces
        const Array4<Real>& Hy_demag = H_demagfield[1].array(mfi);   // Hy_old is the y component at |_y faces
        const Array4<Real>& Hz_demag = H_demagfield[2].array(mfi);   // Hz_old is the z component at |_z faces
        const Array4<Real>& Hx_demag_old = H_demagfield_old[0].array(mfi);   // Hx_old is the x component at |_x faces
        const Array4<Real>& Hy_demag_old = H_demagfield_old[1].array(mfi);   // Hy_old is the y component at |_y faces
        const Array4<Real>& Hz_demag_old = H_demagfield_old[2].array(mfi);   // Hz_old is the z component at |_z faces
        const Array4<Real>& H_exchange_xface_old = H_exchangefield_old[0].array(mfi); // note H_exchange_xface_old include x,y,z components at |_x faces
        const Array4<Real>& H_exchange_yface_old = H_exchangefield_old[1].array(mfi); // note H_exchange_yface_old include x,y,z components at |_y faces
        const Array4<Real>& H_exchange_zface_old = H_exchangefield_old[2].array(mfi); // note H_exchange_zface_old include x,y,z components at |_z faces
        const Array4<Real>& H_DMI_xface_old = H_DMIfield_old[0].array(mfi); // note H_DMI_xface_old include x,y,z components at |_x faces
        const Array4<Real>& H_DMI_yface_old = H_DMIfield_old[1].array(mfi); // note H_DMI_yface_old include x,y,z components at |_y faces
        const Array4<Real>& H_DMI_zface_old = H_DMIfield_old[2].array(mfi); // note H_DMI_zface_old include x,y,z components at |_z faces
        const Array4<Real>& M_xface_old = Mfield_old[0].array(mfi); // note M_xface include x,y,z components at |_x faces
        const Array4<Real>& M_yface_old = Mfield_old[1].array(mfi); // note M_yface include x,y,z components at |_y faces
        const Array4<Real>& M_zface_old = Mfield_old[2].array(mfi); // note M_zface include x,y,z components at |_z faces
        // const Array4<Real>& M_xface = Mfield[0].array(mfi); // note M_xface include x,y,z components at |_x faces
        // const Array4<Real>& M_yface = Mfield[1].array(mfi); // note M_yface include x,y,z components at |_y faces
        // const Array4<Real>& M_zface = Mfield[2].array(mfi); // note M_zface include x,y,z components at |_z faces

        // extract field data of a_temp_static and b_temp_static
        const Array4<Real>& a_temp_static_xface = a_temp_static[0].array(mfi);
        const Array4<Real>& a_temp_static_yface = a_temp_static[1].array(mfi);
        const Array4<Real>& a_temp_static_zface = a_temp_static[2].array(mfi);
        const Array4<Real>& b_temp_static_xface = b_temp_static[0].array(mfi);
        const Array4<Real>& b_temp_static_yface = b_temp_static[1].array(mfi);
        const Array4<Real>& b_temp_static_zface = b_temp_static[2].array(mfi);

        // extract tileboxes for which to loop
        amrex::IntVect Mxface_stag = Mfield[0].ixType().toIntVect();
        amrex::IntVect Myface_stag = Mfield[1].ixType().toIntVect();
        amrex::IntVect Mzface_stag = Mfield[2].ixType().toIntVect();
        // extract tileboxes for which to loop
        amrex::IntVect H_demag_stag = H_demagfield[0].ixType().toIntVect();
        // extract tileboxes for which to loop
        Box const &tbx = mfi.tilebox(Mfield_old[0].ixType().toIntVect());
        Box const &tby = mfi.tilebox(Mfield_old[1].ixType().toIntVect());
        Box const &tbz = mfi.tilebox(Mfield_old[2].ixType().toIntVect());

        // loop over cells and update fields
        amrex::ParallelFor(tbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                // determine if the material is nonmagnetic or not
                if (Ms_xface_arr(i,j,k) > 0.){

                    // when working on M_xface(i,j,k, 0:2) we have direct access to M_xface(i,j,k,0:2) and Hx(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    amrex::Real Hx_eff_old = H_bias_xface(i,j,k,0);
                    amrex::Real Hy_eff_old = H_bias_xface(i,j,k,1);
                    amrex::Real Hz_eff_old = H_bias_xface(i,j,k,2);

                    if (demag_coupling == 1){
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                        // H_maxwell - use H^(old_time)
                        Hx_eff_old += cc_avg_to_face(i, j, k, 0, H_demag_stag, Mxface_stag, Hx_demag_old);
                        Hy_eff_old += cc_avg_to_face(i, j, k, 0, H_demag_stag, Mxface_stag, Hy_demag_old);
                        Hz_eff_old += cc_avg_to_face(i, j, k, 0, H_demag_stag, Mxface_stag, Hz_demag_old);
                    }

                    if (exchange_coupling == 1){

                        if (exchange_xface_arr(i,j,k) == 0.) amrex::Abort("The exchange_xface_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                        // H_exchange - use M^(old_time)

                        Hx_eff_old += H_exchange_xface_old(i, j, k, 0);
                        Hy_eff_old += H_exchange_xface_old(i, j, k, 1);
                        Hz_eff_old += H_exchange_xface_old(i, j, k, 2);

                    }

                    if (DMI_coupling == 1){

                        if (DMI_xface_arr(i,j,k) == 0.) amrex::Abort("The DMI_xface_arr(i,j,k) is 0.0 while including the DMI coupling term H_DMI for H_eff");
                        if (exchange_xface_arr(i,j,k) == 0.) amrex::Abort("The exchange_xface_arr(i,j,k) is 0.0 while DMI is turned on");

                        // H_DMI - use M^(old_time)

                        Hx_eff_old += H_DMI_xface_old(i, j, k, 0);
                        Hy_eff_old += H_DMI_xface_old(i, j, k, 1);
                        Hz_eff_old += H_DMI_xface_old(i, j, k, 2);

                    }

                    if (anisotropy_coupling == 1){

                        if (anisotropy_xface_arr(i,j,k) == 0.) amrex::Abort("The anisotropy_xface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                        // H_anisotropy - use M^(old_time)
                        amrex::Real M_dot_anisotropy_axis = 0.0;
                        for (int comp=0; comp<3; ++comp) {
                            M_dot_anisotropy_axis += M_xface_old(i, j, k, comp) * anisotropy_axis[comp];
                        }
                        amrex::Real const H_anisotropy_coeff = - 2.0 * anisotropy_xface_arr(i,j,k) / mu0 / Ms_xface_arr(i,j,k) / Ms_xface_arr(i,j,k);
                        Hx_eff_old += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                        Hy_eff_old += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                        Hz_eff_old += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                    }

                    // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                    amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_xface_old(i, j, k, 0), 2.) + std::pow(M_xface_old(i, j, k, 1), 2.) + std::pow(M_xface_old(i, j, k, 2), 2.))
                                                              : Ms_xface_arr(i,j,k);
                    // a_temp_static_coeff does not change in the current step for SATURATED materials; but it does change for UNSATURATED ones
                    amrex::Real a_temp_static_coeff = alpha_xface_arr(i,j,k) / M_magnitude;

                    // calculate the b_temp_static_coeff (it is divided by 2.0 because the derivation is based on an interger dt)
                    amrex::Real b_temp_static_coeff = - mu0 * amrex::Math::abs(gamma_xface_arr(i,j,k)) / 2.;

                    for (int comp=0; comp<3; ++comp) {
                        // calculate a_temp_static_xface
                        // all components on x-faces of grid
                        a_temp_static_xface(i, j, k, comp) = a_temp_static_coeff * M_xface_old(i, j, k, comp);
                    }

                    // calculate b_temp_static_xface
                    // x component on x-faces of grid
                    b_temp_static_xface(i, j, k, 0) = M_xface_old(i, j, k, 0) + dt * b_temp_static_coeff * (M_xface_old(i, j, k, 1) * Hz_eff_old - M_xface_old(i, j, k, 2) * Hy_eff_old);

                    // y component on x-faces of grid
                    b_temp_static_xface(i, j, k, 1) = M_xface_old(i, j, k, 1) + dt * b_temp_static_coeff * (M_xface_old(i, j, k, 2) * Hx_eff_old - M_xface_old(i, j, k, 0) * Hz_eff_old);

                    // z component on x-faces of grid
                    b_temp_static_xface(i, j, k, 2) = M_xface_old(i, j, k, 2) + dt * b_temp_static_coeff * (M_xface_old(i, j, k, 0) * Hy_eff_old - M_xface_old(i, j, k, 1) * Hx_eff_old);
                }
            });

        amrex::ParallelFor(tby,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                // determine if the material is nonmagnetic or not
                if (Ms_yface_arr(i,j,k) > 0.){

                    // when working on M_yface(i,j,k,0:2) we have direct access to M_yface(i,j,k,0:2) and Hy(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    amrex::Real Hx_eff_old = H_bias_yface(i,j,k,0);
                    amrex::Real Hy_eff_old = H_bias_yface(i,j,k,1);
                    amrex::Real Hz_eff_old = H_bias_yface(i,j,k,2);

                    if (demag_coupling == 1){
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                        // H_maxwell - use H^(old_time)
                        Hx_eff_old += cc_avg_to_face(i, j, k, 0, H_demag_stag, Myface_stag, Hx_demag_old);
                        Hy_eff_old += cc_avg_to_face(i, j, k, 0, H_demag_stag, Myface_stag, Hy_demag_old);
                        Hz_eff_old += cc_avg_to_face(i, j, k, 0, H_demag_stag, Myface_stag, Hz_demag_old);
                    }

                    if (exchange_coupling == 1){

                        if (exchange_yface_arr(i,j,k) == 0.) amrex::Abort("The exchange_yface_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");
                        
                        // H_exchange - use M^(old_time)
                        Hx_eff_old += H_exchange_yface_old(i, j, k, 0);
                        Hy_eff_old += H_exchange_yface_old(i, j, k, 1);
                        Hz_eff_old += H_exchange_yface_old(i, j, k, 2);
                    }

                    if (DMI_coupling == 1){

                        if (DMI_yface_arr(i,j,k) == 0.) amrex::Abort("The DMI_yface_arr(i,j,k) is 0.0 while including the DMI coupling term H_DMI for H_eff");
                        if (exchange_yface_arr(i,j,k) == 0.) amrex::Abort("The exchange_yface_arr(i,j,k) is 0.0 while DMI is turned on");

                        // H_DMI - use M^(old_time)
                        Hx_eff_old += H_DMI_yface_old(i, j, k, 0);
                        Hy_eff_old += H_DMI_yface_old(i, j, k, 1);
                        Hz_eff_old += H_DMI_yface_old(i, j, k, 2);
                    }

                    if (anisotropy_coupling == 1){

                        if (anisotropy_yface_arr(i,j,k) == 0.) amrex::Abort("The anisotropy_yface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                        // H_anisotropy - use M^(old_time)
                        amrex::Real M_dot_anisotropy_axis = 0.0;
                        for (int comp=0; comp<3; ++comp) {
                            M_dot_anisotropy_axis += M_yface_old(i, j, k, comp) * anisotropy_axis[comp];
                        }
                        amrex::Real const H_anisotropy_coeff = - 2.0 * anisotropy_yface_arr(i,j,k) / mu0 / Ms_yface_arr(i,j,k) / Ms_yface_arr(i,j,k);
                        Hx_eff_old += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                        Hy_eff_old += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                        Hz_eff_old += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                    }

                    // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                    // note the unsaturated case is less usefull in real devices
                    amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_yface_old(i, j, k, 0), 2.) + std::pow(M_yface_old(i, j, k, 1), 2.) + std::pow(M_yface_old(i, j, k, 2), 2.))
                                                              : Ms_yface_arr(i,j,k);
                    amrex::Real a_temp_static_coeff = alpha_yface_arr(i,j,k) / M_magnitude;

                    // calculate the b_temp_static_coeff (it is divided by 2.0 because the derivation is based on an interger dt)
                    amrex::Real b_temp_static_coeff = - mu0 * amrex::Math::abs(gamma_yface_arr(i,j,k)) / 2.;

                    for (int comp=0; comp<3; ++comp) {
                        // calculate a_temp_static_yface
                        // all component on y-faces of grid
                        a_temp_static_yface(i, j, k, comp) = a_temp_static_coeff * M_yface_old(i, j, k, comp);
                    }

                    // calculate b_temp_static_yface
                    // x component on y-faces of grid
                    b_temp_static_yface(i, j, k, 0) = M_yface_old(i, j, k, 0) + dt * b_temp_static_coeff * (M_yface_old(i, j, k, 1) * Hz_eff_old - M_yface_old(i, j, k, 2) * Hy_eff_old);

                    // y component on y-faces of grid
                    b_temp_static_yface(i, j, k, 1) = M_yface_old(i, j, k, 1) + dt * b_temp_static_coeff * (M_yface_old(i, j, k, 2) * Hx_eff_old - M_yface_old(i, j, k, 0) * Hz_eff_old);

                    // z component on y-faces of grid
                    b_temp_static_yface(i, j, k, 2) = M_yface_old(i, j, k, 2) + dt * b_temp_static_coeff * (M_yface_old(i, j, k, 0) * Hy_eff_old - M_yface_old(i, j, k, 1) * Hx_eff_old);
                }
            });

        amrex::ParallelFor(tbz,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                // determine if the material is nonmagnetic or not
                if (Ms_zface_arr(i,j,k) > 0.){

                    // when working on M_zface(i,j,k,0:2) we have direct access to M_zface(i,j,k,0:2) and Hz(i,j,k)
                    // Hy and Hz can be acquired by interpolation

                    // H_bias
                    amrex::Real Hx_eff_old = H_bias_zface(i,j,k,0);
                    amrex::Real Hy_eff_old = H_bias_zface(i,j,k,1);
                    amrex::Real Hz_eff_old = H_bias_zface(i,j,k,2);

                    if (demag_coupling == 1){
                        // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                        // H_maxwell - use H^(old_time)
                        Hx_eff_old += cc_avg_to_face(i, j, k, 0, H_demag_stag, Mzface_stag, Hx_demag_old);
                        Hy_eff_old += cc_avg_to_face(i, j, k, 0, H_demag_stag, Mzface_stag, Hy_demag_old);
                        Hz_eff_old += cc_avg_to_face(i, j, k, 0, H_demag_stag, Mzface_stag, Hz_demag_old);
                    }

                    if (exchange_coupling == 1){

                        if (exchange_zface_arr(i,j,k) == 0.) amrex::Abort("The exchange_zface_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                        // H_exchange - use M^(old_time)

                        Hx_eff_old += H_exchange_zface_old(i, j, k, 0);
                        Hy_eff_old += H_exchange_zface_old(i, j, k, 1);
                        Hz_eff_old += H_exchange_zface_old(i, j, k, 2);

                    }

                    if (DMI_coupling == 1){

                        if (DMI_zface_arr(i,j,k) == 0.) amrex::Abort("The DMI_zface_arr(i,j,k) is 0.0 while including the DMI coupling term H_DMI for H_eff");
                        if (exchange_zface_arr(i,j,k) == 0.) amrex::Abort("The exchange_zface_arr(i,j,k) is 0.0 while DMI is turned on");

                        // H_DMI - use M^(old_time)
                        Hx_eff_old += H_DMI_zface_old(i, j, k, 0);
                        Hy_eff_old += H_DMI_zface_old(i, j, k, 1);
                        Hz_eff_old += H_DMI_zface_old(i, j, k, 2);

                    }

                    if (anisotropy_coupling == 1){

                        if (anisotropy_zface_arr(i,j,k) == 0.) amrex::Abort("The anisotropy_zface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                        // H_anisotropy - use M^(old_time)
                        amrex::Real M_dot_anisotropy_axis = 0.0;
                        for (int comp=0; comp<3; ++comp) {
                            M_dot_anisotropy_axis += M_zface_old(i, j, k, comp) * anisotropy_axis[comp];
                        }
                        amrex::Real const H_anisotropy_coeff = - 2.0 * anisotropy_zface_arr(i,j,k) / mu0 / Ms_zface_arr(i,j,k) / Ms_zface_arr(i,j,k);
                        Hx_eff_old += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                        Hy_eff_old += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                        Hz_eff_old += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                    }

                    // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                    amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_zface_old(i, j, k, 0), 2.) + std::pow(M_zface_old(i, j, k, 1), 2.) + std::pow(M_zface_old(i, j, k, 2), 2.))
                                                              : Ms_zface_arr(i,j,k);
                    amrex::Real a_temp_static_coeff = alpha_zface_arr(i,j,k) / M_magnitude;

                    // calculate the b_temp_static_coeff (it is divided by 2.0 because the derivation is based on an interger dt,
                    // while in real simulations, the input dt is actually dt/2.0)
                    amrex::Real b_temp_static_coeff = - mu0 * amrex::Math::abs(gamma_zface_arr(i,j,k)) / 2.;

                    for (int comp=0; comp<3; ++comp) {
                        // calculate a_temp_static_zface
                        // all components on z-faces of grid
                        a_temp_static_zface(i, j, k, comp) = a_temp_static_coeff * M_zface_old(i, j, k, comp);
                    }

                    // calculate b_temp_static_zface
                    // x component on z-faces of grid
                    b_temp_static_zface(i, j, k, 0) = M_zface_old(i, j, k, 0) + dt * b_temp_static_coeff * (M_zface_old(i, j, k, 1) * Hz_eff_old - M_zface_old(i, j, k, 2) * Hy_eff_old);

                    // y component on z-faces of grid
                    b_temp_static_zface(i, j, k, 1) = M_zface_old(i, j, k, 1) + dt * b_temp_static_coeff * (M_zface_old(i, j, k, 2) * Hx_eff_old - M_zface_old(i, j, k, 0) * Hz_eff_old);

                    // z component on z-faces of grid
                    b_temp_static_zface(i, j, k, 2) = M_zface_old(i, j, k, 2) + dt * b_temp_static_coeff * (M_zface_old(i, j, k, 0) * Hy_eff_old - M_zface_old(i, j, k, 1) * Hx_eff_old);
                }
            });
    }

    // initialize M_max_iter, M_iter, M_tol, M_iter_error
    // maximum number of iterations allowed
    int M_max_iter = 100;
    int M_iter = 0;
    // relative tolerance stopping criteria for 2nd-order iterative algorithm
    amrex::Real M_tol = 1.e-6;
    int stop_iter = 0;

    // begin the iteration
    while (!stop_iter){

        for (MFIter mfi(Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi){

            const Box& bx = mfi.growntilebox(1); 

            // extract dx from the geometry object
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

            const Array4<Real>& alpha_xface_arr = alpha[0].array(mfi);
            const Array4<Real>& alpha_yface_arr = alpha[1].array(mfi);
            const Array4<Real>& alpha_zface_arr = alpha[2].array(mfi);

            const Array4<Real>& gamma_xface_arr = gamma[0].array(mfi);
            const Array4<Real>& gamma_yface_arr = gamma[1].array(mfi);
            const Array4<Real>& gamma_zface_arr = gamma[2].array(mfi);

            const Array4<Real>& Ms_xface_arr = Ms[0].array(mfi);
            const Array4<Real>& Ms_yface_arr = Ms[1].array(mfi);
            const Array4<Real>& Ms_zface_arr = Ms[2].array(mfi);

            const Array4<Real>& exchange_xface_arr = exchange[0].array(mfi);
            const Array4<Real>& exchange_yface_arr = exchange[1].array(mfi);
            const Array4<Real>& exchange_zface_arr = exchange[2].array(mfi);

            const Array4<Real>& DMI_xface_arr = DMI[0].array(mfi);
            const Array4<Real>& DMI_yface_arr = DMI[1].array(mfi);
            const Array4<Real>& DMI_zface_arr = DMI[2].array(mfi);

            const Array4<Real>& anisotropy_xface_arr = anisotropy[0].array(mfi);
            const Array4<Real>& anisotropy_yface_arr = anisotropy[1].array(mfi);
            const Array4<Real>& anisotropy_zface_arr = anisotropy[2].array(mfi);

            // extract field data
            const Array4<Real>& M_xface = Mfield[0].array(mfi);      // note M_xface include x,y,z components at |_x faces
            const Array4<Real>& M_yface = Mfield[1].array(mfi);      // note M_yface include x,y,z components at |_y faces
            const Array4<Real>& M_zface = Mfield[2].array(mfi);      // note M_zface include x,y,z components at |_z faces
            const Array4<Real>& H_bias_xface = H_biasfield[0].array(mfi); // Hx_bias is the x component at |_x faces
            const Array4<Real>& H_bias_yface = H_biasfield[1].array(mfi); // Hy_bias is the y component at |_y faces
            const Array4<Real>& H_bias_zface = H_biasfield[2].array(mfi); // Hz_bias is the z component at |_z faces
            const Array4<Real>& Hx_demag_prev = H_demagfield_prev[0].array(mfi);           // Hx is the x component at |_x faces
            const Array4<Real>& Hy_demag_prev = H_demagfield_prev[1].array(mfi);           // Hy is the y component at |_y faces
            const Array4<Real>& Hz_demag_prev = H_demagfield_prev[2].array(mfi);           // Hz is the z component at |_z faces
            const Array4<Real>& H_exchange_xface_prev = H_exchangefield_prev[0].array(mfi);           // H_exchange_xface include x,y,z component at |_x faces
            const Array4<Real>& H_exchange_yface_prev = H_exchangefield_prev[1].array(mfi);           // H_exchange_yface include x,y,z component at |_y faces
            const Array4<Real>& H_exchange_zface_prev = H_exchangefield_prev[2].array(mfi);           // H_exchange_zface include x,y,z component at |_z faces
            const Array4<Real>& H_DMI_xface_prev = H_DMIfield_prev[0].array(mfi);           // H_DMI_xface include x,y,z component at |_x faces
            const Array4<Real>& H_DMI_yface_prev = H_DMIfield_prev[1].array(mfi);           // H_DMI_yface include x,y,z component at |_y faces
            const Array4<Real>& H_DMI_zface_prev = H_DMIfield_prev[2].array(mfi);           // H_DMI_zface include x,y,z component at |_z faces

            // extract field data of Mfield_prev, Mfield_error, a_temp, a_temp_static, and b_temp_static
            const Array4<Real>& M_xface_prev = Mfield_prev[0].array(mfi);
            const Array4<Real>& M_yface_prev = Mfield_prev[1].array(mfi);
            const Array4<Real>& M_zface_prev = Mfield_prev[2].array(mfi);
            const Array4<Real>& M_xface_old = Mfield_old[0].array(mfi); // note M_xface include x,y,z components at |_x faces
            const Array4<Real>& M_yface_old = Mfield_old[1].array(mfi); // note M_yface include x,y,z components at |_y faces
            const Array4<Real>& M_zface_old = Mfield_old[2].array(mfi); // note M_zface include x,y,z components at |_z faces
            const Array4<Real>& M_xface_error = Mfield_error[0].array(mfi);
            const Array4<Real>& M_yface_error = Mfield_error[1].array(mfi);
            const Array4<Real>& M_zface_error = Mfield_error[2].array(mfi);
            const Array4<Real>& a_temp_xface = a_temp[0].array(mfi);
            const Array4<Real>& a_temp_yface = a_temp[1].array(mfi);
            const Array4<Real>& a_temp_zface = a_temp[2].array(mfi);
            const Array4<Real>& a_temp_static_xface = a_temp_static[0].array(mfi);
            const Array4<Real>& a_temp_static_yface = a_temp_static[1].array(mfi);
            const Array4<Real>& a_temp_static_zface = a_temp_static[2].array(mfi);
            const Array4<Real>& b_temp_static_xface = b_temp_static[0].array(mfi);
            const Array4<Real>& b_temp_static_yface = b_temp_static[1].array(mfi);
            const Array4<Real>& b_temp_static_zface = b_temp_static[2].array(mfi);

            // extract tileboxes for which to loop
            amrex::IntVect H_demag_stag = H_demagfield_prev[0].ixType().toIntVect();
            // extract tileboxes for which to loop
            amrex::IntVect Mxface_stag = Mfield[0].ixType().toIntVect();
            amrex::IntVect Myface_stag = Mfield[1].ixType().toIntVect();
            amrex::IntVect Mzface_stag = Mfield[2].ixType().toIntVect();
            // extract tileboxes for which to loop
            Box const &tbx = mfi.tilebox(Mfield[0].ixType().toIntVect());
            Box const &tby = mfi.tilebox(Mfield[1].ixType().toIntVect());
            Box const &tbz = mfi.tilebox(Mfield[2].ixType().toIntVect());

            // loop over cells and update fields
            amrex::ParallelFor(tbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                    // determine if the material is nonmagnetic or not
                    if (Ms_xface_arr(i,j,k) > 0.){

                        // when working on M_xface(i,j,k, 0:2) we have direct access to M_xface(i,j,k,0:2) and Hx(i,j,k)
                        // Hy and Hz can be acquired by interpolation

                        // H_bias
                        amrex::Real Hx_eff_prev = H_bias_xface(i,j,k,0);
                        amrex::Real Hy_eff_prev = H_bias_xface(i,j,k,1);
                        amrex::Real Hz_eff_prev = H_bias_xface(i,j,k,2);

                        if (demag_coupling == 1){
                            // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                            // H_maxwell - use H^[(new_time),r-1]
                            Hx_eff_prev += cc_avg_to_face(i, j, k, 0, H_demag_stag, Mxface_stag, Hx_demag_prev);
                            Hy_eff_prev += cc_avg_to_face(i, j, k, 0, H_demag_stag, Mxface_stag, Hy_demag_prev);
                            Hz_eff_prev += cc_avg_to_face(i, j, k, 0, H_demag_stag, Mxface_stag, Hz_demag_prev);
                        }

                        if (exchange_coupling == 1){

                            if (exchange_xface_arr(i,j,k) == 0.) amrex::Abort("The exchange_xface_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                            // H_exchange - use M^[(new_time),r-1]
                            
                            Hx_eff_prev += H_exchange_xface_prev(i, j, k, 0);
                            Hy_eff_prev += H_exchange_xface_prev(i, j, k, 1);
                            Hz_eff_prev += H_exchange_xface_prev(i, j, k, 2);
                        }

                        if (DMI_coupling == 1){

                            if (DMI_xface_arr(i,j,k) == 0.) amrex::Abort("The DMI_xface_arr(i,j,k) is 0.0 while including the DMI coupling term H_DMI for H_eff");
                            if (exchange_xface_arr(i,j,k) == 0.) amrex::Abort("The exchange_xface_arr(i,j,k) is 0.0 while DMI is turned on");

                            // H_DMI - use M^[(new_time),r-1]

                            Hx_eff_prev += H_DMI_xface_prev(i,j,k,0);
                            Hy_eff_prev += H_DMI_xface_prev(i,j,k,1);
                            Hz_eff_prev += H_DMI_xface_prev(i,j,k,2);
                        }

                        if (anisotropy_coupling == 1){

                            if (anisotropy_xface_arr(i,j,k) == 0.) amrex::Abort("The anisotropy_xface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                            // H_anisotropy - use M^[(new_time),r-1]
                            amrex::Real M_dot_anisotropy_axis = 0.0;
                            for (int comp=0; comp<3; ++comp) {
                                M_dot_anisotropy_axis += M_xface_prev(i, j, k, comp) * anisotropy_axis[comp];
                            }
                            amrex::Real const H_anisotropy_coeff = - 2.0 * anisotropy_xface_arr(i,j,k) / mu0 / Ms_xface_arr(i,j,k) / Ms_xface_arr(i,j,k);
                            Hx_eff_prev += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                            Hy_eff_prev += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                            Hz_eff_prev += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                        }

                        // calculate the a_temp_dynamic_coeff (it is divided by 2.0 because the derivation is based on an interger dt,
                        // while in real simulations, the input dt is actually dt/2.0)
                        amrex::Real a_temp_dynamic_coeff = mu0 * amrex::Math::abs(gamma_xface_arr(i,j,k)) / 2.;

                        amrex::GpuArray<amrex::Real,3> H_eff_prev;
                        H_eff_prev[0] = Hx_eff_prev;
                        H_eff_prev[1] = Hy_eff_prev;
                        H_eff_prev[2] = Hz_eff_prev;

                        for (int comp=0; comp<3; ++comp) {
                            // calculate a_temp_xface
                            // all components on x-faces of grid
                            a_temp_xface(i, j, k, comp) = (M_normalization != 0) ? -(dt * a_temp_dynamic_coeff * H_eff_prev[comp] + a_temp_static_xface(i, j, k, comp))
                                                                                 : -(dt * a_temp_dynamic_coeff * H_eff_prev[comp] + 0.5 * a_temp_static_xface(i, j, k, comp)
                                                                                     + 0.5 * alpha_xface_arr(i,j,k) * 1. / std::sqrt(std::pow(M_xface(i, j, k, 0), 2.) + std::pow(M_xface(i, j, k, 1), 2.) + std::pow(M_xface(i, j, k, 2), 2.)) * M_xface_old(i, j, k, comp));
                        }

                        for (int comp=0; comp<3; ++comp) {
                            // update M_xface from a and b using the updateM_field
                            // all components on x-faces of grid
                            M_xface(i, j, k, comp) = updateM_field(i, j, k, comp, a_temp_xface, b_temp_static_xface);
                        }

                        // temporary normalized magnitude of M_xface field at the fixed point
                        // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                        amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_xface(i, j, k, 0), 2.) + std::pow(M_xface(i, j, k, 1), 2.) + std::pow(M_xface(i, j, k, 2), 2.)) / Ms_xface_arr(i,j,k);
                        amrex::Real normalized_error = 0.1;

                        if (M_normalization == 1){
                            // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                            // check the normalized error
                            if (amrex::Math::abs(1. - M_magnitude_normalized) > normalized_error){
                                amrex::Abort("Exceed the normalized error of the M_xface field");
                            }
                            // normalize the M_xface field
                            M_xface(i, j, k, 0) /= M_magnitude_normalized;
                            M_xface(i, j, k, 1) /= M_magnitude_normalized;
                            M_xface(i, j, k, 2) /= M_magnitude_normalized;
                        }
                        else if (M_normalization == 0){
                            // check the normalized error
                            if (M_magnitude_normalized > (1. + normalized_error)){
                                amrex::Abort("Caution: Unsaturated material has M_xface exceeding the saturation magnetization");
                            }
                            else if (M_magnitude_normalized > 1. && M_magnitude_normalized <= 1. + normalized_error){
                                // normalize the M_xface field
                                M_xface(i, j, k, 0) /= M_magnitude_normalized;
                                M_xface(i, j, k, 1) /= M_magnitude_normalized;
                                M_xface(i, j, k, 2) /= M_magnitude_normalized;
                            }
                        }

                        // calculate M_xface_error
                        // x,y,z component on M-error on x-faces of grid
                        for (int icomp = 0; icomp < 3; ++icomp) {
                            M_xface_error(i, j, k, icomp) = amrex::Math::abs((M_xface(i, j, k, icomp) - M_xface_prev(i, j, k, icomp))) / Ms_xface_arr(i,j,k);
                        }
                    }
                });

            amrex::ParallelFor(tby,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                    // determine if the material is nonmagnetic or not
                    if (Ms_yface_arr(i,j,k) > 0.){

                        // when working on M_yface(i,j,k,0:2) we have direct access to M_yface(i,j,k,0:2) and Hy(i,j,k)
                        // Hy and Hz can be acquired by interpolation

                        // H_bias
                        amrex::Real Hx_eff_prev = H_bias_yface(i,j,k,0);
                        amrex::Real Hy_eff_prev = H_bias_yface(i,j,k,1);
                        amrex::Real Hz_eff_prev = H_bias_yface(i,j,k,2);

                        if (demag_coupling == 1){
                            // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                            // H_maxwell - use H^[(new_time),r-1]
                            Hx_eff_prev += cc_avg_to_face(i, j, k, 0, H_demag_stag, Myface_stag, Hx_demag_prev);
                            Hy_eff_prev += cc_avg_to_face(i, j, k, 0, H_demag_stag, Myface_stag, Hy_demag_prev);
                            Hz_eff_prev += cc_avg_to_face(i, j, k, 0, H_demag_stag, Myface_stag, Hz_demag_prev);
                        }

                        if (exchange_coupling == 1){

                            if (exchange_yface_arr(i,j,k) == 0.) amrex::Abort("The exchange_yface_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                            // H_exchange - use M^[(new_time),r-1]
                            
                            Hx_eff_prev += H_exchange_yface_prev(i, j, k, 0);
                            Hy_eff_prev += H_exchange_yface_prev(i, j, k, 1);
                            Hz_eff_prev += H_exchange_yface_prev(i, j, k, 2);
                        }

                        if (DMI_coupling == 1){

                            if (DMI_yface_arr(i,j,k) == 0.) amrex::Abort("The DMI_yface_arr(i,j,k) is 0.0 while including the DMI coupling term H_DMI for H_eff");
                            if (exchange_yface_arr(i,j,k) == 0.) amrex::Abort("The exchange_yface_arr(i,j,k) is 0.0 while DMI is turned on");

                            // H_DMI - use M^[(new_time),r-1]
                            Hx_eff_prev += H_DMI_yface_prev(i,j,k,0);
                            Hy_eff_prev += H_DMI_yface_prev(i,j,k,1);
                            Hz_eff_prev += H_DMI_yface_prev(i,j,k,2);
                        }

                        if (anisotropy_coupling == 1){

                            if (anisotropy_yface_arr(i,j,k) == 0.) amrex::Abort("The anisotropy_yface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                            // H_anisotropy - use M^[(new_time),r-1]
                            amrex::Real M_dot_anisotropy_axis = 0.0;
                            for (int comp=0; comp<3; ++comp) {
                                M_dot_anisotropy_axis += M_yface_prev(i, j, k, comp) * anisotropy_axis[comp];
                            }
                            amrex::Real const H_anisotropy_coeff = - 2.0 * anisotropy_yface_arr(i,j,k) / mu0 / Ms_yface_arr(i,j,k) / Ms_yface_arr(i,j,k);
                            Hx_eff_prev += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                            Hy_eff_prev += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                            Hz_eff_prev += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                        }

                        // calculate the a_temp_dynamic_coeff (it is divided by 2.0 because the derivation is based on an interger dt,
                        // while in real simulations, the input dt is actually dt/2.0)
                        amrex::Real a_temp_dynamic_coeff = mu0 * amrex::Math::abs(gamma_yface_arr(i,j,k)) / 2.;

                        amrex::GpuArray<amrex::Real,3> H_eff_prev;
                        H_eff_prev[0] = Hx_eff_prev;
                        H_eff_prev[1] = Hy_eff_prev;
                        H_eff_prev[2] = Hz_eff_prev;

                        for (int comp=0; comp<3; ++comp) {
                            // calculate a_temp_yface
                            // all components on y-faces of grid
                            a_temp_yface(i, j, k, comp) = (M_normalization != 0) ? -(dt * a_temp_dynamic_coeff * H_eff_prev[comp] + a_temp_static_yface(i, j, k, comp))
                                                                                 : -(dt * a_temp_dynamic_coeff * H_eff_prev[comp] + 0.5 * a_temp_static_yface(i, j, k, comp)
                                                                                     + 0.5 * alpha_yface_arr(i,j,k) * 1. / std::sqrt(std::pow(M_yface(i, j, k, 0), 2.) + std::pow(M_yface(i, j, k, 1), 2.) + std::pow(M_yface(i, j, k, 2), 2.)) * M_yface_old(i, j, k, comp));
                        }

                        for (int comp=0; comp<3; ++comp) {
                            // update M_yface from a and b using the updateM_field
                            // all components on y-faces of grid
                            M_yface(i, j, k, comp) = updateM_field(i, j, k, comp, a_temp_yface, b_temp_static_yface);
                        }

                        // temporary normalized magnitude of M_yface field at the fixed point
                        // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                        amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_yface(i, j, k, 0), 2.) + std::pow(M_yface(i, j, k, 1), 2.) + std::pow(M_yface(i, j, k, 2), 2.)) / Ms_yface_arr(i,j,k);
                        amrex::Real normalized_error = 0.1;

                        if (M_normalization == 1){
                            // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                            // check the normalized error
                            if (amrex::Math::abs(1. - M_magnitude_normalized) > normalized_error){
                                amrex::Abort("Exceed the normalized error of the M_yface field");
                            }
                            // normalize the M_yface field
                            M_yface(i, j, k, 0) /= M_magnitude_normalized;
                            M_yface(i, j, k, 1) /= M_magnitude_normalized;
                            M_yface(i, j, k, 2) /= M_magnitude_normalized;
                        }
                        else if (M_normalization == 0){
                            // check the normalized error
                            if (M_magnitude_normalized > 1. + normalized_error){
                                amrex::Abort("Caution: Unsaturated material has M_yface exceeding the saturation magnetization");
                            }
                            else if (M_magnitude_normalized > 1. && M_magnitude_normalized <= 1. + normalized_error){
                                // normalize the M_yface field
                                M_yface(i, j, k, 0) /= M_magnitude_normalized;
                                M_yface(i, j, k, 1) /= M_magnitude_normalized;
                                M_yface(i, j, k, 2) /= M_magnitude_normalized;
                            }
                        }

                        // calculate M_yface_error
                        // x,y,z component on y-faces of grid
                        for (int icomp = 0; icomp < 3; ++icomp) {
                            M_yface_error(i, j, k, icomp) = amrex::Math::abs((M_yface(i, j, k, icomp) - M_yface_prev(i, j, k, icomp))) / Ms_yface_arr(i,j,k);
                        }
                    }
                });

            amrex::ParallelFor(tbz,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                    // determine if the material is nonmagnetic or not
                    if (Ms_zface_arr(i,j,k) > 0.){

                        // when working on M_zface(i,j,k,0:2) we have direct access to M_zface(i,j,k,0:2) and Hz(i,j,k)
                        // Hy and Hz can be acquired by interpolation

                        // H_bias
                        amrex::Real Hx_eff_prev = H_bias_zface(i,j,k,0);
                        amrex::Real Hy_eff_prev = H_bias_zface(i,j,k,1);
                        amrex::Real Hz_eff_prev = H_bias_zface(i,j,k,2);

                        if (demag_coupling == 1){
                            // H_eff = H_maxwell + H_bias + H_exchange + H_anisotropy

                            // H_maxwell - use H^[(new_time),r-1]
                            Hx_eff_prev += cc_avg_to_face(i, j, k, 0, H_demag_stag, Mzface_stag, Hx_demag_prev);
                            Hy_eff_prev += cc_avg_to_face(i, j, k, 0, H_demag_stag, Mzface_stag, Hy_demag_prev);
                            Hz_eff_prev += cc_avg_to_face(i, j, k, 0, H_demag_stag, Mzface_stag, Hz_demag_prev);
                        }

                        if (exchange_coupling == 1){

                            if (exchange_zface_arr(i,j,k) == 0.) amrex::Abort("The exchange_zface_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                            // H_exchange - use M^[(new_time),r-1]
                            
                            Hx_eff_prev += H_exchange_zface_prev(i, j, k, 0);
                            Hy_eff_prev += H_exchange_zface_prev(i, j, k, 1);
                            Hz_eff_prev += H_exchange_zface_prev(i, j, k, 2);
                        }

                        if (DMI_coupling == 1){

                            if (DMI_zface_arr(i,j,k) == 0.) amrex::Abort("The DMI_zface_arr(i,j,k) is 0.0 while including the DMI coupling term H_DMI for H_eff");
                            if (exchange_zface_arr(i,j,k) == 0.) amrex::Abort("The exchange_zface_arr(i,j,k) is 0.0 while DMI is turned on");

                            // H_DMI - use M^[(new_time),r-1]
                            Hx_eff_prev += H_DMI_zface_prev(i,j,k,0);
                            Hy_eff_prev += H_DMI_zface_prev(i,j,k,1);
                            Hz_eff_prev += H_DMI_zface_prev(i,j,k,2);
                        }

                        if (anisotropy_coupling == 1){

                            if (anisotropy_zface_arr(i,j,k) == 0.) amrex::Abort("The anisotropy_zface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                            // H_anisotropy - use M^[(new_time),r-1]
                            amrex::Real M_dot_anisotropy_axis = 0.0;
                            for (int comp=0; comp<3; ++comp) {
                                M_dot_anisotropy_axis += M_zface_prev(i, j, k, comp) * anisotropy_axis[comp];
                            }
                            amrex::Real const H_anisotropy_coeff = - 2.0 * anisotropy_zface_arr(i,j,k) / mu0 / Ms_zface_arr(i,j,k) / Ms_zface_arr(i,j,k);
                            Hx_eff_prev += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                            Hy_eff_prev += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                            Hz_eff_prev += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];
                        }

                        // calculate the a_temp_dynamic_coeff (it is divided by 2.0 because the derivation is based on an interger dt,
                        // while in real simulations, the input dt is actually dt/2.0)
                        amrex::Real a_temp_dynamic_coeff = mu0 * amrex::Math::abs(gamma_zface_arr(i,j,k)) / 2.;

                        amrex::GpuArray<amrex::Real,3> H_eff_prev;
                        H_eff_prev[0] = Hx_eff_prev;
                        H_eff_prev[1] = Hy_eff_prev;
                        H_eff_prev[2] = Hz_eff_prev;

                        for (int comp=0; comp<3; ++comp) {
                            // calculate a_temp_zface
                            // all components on z-faces of grid
                            a_temp_zface(i, j, k, comp) = (M_normalization != 0) ? -(dt * a_temp_dynamic_coeff * H_eff_prev[comp] + a_temp_static_zface(i, j, k, comp))
                                                                                 : -(dt * a_temp_dynamic_coeff * H_eff_prev[comp] + 0.5 * a_temp_static_zface(i, j, k, comp)
                                                                                  + 0.5 * alpha_zface_arr(i,j,k) * 1. / std::sqrt(std::pow(M_zface(i, j, k, 0), 2.) + std::pow(M_zface(i, j, k, 1), 2.) + std::pow(M_zface(i, j, k, 2), 2.)) * M_zface_old(i, j, k, comp));
                        }

                        for (int comp=0; comp<3; ++comp) {
                            // update M_zface from a and b using the updateM_field
                            // all components on z-faces of grid
                            M_zface(i, j, k, comp) = updateM_field(i, j, k, comp, a_temp_zface, b_temp_static_zface);
                        }

                        // temporary normalized magnitude of M_zface field at the fixed point
                        // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                        amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_zface(i, j, k, 0), 2.) + std::pow(M_zface(i, j, k, 1), 2.) + std::pow(M_zface(i, j, k, 2), 2.)) / Ms_zface_arr(i,j,k);
                        amrex::Real normalized_error = 0.1;

                        if (M_normalization == 1){
                            // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                            // check the normalized error
                            if (amrex::Math::abs(1. - M_magnitude_normalized) > normalized_error){
                                amrex::Abort("Exceed the normalized error of the M_zface field");
                            }
                            // normalize the M_zface field
                            M_zface(i, j, k, 0) /= M_magnitude_normalized;
                            M_zface(i, j, k, 1) /= M_magnitude_normalized;
                            M_zface(i, j, k, 2) /= M_magnitude_normalized;
                        }
                        else if (M_normalization == 0){
                            // check the normalized error
                            if (M_magnitude_normalized > 1. + normalized_error){
                                amrex::Abort("Caution: Unsaturated material has M_zface exceeding the saturation magnetization");
                            }
                            else if (M_magnitude_normalized > 1. && M_magnitude_normalized <= 1. + normalized_error){
                                // normalize the M_zface field
                                M_zface(i, j, k, 0) /= M_magnitude_normalized;
                                M_zface(i, j, k, 1) /= M_magnitude_normalized;
                                M_zface(i, j, k, 2) /= M_magnitude_normalized;
                            }
                        }

                        // calculate M_zface_error
                        // x,y,z component on z-faces of grid
                        for (int icomp = 0; icomp < 3; ++icomp) {
                            M_zface_error(i, j, k, icomp) = amrex::Math::abs((M_zface(i, j, k, icomp) - M_zface_prev(i, j, k, icomp))) / Ms_zface_arr(i,j,k);
                        }
                    }
                });
        }

        // update H_demag
	   if(demag_coupling == 1)
	   {
	      //Solve Poisson's equation laplacian(Phi) = div(M) and get H_demagfield = -grad(Phi)
	      //Compute RHS of Poisson equation
	      ComputePoissonRHS(PoissonRHS, Mfield, Ms, geom);
                    
	      //Initial guess for phi
	      PoissonPhi.setVal(0.);
	      openbc.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
    
	      // Calculate H from Phi
	      ComputeHfromPhi(PoissonPhi, H_demagfield, prob_lo, prob_hi, geom);
	   }

       if (exchange_coupling == 1){
          CalculateH_exchange(Mfield, H_exchangefield, Ms, exchange, DMI, exchange_coupling, DMI_coupling, mu0, geom);
       }

       if (DMI_coupling == 1){
          CalculateH_DMI(Mfield, H_DMIfield, Ms, exchange, DMI, exchange_coupling, DMI_coupling, mu0, geom);
       }

        // Check the error between Mfield and Mfield_prev and decide whether another iteration is needed
        amrex::Real M_iter_maxerror = -1.;
        for (int iface = 0; iface < 3; iface++){
            for (int jcomp = 0; jcomp < 3; jcomp++){
                Real M_iter_error = Mfield_error[iface].norm0(jcomp);
                if (M_iter_error >= M_iter_maxerror){
                    M_iter_maxerror = M_iter_error;
                }
            }
        }

        if (M_iter_maxerror <= M_tol){

            stop_iter = 1;

            // normalize M
            if (M_normalization == 2){

                for (MFIter mfi(Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi){

                    const Array4<Real>& Ms_xface_arr = Ms[0].array(mfi);
                    const Array4<Real>& Ms_yface_arr = Ms[1].array(mfi);
                    const Array4<Real>& Ms_zface_arr = Ms[2].array(mfi);

                    // extract field data
                    const Array4<Real>& M_xface = Mfield[0].array(mfi); // note M_xface include x,y,z components at |_x faces
                    const Array4<Real>& M_yface = Mfield[1].array(mfi); // note M_yface include x,y,z components at |_y faces
                    const Array4<Real>& M_zface = Mfield[2].array(mfi); // note M_zface include x,y,z components at |_z faces

                    // extract tileboxes for which to loop
                    // extract tileboxes for which to loop
                    Box const &tbx = mfi.tilebox(Mfield[0].ixType().toIntVect());
                    Box const &tby = mfi.tilebox(Mfield[1].ixType().toIntVect());
                    Box const &tbz = mfi.tilebox(Mfield[2].ixType().toIntVect());

                    // loop over cells and update fields
                    amrex::ParallelFor(tbx, tby, tbz,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                            if (Ms_xface_arr(i,j,k) > 0.){
                                // temporary normalized magnitude of M_xface field at the fixed point
                                // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                                amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_xface(i, j, k, 0), 2.) + std::pow(M_xface(i, j, k, 1), 2.) +
                                                                               std::pow(M_xface(i, j, k, 2), 2.)) /
                                                                     Ms_xface_arr(i,j,k);
                                amrex::Real normalized_error = 0.1;
                                // check the normalized error
                                if (amrex::Math::abs(1. - M_magnitude_normalized) > normalized_error){
                                    amrex::Abort("Exceed the normalized error of the M_xface field");
                                }
                                // normalize the M_xface field
                                M_xface(i, j, k, 0) /= M_magnitude_normalized;
                                M_xface(i, j, k, 1) /= M_magnitude_normalized;
                                M_xface(i, j, k, 2) /= M_magnitude_normalized;
                            }
                        },

                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                            if (Ms_yface_arr(i,j,k) > 0.){
                                // temporary normalized magnitude of M_yface field at the fixed point
                                // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                                amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_yface(i, j, k, 0), 2.) + std::pow(M_yface(i, j, k, 1), 2.) +
                                                                               std::pow(M_yface(i, j, k, 2), 2.)) /
                                                                     Ms_yface_arr(i,j,k);
                                amrex::Real normalized_error = 0.1;
                                // check the normalized error
                                if (amrex::Math::abs(1. - M_magnitude_normalized) > normalized_error){
                                    amrex::Abort("Exceed the normalized error of the M_yface field");
                                }
                                // normalize the M_yface field
                                M_yface(i, j, k, 0) /= M_magnitude_normalized;
                                M_yface(i, j, k, 1) /= M_magnitude_normalized;
                                M_yface(i, j, k, 2) /= M_magnitude_normalized;
                            }
                        },

                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                            if (Ms_zface_arr(i,j,k) > 0.){
                                // temporary normalized magnitude of M_zface field at the fixed point
                                // re-investigate the way we do Ms interp, in case we encounter the case where Ms changes across two adjacent cells that you are doing interp
                                amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_zface(i, j, k, 0), 2.) + std::pow(M_zface(i, j, k, 1), 2.) +
                                                                               std::pow(M_zface(i, j, k, 2), 2.)) /
                                                                     Ms_zface_arr(i,j,k);
                                amrex::Real normalized_error = 0.1;
                                // check the normalized error
                                if (amrex::Math::abs(1. - M_magnitude_normalized) > normalized_error){
                                    amrex::Abort("Exceed the normalized error of the M_zface field");
                                }
                                // normalize the M_zface field
                                M_zface(i, j, k, 0) /= M_magnitude_normalized;
                                M_zface(i, j, k, 1) /= M_magnitude_normalized;
                                M_zface(i, j, k, 2) /= M_magnitude_normalized;
                            }
                        });
                }
            }
        }
        else{

            // Copy Mfield to Mfield_previous and fill periodic/interior ghost cells
            for (int i = 0; i < 3; i++){
                MultiFab::Copy(Mfield_prev[i], Mfield[i], 0, 0, 3, Mfield[i].nGrow());
                MultiFab::Copy(H_demagfield_prev[i], H_demagfield[i], 0, 0, 1, H_demagfield[i].nGrow());
                MultiFab::Copy(H_exchangefield_prev[i], H_exchangefield[i], 0, 0, 3, H_exchangefield[i].nGrow());
                MultiFab::Copy(H_DMIfield_prev[i], H_DMIfield[i], 0, 0, 3, H_DMIfield[i].nGrow());
                Mfield_prev[i].FillBoundary(geom.periodicity());
                H_demagfield_prev[i].FillBoundary(geom.periodicity());
                H_exchangefield_prev[i].FillBoundary(geom.periodicity());
                H_DMIfield_prev[i].FillBoundary(geom.periodicity());
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
