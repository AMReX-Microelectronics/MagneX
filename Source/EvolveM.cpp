#include "EvolveM.H"
#include "MagLaplacian.H"
#include "ComputeLLG_RHS.H"

void Compute_LLG_RHS(std::array< MultiFab, AMREX_SPACEDIM >&  LLG_RHS,
                   std::array< MultiFab, AMREX_SPACEDIM >&   Mfield_old,
                   std::array< MultiFab, AMREX_SPACEDIM >&   H_demagfield,
                   std::array< MultiFab, AMREX_SPACEDIM >&   H_biasfield,
                   std::array< MultiFab, AMREX_SPACEDIM >&   alpha,
                   std::array< MultiFab, AMREX_SPACEDIM >&   Ms,
                   std::array< MultiFab, AMREX_SPACEDIM >&   gamma,
                   std::array< MultiFab, AMREX_SPACEDIM >&   exchange,
                   std::array< MultiFab, AMREX_SPACEDIM >&   anisotropy,
                   int demag_coupling,
                   int exchange_coupling,
                   int anisotropy_coupling,
                   amrex::GpuArray<amrex::Real, 3>& anisotropy_axis,
                   int M_normalization, 
                   Real mu0,
                   const Geometry& geom)
{
        //for (MFIter mfi(*Mfield[0]); mfi.isValid(); ++mfi)
        for (MFIter mfi(Mfield_old[0]); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.growntilebox(1); 

            // extract dx from the geometry object
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

            // extract field data
            const Array4<Real>& Hx_demag= H_demagfield[0].array(mfi);
            const Array4<Real>& Hy_demag= H_demagfield[1].array(mfi);
            const Array4<Real>& Hz_demag= H_demagfield[2].array(mfi);

            const Array4<Real>& LLG_rhs_xface = LLG_RHS[0].array(mfi);         
            const Array4<Real>& LLG_rhs_yface = LLG_RHS[1].array(mfi);         
            const Array4<Real>& LLG_rhs_zface = LLG_RHS[2].array(mfi);         

            const Array4<Real>& M_xface_old = Mfield_old[0].array(mfi); 
            const Array4<Real>& M_yface_old = Mfield_old[1].array(mfi); 
            const Array4<Real>& M_zface_old = Mfield_old[2].array(mfi); 

            const Array4<Real>& H_bias_xface = H_biasfield[0].array(mfi);
            const Array4<Real>& H_bias_yface = H_biasfield[1].array(mfi);
            const Array4<Real>& H_bias_zface = H_biasfield[2].array(mfi);
          
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

            const Array4<Real>& anisotropy_xface_arr = anisotropy[0].array(mfi);
            const Array4<Real>& anisotropy_yface_arr = anisotropy[1].array(mfi);
            const Array4<Real>& anisotropy_zface_arr = anisotropy[2].array(mfi);

            amrex::IntVect Mxface_stag = Mfield_old[0].ixType().toIntVect();
            amrex::IntVect Myface_stag = Mfield_old[1].ixType().toIntVect();
            amrex::IntVect Mzface_stag = Mfield_old[2].ixType().toIntVect();

            // extract tileboxes for which to loop
            Box const &tbx = mfi.tilebox(Mfield_old[0].ixType().toIntVect());
            Box const &tby = mfi.tilebox(Mfield_old[1].ixType().toIntVect());
            Box const &tbz = mfi.tilebox(Mfield_old[2].ixType().toIntVect());

            //xface 
            amrex::ParallelFor( tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (Ms_xface_arr(i,j,k) > 0._rt)
                {
                    amrex::Real Hx_eff = H_bias_xface(i,j,k,0);
                    amrex::Real Hy_eff = H_bias_xface(i,j,k,1);
                    amrex::Real Hz_eff = H_bias_xface(i,j,k,2);

                    // PSSW validation
                    // amrex::Real z = prob_lo[2] + (k+0.5) * dx[2];
                    
                    // Hx_bias(i,j,k) = 24.0 * (exp(-(time-3.* TP)*(time-3.* TP)/(2*TP*TP))*cos(2*pi*frequency*time)) * cos(z / 345.0e-9 * pi);
                    // Hy_bias(i,j,k) = 2.4e4;
                    // Hz_bias(i,j,k) = 0.;

                    if(demag_coupling == 1)
                    {
                        Hx_eff += face_avg_to_face(i, j, k, 0, Mxface_stag, Mxface_stag, Hx_demag);
                        Hy_eff += face_avg_to_face(i, j, k, 0, Myface_stag, Mxface_stag, Hy_demag);
                        Hz_eff += face_avg_to_face(i, j, k, 0, Mzface_stag, Mxface_stag, Hz_demag);
                    }

                    if(exchange_coupling == 1)
                    { 
                    //Add exchange term
                      if (exchange_xface_arr(i,j,k) == 0._rt) amrex::Abort("The exchange_yface_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                      // H_exchange - use M^(old_time)
                      amrex::Real const H_exchange_coeff = 2.0 * exchange_xface_arr(i,j,k) / mu0 / Ms_xface_arr(i,j,k) / Ms_xface_arr(i,j,k);

                      amrex::Real Ms_lo_x = Ms_xface_arr(i-1, j, k); 
                      amrex::Real Ms_hi_x = Ms_xface_arr(i+1, j, k); 
                      amrex::Real Ms_lo_y = Ms_xface_arr(i, j-1, k); 
                      amrex::Real Ms_hi_y = Ms_xface_arr(i, j+1, k); 
                      amrex::Real Ms_lo_z = Ms_xface_arr(i, j, k-1);
                      amrex::Real Ms_hi_z = Ms_xface_arr(i, j, k+1);

		      //if(i == 31 && j == 31 && k == 31) printf("Laplacian_x = %g \n", Laplacian_Mag(M_xface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 0, 0));
                      Hx_eff += H_exchange_coeff * Laplacian_Mag(M_xface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 0, 0);
                      Hy_eff += H_exchange_coeff * Laplacian_Mag(M_xface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 1, 0);
                      Hz_eff += H_exchange_coeff * Laplacian_Mag(M_xface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 2, 0);

                    }
                 
                    if(anisotropy_coupling == 1)
                    {
                     //Add anisotropy term
 
                     if (anisotropy_xface_arr(i,j,k) == 0._rt) amrex::Abort("The anisotropy_xface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                      // H_anisotropy - use M^(old_time)
                      amrex::Real M_dot_anisotropy_axis = 0.0;
                      for (int comp=0; comp<3; ++comp) {
                          M_dot_anisotropy_axis += M_xface_old(i, j, k, comp) * anisotropy_axis[comp];
                      }
                      amrex::Real const H_anisotropy_coeff = - 2.0 * anisotropy_xface_arr(i,j,k) / mu0 / Ms_xface_arr(i,j,k) / Ms_xface_arr(i,j,k);
                      Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                      Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                      Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];

                    }

                   //Update M

                   // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                   amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_xface_old(i, j, k, 0), 2._rt) + std::pow(M_xface_old(i, j, k, 1), 2._rt) + std::pow(M_xface_old(i, j, k, 2), 2._rt))
                                                              : Ms_xface_arr(i,j,k); 

                   // x component on x-faces of grid
                   LLG_rhs_xface(i, j, k, 0) = LLG_RHS_x(M_xface_old, alpha_xface_arr, gamma_xface_arr, M_magnitude, mu0, Hx_eff, Hy_eff, Hz_eff, i, j, k);

                   // y component on x-faces of grid
                   LLG_rhs_xface(i, j, k, 1) = LLG_RHS_y(M_xface_old, alpha_xface_arr, gamma_xface_arr, M_magnitude, mu0, Hx_eff, Hy_eff, Hz_eff, i, j, k);

                   // z component on x-faces of grid
                   LLG_rhs_xface(i, j, k, 2) = LLG_RHS_z(M_xface_old, alpha_xface_arr, gamma_xface_arr, M_magnitude, mu0, Hx_eff, Hy_eff, Hz_eff, i, j, k);

                 }   
 
            });     
                      
            //yface 
            amrex::ParallelFor( tby, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (Ms_yface_arr(i,j,k) > 0._rt)
                {
                    amrex::Real Hx_eff = H_bias_yface(i,j,k,0);
                    amrex::Real Hy_eff = H_bias_yface(i,j,k,1);
                    amrex::Real Hz_eff = H_bias_yface(i,j,k,2);
                 
                    if(demag_coupling == 1)
                    {
                      Hx_eff += face_avg_to_face(i, j, k, 0, Mxface_stag, Myface_stag, Hx_demag);
                      Hy_eff += face_avg_to_face(i, j, k, 0, Myface_stag, Myface_stag, Hy_demag);
                      Hz_eff += face_avg_to_face(i, j, k, 0, Mzface_stag, Myface_stag, Hz_demag);
                    }

                    if(exchange_coupling == 1)
                    { 
                    //Add exchange term
                      if (exchange_yface_arr(i,j,k) == 0._rt) amrex::Abort("The exchange_yface_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                      // H_exchange - use M^(old_time)
                      amrex::Real const H_exchange_coeff = 2.0 * exchange_yface_arr(i,j,k) / mu0 / Ms_yface_arr(i,j,k) / Ms_yface_arr(i,j,k);

                      amrex::Real Ms_lo_x = Ms_yface_arr(i-1, j, k); 
                      amrex::Real Ms_hi_x = Ms_yface_arr(i+1, j, k); 
                      amrex::Real Ms_lo_y = Ms_yface_arr(i, j-1, k); 
                      amrex::Real Ms_hi_y = Ms_yface_arr(i, j+1, k); 
                      amrex::Real Ms_lo_z = Ms_yface_arr(i, j, k-1);
                      amrex::Real Ms_hi_z = Ms_yface_arr(i, j, k+1);

		      //if(i == 31 && j == 31 && k == 31) printf("Laplacian_x = %g \n", Laplacian_Mag(M_yface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 0, 0));
                      Hx_eff += H_exchange_coeff * Laplacian_Mag(M_yface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 0, 1);
                      Hy_eff += H_exchange_coeff * Laplacian_Mag(M_yface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 1, 1);
                      Hz_eff += H_exchange_coeff * Laplacian_Mag(M_yface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 2, 1);

                    }
                 
                    if(anisotropy_coupling == 1)
                    {
                     //Add anisotropy term
 
                     if (anisotropy_yface_arr(i,j,k) == 0._rt) amrex::Abort("The anisotropy_yface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                      // H_anisotropy - use M^(old_time)
                      amrex::Real M_dot_anisotropy_axis = 0.0;
                      for (int comp=0; comp<3; ++comp) {
                          M_dot_anisotropy_axis += M_yface_old(i, j, k, comp) * anisotropy_axis[comp];
                      }
                      amrex::Real const H_anisotropy_coeff = - 2.0 * anisotropy_yface_arr(i,j,k) / mu0 / Ms_yface_arr(i,j,k) / Ms_yface_arr(i,j,k);
                      Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                      Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                      Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];

                    }

                   //Update M

                   // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                   amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_yface_old(i, j, k, 0), 2._rt) + std::pow(M_yface_old(i, j, k, 1), 2._rt) + std::pow(M_yface_old(i, j, k, 2), 2._rt))
                                                              : Ms_yface_arr(i,j,k); 

                   // x component on y-faces of grid
                   LLG_rhs_yface(i, j, k, 0) = LLG_RHS_x(M_yface_old, alpha_yface_arr, gamma_yface_arr, M_magnitude, mu0, Hx_eff, Hy_eff, Hz_eff, i, j, k);

                   // y component on y-faces of grid
                   LLG_rhs_yface(i, j, k, 1) = LLG_RHS_y(M_yface_old, alpha_yface_arr, gamma_yface_arr, M_magnitude, mu0, Hx_eff, Hy_eff, Hz_eff, i, j, k);

                   // z component on y-faces of grid
                   LLG_rhs_yface(i, j, k, 2) = LLG_RHS_z(M_yface_old, alpha_yface_arr, gamma_yface_arr, M_magnitude, mu0, Hx_eff, Hy_eff, Hz_eff, i, j, k);

                 }   
 
            });     
                      
            //zface 
            amrex::ParallelFor( tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (Ms_zface_arr(i,j,k) > 0._rt)
                {
                    amrex::Real Hx_eff = H_bias_zface(i,j,k,0);
                    amrex::Real Hy_eff = H_bias_zface(i,j,k,1);
                    amrex::Real Hz_eff = H_bias_zface(i,j,k,2);
                 
                    if(demag_coupling == 1)
                    {
                      Hx_eff += face_avg_to_face(i, j, k, 0, Mxface_stag, Mzface_stag, Hx_demag);
                      Hy_eff += face_avg_to_face(i, j, k, 0, Myface_stag, Mzface_stag, Hy_demag);
                      Hz_eff += face_avg_to_face(i, j, k, 0, Mzface_stag, Mzface_stag, Hz_demag);
                    }

                    if(exchange_coupling == 1)
                    { 
                    //Add exchange term
                      if (exchange_zface_arr(i,j,k) == 0._rt) amrex::Abort("The exchange_zface_arr(i,j,k) is 0.0 while including the exchange coupling term H_exchange for H_eff");

                      // H_exchange - use M^(old_time)
                      amrex::Real const H_exchange_coeff = 2.0 * exchange_zface_arr(i,j,k) / mu0 / Ms_zface_arr(i,j,k) / Ms_zface_arr(i,j,k);

                      amrex::Real Ms_lo_x = Ms_zface_arr(i-1, j, k); 
                      amrex::Real Ms_hi_x = Ms_zface_arr(i+1, j, k); 
                      amrex::Real Ms_lo_y = Ms_zface_arr(i, j-1, k); 
                      amrex::Real Ms_hi_y = Ms_zface_arr(i, j+1, k); 
                      amrex::Real Ms_lo_z = Ms_zface_arr(i, j, k-1);
                      amrex::Real Ms_hi_z = Ms_zface_arr(i, j, k+1);

		      //if(i == 31 && j == 31 && k == 31) printf("Laplacian_x = %g \n", Laplacian_Mag(M_zface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 0, 0));
                      Hx_eff += H_exchange_coeff * Laplacian_Mag(M_zface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 0, 2);
                      Hy_eff += H_exchange_coeff * Laplacian_Mag(M_zface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 1, 2);
                      Hz_eff += H_exchange_coeff * Laplacian_Mag(M_zface_old, Ms_lo_x, Ms_hi_x, Ms_lo_y, Ms_hi_y, Ms_lo_z, Ms_hi_z, i, j, k, dx, 2, 2);

                    }
                 
                    if(anisotropy_coupling == 1)
                    {
                     //Add anisotropy term
 
                     if (anisotropy_zface_arr(i,j,k) == 0._rt) amrex::Abort("The anisotropy_zface_arr(i,j,k) is 0.0 while including the anisotropy coupling term H_anisotropy for H_eff");

                      // H_anisotropy - use M^(old_time)
                      amrex::Real M_dot_anisotropy_axis = 0.0;
                      for (int comp=0; comp<3; ++comp) {
                          M_dot_anisotropy_axis += M_zface_old(i, j, k, comp) * anisotropy_axis[comp];
                      }
                      amrex::Real const H_anisotropy_coeff = - 2.0 * anisotropy_zface_arr(i,j,k) / mu0 / Ms_zface_arr(i,j,k) / Ms_zface_arr(i,j,k);
                      Hx_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[0];
                      Hy_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[1];
                      Hz_eff += H_anisotropy_coeff * M_dot_anisotropy_axis * anisotropy_axis[2];

                    }

                   //Update M

                   // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                   amrex::Real M_magnitude = (M_normalization == 0) ? std::sqrt(std::pow(M_zface_old(i, j, k, 0), 2._rt) + std::pow(M_zface_old(i, j, k, 1), 2._rt) + std::pow(M_zface_old(i, j, k, 2), 2._rt))
                                                              : Ms_zface_arr(i,j,k); 

                   // x component on z-faces of grid
                   LLG_rhs_zface(i, j, k, 0) = LLG_RHS_x(M_zface_old, alpha_zface_arr, gamma_zface_arr, M_magnitude, mu0, Hx_eff, Hy_eff, Hz_eff, i, j, k);

                   // y component on z-faces of grid
                   LLG_rhs_zface(i, j, k, 1) = LLG_RHS_y(M_zface_old, alpha_zface_arr, gamma_zface_arr, M_magnitude, mu0, Hx_eff, Hy_eff, Hz_eff, i, j, k);

                   // z component on z-faces of grid
                   LLG_rhs_zface(i, j, k, 2) = LLG_RHS_z(M_zface_old, alpha_zface_arr, gamma_zface_arr, M_magnitude, mu0, Hx_eff, Hy_eff, Hz_eff, i, j, k);

                 }   
 
            });    
            
        }
}

void NormalizeM(std::array< MultiFab, AMREX_SPACEDIM >& Mfield, std::array< MultiFab, AMREX_SPACEDIM >& Ms, int M_normalization)
{
        //for (MFIter mfi(*Mfield[0]); mfi.isValid(); ++mfi)
        for (MFIter mfi(Mfield[0]); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.growntilebox(1); 

            // extract field data
            const Array4<Real>& M_xface = Mfield[0].array(mfi);         
            const Array4<Real>& M_yface = Mfield[1].array(mfi);         
            const Array4<Real>& M_zface = Mfield[2].array(mfi);         

            const Array4<Real>& Ms_xface_arr = Ms[0].array(mfi);
            const Array4<Real>& Ms_yface_arr = Ms[1].array(mfi);
            const Array4<Real>& Ms_zface_arr = Ms[2].array(mfi);

            // extract tileboxes for which to loop
            Box const &tbx = mfi.tilebox(Mfield[0].ixType().toIntVect());
            Box const &tby = mfi.tilebox(Mfield[1].ixType().toIntVect());
            Box const &tbz = mfi.tilebox(Mfield[2].ixType().toIntVect());

            //xface 
            amrex::ParallelFor( tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (Ms_xface_arr(i,j,k) > 0._rt)
                {
                   // temporary normalized magnitude of M_xface field at the fixed point
                   amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_xface(i, j, k, 0), 2._rt) + std::pow(M_xface(i, j, k, 1), 2._rt) + std::pow(M_xface(i, j, k, 2), 2._rt)) / Ms_xface_arr(i,j,k);
                   amrex::Real normalized_error = 0.1;

                   if (M_normalization > 0)
                   {
                       // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                       // check the normalized error
                       if (amrex::Math::abs(1._rt - M_magnitude_normalized) > normalized_error)
                       {
  			   printf("y-face M_magnitude_normalized = %g \n", M_magnitude_normalized);
                           printf("i = %d, j = %d, k = %d \n", i, j,k);
                           amrex::Abort("Exceed the normalized error of the Mx field");
                       }
                       // normalize the M field
                       M_xface(i, j, k, 0) /= M_magnitude_normalized;
                       M_xface(i, j, k, 1) /= M_magnitude_normalized;
                       M_xface(i, j, k, 2) /= M_magnitude_normalized;
                   }
                   else if (M_normalization == 0)
                   {   
		       if(i == 1 && j == 1 && k == 1) printf("Here ??? \n");
                       // check the normalized error
                       if (M_magnitude_normalized > (1._rt + normalized_error))
                       {
                           amrex::Abort("Caution: Unsaturated material has M_xface exceeding the saturation magnetization");
                       }
                       else if (M_magnitude_normalized > 1._rt && M_magnitude_normalized <= (1._rt + normalized_error) )
                       {
                           // normalize the M field
                           M_xface(i, j, k, 0) /= M_magnitude_normalized;
                           M_xface(i, j, k, 1) /= M_magnitude_normalized;
                           M_xface(i, j, k, 2) /= M_magnitude_normalized;
                       }
                   }  
                }
            });     
                      
            //yface 
            amrex::ParallelFor( tby, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (Ms_yface_arr(i,j,k) > 0._rt)
                {
                   // temporary normalized magnitude of M_xface field at the fixed point
                   amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_yface(i, j, k, 0), 2._rt) + std::pow(M_yface(i, j, k, 1), 2._rt) + std::pow(M_yface(i, j, k, 2), 2._rt)) / Ms_yface_arr(i,j,k);
                   amrex::Real normalized_error = 0.1;

                   if (M_normalization > 0)
                   {
                       // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                       // check the normalized error
                       if (amrex::Math::abs(1._rt - M_magnitude_normalized) > normalized_error)
                       {
                           printf("y-face M_magnitude_normalized = %g \n", M_magnitude_normalized);
                           printf("i = %d, j = %d, k = %d \n", i, j,k);
                           amrex::Abort("Exceed the normalized error of the Mx field");
                       }
                       // normalize the M field
                       M_yface(i, j, k, 0) /= M_magnitude_normalized;
                       M_yface(i, j, k, 1) /= M_magnitude_normalized;
                       M_yface(i, j, k, 2) /= M_magnitude_normalized;
                   }
                   else if (M_normalization == 0)
                   {   
		       if(i == 1 && j == 1 && k == 1) printf("Here ??? \n");
                       // check the normalized error
                       if (M_magnitude_normalized > (1._rt + normalized_error))
                       {
                           amrex::Abort("Caution: Unsaturated material has M_xface exceeding the saturation magnetization");
                       }
                       else if (M_magnitude_normalized > 1._rt && M_magnitude_normalized <= (1._rt + normalized_error) )
                       {
                           // normalize the M field
                           M_yface(i, j, k, 0) /= M_magnitude_normalized;
                           M_yface(i, j, k, 1) /= M_magnitude_normalized;
                           M_yface(i, j, k, 2) /= M_magnitude_normalized;
                       }
                   }
                }
            });     
                      
            //zface 
            amrex::ParallelFor( tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (Ms_zface_arr(i,j,k) > 0._rt)
                {
                   // temporary normalized magnitude of M_xface field at the fixed point
                   amrex::Real M_magnitude_normalized = std::sqrt(std::pow(M_zface(i, j, k, 0), 2._rt) + std::pow(M_zface(i, j, k, 1), 2._rt) + std::pow(M_zface(i, j, k, 2), 2._rt)) / Ms_zface_arr(i,j,k);
                   amrex::Real normalized_error = 0.1;

                   if (M_normalization > 0)
                   {
                       // saturated case; if |M| has drifted from M_s too much, abort.  Otherwise, normalize
                       // check the normalized error
                       if (amrex::Math::abs(1._rt - M_magnitude_normalized) > normalized_error)
                       {
			   printf("z-face M_magnitude_normalized = %g \n", M_magnitude_normalized);
                           printf("i = %d, j = %d, k = %d \n", i, j,k);
                           amrex::Abort("Exceed the normalized error of the Mx field");
                       }
                       // normalize the M field
                       M_zface(i, j, k, 0) /= M_magnitude_normalized;
                       M_zface(i, j, k, 1) /= M_magnitude_normalized;
                       M_zface(i, j, k, 2) /= M_magnitude_normalized;
                   }
                   else if (M_normalization == 0)
                   {   
		       if(i == 1 && j == 1 && k == 1) printf("Here ??? \n");
                       // check the normalized error
                       if (M_magnitude_normalized > (1._rt + normalized_error))
                       {
                           amrex::Abort("Caution: Unsaturated material has M_xface exceeding the saturation magnetization");
                       }
                       else if (M_magnitude_normalized > 1._rt && M_magnitude_normalized <= (1._rt + normalized_error) )
                       {
                           // normalize the M field
                           M_zface(i, j, k, 0) /= M_magnitude_normalized;
                           M_zface(i, j, k, 1) /= M_magnitude_normalized;
                           M_zface(i, j, k, 2) /= M_magnitude_normalized;
                       }
                   }
                }
            });    
            
        }
}
