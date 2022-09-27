#include "Initialization.H"

void InitializeMagneticProperties(std::array< MultiFab, AMREX_SPACEDIM >&  alpha,
                   std::array< MultiFab, AMREX_SPACEDIM >&   Ms,
                   std::array< MultiFab, AMREX_SPACEDIM >&   gamma,
                   std::array< MultiFab, AMREX_SPACEDIM >&   exchange,
                   std::array< MultiFab, AMREX_SPACEDIM >&   anisotropy,
                   Real        alpha_val,
                   Real        Ms_val,
                   Real        gamma_val,
                   Real        exchange_val,
                   Real        anisotropy_val,
                   amrex::GpuArray<amrex::Real, 3> prob_lo,
                   amrex::GpuArray<amrex::Real, 3> prob_hi,
                   amrex::GpuArray<amrex::Real, 3> mag_lo,
                   amrex::GpuArray<amrex::Real, 3> mag_hi,
                   const       Geometry& geom)
{

    for(int i = 0; i < 3; i++){
       alpha[i].setVal(0.);
       Ms[i].setVal(0.);
       gamma[i].setVal(0.);
       exchange[i].setVal(0.);
       anisotropy[i].setVal(0.);
    }

    // loop over boxes
    for (MFIter mfi(alpha[0]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        Box const &tbx = mfi.tilebox(alpha[0].ixType().toIntVect());
        Box const &tby = mfi.tilebox(alpha[1].ixType().toIntVect());
        Box const &tbz = mfi.tilebox(alpha[2].ixType().toIntVect());

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

        const Array4<Real>& anisotropy_xface_arr = anisotropy[0].array(mfi);
        const Array4<Real>& anisotropy_yface_arr = anisotropy[1].array(mfi);
        const Array4<Real>& anisotropy_zface_arr = anisotropy[2].array(mfi);

        //xface
        amrex::ParallelFor( tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real x = prob_lo[0] + i * dx[0];
            Real y = prob_lo[1] + (j+0.5) * dx[1];
            Real z = prob_lo[2] + (k+0.5) * dx[2];
             
            if (x > mag_lo[0] && x < mag_hi[0]){
               if (y > mag_lo[1] && y < mag_hi[1]){
                  if (z > mag_lo[2] && z < mag_hi[2]){
                     alpha_xface_arr(i,j,k) = alpha_val;
                     gamma_xface_arr(i,j,k) = gamma_val;
                     Ms_xface_arr(i,j,k) = Ms_val;
                     exchange_xface_arr(i,j,k) = exchange_val;
                     anisotropy_xface_arr(i,j,k) = anisotropy_val;
                  }
               }
            }
        }); 

        //yface
        amrex::ParallelFor( tby, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real x = prob_lo[0] + (i+0.5) * dx[0];
            Real y = prob_lo[1] + j * dx[1];
            Real z = prob_lo[2] + (k+0.5) * dx[2];
             
            if (x > mag_lo[0] && x < mag_hi[0]){
               if (y > mag_lo[1] && y < mag_hi[1]){
                  if (z > mag_lo[2] && z < mag_hi[2]){
                     alpha_yface_arr(i,j,k) = alpha_val;
                     gamma_yface_arr(i,j,k) = gamma_val;
                     Ms_yface_arr(i,j,k) = Ms_val;
                     exchange_yface_arr(i,j,k) = exchange_val;
                     anisotropy_yface_arr(i,j,k) = anisotropy_val;
                  }
               }
            }
        }); 

        //zface
        amrex::ParallelFor( tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real x = prob_lo[0] + (i+0.5) * dx[0];
            Real y = prob_lo[1] + (j+0.5) * dx[1];
            Real z = prob_lo[2] + k * dx[2];
             
            if (x > mag_lo[0] && x < mag_hi[0]){
               if (y > mag_lo[1] && y < mag_hi[1]){
                  if (z > mag_lo[2] && z < mag_hi[2]){
                     alpha_zface_arr(i,j,k) = alpha_val;
                     gamma_zface_arr(i,j,k) = gamma_val;
                     Ms_zface_arr(i,j,k) = Ms_val;
                     exchange_zface_arr(i,j,k) = exchange_val;
                     anisotropy_zface_arr(i,j,k) = anisotropy_val;
                  }
               }
            }
        }); 
     }
     // fill periodic ghost cells for Ms. Used to calculate Ms_lo(hi)_x(y,z) for exchange field calculation
     for(int i = 0; i < 3; i++){
       Ms[i].FillBoundary(geom.periodicity());
    }

}

//Initialize fields

void InitializeFields(std::array< MultiFab, AMREX_SPACEDIM >&  Mfield,
                   std::array< MultiFab, AMREX_SPACEDIM >&   H_biasfield,
                   std::array< MultiFab, AMREX_SPACEDIM >&   Ms,
                   amrex::GpuArray<amrex::Real, 3> prob_lo,
                   amrex::GpuArray<amrex::Real, 3> prob_hi,
                   const       Geometry& geom)
{
    //for (MFIter mfi(*Mfield[0]); mfi.isValid(); ++mfi)
    for (MFIter mfi(Mfield[0]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(1);
 
        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        // extract field data
        const Array4<Real>& M_xface = Mfield[0].array(mfi);         
        const Array4<Real>& M_yface = Mfield[1].array(mfi);         
        const Array4<Real>& M_zface = Mfield[2].array(mfi);
         
        const Array4<Real>& H_bias_xface = H_biasfield[0].array(mfi);
        const Array4<Real>& H_bias_yface = H_biasfield[1].array(mfi);
        const Array4<Real>& H_bias_zface = H_biasfield[2].array(mfi);
      
        const Array4<Real>& Ms_xface_arr = Ms[0].array(mfi);
        const Array4<Real>& Ms_yface_arr = Ms[1].array(mfi);
        const Array4<Real>& Ms_zface_arr = Ms[2].array(mfi);

        // extract tileboxes for which to loop
        Box const &tbx = mfi.tilebox(Mfield[0].ixType().toIntVect());
        Box const &tby = mfi.tilebox(Mfield[1].ixType().toIntVect());
        Box const &tbz = mfi.tilebox(Mfield[2].ixType().toIntVect());

        amrex::ParallelFor( tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

            if (Ms_xface_arr(i,j,k) > 0._rt)
            {

                Real x = prob_lo[0] + i * dx[0];
                Real y = prob_lo[1] + (j+0.5) * dx[1];
                Real z = prob_lo[2] + (k+0.5) * dx[2];
               
                //x_face 
                M_xface(i,j,k,0) = (y < 0) ? 1.4e5 : 0.;
                M_xface(i,j,k,1) = 0._rt;
                M_xface(i,j,k,2) = (y >= 0) ? 1.4e5 : 0.;

                H_bias_xface(i,j,k,0) = 0._rt;         
                H_bias_xface(i,j,k,1) = 3.7e4;
                H_bias_xface(i,j,k,2) = 0._rt;

             } else {
             
                //x_face 
                M_xface(i,j,k,0) = 0.0; 
                M_xface(i,j,k,1) = 0.0;
                M_xface(i,j,k,2) = 0.0;

                H_bias_xface(i,j,k,0) = 0.0;         
                H_bias_xface(i,j,k,1) = 0.0;
                H_bias_xface(i,j,k,2) = 0.0;

	     }
 

        });

        amrex::ParallelFor( tby, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

            if (Ms_yface_arr(i,j,k) > 0._rt)
            {

                Real x = prob_lo[0] + (i+0.5) * dx[0];
                Real y = prob_lo[1] + j * dx[1];
                Real z = prob_lo[2] + (k+0.5) * dx[2];
               
                //y_face
                M_yface(i,j,k,0) = (y < 0) ? 1.4e5 : 0.;
                M_yface(i,j,k,1) = 0._rt;
                M_yface(i,j,k,2) = (y >= 0) ? 1.4e5 : 0.;

                H_bias_yface(i,j,k,0) = 0._rt;         
                H_bias_yface(i,j,k,1) = 3.7e4;
                H_bias_yface(i,j,k,2) = 0._rt;

             } else {
             
                //y_face
                M_yface(i,j,k,0) = 0.0;
                M_yface(i,j,k,1) = 0.0;
                M_yface(i,j,k,2) = 0.0;

                H_bias_yface(i,j,k,0) = 0.0;         
                H_bias_yface(i,j,k,1) = 0.0;
                H_bias_yface(i,j,k,2) = 0.0;

	     }

        });

        amrex::ParallelFor( tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

            if (Ms_zface_arr(i,j,k) > 0._rt)
            {

                Real x = prob_lo[0] + (i+0.5) * dx[0];
                Real y = prob_lo[1] + (j+0.5) * dx[1];
                Real z = prob_lo[2] + k * dx[2];
               
                //z_face
                M_zface(i,j,k,0) = (y < 0) ? 1.4e5 : 0.;
                M_zface(i,j,k,1) = 0._rt;
                M_zface(i,j,k,2) = (y >= 0) ? 1.4e5 : 0.;

                H_bias_zface(i,j,k,0) = 0._rt;         
                H_bias_zface(i,j,k,1) = 3.7e4;
                H_bias_zface(i,j,k,2) = 0._rt;

             } else {
             
                //z_face
                M_zface(i,j,k,0) = 0.0;
                M_zface(i,j,k,1) = 0.0;
                M_zface(i,j,k,2) = 0.0;

                H_bias_zface(i,j,k,0) = 0.0;         
                H_bias_zface(i,j,k,1) = 0.0;
                H_bias_zface(i,j,k,2) = 0.0;

	     }

        });
    }
}
