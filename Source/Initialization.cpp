#include "Initialization.H"

void InitializeMagneticProperties(std::array< MultiFab, AMREX_SPACEDIM >&  alpha,
                   std::array< MultiFab, AMREX_SPACEDIM >&   Ms,
                   std::array< MultiFab, AMREX_SPACEDIM >&   gamma,
                   std::array< MultiFab, AMREX_SPACEDIM >&   exchange,
                   std::array< MultiFab, AMREX_SPACEDIM >&   DMI,
                   std::array< MultiFab, AMREX_SPACEDIM >&   anisotropy,
                   Real        alpha_val,
                   Real        Ms_val,
                   Real        gamma_val,
                   Real        exchange_val,
                   Real        DMI_val,
                   Real        anisotropy_val,
                   amrex::GpuArray<amrex::Real, 3> prob_lo,
                   amrex::GpuArray<amrex::Real, 3> prob_hi,
                   amrex::GpuArray<amrex::Real, 3> mag_lo,
                   amrex::GpuArray<amrex::Real, 3> mag_hi,
                   const       Geometry& geom)
{

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> ddx;
    for (int i = 0; i < 3; ++i) {
      ddx[i] = dx[i] / 1.e6;
    }

    for(int i = 0; i < 3; i++){
       alpha[i].setVal(0.);
       Ms[i].setVal(0.);
       gamma[i].setVal(0.);
       exchange[i].setVal(0.);
       DMI[i].setVal(0.);
       anisotropy[i].setVal(0.);
    }

    // loop over boxes
    for (MFIter mfi(alpha[0]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        Box const &tbx = mfi.tilebox(alpha[0].ixType().toIntVect());
        Box const &tby = mfi.tilebox(alpha[1].ixType().toIntVect());
        Box const &tbz = mfi.tilebox(alpha[2].ixType().toIntVect());
        
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

        //xface
        amrex::ParallelFor( tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real x = prob_lo[0] + i * dx[0];
            Real y = prob_lo[1] + (j+0.5) * dx[1];
            Real z = prob_lo[2] + (k+0.5) * dx[2];

            if (x > mag_lo[0]-ddx[0] && x < mag_hi[0]+ddx[0]){
               if (y > mag_lo[1]-ddx[1] && y < mag_hi[1]+ddx[1]){
                  if (z > mag_lo[2]-ddx[2] && z < mag_hi[2]+ddx[2]){
                     alpha_xface_arr(i,j,k) = alpha_val;
                     gamma_xface_arr(i,j,k) = gamma_val;
                     Ms_xface_arr(i,j,k) = Ms_val;
                     exchange_xface_arr(i,j,k) = exchange_val;
                     DMI_xface_arr(i,j,k) = DMI_val;
                     anisotropy_xface_arr(i,j,k) = anisotropy_val;
                     if (Ms_xface_arr(i,j,k) < Ms_val) {
                     printf("i= %d, j = %d, k = %d, Ms_xface = %g \n", i, j, k, Ms_xface_arr(i,j,k));
                     }
                     // amrex::Print() << "i=" << i << "j=" << j << "k=" << k << Ms_xface_arr(i,j,k) << "\n";
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

            if (x > mag_lo[0]-ddx[0] && x < mag_hi[0]+ddx[0]){
               if (y > mag_lo[1]-ddx[1] && y < mag_hi[1]+ddx[1]){
                  if (z > mag_lo[2]-ddx[2] && z < mag_hi[2]+ddx[2]){
                     alpha_yface_arr(i,j,k) = alpha_val;
                     gamma_yface_arr(i,j,k) = gamma_val;
                     Ms_yface_arr(i,j,k) = Ms_val;
                     exchange_yface_arr(i,j,k) = exchange_val;
                     DMI_yface_arr(i,j,k) = DMI_val;
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

            if (x > mag_lo[0]-ddx[0] && x < mag_hi[0]+ddx[0]){
               if (y > mag_lo[1]-ddx[1] && y < mag_hi[1]+ddx[1]){
                  if (z > mag_lo[2]-ddx[2] && z < mag_hi[2]+ddx[2]){
                     alpha_zface_arr(i,j,k) = alpha_val;
                     gamma_zface_arr(i,j,k) = gamma_val;
                     Ms_zface_arr(i,j,k) = Ms_val;
                     exchange_zface_arr(i,j,k) = exchange_val;
                     DMI_zface_arr(i,j,k) = DMI_val;
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

void InitializeFields(amrex::Vector<MultiFab>& Mfield, //std::array< MultiFab, AMREX_SPACEDIM >&  Mfield,
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
                M_xface(i,j,k,0) = 0._rt;
                M_xface(i,j,k,1) = Ms_xface_arr(i,j,k);
                M_xface(i,j,k,2) = 0._rt;
               //  M_xface(i,j,k,0) = (z < 0) ? 1.392605752054084e5 : 0.;
               //  M_xface(i,j,k,1) = 0._rt;
               //  M_xface(i,j,k,2) = (z >= 0) ? 1.392605752054084e5 : 0.;
               // M_xface(i,j,k,0) = 8.0e5 /sqrt(3.0);
               // M_xface(i,j,k,1) = 8.0e5 /sqrt(3.0);
               // M_xface(i,j,k,2) = 8.0e5 /sqrt(3.0);

                H_bias_xface(i,j,k,0) = 0._rt;         
                H_bias_xface(i,j,k,1) = 2.387324146378430e4;
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
                M_yface(i,j,k,0) = 0._rt;
                M_yface(i,j,k,1) = Ms_yface_arr(i,j,k);
                M_yface(i,j,k,2) = 0._rt;
               //  M_yface(i,j,k,0) = (z < 0) ? 1.392605752054084e5 : 0.;
               //  M_yface(i,j,k,1) = 0._rt;
               //  M_yface(i,j,k,2) = (z >= 0) ? 1.392605752054084e5 : 0.;
               //  M_yface(i,j,k,0) = 8.0e5 /sqrt(3.0);
               //  M_yface(i,j,k,1) = 8.0e5 /sqrt(3.0);
               //  M_yface(i,j,k,2) = 8.0e5 /sqrt(3.0);

                H_bias_yface(i,j,k,0) = 0._rt;         
                H_bias_yface(i,j,k,1) = 2.387324146378430e4;
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
                M_zface(i,j,k,0) = 0._rt;
                M_zface(i,j,k,1) = Ms_zface_arr(i,j,k);
                M_zface(i,j,k,2) = 0._rt;
               //  M_zface(i,j,k,0) = (z < 0) ? 1.392605752054084e5 : 0.;
               //  M_zface(i,j,k,1) = 0._rt;
               //  M_zface(i,j,k,2) = (z >= 0) ? 1.392605752054084e5 : 0.;
               //  M_zface(i,j,k,0) = 8.0e5 /sqrt(3.0);
               //  M_zface(i,j,k,1) = 8.0e5 /sqrt(3.0);
               //  M_zface(i,j,k,2) = 8.0e5 /sqrt(3.0);

                H_bias_zface(i,j,k,0) = 0._rt;
                H_bias_zface(i,j,k,1) = 2.387324146378430e4;
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
