#include "Initialization.H"

#include "AMReX_ParmParse.H"
#include "AMReX_Parser.H"

void InitializeMagneticProperties(MultiFab&  alpha,
                   MultiFab&   Ms,
                   MultiFab&   gamma,
                   MultiFab&   exchange,
                   MultiFab&   DMI,
                   MultiFab&   anisotropy,
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

    alpha.setVal(0.);
    Ms.setVal(0.);
    gamma.setVal(0.);
    exchange.setVal(0.);
    DMI.setVal(0.);
    anisotropy.setVal(0.);

    ParmParse pp;
    std::string Ms_parser_string;

    pp.get("Ms_parser(x,y,z)",Ms_parser_string);
    Print() << "HACK " << Ms_parser_string << std::endl;

    Parser Ms_parser(Ms_parser_string);
    auto Ms_p = Ms_parser.compile<3>();
    
    // loop over boxes
    for (MFIter mfi(alpha); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real>& alpha_arr = alpha.array(mfi);
        const Array4<Real>& gamma_arr = gamma.array(mfi);
        const Array4<Real>& Ms_arr = Ms.array(mfi);
        const Array4<Real>& exchange_arr = exchange.array(mfi);
        const Array4<Real>& DMI_arr = DMI.array(mfi);
        const Array4<Real>& anisotropy_arr = anisotropy.array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real x = prob_lo[0] + (i+0.5) * dx[0];
            Real y = prob_lo[1] + (j+0.5) * dx[1];
            Real z = prob_lo[2] + (k+0.5) * dx[2];

            if (x > mag_lo[0]-ddx[0] && x < mag_hi[0]+ddx[0]){
                if (y > mag_lo[1]-ddx[1] && y < mag_hi[1]+ddx[1]){
                    if (z > mag_lo[2]-ddx[2] && z < mag_hi[2]+ddx[2]){
                        alpha_arr(i,j,k) = alpha_val;
                        gamma_arr(i,j,k) = gamma_val;
                        Ms_arr(i,j,k) = Ms_val;
                        exchange_arr(i,j,k) = exchange_val;
                        DMI_arr(i,j,k) = DMI_val;
                        anisotropy_arr(i,j,k) = anisotropy_val;
                        if (Ms_arr(i,j,k) < Ms_val) {
                            printf("i= %d, j = %d, k = %d, Ms = %g \n", i, j, k, Ms_arr(i,j,k));
                        }
                        // amrex::Print() << "i=" << i << "j=" << j << "k=" << k << Ms_xface_arr(i,j,k) << "\n";
                    }
                }
            }
        }); 
    }
    // fill periodic ghost cells for Ms. Used to calculate Ms_lo(hi)_x(y,z) for exchange field calculation
    Ms.FillBoundary(geom.periodicity());

}

//Initialize fields

void InitializeFields(Array< MultiFab, AMREX_SPACEDIM >&  Mfield,
                      Array< MultiFab, AMREX_SPACEDIM >&  H_biasfield,
                      MultiFab&  Ms,
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
        const Array4<Real>& Mx = Mfield[0].array(mfi);         
        const Array4<Real>& My = Mfield[1].array(mfi);         
        const Array4<Real>& Mz = Mfield[2].array(mfi);
         
        const Array4<Real>& Hx_bias = H_biasfield[0].array(mfi);
        const Array4<Real>& Hy_bias = H_biasfield[1].array(mfi);
        const Array4<Real>& Hz_bias = H_biasfield[2].array(mfi);
      
        const Array4<Real>& Ms_arr = Ms.array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

            if (Ms_arr(i,j,k) > 0._rt)
            {

                Real x = prob_lo[0] + (i+0.5) * dx[0];
                Real y = prob_lo[1] + (j+0.5) * dx[1];
                Real z = prob_lo[2] + (k+0.5) * dx[2];
               
                //Mx(i,j,k) = 0._rt;
                //My(i,j,k) = 0.11 * Ms_arr(i,j,k);
                //Mz(i,j,k) = 0.99 * Ms_arr(i,j,k);
                //  M_xface(i,j,k,0) = (z < 0) ? Ms_xface_arr(i,j,k) : 0.;
                //  M_xface(i,j,k,1) = 0._rt;
                //  M_xface(i,j,k,2) = (z >= 0) ? Ms_xface_arr(i,j,k) : 0.;
                // M_xface(i,j,k,0) = 8.0e5 /sqrt(3.0);
                // M_xface(i,j,k,1) = 8.0e5 /sqrt(3.0);
                // M_xface(i,j,k,2) = 8.0e5 /sqrt(3.0);
                // Mx(i,j,k) = (y < 0) ? 1.392605752054084e5 : 0.;
                // My(i,j,k) = 0._rt;
                // Mz(i,j,k) = (y >= 0) ? 1.392605752054084e5 : 0.;
                // Hxbias(i,j,k) = 0._rt;         
                // Hybias(i,j,k) = 3.7e4; //0._rt;
                // Hzbias(i,j,k) = 0._rt;

                Mx(i,j,k) = 0._rt;
                My(i,j,k) = Ms_arr(i,j,k);
                Mz(i,j,k) = 0._rt;
                Hx_bias(i,j,k) = 0._rt;         
                Hy_bias(i,j,k) = 2.387324146378430e4;
                Hz_bias(i,j,k) = 0._rt;

            } else {
             
                //x_face 
                Mx(i,j,k) = 0.0; 
                My(i,j,k) = 0.0;
                Mz(i,j,k) = 0.0;

                Hx_bias(i,j,k) = 0.0;         
                Hy_bias(i,j,k) = 0.0;
                Hz_bias(i,j,k) = 0.0;

	        }
        });
    }
}
