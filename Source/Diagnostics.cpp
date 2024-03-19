#include "MagneX.H"
#include<cmath>

long CountMagneticCells(MultiFab& Ms)
{
    // timer for profiling
    BL_PROFILE_VAR("CountMagneticCells()",CountMagneticCells);

    ReduceOps<ReduceOpSum> reduce_op;
    
    ReduceData<long> reduce_data(reduce_op);

    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(Ms,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.tilebox();

        auto const& fab = Ms.array(mfi);

        reduce_op.eval(bx, reduce_data,
                       [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            if (fab(i,j,k) > 0.) {
                return {1};
            } else {
                return {0};
            }
        });
    }

    long sum = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceLongSum(sum);

    return sum;
}


Real SumNormalizedM(MultiFab& Ms,
                    MultiFab& Mfield)
{
    // timer for profiling
    BL_PROFILE_VAR("SumNormalizedM()",SumNormalizedM);

    ReduceOps<ReduceOpSum> reduce_op;
    
    ReduceData<Real> reduce_data(reduce_op);

    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(Ms,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.tilebox();

        auto const& fab = Ms.array(mfi);
        auto const& M = Mfield.array(mfi);
        
	reduce_op.eval(bx, reduce_data,
                       [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            if (fab(i,j,k) > 0.) {
                return {M(i,j,k)/fab(i,j,k)};
            } else {
                return {0.};
            }
	});
    }

    Real sum = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceRealSum(sum);

    return sum;
}

Real SumHeff(MultiFab& H_demagfield,
	     MultiFab& H_exchangefield,
	     MultiFab& H_biasfield)
{
    // timer for profiling
    BL_PROFILE_VAR("SumNormalizedM()",SumNormalizedM);

    ReduceOps<ReduceOpSum> reduce_op;
    
    ReduceData<Real> reduce_data(reduce_op);

    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(H_demagfield,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.tilebox();

        // extract field data
        auto const& H_demag = H_demagfield.array(mfi);
	auto const& H_bias = H_biasfield.array(mfi);
        auto const& H_exch = H_exchangefield.array(mfi);
	// auto const& H_anis = H_anisotropyfield.array(mfi);

        reduce_op.eval(bx, reduce_data,
                       [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            return {H_demag(i,j,k) + H_bias(i,j,k) + H_exch(i,j,k)};
	});
    }

    Real sum = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceRealSum(sum);

    return sum;
}

/*
Real Coercivity(Array< MultiFab, AMREX_SPACEDIM >& Mfield,
	        Array< MultiFab, AMREX_SPACEDIM >& Heff,
		IntVect& location)

{

	Real magnitude = 0.0;

	for ( MFIter mfi(Mfield[0]); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

	    // Real& magnitude_ptr = magnitude;

	    const Array4<Real const>& Mx_ptr = Mfield[0].array(mfi);
            const Array4<Real const>& My_ptr = Mfield[1].array(mfi);
            const Array4<Real const>& Mz_ptr = Mfield[2].array(mfi);

            const Array4<Real const>& Hx_ptr = Heff[0].array(mfi);
            const Array4<Real const>& Hy_ptr = Heff[1].array(mfi);
            const Array4<Real const>& Hz_ptr = Heff[2].array(mfi);

            amrex::ParallelFor( bx, [&] AMREX_GPU_DEVICE (int i, int j, int k)
            {

                // Check if the dot product of M and H is zero
	        // If so, record the point in the field that this occurs, and the magnitude of the field at that point
	        if(fabs(Mx_ptr(i,j,k)*Hx_ptr(i,j,k) + My_ptr(i,j,k)*Hx_ptr(i,j,k) + Mz_ptr(i,j,k)*Hz_ptr(i,j,k)) <= 1e-5) {
		    Real temp_magnitude = sqrt((Mx_ptr(i,j,k)*Mx_ptr(i,j,k)) + (My_ptr(i,j,k)*My_ptr(i,j,k)) + (Mz_ptr(i,j,k)*Mz_ptr(i,j,k)));

		    // location = (i,j,k);
	            magnitude = temp_magnitude;
	            //indicate that the coercivity exists
	        }
    	    });
         }
	return magnitude;
}
*/

Real AnisotropyEnergy(MultiFab& Ms,
                      MultiFab& Mfield_x,
                      MultiFab& Mfield_y,
                      MultiFab& Mfield_z,
		      MultiFab& anisotropy,
		      long angle[])
{
    // timer for profiling
    // BL_PROFILE_VAR("SumNormalizedM()",SumNormalizedM);

    ReduceOps<ReduceOpSum> reduce_op;
    
    ReduceData<Real> reduce_data(reduce_op);

    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(Ms,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.tilebox();

        auto const& fab = Ms.array(mfi);
        auto const& Mx = Mfield_x.array(mfi);
        auto const& My = Mfield_y.array(mfi);
        auto const& Mz = Mfield_z.array(mfi);
        auto const& anis = anisotropy.array(mfi);

        reduce_op.eval(bx, reduce_data,
                       [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            if (fab(i,j,k) > 0.) {
		return {-(anis(i,j,k)) * std::pow((Mx(i,j,k)*angle[0] + My(i,j,k)*angle[1] + Mz(i,j,k)*angle[2]), 2)};
            } else {
                return {0.};
            }
        });
    }

    Real sum = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceRealSum(sum);

    return sum;
}

