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

Real AnisotropyEnergy(MultiFab& Ms,
                      MultiFab& Mfield_x,
                      MultiFab& Mfield_y,
                      MultiFab& Mfield_z,
		      MultiFab& anisotropy)
{
    // timer for profiling
    // BL_PROFILE_VAR("SumNormalizedM()",SumNormalizedM);

    ReduceOps<ReduceOpSum> reduce_op;
    
    ReduceData<Real> reduce_data(reduce_op);

    using ReduceTuple = typename decltype(reduce_data)::Type;

    int comp=0;
    Real K = anisotropy.max(comp);

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
		return {-(K) * std::pow((Mx(i,j,k)*anisotropy_axis[0] + My(i,j,k)*anisotropy_axis[1] + Mz(i,j,k)*anisotropy_axis[2]), 2)};
            } else {
                return {0.};
            }
        });
    }

    Real sum = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceRealSum(sum);

    return sum;
}

