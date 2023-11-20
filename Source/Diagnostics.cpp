#include "Diagnostics.H"

long CountMagneticCells(MultiFab& Ms)
{
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
