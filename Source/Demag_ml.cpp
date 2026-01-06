#include "MagneX.H"
#include <torch/script.h>

using namespace amrex;

void CalculateH_demag_ML(const Array<MultiFab, AMREX_SPACEDIM>& Mfield,
                         torch::jit::script::Module& x_norm_module,
                         torch::jit::script::Module& ml_module,
                         torch::jit::script::Module& y_norm_module,
                         Array<MultiFab, AMREX_SPACEDIM>& H_demagfield)
{
    BL_PROFILE_VAR("CalculateH_demag_ML()", CalculateH_demag_ML);

    for (MFIter mfi(Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.validbox();

        const auto& Mx = Mfield[0].const_array(mfi);
        const auto& My = Mfield[1].const_array(mfi);
        const auto& Mz = Mfield[2].const_array(mfi);

        auto Hx_demag = H_demagfield[0].array(mfi);
        auto Hy_demag = H_demagfield[1].array(mfi);
        auto Hz_demag = H_demagfield[2].array(mfi);

        const IntVect bx_lo = bx.smallEnd();
        const IntVect nbox  = bx.size();

#if AMREX_SPACEDIM == 2
        const int ncell = nbox[0] * nbox[1];
#else
        const int ncell = nbox[0] * nbox[1] * nbox[2];
#endif

        // Host-visible (Managed) buffers filled on GPU
        amrex::Gpu::ManagedVector<Real> aux_Mx(ncell);
        amrex::Gpu::ManagedVector<Real> aux_My(ncell);
        amrex::Gpu::ManagedVector<Real> aux_Mz(ncell);

        Real* AMREX_RESTRICT auxPtr_Mx = aux_Mx.dataPtr();
        Real* AMREX_RESTRICT auxPtr_My = aux_My.dataPtr();
        Real* AMREX_RESTRICT auxPtr_Mz = aux_Mz.dataPtr();

        // Fill aux buffers from MultiFab on GPU
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const int ii = i - bx_lo[0];
            const int jj = j - bx_lo[1];

#if AMREX_SPACEDIM == 2
            const int index = jj + ii * nbox[1];
#else
            const int kk = k - bx_lo[2];
            const int index = kk + jj * nbox[2] + ii * nbox[2] * nbox[1];
#endif

            auxPtr_Mx[index] = Mx(i, j, k);
            auxPtr_My[index] = My(i, j, k);
            auxPtr_Mz[index] = Mz(i, j, k);
        });

        // Make sure aux buffers are ready for from_blob on host side
        amrex::Gpu::streamSynchronize();

        // Wrap buffers as CPU tensors (no copy)
        at::Tensor inputs_torch_Mx = torch::from_blob(auxPtr_Mx, {ncell, 1}, torch::kFloat64);
        at::Tensor inputs_torch_My = torch::from_blob(auxPtr_My, {ncell, 1}, torch::kFloat64);
        at::Tensor inputs_torch_Mz = torch::from_blob(auxPtr_Mz, {ncell, 1}, torch::kFloat64);

        // Reshape (assumes ncell == 128*128*4)
        at::Tensor reshaped_Mx = inputs_torch_Mx.reshape({128, 128, 4});
        at::Tensor reshaped_My = inputs_torch_My.reshape({128, 128, 4});
        at::Tensor reshaped_Mz = inputs_torch_Mz.reshape({128, 128, 4});

        // Stack into [3, 128, 128, 4] then add batch dim -> [1, 3, 128, 128, 4]
        at::Tensor final_tensor_M = torch::stack({reshaped_Mx, reshaped_My, reshaped_Mz}, 0);
        final_tensor_M = final_tensor_M.to(torch::kCUDA).to(torch::kFloat32);
        final_tensor_M = final_tensor_M.unsqueeze(0);

        // Normalize -> model -> denormalize
        at::Tensor norm_torch    = x_norm_module.get_method("encode")({final_tensor_M}).toTensor();
        at::Tensor outputs_torch = ml_module.forward({norm_torch}).toTensor();
        at::Tensor denorm_torch  = y_norm_module.get_method("decode")({outputs_torch}).toTensor();

        // Convert to float64 for accessor<Real,...> usage (same as your original)
        denorm_torch = denorm_torch.to(torch::kFloat64);

        // Extract H components: denorm_torch shape assumed [1, 3, 128, 128, 4]
        at::Tensor denorm_torch_Hx = denorm_torch.select(0, 0).select(0, 0).flatten();
        at::Tensor denorm_torch_Hy = denorm_torch.select(0, 0).select(0, 1).flatten();
        at::Tensor denorm_torch_Hz = denorm_torch.select(0, 0).select(0, 2).flatten();

#ifdef AMREX_USE_CUDA
        auto denorm_torch_Hx_acc = denorm_torch_Hx.packed_accessor64<Real, 1>();
        auto denorm_torch_Hy_acc = denorm_torch_Hy.packed_accessor64<Real, 1>();
        auto denorm_torch_Hz_acc = denorm_torch_Hz.packed_accessor64<Real, 1>();
#endif

        // Copy tensor data back into demag field
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const int ii = i - bx_lo[0];
            const int jj = j - bx_lo[1];
            const int kk = k - bx_lo[2];
            const int index = kk + jj * nbox[2] + ii * nbox[2] * nbox[1];

            Hx_demag(i, j, k) = denorm_torch_Hx_acc[index];
            Hy_demag(i, j, k) = denorm_torch_Hy_acc[index];
            Hz_demag(i, j, k) = denorm_torch_Hz_acc[index];
        });

        amrex::Gpu::streamSynchronize();
    }
}