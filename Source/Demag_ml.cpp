#ifdef AMREX_USE_ML

// MagneX_ML_Infer_Dynamic.cpp
#include "MagneX.H"
#include <torch/script.h>
#include <AMReX_Gpu.H>
// #include <AMReX_ParallelFor.H>
#include <AMReX_Print.H>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAContext.h>
using namespace amrex;

// ------------------------------------------------------------
// Helper: Move all parameters + buffers of a TorchScript module
// to the same device (e.g., cuda:3) to avoid device mismatch.
// ------------------------------------------------------------
void MoveModuleToDevice(torch::jit::script::Module& m,
                                      const torch::Device& device)
{
        m.to(device);
//     for (auto& p : m.named_parameters(true)) {
//         p.value().set_data(p.value().to(device));
//     }
//     for (auto& b : m.named_buffers(true)) {
//         b.value().set_data(b.value().to(device));
//     }
}

// ------------------------------------------------------------
// Helper: read expected_spatial = [nx, ny, nz] from normalizer.
// normalizer must have exported method get_expected_spatial().
// ------------------------------------------------------------
amrex::IntVect GetExpectedSpatial(torch::jit::script::Module& x_norm_module)
{
    at::Tensor t = x_norm_module.get_method("get_expected_spatial")({}).toTensor();
    t = t.to(torch::kCPU).to(torch::kLong).contiguous();

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(t.numel() == 3,
        "get_expected_spatial() must return a tensor of 3 elements: [nx, ny, nz]");

    const int nx = static_cast<int>(t[0].item<int64_t>());
    const int ny = static_cast<int>(t[1].item<int64_t>());
    const int nz = static_cast<int>(t[2].item<int64_t>());

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nx > 0 && ny > 0 && nz > 0,
        "expected_spatial invalid (must be >0)");

    return amrex::IntVect(nx, ny, nz);
}

// ------------------------------------------------------------
// Bundle: holds model + normalizers + expected shape meta.
// ------------------------------------------------------------
struct MLBundle
{
    torch::jit::script::Module x_norm;
    torch::jit::script::Module y_norm;
    torch::jit::script::Module model;

    amrex::IntVect expected_spatial{0}; // (nx,ny,nz)
    int device_id = 0;

    bool initialized = false;
};

// ------------------------------------------------------------
// Load bundle from .pt files; move everything to cuda:device_id;
// read expected_spatial from x_norm.
// ------------------------------------------------------------
static inline MLBundle LoadMLBundle(const std::string& x_norm_pt,
                                    const std::string& y_norm_pt,
                                    const std::string& model_pt,
                                    int device_id)
{
    MLBundle b;
    b.device_id = device_id;

    // Load modules (default loads to CPU)
    b.x_norm = torch::jit::load(x_norm_pt);
    b.y_norm = torch::jit::load(y_norm_pt);
    b.model  = torch::jit::load(model_pt);

    // Move to target GPU
    torch::Device dev(torch::kCUDA, device_id);
    MoveModuleToDevice(b.x_norm, dev);
    MoveModuleToDevice(b.y_norm, dev);
    MoveModuleToDevice(b.model,  dev);

    // Read expected spatial shape from normalizer
    b.expected_spatial = GetExpectedSpatial(b.x_norm);

    b.initialized = true;

    amrex::Print() << "[MLBundle] Loaded modules on cuda:" << device_id
                   << " expected_spatial = ("
                   << b.expected_spatial[0] << ", "
                   << b.expected_spatial[1] << ", "
                   << b.expected_spatial[2] << ")\n";

    return b;
}

// ------------------------------------------------------------
// Pack Mfield MultiFab (Mx,My,Mz) into Torch tensor [1,3,nx,ny,nz]
// Dynamically sized using bx.size() and expected_spatial.
// ------------------------------------------------------------
at::Tensor PackMfieldToTensorDynamic(
    const Array<MultiFab, AMREX_SPACEDIM>& Mfield,
    const MFIter& mfi,
    const Box& bx,
    const amrex::IntVect& expected_spatial,
    int device_id)
{
#if AMREX_SPACEDIM != 3
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false, "PackMfieldToTensorDynamic expects 3D");
#endif

    const auto& Mx = Mfield[0].const_array(mfi);
    const auto& My = Mfield[1].const_array(mfi);
    const auto& Mz = Mfield[2].const_array(mfi);

    const IntVect bx_lo = bx.smallEnd();
    const IntVect nbox  = bx.size(); // (nx,ny,nz)

    const int nx = nbox[0];
    const int ny = nbox[1];
    const int nz = nbox[2];
    const int ncell = nx * ny * nz;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        nx == expected_spatial[0] && ny == expected_spatial[1] && nz == expected_spatial[2],
        "PackMfieldToTensorDynamic: bx size != expected_spatial from normalizer"
    );

    amrex::Gpu::ManagedVector<Real> aux_Mx(ncell), aux_My(ncell), aux_Mz(ncell);
    Real* AMREX_RESTRICT auxPtr_Mx = aux_Mx.dataPtr();
    Real* AMREX_RESTRICT auxPtr_My = aux_My.dataPtr();
    Real* AMREX_RESTRICT auxPtr_Mz = aux_Mz.dataPtr();

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        const int ii = i - bx_lo[0];
        const int jj = j - bx_lo[1];
        const int kk = k - bx_lo[2];

        // flatten index consistent with (nx,ny,nz) reshape below
        const int index = kk + jj * nz + ii * nz * ny;

        auxPtr_Mx[index] = Mx(i, j, k);
        auxPtr_My[index] = My(i, j, k);
        auxPtr_Mz[index] = Mz(i, j, k);
    });

    amrex::Gpu::streamSynchronize();

    // Wrap managed memory; then we will immediately copy to CUDA tensor
    at::Tensor tMx = torch::from_blob(auxPtr_Mx, {ncell}, torch::kFloat64);
    at::Tensor tMy = torch::from_blob(auxPtr_My, {ncell}, torch::kFloat64);
    at::Tensor tMz = torch::from_blob(auxPtr_Mz, {ncell}, torch::kFloat64);

    at::Tensor rMx = tMx.reshape({nx, ny, nz});
    at::Tensor rMy = tMy.reshape({nx, ny, nz});
    at::Tensor rMz = tMz.reshape({nx, ny, nz});

    at::Tensor M = torch::stack({rMx, rMy, rMz}, 0); // [3,nx,ny,nz]

    torch::Device dev(torch::kCUDA, device_id);
    M = M.to(dev).to(torch::kFloat32).unsqueeze(0); // [1,3,nx,ny,nz]

    return M;
}

// ------------------------------------------------------------
// Normalize, Forward, Denormalize
// ------------------------------------------------------------
at::Tensor NormalizeInput(const at::Tensor& M_cuda_f32,
                          torch::jit::script::Module& x_norm_module)
{
    BL_PROFILE("NormalizeInput");
    return x_norm_module.get_method("encode")({M_cuda_f32}).toTensor();
}

// at::Tensor MLForwardOnly(const at::Tensor& norm_tensor,
//                          torch::jit::script::Module& ml_module)
// {
//     BL_PROFILE("MLForwardOnly");
//     auto out = ml_module.forward({norm_tensor}).toTensor();

// #ifdef AMREX_USE_CUDA
//     // 把 forward 的 GPU 时间“结算”在这里（用于 profiling 验证）
//     amrex::Gpu::streamSynchronize();
// #endif

//     return out;
// }
at::Tensor MLForwardOnly(const at::Tensor& norm_tensor,
                         torch::jit::script::Module& ml_module)
{
    BL_PROFILE("MLForwardOnly");

#ifdef AMREX_USE_CUDA
    at::cuda::CUDAEvent start(/*enable_timing=*/true);
    at::cuda::CUDAEvent stop (/*enable_timing=*/true);

    auto stream = at::cuda::getDefaultCUDAStream();

    stream.synchronize();

    start.record(stream);

    auto out = ml_module.forward({norm_tensor}).toTensor();

    stop.record(stream);
    stop.synchronize();

    float ms = start.elapsed_time(stop);
    amrex::Print() << "Forward-only time (ms) = " << ms << "\n";

    return out;
#else
    return ml_module.forward({norm_tensor}).toTensor();
#endif
}


at::Tensor DenormalizeOutput(const at::Tensor& y_tensor,
                             torch::jit::script::Module& y_norm_module)
{
    BL_PROFILE("DenormalizeOutput");
    // Keep device, convert to float64 for AMReX Real=double path
    return y_norm_module.get_method("decode")({y_tensor}).toTensor().to(torch::kFloat64);
}

// ------------------------------------------------------------
// Unpack Torch tensor [1,3,nx,ny,nz] into H_demagfield MultiFabs.
// ------------------------------------------------------------
void UnpackTensorToHfieldDynamic(
    const at::Tensor& denorm_torch_f64, // [1,3,nx,ny,nz]
    Array<MultiFab, AMREX_SPACEDIM>& H_demagfield,
    const MFIter& mfi,
    const Box& bx,
    const amrex::IntVect& expected_spatial)
{
#if AMREX_SPACEDIM != 3
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false, "UnpackTensorToHfieldDynamic expects 3D");
#endif

    auto Hx_demag = H_demagfield[0].array(mfi);
    auto Hy_demag = H_demagfield[1].array(mfi);
    auto Hz_demag = H_demagfield[2].array(mfi);

    const IntVect bx_lo = bx.smallEnd();
    const IntVect nbox  = bx.size();

    const int nx = nbox[0];
    const int ny = nbox[1];
    const int nz = nbox[2];

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        nx == expected_spatial[0] && ny == expected_spatial[1] && nz == expected_spatial[2],
        "UnpackTensorToHfieldDynamic: bx size != expected_spatial from normalizer"
    );

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(denorm_torch_f64.dim() == 5, "denorm must be [1,3,nx,ny,nz]");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(denorm_torch_f64.size(0) == 1, "batch must be 1");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(denorm_torch_f64.size(1) == 3, "channel must be 3");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(denorm_torch_f64.size(2) == nx &&
                                    denorm_torch_f64.size(3) == ny &&
                                    denorm_torch_f64.size(4) == nz,
                                    "denorm tensor spatial mismatch");

    // Flatten each component to 1D for packed_accessor
    at::Tensor Hx = denorm_torch_f64.select(0, 0).select(0, 0).contiguous().view({-1});
    at::Tensor Hy = denorm_torch_f64.select(0, 0).select(0, 1).contiguous().view({-1});
    at::Tensor Hz = denorm_torch_f64.select(0, 0).select(0, 2).contiguous().view({-1});

#ifdef AMREX_USE_CUDA
    auto Hx_acc = Hx.packed_accessor64<Real, 1>();
    auto Hy_acc = Hy.packed_accessor64<Real, 1>();
    auto Hz_acc = Hz.packed_accessor64<Real, 1>();
#endif

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        const int ii = i - bx_lo[0];
        const int jj = j - bx_lo[1];
        const int kk = k - bx_lo[2];

        const int index = kk + jj * nz + ii * nz * ny;

        Hx_demag(i, j, k) = Hx_acc[index];
        Hy_demag(i, j, k) = Hy_acc[index];
        Hz_demag(i, j, k) = Hz_acc[index];
    });

    amrex::Gpu::streamSynchronize();
}

// ------------------------------------------------------------
// One-call wrapper: pack -> encode -> forward -> decode -> unpack
// ------------------------------------------------------------
void RunMLDemagOnBox(
    MLBundle& b,
    const Array<MultiFab, AMREX_SPACEDIM>& Mfield,
    Array<MultiFab, AMREX_SPACEDIM>& H_demagfield,
    const MFIter& mfi,
    const Box& bx)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(b.initialized, "MLBundle not initialized");

    // 1) Pack raw M to [1,3,nx,ny,nz] on cuda:device_id
    at::Tensor M = PackMfieldToTensorDynamic(Mfield, mfi, bx, b.expected_spatial, b.device_id);

    // 2) Normalize (encode) on same GPU
    at::Tensor norm = NormalizeInput(M, b.x_norm);

    // 3) Forward
    at::Tensor pred = MLForwardOnly(norm, b.model);

    // 4) Denormalize
    at::Tensor denorm = DenormalizeOutput(pred, b.y_norm);

    // 5) Unpack back to MultiFab
    UnpackTensorToHfieldDynamic(denorm, H_demagfield, mfi, bx, b.expected_spatial);
}

#endif
