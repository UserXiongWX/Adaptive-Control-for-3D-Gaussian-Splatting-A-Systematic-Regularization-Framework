#include "spatial.h"
#include "simple_knn.h"

torch::Tensor
distCUDA2(const torch::Tensor& points)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);
  
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());

  return means;
}

std::tuple<torch::Tensor, torch::Tensor>
knn_indices_cuda(
    const torch::Tensor& points,
    const int K_neighbors)
{
    const int P = points.size(0);
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must be Px3");
    TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
    TORCH_CHECK(points.is_contiguous(), "points must be contiguous");
    TORCH_CHECK(K_neighbors > 0, "K_neighbors must be positive");

    auto uint_opts = points.options().dtype(torch::kInt32);
    auto float_opts = points.options().dtype(torch::kFloat32);

    torch::Tensor out_indices = torch::empty({P, K_neighbors}, uint_opts);
    torch::Tensor out_sq_dists = torch::empty({P, K_neighbors}, float_opts);

    SimpleKNN::knn_indices(
        P,
        K_neighbors,
        (float3*)points.data_ptr<float>(),
        (uint32_t*)out_indices.data_ptr<int32_t>(),
        (float*)out_sq_dists.data_ptr<float>()
    );

    return std::make_tuple(out_indices, out_sq_dists);
}