#include <torch/extension.h>
#include <tuple>

torch::Tensor distCUDA2(const torch::Tensor& points);
std::tuple<torch::Tensor, torch::Tensor> knn_indices_cuda(const torch::Tensor& points, int K_neighbors);