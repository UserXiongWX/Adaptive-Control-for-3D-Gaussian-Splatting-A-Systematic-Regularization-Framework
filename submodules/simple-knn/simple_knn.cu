#define BOX_SIZE 1024

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "simple_knn.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <vector>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#define __CUDACC__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

struct CustomMin
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
	}
};

struct CustomMax
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
	}
};

__host__ __device__ uint32_t prepMorton(uint32_t x)
{
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;
	return x;
}

__host__ __device__ uint32_t coord2Morton(float3 coord, float3 minn, float3 maxx)
{
	uint32_t x = prepMorton(((coord.x - minn.x) / (maxx.x - minn.x)) * ((1 << 10) - 1));
	uint32_t y = prepMorton(((coord.y - minn.y) / (maxx.y - minn.y)) * ((1 << 10) - 1));
	uint32_t z = prepMorton(((coord.z - minn.z) / (maxx.z - minn.z)) * ((1 << 10) - 1));

	return x | (y << 1) | (z << 2);
}

__global__ void coord2Morton(int P, const float3* points, float3 minn, float3 maxx, uint32_t* codes)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	codes[idx] = coord2Morton(points[idx], minn, maxx);
}

struct MinMax
{
	float3 minn;
	float3 maxx;
};

__global__ void boxMinMax(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes)
{
	auto idx = cg::this_grid().thread_rank();

	MinMax me;
	if (idx < P)
	{
		me.minn = points[indices[idx]];
		me.maxx = points[indices[idx]];
	}
	else
	{
		me.minn = { FLT_MAX, FLT_MAX, FLT_MAX };
		me.maxx = { -FLT_MAX,-FLT_MAX,-FLT_MAX };
	}

	__shared__ MinMax redResult[BOX_SIZE];

	for (int off = BOX_SIZE / 2; off >= 1; off /= 2)
	{
		if (threadIdx.x < 2 * off)
			redResult[threadIdx.x] = me;
		__syncthreads();

		if (threadIdx.x < off)
		{
			MinMax other = redResult[threadIdx.x + off];
			me.minn.x = min(me.minn.x, other.minn.x);
			me.minn.y = min(me.minn.y, other.minn.y);
			me.minn.z = min(me.minn.z, other.minn.z);
			me.maxx.x = max(me.maxx.x, other.maxx.x);
			me.maxx.y = max(me.maxx.y, other.maxx.y);
			me.maxx.z = max(me.maxx.z, other.maxx.z);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
		boxes[blockIdx.x] = me;
}

__device__ __host__ float distBoxPoint(const MinMax& box, const float3& p)
{
	float3 diff = { 0, 0, 0 };
	if (p.x < box.minn.x || p.x > box.maxx.x)
		diff.x = min(abs(p.x - box.minn.x), abs(p.x - box.maxx.x));
	if (p.y < box.minn.y || p.y > box.maxx.y)
		diff.y = min(abs(p.y - box.minn.y), abs(p.y - box.maxx.y));
	if (p.z < box.minn.z || p.z > box.maxx.z)
		diff.z = min(abs(p.z - box.minn.z), abs(p.z - box.maxx.z));
	return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

template<int K>
__device__ void updateKBest(const float3& ref, const float3& point, float* knn)
{
	float3 d = { point.x - ref.x, point.y - ref.y, point.z - ref.z };
	float dist = d.x * d.x + d.y * d.y + d.z * d.z;
	for (int j = 0; j < K; j++)
	{
		if (knn[j] > dist)
		{
			float t = knn[j];
			knn[j] = dist;
			dist = t;
		}
	}
}

__global__ void boxMeanDist(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes, float* dists)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 point = points[indices[idx]];
	float best[3] = { FLT_MAX, FLT_MAX, FLT_MAX };

	for (int i = max(0, idx - 3); i <= min(P - 1, idx + 3); i++)
	{
		if (i == idx)
			continue;
		updateKBest<3>(point, points[indices[i]], best);
	}

	float reject = best[2];
	best[0] = FLT_MAX;
	best[1] = FLT_MAX;
	best[2] = FLT_MAX;

	for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++)
	{
		MinMax box = boxes[b];
		float dist = distBoxPoint(box, point);
		if (dist > reject || dist > best[2])
			continue;

		for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++)
		{
			if (i == idx)
				continue;
			updateKBest<3>(point, points[indices[i]], best);
		}
	}
	dists[indices[idx]] = (best[0] + best[1] + best[2]) / 3.0f;
}


void SimpleKNN::knn(int P, float3* points, float* meanDists)
{
	float3* result;
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;

	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

	thrust::device_vector<uint32_t> morton(P);
	thrust::device_vector<uint32_t> morton_sorted(P);
	coord2Morton << <(P + 255) / 256, 256 >> > (P, points, minn, maxx, morton.data().get());

	thrust::device_vector<uint32_t> indices(P);
	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<uint32_t> indices_sorted(P);

	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	temp_storage.resize(temp_storage_bytes);

	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);

	uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
	thrust::device_vector<MinMax> boxes(num_boxes);
	boxMinMax << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get());
	boxMeanDist << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get(), meanDists);

	cudaFree(result);
}

template<int K_NEIGHBORS>
__device__ void updateKBestWithIndices(
    const float3& ref_point,
    const float3& query_point,
    const uint32_t query_point_original_idx,
    float* k_best_sq_dists,
    uint32_t* k_best_indices)
{
    float3 d = { query_point.x - ref_point.x, query_point.y - ref_point.y, query_point.z - ref_point.z };
    float dist_sq = d.x * d.x + d.y * d.y + d.z * d.z;

    for (int j = 0; j < K_NEIGHBORS; j++) {
        if (dist_sq < k_best_sq_dists[j]) {
            for (int l = K_NEIGHBORS - 1; l > j; l--) {
                k_best_sq_dists[l] = k_best_sq_dists[l - 1];
                k_best_indices[l] = k_best_indices[l - 1];
            }
            k_best_sq_dists[j] = dist_sq;
            k_best_indices[j] = query_point_original_idx;
            break;
        }
    }
}

template<int K_NEIGHBORS>
__global__ void boxKnnIndices(
    uint32_t P,
    const float3* points,
    const uint32_t* sorted_indices,
    const MinMax* boxes,
    const uint32_t num_total_boxes,
    uint32_t* out_knn_indices_flat,
    float* out_knn_sq_dists_flat)
{
    int sorted_idx = cg::this_grid().thread_rank();
    if (sorted_idx >= P)
        return;

    uint32_t original_ref_idx = sorted_indices[sorted_idx];
    float3 ref_point = points[original_ref_idx];

    float k_best_sq_dists_local[K_NEIGHBORS];
    uint32_t k_best_indices_local[K_NEIGHBORS];
    for (int k = 0; k < K_NEIGHBORS; ++k) {
        k_best_sq_dists_local[k] = FLT_MAX;
        k_best_indices_local[k] = P;
    }

    const int local_search_window = K_NEIGHBORS * 2 + 10;
    for (int offset = -local_search_window / 2; offset <= local_search_window / 2; ++offset) {
        if (offset == 0) continue;
        int neighbor_sorted_idx = sorted_idx + offset;
        if (neighbor_sorted_idx >= 0 && neighbor_sorted_idx < P) {
            uint32_t original_query_idx = sorted_indices[neighbor_sorted_idx];
            float3 query_point = points[original_query_idx];
            updateKBestWithIndices<K_NEIGHBORS>(ref_point, query_point, original_query_idx, k_best_sq_dists_local, k_best_indices_local);
        }
    }
    
    float reject_sq_dist = k_best_sq_dists_local[K_NEIGHBORS - 1];

    for (uint32_t b = 0; b < num_total_boxes; ++b) {
        MinMax box = boxes[b];
        float dist_to_box_sq = distBoxPoint(box, ref_point);
        if (dist_to_box_sq < reject_sq_dist) {
            for (int i_in_box_sorted = b * BOX_SIZE; i_in_box_sorted < min(P, (b + 1) * BOX_SIZE); ++i_in_box_sorted) {
                uint32_t original_query_idx = sorted_indices[i_in_box_sorted];
                if (original_query_idx == original_ref_idx) continue;

                float3 query_point = points[original_query_idx];
                updateKBestWithIndices<K_NEIGHBORS>(ref_point, query_point, original_query_idx, k_best_sq_dists_local, k_best_indices_local);
                reject_sq_dist = k_best_sq_dists_local[K_NEIGHBORS - 1];
            }
        }
    }

    for (int k = 0; k < K_NEIGHBORS; ++k) {
        out_knn_indices_flat[original_ref_idx * K_NEIGHBORS + k] = k_best_indices_local[k];
        if (out_knn_sq_dists_flat != nullptr) {
            out_knn_sq_dists_flat[original_ref_idx * K_NEIGHBORS + k] = k_best_sq_dists_local[k];
        }
    }
}

void SimpleKNN::knn_indices(
    int P,
    int K_neighbors,
    float3* points,
    uint32_t* out_knn_indices,
    float* out_knn_sq_dists)
{
    if (K_neighbors <= 0) return;

    float3* result_minmax_gpu;
    cudaMalloc(&result_minmax_gpu, sizeof(float3));
    size_t temp_storage_bytes;
    float3 init_min = { FLT_MAX, FLT_MAX, FLT_MAX };
	float3 init_max = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    float3 min_coord, max_coord;

    cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result_minmax_gpu, P, CustomMin(), init_min);
    thrust::device_vector<char> temp_storage_reduce(temp_storage_bytes);
    cub::DeviceReduce::Reduce(temp_storage_reduce.data().get(), temp_storage_bytes, points, result_minmax_gpu, P, CustomMin(), init_min);
    cudaMemcpy(&min_coord, result_minmax_gpu, sizeof(float3), cudaMemcpyDeviceToHost);

    cub::DeviceReduce::Reduce(temp_storage_reduce.data().get(), temp_storage_bytes, points, result_minmax_gpu, P, CustomMax(), init_max);
    cudaMemcpy(&max_coord, result_minmax_gpu, sizeof(float3), cudaMemcpyDeviceToHost);
    cudaFree(result_minmax_gpu);

    thrust::device_vector<uint32_t> morton_codes_gpu(P);
    thrust::device_vector<uint32_t> morton_codes_sorted_gpu(P);
    coord2Morton << <(P + 255) / 256, 256 >> > (P, points, min_coord, max_coord, morton_codes_gpu.data().get());

    thrust::device_vector<uint32_t> original_indices_gpu(P);
    thrust::sequence(original_indices_gpu.begin(), original_indices_gpu.end());
    thrust::device_vector<uint32_t> sorted_original_indices_gpu(P);

    size_t temp_storage_sort_bytes;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_sort_bytes, morton_codes_gpu.data().get(), morton_codes_sorted_gpu.data().get(), original_indices_gpu.data().get(), sorted_original_indices_gpu.data().get(), P);
    thrust::device_vector<char> temp_storage_sort(temp_storage_sort_bytes);
    cub::DeviceRadixSort::SortPairs(temp_storage_sort.data().get(), temp_storage_sort_bytes, morton_codes_gpu.data().get(), morton_codes_sorted_gpu.data().get(), original_indices_gpu.data().get(), sorted_original_indices_gpu.data().get(), P);

    uint32_t num_spatial_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
    thrust::device_vector<MinMax> spatial_boxes_gpu(num_spatial_boxes);
    boxMinMax << <num_spatial_boxes, BOX_SIZE >> > (P, points, sorted_original_indices_gpu.data().get(), spatial_boxes_gpu.data().get());
    cudaDeviceSynchronize();

    if (K_neighbors == 5) {
        boxKnnIndices<5><<<(P + 255) / 256, 256>>>(
            P,
            points,
            sorted_original_indices_gpu.data().get(),
            spatial_boxes_gpu.data().get(),
            num_spatial_boxes,
            out_knn_indices,
            out_knn_sq_dists);
    } else if (K_neighbors == 3) {
         boxKnnIndices<3><<<(P + 255) / 256, 256>>>(
            P,
            points,
            sorted_original_indices_gpu.data().get(),
            spatial_boxes_gpu.data().get(),
            num_spatial_boxes,
            out_knn_indices,
            out_knn_sq_dists);
    }
    cudaDeviceSynchronize();
}