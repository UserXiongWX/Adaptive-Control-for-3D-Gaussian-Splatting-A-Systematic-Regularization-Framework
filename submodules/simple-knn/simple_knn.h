#ifndef SIMPLEKNN_H_INCLUDED
#define SIMPLEKNN_H_INCLUDED
#include <cstdint>
class SimpleKNN
{
public:
	static void knn(int P, float3* points, float* meanDists);
	static void knn_indices(int P, int K_neighbors, float3* points, uint32_t* out_knn_indices, float* out_knn_sq_dists /* optional */);
};

#endif