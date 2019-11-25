#include "filter.cuh"

__global__ void filter(Candidate* dst, double* ndvi, double* ts, double* net_radiation,
					   double* soil_heat, double* ho, int* nvalid, int line, int size){

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	while(i < size){
		if(isnan(ndvi[i]) && ndvi[i] > 0.15 && ndvi[i] < 0.20 && ts[i] > 273.16){
			dst[atomicAdd(nvalid, 1)] = Candidate(ndvi[i], ts[i], net_radiation[i], soil_heat[i], ho[i], line, i);
		}

		i += blockDim.x * gridDim.x;

	}

}
