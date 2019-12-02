#include "filter.cuh"

__global__ void filterHot(Candidate* dst, double* ndvi, double* ts, double* net_radiation,
					   double* soil_heat, double* ho, int* nvalid, int line, int size){

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	while(i < size){

		if(!isnan(ndvi[i]) && ndvi[i] > 0.15 && ndvi[i] < 0.20 && ts[i] > 273.16){
			dst[atomicAdd(nvalid, 1)] = Candidate(ndvi[i], ts[i], net_radiation[i], soil_heat[i], ho[i], line, i);
		}

		i += blockDim.x * gridDim.x;

	}

}

__global__ void finalFilterHot(Candidate* dst, double* ndvi, double* ts, double* net_radiation,
					   double* soil_heat, double* ho, int* nvalid, double ho_max, double ho_min, double surfaceTempHot, int line, int size){

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	while(i < size){

		if(definitelyGreaterThanDev(ho[i], ho_min) && definitelyLessThanDev(ho[i], ho_max) && essentiallyEqualDev(ts[i], surfaceTempHot)){
			dst[atomicAdd(nvalid, 1)] = Candidate(ndvi[i], ts[i], net_radiation[i], soil_heat[i], ho[i], line, i);
		}

		i += blockDim.x * gridDim.x;

	}

}

__global__ void filterCold(Candidate* dst, double* ndvi, double* ts, double* net_radiation,
					   double* soil_heat, double* ho, int* nvalid, int line, int size){

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	while(i < size){

		if(!isnan(ndvi[i]) && !isnan(ho[i]) && ndvi[i] < 0 && ts[i] > 273.16){
			dst[atomicAdd(nvalid, 1)] = Candidate(ndvi[i], ts[i], net_radiation[i], soil_heat[i], ho[i], line, i);
		}

		i += blockDim.x * gridDim.x;

	}

}


/**
 * @brief  Determines if a and b are approximately equals based on a epsilon.
 * @param  a: First value.
 * @param  b: Second value.
 * @retval TRUE if they are approximately equals, and FALSE otherwise.
 */
__device__ bool approximatelyEqualDev(double a, double b) {
	return fabs(a - b) <= ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * 1e-7);
}

/**
 * @brief  Determines if a and b are essentially equals based on a epsilon.
 * @param  a: First value.
 * @param  b: Second value.
 * @retval TRUE if they are essentially equals, and FALSE otherwise.
 */
__device__ bool essentiallyEqualDev(double a, double b) {
	return fabs(a - b) <= ((fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * 1e-7);
}

/**
 * @brief  Determines if a is definitely greater than b based on a epsilon.
 * @param  a: First value.
 * @param  b: Second value.
 * @retval TRUE if a is definitely greater than b, and FALSE otherwise.
 */
__device__ bool definitelyGreaterThanDev(double a, double b) {
	return (a - b) > ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * 1e-7);
}

/**
 * @brief  Determines if a is definitely less than b based on a epsilon.
 * @param  a: First value.
 * @param  b: Second value.
 * @retval TRUE if a is definitely less than b, and FALSE otherwise.
 */
__device__ bool definitelyLessThanDev(double a, double b) {
	return (b - a) > ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * 1e-7);
}
