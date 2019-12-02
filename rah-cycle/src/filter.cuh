#include "cuda-utils.h"
#include "types.h"
#include "device_functions.h"
#include "candidate.h"

__global__ void filterHot(Candidate* dst, double* ndvi, double* ts, double* net_radiation,
					   double* soil_heat, double* ho, int* nvalid,  int line, int size);

__global__ void finalFilterHot(Candidate* dst, double* ndvi, double* ts, double* net_radiation,
					   double* soil_heat, double* ho, int* nvalid, double ho_max, double ho_min, double surfaceTempHot, int line, int size);

__global__ void filterCold(Candidate* dst, double* ndvi, double* ts, double* net_radiation,
					   double* soil_heat, double* ho, int* nvalid,  int line, int size);


/**
 * @brief  Determines if a and b are approximately equals based on a epsilon.
 * @param  a: First value.
 * @param  b: Second value.
 * @retval TRUE if they are approximately equals, and FALSE otherwise.
 */
__device__ bool approximatelyEqualDev(double a, double b);

/**
 * @brief  Determines if a and b are essentially equals based on a epsilon.
 * @param  a: First value.
 * @param  b: Second value.
 * @retval TRUE if they are essentially equals, and FALSE otherwise.
 */
__device__ bool essentiallyEqualDev(double a, double b);

/**
 * @brief  Determines if a is definitely greater than b based on a epsilon.
 * @param  a: First value.
 * @param  b: Second value.
 * @retval TRUE if a is definitely greater than b, and FALSE otherwise.
 */
__device__ bool definitelyGreaterThanDev(double a, double b);

/**
 * @brief  Determines if a is definitely less than b based on a epsilon.
 * @param  a: First value.
 * @param  b: Second value.
 * @retval TRUE if a is definitely less than b, and FALSE otherwise.
 */
__device__ bool definitelyLessThanDev(double a, double b);
