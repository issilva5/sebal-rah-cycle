#include "cuda-utils.h"
#include "types.h"
#include "device_functions.h"
#include <thrust/device_vector.h>
#include "candidate.h"

__global__ void filterHot(Candidate* dst, double* ndvi, double* ts, double* net_radiation,
					   double* soil_heat, double* ho, int* nvalid,  int line, int size);

__global__ void finalFilterHot(Candidate* dst, double* ndvi, double* ts, double* net_radiation,
					   double* soil_heat, double* ho, int* nvalid, double ho_max, double ho_min, double surfaceTempHot, int line, int size);

__global__ void filterCold(Candidate* dst, double* ndvi, double* ts, double* net_radiation,
					   double* soil_heat, double* ho, int* nvalid,  int line, int size);

__global__ void asebalFilterCold(Candidate* dst, double* ndvi, double* ts, double* albedo, double* net_radiation, double* soil_heat, double* ho,
		int* nvalid, double* albedoQuartile, double* ndviQuartile, double* tsQuartile, int line, int size);

__global__ void asebalFilterHot(Candidate* dst, double* ndvi, double* ts, double* albedo, double* net_radiation, double* soil_heat, double* ho,
		int* nvalid, double* albedoQuartile, double* ndviQuartile, double* tsQuartile, int line, int size);

__device__ bool checkLandCode(int value);

__global__ void landCoverHomogeneityKernel(double* inputBuffer, int* output, int line, int numCol, int numLine);

/**
 * @brief   Tests the ndvi, surface_temperature and albedo homogeneity of a pixel.
 * @note    A pixel is homogeneous in these criteria if inside a 7x7 window the coefficient of variation of the albedo and ndvi is less or equal than 25%
 *          and the surface temperature has a standard deviation less or equal than 1.5 K.
 * @param   ndvi: NDVI TIFF.
 * @param   surface_temperature: TS TIFF.
 * @param   albedo: Albedo TIFF.
 * @param   maskLC: A binary TIFF conteining the data of the land cover homogeneity.
 * @param   output: A binary TIFF, where pixels with 1 means that is a homogeneous pixel in land cover, ndvi, surface temperature and albedo, and 0 means otherwise.
 */
__global__ void testHomogeneityKernel(double* ndviBuffer, double* tsBuffer, double* albedoBuffer, int* lcBuffer, int* output, int line, int numCol, int numLine);

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
