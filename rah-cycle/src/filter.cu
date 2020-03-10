#include "filter.cuh"

// Our Sebal

__global__ void filterHot(Candidate* dst, double* ndvi, double* ts, double* net_radiation, double* soil_heat, double* ho, int* nvalid, int line,
		int size) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	while (i < size) {

		if (!isnan(ndvi[i]) && ndvi[i] > 0.15 && ndvi[i] < 0.20 && ts[i] > 273.16) {
			dst[atomicAdd(nvalid, 1)] = Candidate(ndvi[i], ts[i], net_radiation[i], soil_heat[i], ho[i], line, i);
		}

		i += blockDim.x * gridDim.x;

	}

}

__global__ void finalFilterHot(Candidate* dst, double* ndvi, double* ts, double* net_radiation, double* soil_heat, double* ho, int* nvalid,
		double ho_max, double ho_min, double surfaceTempHot, int line, int size) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	while (i < size) {

		if (definitelyGreaterThanDev(ho[i], ho_min) && definitelyLessThanDev(ho[i], ho_max) && essentiallyEqualDev(ts[i], surfaceTempHot)) {
			dst[atomicAdd(nvalid, 1)] = Candidate(ndvi[i], ts[i], net_radiation[i], soil_heat[i], ho[i], line, i);
		}

		i += blockDim.x * gridDim.x;

	}

}

__global__ void filterCold(Candidate* dst, double* ndvi, double* ts, double* net_radiation, double* soil_heat, double* ho, int* nvalid, int line,
		int size) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	while (i < size) {

		if (!isnan(ndvi[i]) && !isnan(ho[i]) && ndvi[i] < 0 && ts[i] > 273.16) {
			dst[atomicAdd(nvalid, 1)] = Candidate(ndvi[i], ts[i], net_radiation[i], soil_heat[i], ho[i], line, i);
		}

		i += blockDim.x * gridDim.x;

	}

}

//ASEBAL

__global__ void asebalFilterCold(Candidate* dst, double* ndvi, double* ts, double* albedo, double* net_radiation, double* soil_heat, double* ho,
		int* nvalid, double* albedoQuartile, double* ndviQuartile, double* tsQuartile, int line, int size) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	while (i < size) {

		bool albedoValid = !isnan(albedo[i]) && albedo[i] < albedoQuartile[1];
		bool ndviValid = !isnan(ndvi[i]) && ndvi[i] >= ndviQuartile[2]; //ndvi_line[col] >= ndviQuartile[3];
		bool tsValid = !isnan(ts[i]) && ts[i] < tsQuartile[0];

		if (albedoValid && ndviValid && tsValid) {

			dst[atomicAdd(nvalid, 1)] = Candidate(ndvi[i], ts[i], net_radiation[i], soil_heat[i], ho[i], line, i);

		}

		i += blockDim.x * gridDim.x;

	}

}

__global__ void asebalFilterHot(Candidate* dst, double* ndvi, double* ts, double* albedo, double* net_radiation, double* soil_heat, double* ho,
		int* nvalid, double* albedoQuartile, double* ndviQuartile, double* tsQuartile, int line, int size) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	while (i < size) {

		bool albedoValid = !isnan(albedo[i]) && albedo[i] > albedoQuartile[2];
		bool ndviValid = !isnan(ndvi[i]) && ndvi[i] < ndviQuartile[0];
		bool tsValid = !isnan(ts[i]) && ts[i] > tsQuartile[2];

		if (albedoValid && ndviValid && tsValid) {

			dst[atomicAdd(nvalid, 1)] = Candidate(ndvi[i], ts[i], net_radiation[i], soil_heat[i], ho[i], line, i);

		}

		i += blockDim.x * gridDim.x;

	}

}

//ESA SEBAL

__device__ bool checkLandCode(int value){

    return (value == AGP) || (value == PAS) || (value == AGR) || (value == CAP) || (value == CSP) || (value == MAP);

}

__global__ void landCoverHomogeneityKernel(double* inputBuffer, int* output, int line, int numCol, int numLine){

	int column = threadIdx.x + blockIdx.x * blockDim.x;
	double pixel_value;
	int aux;

	while (column < numCol) {

		aux = line % 7;

		pixel_value = inputBuffer[aux * numCol + column];

		output[column] = false;

		if(checkLandCode(pixel_value)) { //Verify if the pixel is an AGR pixel

			output[column] = true;

			for(int i = -3; i <= 3 && output[column]; i++){

				for(int j = -3; j <= 3 && output[column]; j++){

					// Check if the neighbor is AGR too

					if (column + i >= 0 && column + i < numCol && line + j >= 0 && line + j < numLine) {

						aux = (line + j) % 7;

						pixel_value = inputBuffer[aux * numCol + column + i];

						if(!isnan(pixel_value))
							if(!checkLandCode(pixel_value))
								output[column] = false;

					}

				}

			}

		}

		column += blockDim.x * gridDim.x;

	}

}

__global__ void testHomogeneityKernel(double* ndviBuffer, double* tsBuffer, double* albedoBuffer, int* lcBuffer, int* output, int line, int numCol, int numLine){

    int column = threadIdx.x + blockIdx.x * blockDim.x;
    double pixel_value;
    int aux;

    int ndvi_size, ts_size, albedo_size;

	while(column < numCol) {

		double ndvi_neighbors[60], ts_neighbors[60], albedo_neighbors[60];

		aux = line % 7;
		ndvi_size = 0;
		ts_size = 0;
		albedo_size = 0;

		if(lcBuffer[column] == true) { //Verify if the pixel passed the land cover test

			if(!isnan(ndviBuffer[aux * numCol + column])){

				for(int i = -3; i <= 3; i++){

					for(int j = -3; j <= 3; j++){

						// Add for the NDVI, TS and Albedo the value of neighbors pixels into the respective vector

						if (column + i >= 0 && column + i < numCol && line + j >= 0 && line + j < numLine) {

							aux = (line + j) % 7;

							pixel_value = ndviBuffer[aux * numCol + column + i];
							if(!isnan(pixel_value)) {
								ndvi_neighbors[ndvi_size] = pixel_value;
								ndvi_size++;
							}

							pixel_value = tsBuffer[aux * numCol + column + i];
							if(!isnan(pixel_value)) {
								ts_neighbors[ts_size] = pixel_value;
								ts_size++;
							}

							pixel_value = albedoBuffer[aux * numCol + column + i];
							if(!isnan(pixel_value)) {
								albedo_neighbors[albedo_size] = pixel_value;
								albedo_size++;
							}

						}

					}

				}

				// Do the calculation of the dispersion measures from the NDVI, TS and Albedo

				double meanNDVI, meanTS, meanAlb;
				double sdNDVI, sdTS, sdAlb;
				double cvNDVI, cvAlb;
				double sumNDVI = 0, sumTS = 0, sumAlb = 0;

				for(int i = 0; i < ndvi_size; i++) {

					sumNDVI += ndvi_neighbors[i];
					sumTS += ts_neighbors[i];
					sumAlb += albedo_neighbors[i];

				}

				meanNDVI = sumNDVI / ndvi_size;
				meanTS = sumTS / ts_size;
				meanAlb = sumAlb / albedo_size;

				sumNDVI = 0, sumTS = 0, sumAlb = 0;

				for(int i = 0; i < ndvi_size; i++) {

					sumNDVI += (ndvi_neighbors[i] - meanNDVI) * (ndvi_neighbors[i] - meanNDVI);
					sumTS += (ts_neighbors[i] - meanTS) * (ts_neighbors[i] - meanTS);
					sumAlb += (albedo_neighbors[i] - meanAlb) * (albedo_neighbors[i] - meanAlb);

				}

				sdNDVI = sqrt(sumNDVI / ndvi_size);
				sdTS = sqrt(sumTS / ts_size);
				sdAlb = sqrt(sumAlb / albedo_size);

				cvNDVI = sdNDVI / meanNDVI;
				cvAlb = sdAlb / meanAlb;


				// Check if the pixel is eligible
				output[column] = (cvNDVI < 0.25) && (cvAlb < 0.25) && (sdTS < 1.5);

			} else {

				output[column] = false;

			}

		} else {

			output[column] = false;

		}

		column += blockDim.x * gridDim.x;

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
