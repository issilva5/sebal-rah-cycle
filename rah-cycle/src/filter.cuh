#include "cuda-utils.h"
#include "types.h"
#include "device_functions.h"
#include "candidate.h"

__global__ void filterHot(Candidate* dst, double* ndvi, double* ts, double* net_radiation,
					   double* soil_heat, double* ho, int* nvalid,  int line, int size);

__global__ void filterCold(Candidate* dst, double* ndvi, double* ts, double* net_radiation,
					   double* soil_heat, double* ho, int* nvalid,  int line, int size);
