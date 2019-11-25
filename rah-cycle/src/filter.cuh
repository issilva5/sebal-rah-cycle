#include "cuda-utils.h"
#include "types.h"
#include "device_functions.h"
#include "candidate.h"

__global__ void filter(Candidate* dst, double* ndvi, double* ts, double* net_radiation,
					   double* soil_heat, double* ho, double* nvalid, int size);
