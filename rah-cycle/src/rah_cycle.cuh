#include "cuda-utils.h"
#include "types.h"

__global__ void correctionCycle(double* surfaceTemperatureLine, double* zomLine,
		double* ustarRLine, double* ustarWLine, double* rahRLine,
		double* rahWLine, double *a, double *b, double *u200);

