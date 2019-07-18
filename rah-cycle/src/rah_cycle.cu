#include "rah_cycle.cuh"

__global__ void correctionCycle(double* surfaceTemperatureLine, double* zomLine, double* ustarRLine, double* ustarWLine, double* rahRLine,
		double* rahWLine, double *a, double *b, double *u200) {

	//Identify position
	int pos = blockIdx.x;

	double sensibleHeatFlux = RHO * SPECIFIC_HEAT_AIR * (*a + *b * (surfaceTemperatureLine[pos] - 273.15)) / rahRLine[pos];

	double L = -1 * ((RHO * SPECIFIC_HEAT_AIR * pow(ustarRLine[pos], 3) * surfaceTemperatureLine[pos]) / (VON_KARMAN * GRAVITY * sensibleHeatFlux));

	double y01 = pow((1 - (16 * 0.1) / L), 0.25);
	double y2 = pow((1 - (16 * 2) / L), 0.25);
	double x200 = pow((1 - (16 * 200) / L), 0.25);

	double psi01, psi2, psi200;

	if (!isnan(L) && L > 0) {

		psi01 = -5 * (0.1 / L);
		psi2 = -5 * (2 / L);
		psi200 = -5 * (2 / L);

	} else {

		psi01 = 2 * log((1 + y01 * y01) / 2);

		psi2 = 2 * log((1 + y2 * y2) / 2);

		psi200 = 2 * log((1 + x200) / 2) + log((1 + x200 * x200) / 2) - 2 * atan(x200) + 0.5 * M_PI;

	}

	ustarWLine[pos] = (VON_KARMAN * *u200) / (log(200 / zomLine[pos]) - psi200);

	rahWLine[pos] = (log(2 / 0.1) - psi2 + psi01) / (ustarWLine[pos] * VON_KARMAN);

}
