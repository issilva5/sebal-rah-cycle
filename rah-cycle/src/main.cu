/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include "cuda-utils.h"
#include "utils.h"

void allocateArraysDevice(std::vector<double*> arrays, uint32 size) {

	for (unsigned i = 0; i < arrays.size(); i++) {
		HANDLE_ERROR(cudaMalloc((void** ) &arrays[i], size * sizeof(double)));
	}

}

__global__ void correctionCycle(double* surfaceTemperatureLine, double* zomLine,
		double* ustarRLine, double* ustarWLine, double* rahRLine,
		double* rahWLine, double *a, double *b, double *u200) {

	//Identify position
	int pos = 0;

	double sensibleHeatFlux = RHO * SPECIFIC_HEAT_AIR
			* (*a + *b * (surfaceTemperatureLine[pos] - 273.15)) / rahRLine[pos];

	double ustarPow3 = ustarRLine[pos] * ustarRLine[pos] * ustarRLine[pos];

	double L = -1
			* ((RHO * SPECIFIC_HEAT_AIR * ustarPow3
					* surfaceTemperatureLine[pos])
					/ (VON_KARMAN * GRAVITY * sensibleHeatFlux));

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

		psi200 = 2 * log((1 + x200) / 2) + log((1 + x200 * x200) / 2)
				- 2 * atan(x200) + 0.5 * M_PI;

	}

	ustarWLine[pos] = (VON_KARMAN * *u200) / (log(200 / zomLine[pos]) - psi200);

	rahWLine[pos] = (log(2 / 0.1) - psi2 + psi01)
			/ (ustarWLine[pos] * VON_KARMAN);

}

int main(int argc, char **argv) {

	/*********** FIRST ZOM, USTAR E RAH VALUES BEGIN **********/

	std::string albedoPath, zomPath, ustarPath0, ustarPath1, rahPath0, rahPath1,
			sensibleHeatPath, surfaceTemperaturePath; //TODO insert path

	//Albedo base TIFF
	TIFF *albedo;
	albedo = TIFFOpen(albedoPath.c_str(), "rm");

	uint32 heightBand, widthBand;
	TIFFGetField(albedo, TIFFTAG_IMAGELENGTH, &heightBand);
	TIFFGetField(albedo, TIFFTAG_IMAGEWIDTH, &widthBand);

	//Auxiliary products TIFFs
	TIFF *zom, *ustar, *aerodynamicResistance;
	zom = TIFFOpen(zomPath.c_str(), "w8m");
	setup(zom, albedo);

	ustar = TIFFOpen(ustarPath0.c_str(), "w8m");
	setup(ustar, albedo);

	aerodynamicResistance = TIFFOpen(rahPath0.c_str(), "w8m");
	setup(aerodynamicResistance, albedo);

	//Calculates initial values of zom, ustar and aerodynamic_resistance TODO

	TIFFClose(albedo);
	TIFFClose(zom);
	TIFFClose(ustar);
	TIFFClose(aerodynamicResistance);

	/*********** FIRST ZOM, USTAR E RAH VALUES END **********/

	/*********** DEALING WITH HOT PIXEL BEGIN **********/

	aerodynamicResistance = TIFFOpen(rahPath0.c_str(), "rm");

	//Extract the hot pixel aerodynamic_resistance //TODO deal with hot pixel
	//hot_pixel.aerodynamic_resistance.push_back(read_position_tiff(aerodynamic_resistance, hot_pixel.col, hot_pixel.line));
	//double H_hot = hot_pixel.net_radiation - hot_pixel.soil_heat_flux;
	double hHot = 154564;

	TIFFClose(aerodynamicResistance);

	/*********** DEALING WITH HOT PIXEL END **********/

	/********** RAH CYCLE BEGIN **********/

	TIFF *ustarR, *aerodynamicResistanceR;
	TIFF *ustarW, *aerodynamicResistanceW, *sensibleHeatFlux;
	TIFF *surfaceTemperature;
	zom = TIFFOpen(zomPath.c_str(), "rm"); //It's not modified into the rah cycle
	surfaceTemperature = TIFFOpen(surfaceTemperaturePath.c_str(), "rm");

	//It's only written into the rah cycle
	sensibleHeatFlux = TIFFOpen(sensibleHeatPath.c_str(), "w8m");
	setup(sensibleHeatFlux, zom);

	//Auxiliaries arrays calculation
	double zomLine[widthBand], surfaceTemperatureLine[widthBand];
	double ustarReadLine[widthBand], ustarWriteLine[widthBand];
	double aerodynamicResistanceReadLine[widthBand],
			aerodynamicResistanceWriteLine[widthBand];

	//Auxiliaries arrays calculation to device
	double *devZom, *devTS;
	double *devUstarR, *devUstarW;
	double *devRahR, *devRahW;

	//Allocating arrays on device
	allocateArraysDevice(std::vector<double*> { devZom, devTS, devUstarR, devUstarW,
			devRahR, devRahW }, widthBand);

	//Auxiliaries loop variables
	int i = 0;
	bool Erro = true;
	double rahHot0, rahHot;

	while (Erro) {

		rahHot0 = 1245; //TODO hotPixel.aerodynamicResistance[i];

		//Opening the ustar e rah TIFFs for read and write based on i parity
		//If i is even then the TIFFs with 0 in the path name will be readable
		//otherwise TIFFs with 1 will be.

		if (i % 2) {

			//Since ustar is both write and read into the rah cycle, two TIFF will be needed
			ustarR = TIFFOpen(ustarPath0.c_str(), "rm");
			ustarW = TIFFOpen(ustarPath1.c_str(), "w8m");
			setup(ustarW, zom);

			//Since ustar is both write and read into the rah cycle, two TIFF will be needed
			aerodynamicResistanceR = TIFFOpen(rahPath0.c_str(), "rm");
			aerodynamicResistanceW = TIFFOpen(rahPath1.c_str(), "w8m");
			setup(aerodynamicResistanceW, zom);

		} else {

			//Since ustar is both write and read into the rah cycle, two TIFF will be needed
			ustarR = TIFFOpen(ustarPath1.c_str(), "rm");
			ustarW = TIFFOpen(ustarPath0.c_str(), "w8m");
			setup(ustarW, zom);

			//Since ustar is both write and read into the rah cycle, two TIFF will be needed
			aerodynamicResistanceR = TIFFOpen(rahPath1.c_str(), "rm");
			aerodynamicResistanceW = TIFFOpen(rahPath0.c_str(), "w8m");
			setup(aerodynamicResistanceW, zom);

		}

		//Coefficients calculation
//		double dtHot = hHot * rahHot0 / (RHO * SPECIFIC_HEAT_AIR);
//		double b = dtHot / (hotPixel.temperature - coldPixel.temperature);
//		double a = -b * (coldPixel.temperature - 273.15); TODO deal with hot/cold pixel

		double dtHot = hHot * rahHot0 / (RHO * SPECIFIC_HEAT_AIR);
		double b = 154;
		double a = 7864;

		double u200 = 35.654654; //TODO

		double *devA, *devB, *devU200;

		/********** COPY HOST TO DEVICE MEMORY BEGIN TODO **********/

		HANDLE_ERROR(cudaMemcpy(devA, (void**) &a, sizeof(int), cudaMemcpyHostToDevice));

		HANDLE_ERROR(cudaMemcpy(devB, (void**) &b, sizeof(int), cudaMemcpyHostToDevice));

		HANDLE_ERROR(
				cudaMemcpy(devU200, (void**) &u200, sizeof(double),
						cudaMemcpyHostToDevice));

		/********** COPY HOST TO DEVICE MEMORY END **********/

		for (int line = 0; line < heightBand; line++) {

			//Reading data needed
			read_line_tiff(surfaceTemperature, surfaceTemperatureLine, line);
			read_line_tiff(zom, zomLine, line);
			read_line_tiff(ustarR, ustarReadLine, line);
			read_line_tiff(aerodynamicResistanceR,
					aerodynamicResistanceReadLine, line);

			/********** COPY HOST TO DEVICE MEMORY BEGIN **********/

			HANDLE_ERROR(
					cudaMemcpy(devTS, surfaceTemperatureLine,
							widthBand * sizeof(double),
							cudaMemcpyHostToDevice));

			HANDLE_ERROR(
					cudaMemcpy(devZom, zomLine, widthBand * sizeof(double),
							cudaMemcpyHostToDevice));

			HANDLE_ERROR(
					cudaMemcpy(devUstarR, ustarReadLine,
							widthBand * sizeof(double),
							cudaMemcpyHostToDevice));

			HANDLE_ERROR(
					cudaMemcpy(devRahR, aerodynamicResistanceReadLine,
							widthBand * sizeof(double),
							cudaMemcpyHostToDevice));

			/********** COPY HOST TO DEVICE MEMORY END **********/

			/********** KERNEL BEGIN **********/

			correctionCycle<<<10, 10>>>(devTS, devZom, devUstarR, devUstarW, devRahR, devRahW, devA, devB, devU200);

			/********** KERNEL END **********/

			/********** COPY DEVICE TO HOST MEMORY BEGIN **********/

			HANDLE_ERROR(
					cudaMemcpy(ustarWriteLine, devUstarW,
							widthBand * sizeof(double),
							cudaMemcpyDeviceToHost));

			HANDLE_ERROR(
					cudaMemcpy(aerodynamicResistanceWriteLine, devRahW,
							widthBand * sizeof(double),
							cudaMemcpyDeviceToHost));

			/********** COPY DEVICE TO HOST MEMORY END **********/

			save_tiffs(std::vector<double*> { ustarWriteLine,
					aerodynamicResistanceWriteLine }, std::vector<TIFF*> { ustarW,
					aerodynamicResistanceW }, line);

		}

	}

	/********** RAH CYCLE END **********/

	return 0;
}
