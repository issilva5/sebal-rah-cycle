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
#include "products.h"
#include "rah_cycle.cuh"

int main(int argc, char **argv) {

	/*********** PARAMETERS **********/

	Sensor sensor = Sensor();
	Station station = Station();
	MTL mtl = MTL();

	/*********** PARAMETERS **********/

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

	Candidate hotPixel = Candidate();
	Candidate coldPixel = Candidate();

	aerodynamicResistance = TIFFOpen(rahPath0.c_str(), "rm");

	double hHot = hotPixel.net_radiation - hotPixel.soil_heat_flux;

	TIFFClose(aerodynamicResistance);

	/*********** DEALING WITH HOT PIXEL END **********/

	/*********** u200 BEGIN **********/

	double ustarStation = (VON_KARMAN * station.v6)
			/ (log(station.WIND_SPEED / station.SURFACE_ROUGHNESS));
	double u200 = (ustarStation / VON_KARMAN)
			* log(200 / station.SURFACE_ROUGHNESS);

	/*********** u200 END **********/

	/********** RAH CYCLE BEGIN **********/

	TIFF *ustarR, *aerodynamicResistanceR;
	TIFF *ustarW, *aerodynamicResistanceW;
	TIFF *surfaceTemperature;

	zom = TIFFOpen(zomPath.c_str(), "rm");
	surfaceTemperature = TIFFOpen(surfaceTemperaturePath.c_str(), "rm");

	//Auxiliaries arrays calculation
	double zomLine[widthBand], surfaceTemperatureLine[widthBand];
	double ustarReadLine[widthBand], ustarWriteLine[widthBand];
	double aerodynamicResistanceReadLine[widthBand],
			aerodynamicResistanceWriteLine[widthBand];

	/********** ALLOCATING VARIABLES IN DEVICE MEMORY BEGIN **********/

	//Auxiliaries arrays calculation to device
	double *devZom, *devTS;
	double *devUstarR, *devUstarW;
	double *devRahR, *devRahW;
	double *devA, *devB, *devU200;

	HANDLE_ERROR(cudaMalloc((void** ) &devA, sizeof(double)));

	HANDLE_ERROR(cudaMalloc((void** ) &devB, sizeof(double)));

	HANDLE_ERROR(cudaMalloc((void** ) &devU200, sizeof(double)));

	HANDLE_ERROR(cudaMalloc((void** ) &devZom, widthBand * sizeof(double)));

	HANDLE_ERROR(cudaMalloc((void** ) &devTS, widthBand * sizeof(double)));

	HANDLE_ERROR(cudaMalloc((void** ) &devUstarR, widthBand * sizeof(double)));

	HANDLE_ERROR(cudaMalloc((void** ) &devUstarW, widthBand * sizeof(double)));

	HANDLE_ERROR(cudaMalloc((void** ) &devRahR, widthBand * sizeof(double)));

	HANDLE_ERROR(cudaMalloc((void** ) &devRahW, widthBand * sizeof(double)));

	/********** ALLOCATING VARIABLES IN DEVICE MEMORY BEGIN **********/

	//Auxiliaries loop variables
	int i = 0;
	double rahHot0, rahHot;
	double dtHot, a, b;

	while (true) {

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

			rahHot = read_position_tiff(aerodynamicResistanceR, 456, 24564);

		} else {

			//Since ustar is both write and read into the rah cycle, two TIFF will be needed
			ustarR = TIFFOpen(ustarPath1.c_str(), "rm");
			ustarW = TIFFOpen(ustarPath0.c_str(), "w8m");
			setup(ustarW, zom);

			//Since ustar is both write and read into the rah cycle, two TIFF will be needed
			aerodynamicResistanceR = TIFFOpen(rahPath1.c_str(), "rm");
			aerodynamicResistanceW = TIFFOpen(rahPath0.c_str(), "w8m");
			setup(aerodynamicResistanceW, zom);

			rahHot = read_position_tiff(aerodynamicResistanceR, 456, 24564);

		}

		if (i > 0 && fabs(1 - rahHot0 / rahHot) >= 0.05)
			break;

		rahHot0 = rahHot;

		//Coefficients calculation
		dtHot = hHot * rahHot0 / (RHO * SPECIFIC_HEAT_AIR);
		b = dtHot / (hotPixel.temperature - coldPixel.temperature);
		a = -b * (coldPixel.temperature - 273.15);

		/********** COPY VARIABLES FROM HOST TO DEVICE MEMORY BEGIN **********/
		//TODO use constant memory?
		HANDLE_ERROR(cudaMemcpy(devA, &a, sizeof(int), cudaMemcpyHostToDevice));

		HANDLE_ERROR(cudaMemcpy(devB, &b, sizeof(int), cudaMemcpyHostToDevice));

		HANDLE_ERROR(
				cudaMemcpy(devU200, &u200, sizeof(double),
						cudaMemcpyHostToDevice));

		/********** COPY  VARIABLES FROM HOST TO DEVICE MEMORY END **********/

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
					aerodynamicResistanceWriteLine }, std::vector<TIFF*> {
					ustarW, aerodynamicResistanceW }, line);

			TIFFClose(ustarR);
			TIFFClose(ustarW);
			TIFFClose(aerodynamicResistanceR);
			TIFFClose(aerodynamicResistanceW);

			i++;

		}

	}

	/********** DE-ALLOCATING VARIABLES IN DEVICE MEMORY BEGIN **********/

	HANDLE_ERROR(cudaFree(devA));

	HANDLE_ERROR(cudaFree(devB));

	HANDLE_ERROR(cudaFree(devU200));

	HANDLE_ERROR(cudaFree(devZom));

	HANDLE_ERROR(cudaFree(devTS));

	HANDLE_ERROR(cudaFree(devUstarR));

	HANDLE_ERROR(cudaFree(devUstarW));

	HANDLE_ERROR(cudaFree(devRahR));

	HANDLE_ERROR(cudaFree(devRahW));

	/********** DE-ALLOCATING VARIABLES IN DEVICE MEMORY BEGIN **********/

	/********** RAH CYCLE END **********/

	return 0;
}
