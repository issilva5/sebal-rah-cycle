/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include "landsat.h"

/*
    Execution test (after resample function)

    arg 01 - band 01 remsampled path
    arg 02 - band 02 remsampled path
    arg 03 - band 03 remsampled path
    arg 04 - band 04 remsampled path
    arg 05 - band 05 remsampled path
    arg 06 - band 06 remsampled path
    arg 07 - band 07 remsampled path
    arg 08 - metadata path
    arg 09 - tal path
    arg 10 - station data path
    arg 11 - output path

    Flag
    -dist=xxxx - value distance between sun and earth

    ./run input/B2_converted.tif input/B3_converted.tif input/B4_converted.tif input/B5_converted.tif input/B6_converted.tif input/B7_converted.tif input/B10_converted.tif input/MTL.txt tal_converted.tif input/station.csv results -dist=0.98330

*/
int main(int argc, char *argv[]) {

	std::string output_path = argv[11];

	std::string metadata_path = argv[8];
	MTL mtl = MTL(metadata_path);

	std::string station_data_path = argv[10];
	Station station = Station(station_data_path, mtl.image_hour);

	Sensor sensor = Sensor(mtl.number_sensor, mtl.year);

	if (argc == 13) {
		std::string dist_flag = argv[12];
		if (dist_flag.substr(0, 6) == "-dist=")
			mtl.distance_earth_sun = atof(dist_flag.substr(6, dist_flag.size()).c_str());
	}

	std::string tal_path = argv[9];

	TIFF *bands_resampled[8];
	for (int i = 1; i < 8; i++) {
		std::string path_tiff_base = argv[i];
		bands_resampled[i] = TIFFOpen(path_tiff_base.c_str(), "rm");
		check_open_tiff(bands_resampled[i]);
	}

	Landsat landsat = Landsat(tal_path, output_path);
	landsat.process_partial_products(bands_resampled, mtl, station, sensor);
	landsat.process_final_products(station, mtl);

	close_tiffs(bands_resampled, 8);
	return 0;

}
