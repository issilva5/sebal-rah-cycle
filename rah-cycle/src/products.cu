#include "products.h"

/**
 * @brief  The spectral radiance for each band is computed.
 * @param  read_bands[]: Satellite bands.
 * @param  mtl: MTL struct.
 * @param  sensor: Sensor struct.
 * @param  width_band: Band width.
 * @param  line: Line to be calculated.
 * @param  radiance_line[][8]: Auxiliary array for save the calculated value of radiance for each band.
 */
void radiance_function(TIFF* read_bands[], MTL mtl, Sensor sensor, int width_band, int line, double radiance_line[][8], double noData){
    double line_band[width_band];

    if (mtl.number_sensor == 8){
        read_line_tiff(read_bands[7], line_band, line);
        for (int col = 0; col < width_band; col++) {
            radiance_line[col][7] = line_band[col] != noData ? line_band[col] * mtl.rad_mult_10 + mtl.rad_add_10 : NaN;
	}
    }
    else{
        for (int i = 1; i < 8; i++){
            read_line_tiff(read_bands[i], line_band, line);
            for (int col = 0; col < width_band; col++) {
                radiance_line[col][i] = line_band[col] != noData ? line_band[col] * sensor.parameters[i][sensor.GRESCALE] + sensor.parameters[i][sensor.BRESCALE] : NaN;

		if(!std::isnan(radiance_line[col][i]) && radiance_line[col][i] < 0) radiance_line[col][i] = 0;
	    }
        }
    }

};

/**
 * @brief  The reflectivity for each band (ρλ) is computed (model F02).
 * @param  read_bands[]: Satellite bands.
 * @param  mtl: MTL struct.
 * @param  sensor: Sensor struct.
 * @param  radiance_line[][8]: Radiance for the specific line for each band.
 * @param  width_band: Band width.
 * @param  line: Line to be calculated.
 * @param  reflectance_line[][8]: Auxiliary array for save the calculated value of reflectance for each band.
 */
void reflectance_function(TIFF* read_bands[], MTL mtl, Sensor sensor, double radiance_line[][8], int width_band, int line, double reflectance_line[][8], double noData){
    double costheta = sin(mtl.sun_elevation * PI / 180);
    double line_band[width_band];
    //-3.39999995214436425e+38
    for (int i = 1; i < 8; i++){
        read_line_tiff(read_bands[i], line_band, line);
        for (int col = 0; col < width_band; col++){
            if (mtl.number_sensor == 8)
                reflectance_line[col][i] = line_band[col] != noData ? (line_band[col] * sensor.parameters[i][sensor.GRESCALE] + sensor.parameters[i][sensor.BRESCALE]) / costheta : NaN;
            else
                reflectance_line[col][i] = line_band[col] != noData ? (PI * radiance_line[col][i] * mtl.distance_earth_sun * mtl.distance_earth_sun) /
                                           (sensor.parameters[i][sensor.ESUN] * costheta) : NaN;
        }
    }

};

/**
 * @brief  The surface albedo is computed.
 * @param  reflectance_line[][8]: Reflectance for the specific line for each band.
 * @param  sensor: Sensor struct.
 * @param  tal_line[]: Array containing the specified line from the tal TIFF.
 * @param  width_band: Band width.
 * @param  number_sensor: Number of the satellite sensor.
 * @param  albedo_line[]: Auxiliary array for save the calculated value of albedo for the line.
 */
void albedo_function(double reflectance_line[][8], Sensor sensor, double tal_line[], int width_band, int number_sensor,
		double albedo_line[]) {
	int final_tif_calc = number_sensor == 8 ? 6 : 7;

	for (int col = 0; col < width_band; col++) {
		albedo_line[col] = reflectance_line[col][1] * sensor.parameters[1][sensor.WB]
				+ reflectance_line[col][2] * sensor.parameters[2][sensor.WB] + reflectance_line[col][3] * sensor.parameters[3][sensor.WB]
				+ reflectance_line[col][4] * sensor.parameters[4][sensor.WB] + reflectance_line[col][5] * sensor.parameters[5][sensor.WB]
				+ reflectance_line[col][final_tif_calc] * sensor.parameters[final_tif_calc][sensor.WB];
		albedo_line[col] = (albedo_line[col] - 0.03) / (tal_line[col] * tal_line[col]);
	}

}
;

/**
 * @brief  Normalized Difference Vegetation Index (NDVI) is computed.
 * @note   Values for NDVI range between -1 and 1.
 * @param  reflectance_line[][8]: Reflectance for the specific line for each band.
 * @param  width_band: Band width.
 * @param  ndvi_line[]: Auxiliary array for save the calculated value of NDVI for the line.
 */
void ndvi_function(double reflectance_line[][8], int width_band, double ndvi_line[]) {

	for (int col = 0; col < width_band; col++) {
		ndvi_line[col] = (reflectance_line[col][4] - reflectance_line[col][3]) / (reflectance_line[col][4] + reflectance_line[col][3]);
	}

}
;

/**
 * @brief  Leaf Area Index (LAI) is computed.
 * @param  reflectance_line[][8]: Reflectance for the specific line for each band.
 * @param  width_band: Band width.
 * @param  lai_line[]: Auxiliary array for save the calculated value of LAI for the line.
 */
void lai_function(double reflectance_line[][8], int width_band, double lai_line[]) {
	double savi_line[width_band];
	double L = 0.05;

	for (int col = 0; col < width_band; col++) {

		savi_line[col] = ((1 + L) * (reflectance_line[col][4] - reflectance_line[col][3]))
				/ (L + (reflectance_line[col][4] + reflectance_line[col][3]));

		if (!std::isnan(savi_line[col]) && definitelyGreaterThan(savi_line[col], 0.687))
			lai_line[col] = 6;
		else if (!std::isnan(savi_line[col]) && definitelyLessThan(savi_line[col], 0.1))
			lai_line[col] = 0;
		else
			lai_line[col] = -log((0.69 - savi_line[col]) / 0.59) / 0.91;

	}

}
;

/**
 * @brief  Enhanced Vegetation Index (EVI) is computed.
 * @param  reflectance_line[][8]: Reflectance for the specific line for each band.
 * @param  width_band: Band width.
 * @param  evi_line[]: Auxiliary array for save the calculated value of EVI for the line.
 */
void evi_function(double reflectance_line[][8], int width_band, double evi_line[]) {

	for (int col = 0; col < width_band; col++) {
		evi_line[col] = 2.5
				* ((reflectance_line[col][4] - reflectance_line[col][3])
						/ (reflectance_line[col][4] + 6 * reflectance_line[col][3] - 7.5 * reflectance_line[col][1] + 1));
	}

}
;

/**
 * @brief  Calculates emissivity representing surface behavior for thermal emission in the relatively narrow band 6 of Landsat (10.4 to 12.5 µm),
 expressed as enb.
 * @param  lai_line[]: Array containing the specified line from the LAI computation.
 * @param  ndvi_line[]: Array containing the specified line from the NDVI computation.
 * @param  width_band: Band width.
 * @param  enb_emissivity_line[]: Auxiliary array for save the calculated value of Enb for the line.
 */
void enb_emissivity_function(double lai_line[], double ndvi_line[], int width_band, double enb_emissivity_line[]) {

	for (int col = 0; col < width_band; col++) {
		if (definitelyLessThan(ndvi_line[col], 0) || definitelyGreaterThan(lai_line[col], 2.99))
			enb_emissivity_line[col] = 0.98;
		else
			enb_emissivity_line[col] = 0.97 + 0.0033 * lai_line[col];
	}

}
;

/**
 * @brief  Calculates emissivity representing surface behavior for thermal emission in the broad thermal spectrum (6 to 14 µm), expressed as eο.
 * @param  lai_line[]: Array containing the specified line from the LAI computation.
 * @param  ndvi_line[]: Array containing the specified line from the NDVI computation.
 * @param  width_band: Band width.
 * @param  eo_emissivity_line[]: Auxiliary array for save the calculated value of Eo for the line.
 */
void eo_emissivity_function(double lai_line[], double ndvi_line[], int width_band, double eo_emissivity_line[]) {

	for (int col = 0; col < width_band; col++) {
		if (definitelyLessThan(ndvi_line[col], 0) || definitelyGreaterThan(lai_line[col], 2.99))
			eo_emissivity_line[col] = 0.98;
		else
			eo_emissivity_line[col] = 0.95 + 0.01 * lai_line[col];
	}

}
;

/**
 * @brief  Calculates the atmospheric emissivity (ea).
 * @param  tal_line[]: Array containing the specified line from the tal computation.
 * @param  width_band: Band width.
 * @param  ea_emissivity_line[]: Auxiliary array for save the calculated value of Ea for the line.
 */
void ea_emissivity_function(double tal_line[], int width_band, double ea_emissivity_line[]) {

	for (int col = 0; col < width_band; col++)
		ea_emissivity_line[col] = 0.85 * pow((-1 * log(tal_line[col])), 0.09);

}
;

/**
 * @brief  The surface temperature (TS) is computed.
 * @param  radiance_line[][8]: Radiance for the specific line for each band.
 * @param  enb_emissivity_line[]: Array containing the specified line from the Enb computation.
 * @param  number_sensor: Number of the satellite sensor.
 * @param  width_band: Band width.
 * @param  surface_temperature_line[]: Auxiliary array for save the calculated value of TS for the line.
 */
void surface_temperature_function(double radiance_line[][8], double enb_emissivity_line[], int number_sensor, int width_band,
		double surface_temperature_line[]) {
	double k1, k2;

	switch (number_sensor) {
	case 5:
		k1 = 607.76;
		k2 = 1282.71;

		break;
	case 7:
		k1 = 666.09;
		k2 = 1260.56;

		break;
	case 8:
		k1 = 774.8853;
		k2 = 1321.0789;

		break;
	default:
		std::cerr << "Sensor problem!" << std::endl;
		exit(6);
	}

	int radiance_number = (number_sensor == 5) ? 6 : 7;

	for (int col = 0; col < width_band; col++)
		surface_temperature_line[col] = k2 / (log((enb_emissivity_line[col] * k1 / radiance_line[col][radiance_number]) + 1));

}
;

/**
 * @brief  Computes Short Wave Radiation (Rs).
 * @param  tal_line[]: Array containing the specified line from the tal computation.
 * @param  mtl: MTL Struct.
 * @param  width_band: Band width.
 * @param  short_wave_radiation_line[]: Auxiliary array for save the calculated value of Rs for the line.
 */
void short_wave_radiation_function(double tal_line[], MTL mtl, int width_band, double short_wave_radiation_line[]) {
	double costheta = sin(mtl.sun_elevation * PI / 180);

	for (int col = 0; col < width_band; col++) {
		short_wave_radiation_line[col] = (1367 * costheta * tal_line[col]) / (mtl.distance_earth_sun * mtl.distance_earth_sun);
	}
}
;

/**
 * @brief  Computes Large Wave Radiation from Surface (RLSup)
 * @param  eo_emissivity_line[]: Array containing the specified line from the Eo computation.
 * @param  surface_temperature_line[]: Array containing the specified line from the TS computation.
 * @param  width_band: Band width.
 * @param  large_wave_radiation_surface_line[]: Auxiliary array for save the calculated value of RLSup for the line.
 */
void large_wave_radiation_surface_function(double eo_emissivity_line[], double surface_temperature_line[], int width_band,
		double large_wave_radiation_surface_line[]) {

	for (int col = 0; col < width_band; col++) {
		double temperature_pixel = surface_temperature_line[col];
		double surface_temperature_pow_4 = temperature_pixel * temperature_pixel * temperature_pixel * temperature_pixel;
		large_wave_radiation_surface_line[col] = eo_emissivity_line[col] * 5.67 * 1e-8 * surface_temperature_pow_4;
	}

}
;

/**
 * @brief  Computes Large Wave Radiation from Atmosphere (RLatm)
 * @param  ea_emissivity_line[]: Array containing the specified line from the Ea computation.
 * @param  width_band: Band width.
 * @param  temperature: Near surface air temperature in Kelvin.
 * @param  large_wave_radiation_atmosphere_line[]: Auxiliary array for save the calculated value of RLatm for the line.
 */
void large_wave_radiation_atmosphere_function(double ea_emissivity_line[], int width_band, double temperature,
		double large_wave_radiation_atmosphere_line[]) {

	double temperature_kelvin = temperature + 273.15;
	double temperature_kelvin_pow_4 = temperature_kelvin * temperature_kelvin * temperature_kelvin * temperature_kelvin;

	for (int col = 0; col < width_band; col++)
		large_wave_radiation_atmosphere_line[col] = ea_emissivity_line[col] * 5.67 * 1e-8 * temperature_kelvin_pow_4;

}
;

/**
 * @brief  The net surface radiation flux (Rn) is computed.
 * @param  short_wave_radiation_line[]: Array containing the specified line from the Rs computation.
 * @param  large_wave_radiation_surface_line[]: Array containing the specified line from the RLSup computation.
 * @param  large_wave_radiation_atmosphere_line[]: Array containing the specified line from the RLatm computation.
 * @param  albedo_line[]: Array containing the specified line from the albedo computation.
 * @param  eo_emissivity_line[]: Array containing the specified line from the Eo computation.
 * @param  width_band: Band width.
 * @param  net_radiation_line[]: Auxiliary array for save the calculated value of Rn for the line.
 */
void net_radiation_function(double short_wave_radiation_line[], double large_wave_radiation_surface_line[],
		double large_wave_radiation_atmosphere_line[], double albedo_line[], double eo_emissivity_line[], int width_band,
		double net_radiation_line[]) {

	for (int col = 0; col < width_band; col++) {
		net_radiation_line[col] = short_wave_radiation_line[col] - (short_wave_radiation_line[col] * albedo_line[col])
				+ large_wave_radiation_atmosphere_line[col] - large_wave_radiation_surface_line[col]
				- (1 - eo_emissivity_line[col]) * large_wave_radiation_atmosphere_line[col];

		if (definitelyLessThan(net_radiation_line[col], 0))
			net_radiation_line[col] = 0;
	}

}
;

/**
 * @brief  Computes the Soil heat flux (G).
 * @param  ndvi_line[]: Array containing the specified line from the NDVI computation.
 * @param  surface_temperature_line[]: Array containing the specified line from the TS computation.
 * @param  albedo_line[]: Array containing the specified line from the albedo computation.
 * @param  net_radiation_line[]: Array containing the specified line from the Rn computation.
 * @param  width_band: Band width.
 * @param  soil_heat_flux[]: Auxiliary array for save the calculated value of G for the line.
 */
void soil_heat_flux_function(double ndvi_line[], double surface_temperature_line[], double albedo_line[], double net_radiation_line[],
		int width_band, double soil_heat_line[]) {

	for (int col = 0; col < width_band; col++) {
		if (essentiallyEqual(ndvi_line[col], 0) || definitelyGreaterThan(ndvi_line[col], 0)) {
			double ndvi_pixel_pow_4 = ndvi_line[col] * ndvi_line[col] * ndvi_line[col] * ndvi_line[col];
			soil_heat_line[col] = (surface_temperature_line[col] - 273.15) * (0.0038 + 0.0074 * albedo_line[col])
					* (1 - 0.98 * ndvi_pixel_pow_4) * net_radiation_line[col];
		} else
			soil_heat_line[col] = 0.5 * net_radiation_line[col];

		if (definitelyLessThan(soil_heat_line[col], 0))
			soil_heat_line[col] = 0;

	}

}
;

/**
 * @brief  Computes the HO.
 * @param  net_radiation_line[]: Array containing the specified line from the Rn computation.
 * @param  soil_heat_flux[]: Array containing the specified line from the G computation.
 * @param  width_band: Band width.
 * @param  ho_line[]: Auxiliary array for save the calculated value of HO for the line.
 */
void ho_function(double net_radiation_line[], double soil_heat_flux[], int width_band, double ho_line[]) {

	for (int col = 0; col < width_band; col++)
		ho_line[col] = net_radiation_line[col] - soil_heat_flux[col];

}
;

/**
 * @brief  Select the hot pixel.
 * @param  ndvi: NDVI TIFF.
 * @param  surface_temperature: TS TIFF.
 * @param  net_radiation: Rn TIFF.
 * @param  soil_heat: G TIFF.
 * @param  height_band: Band height.
 * @param  width_band: Band width.
 * @retval Candidate struct containing the hot pixel.
 */
Candidate select_hot_pixel(TIFF** ndvi, TIFF** surface_temperature, TIFF** net_radiation, TIFF** soil_heat, int height_band,
		int width_band, int threadNum) {

    //Timing
    std::chrono::steady_clock::time_point begin, end;
    std::chrono::duration< double, std::micro > time_span_us;

	//Auxiliary arrays
	double ndvi_line[width_band], surface_temperature_line[width_band];
	double net_radiation_line[width_band], soil_heat_line[width_band];
	double ho_line[width_band];
	int valid = 0;
	const int MAXC = 10000000;
	Candidate* pre_candidates;
	pre_candidates = (Candidate*) malloc(MAXC * sizeof(Candidate));
	if(pre_candidates == NULL) exit(1);

	//Device variables
	double *ndvi_dev, *ts_dev, *rn_dev, *soil_heat_dev, *ho_dev;
	int *valid_dev;
	Candidate* pre_candidates_dev;

	HANDLE_ERROR(cudaMalloc((void**) &pre_candidates_dev, MAXC * sizeof(Candidate)));
	HANDLE_ERROR(cudaMalloc((void**) &ndvi_dev, width_band * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &ts_dev, width_band * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &rn_dev, width_band * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &soil_heat_dev, width_band * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &ho_dev, width_band * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &valid_dev, sizeof(int)));

	//Contains the candidates with NDVI between 0.15 and 0.20, which surface temperature is greater than 273.16
	//std::vector<Candidate> pre_candidates;
	begin = std::chrono::steady_clock::now();
	    //printf("PHASE 2 - PSH NDVI FILTER BEGIN, %d\n", int(time(NULL)));

	HANDLE_ERROR(cudaMemcpy(pre_candidates_dev, pre_candidates, MAXC*sizeof(Candidate), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(valid_dev, &valid, sizeof(int), cudaMemcpyHostToDevice));

	for (int line = 0; line < height_band; line++) {
		read_line_tiff(*net_radiation, net_radiation_line, line);
		read_line_tiff(*soil_heat, soil_heat_line, line);

		ho_function(net_radiation_line, soil_heat_line, width_band, ho_line);

		read_line_tiff(*ndvi, ndvi_line, line);
		read_line_tiff(*surface_temperature, surface_temperature_line, line);


		HANDLE_ERROR(cudaMemcpy(ndvi_dev, ndvi_line, width_band*sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(ts_dev, surface_temperature_line, width_band*sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(rn_dev, net_radiation_line, width_band*sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(soil_heat_dev, soil_heat_line, width_band*sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(ho_dev, ho_line, width_band*sizeof(double), cudaMemcpyHostToDevice));

		filterHot<<<(width_band + threadNum - 1) / threadNum, threadNum>>>(pre_candidates_dev, ndvi_dev, ts_dev, rn_dev, soil_heat_dev, ho_dev, valid_dev, line, width_band);

	}

	HANDLE_ERROR(cudaMemcpy(&valid, valid_dev, sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pre_candidates, pre_candidates_dev, MAXC * sizeof(Candidate), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(pre_candidates_dev));
	HANDLE_ERROR(cudaFree(ndvi_dev));
	HANDLE_ERROR(cudaFree(ts_dev));
	HANDLE_ERROR(cudaFree(rn_dev));
	HANDLE_ERROR(cudaFree(soil_heat_dev));
	HANDLE_ERROR(cudaFree(ho_dev));
	HANDLE_ERROR(cudaFree(valid_dev));

	end = std::chrono::steady_clock::now();
	time_span_us = std::chrono::duration_cast< std::chrono::duration<double, std::micro> >(end - begin);
	//printf("PHASE 2 - PSH NDVI FILTER DURATION, %.5f\n", time_span_us);
	//printf("VALID: %d\n", valid);

	begin = std::chrono::steady_clock::now();
	//printf("PHASE 2 - PSH SORT BY TEMP BEGIN, %d\n", int(time(NULL)));
	//Sort the candidates by their temperatures and choose the surface temperature of the hot pixel
	
	std::sort(pre_candidates, pre_candidates + valid, compare_candidate_temperature);
	end = std::chrono::steady_clock::now();
	
	if(valid <= 0) {
		std::cerr << "Pixel problem! - There are no precandidates";
		exit(15);
	}

	time_span_us = std::chrono::duration_cast< std::chrono::duration<double, std::micro> >(end - begin);
	//printf("PHASE 2 - PSH SORT BY TEMP DURATION, %.5f\n", time_span_us);
	int pos = floor(0.95 * valid);
	double surfaceTempHot = pre_candidates[pos].temperature;

	begin = std::chrono::steady_clock::now();
	    //printf("PHASE 2 - PSH HO MANIPULATION BEGIN, %d\n", int(time(NULL)));
	//Select only the ones with temperature equals the surface temperature of the hot pixel
	std::vector<double> ho_candidates;
	Candidate lastHOCandidate;
	for (int i = 0; i < valid; i++) {
		if (essentiallyEqual(pre_candidates[i].temperature, surfaceTempHot)) {
			ho_candidates.push_back(pre_candidates[i].ho);
			lastHOCandidate = pre_candidates[i];
		}
	}

	free(pre_candidates);

	if (ho_candidates.size() == 1) {
		return lastHOCandidate;
	}

	//Select the limits of HOs
	sort(ho_candidates.begin(), ho_candidates.end());
	end = std::chrono::steady_clock::now();
	time_span_us = std::chrono::duration_cast< std::chrono::duration<double, std::micro> >(end - begin);
	//printf("PHASE 2 - PSH HO MANIPULATION DURATION, %.5f\n", time_span_us);
	
	if(ho_candidates.size() <= 0) {
		std::cerr << "Pixel problem! - There are no precandidates after HO manipulation";
		exit(15);
	}

	begin = std::chrono::steady_clock::now();
	//printf("PHASE 2 - PSH SELECT FINAL CANDIDATES BEGIN, %d\n", int(time(NULL)));
	double HO_min = ho_candidates[floor(0.25 * ho_candidates.size())];
	double HO_max = ho_candidates[floor(0.75 * ho_candidates.size())];

	//Contains the final candidates which HO is in (HO_min, HO_max) and surface temperature is greater than 273.16
	std::vector<Candidate> final_candidates;

	for (int line = 0; line < height_band; line++) {

		read_line_tiff(*net_radiation, net_radiation_line, line);
		read_line_tiff(*soil_heat, soil_heat_line, line);

		ho_function(net_radiation_line, soil_heat_line, width_band, ho_line);

		read_line_tiff(*ndvi, ndvi_line, line);
		read_line_tiff(*surface_temperature, surface_temperature_line, line);

		for (int col = 0; col < width_band; col++) {
			if (definitelyGreaterThan(ho_line[col], HO_min) && definitelyLessThan(ho_line[col], HO_max)
					&& essentiallyEqual(surface_temperature_line[col], surfaceTempHot)) {
				final_candidates.push_back(
						Candidate(ndvi_line[col], surface_temperature_line[col], net_radiation_line[col], soil_heat_line[col], ho_line[col],
								line, col));
			}
		}

	}

	end = std::chrono::steady_clock::now();
	time_span_us = std::chrono::duration_cast< std::chrono::duration<double, std::micro> >(end - begin);
	//printf("PHASE 2 - PSH SELECT FINAL CANDIDATES DURATION, %.5f\n", time_span_us);
	//printf("VALID: %d\n", valid);
	
	if(final_candidates.size() <= 0) {
		std::cerr << "Pixel problem! - There are no final candidates";
		exit(15);
	}

	begin = std::chrono::steady_clock::now();
	//printf("PHASE 2 - PSH CV EXTRACT BEGIN, %d\n", int(time(NULL)));
	//Calculate the coefficient of variation, after the extract
	for (int i = 0; i < final_candidates.size(); i++) {
		final_candidates[i].extract_coefficient_variation(*ndvi);
	}
	end = std::chrono::steady_clock::now();
	time_span_us = std::chrono::duration_cast< std::chrono::duration<double, std::micro> >(end - begin);
	//printf("PHASE 2 - PSH CV EXTRACT DURATION, %.5f\n", time_span_us);

	begin = std::chrono::steady_clock::now();
	//printf("PHASE 2 - PSH FINAL BEGIN, %d\n", int(time(NULL)));
	//Choose as candidate the pixel with the minor CV
	Candidate choosen = final_candidates[0];

	for (int i = 1; i < final_candidates.size(); i++) {

		if (definitelyLessThan(final_candidates[i].coefficient_variation, choosen.coefficient_variation))
			choosen = final_candidates[i];

	}
	end = std::chrono::steady_clock::now();
	time_span_us = std::chrono::duration_cast< std::chrono::duration<double, std::micro> >(end - begin);
	//printf("PHASE 2 - PSH FINAL DURATION, %.5f\n", time_span_us);
	//choosen.toString();
	return choosen;
}

/**
 * @brief  Select the cold pixel.
 * @param  ndvi: NDVI TIFF.
 * @param  surface_temperature: TS TIFF.
 * @param  net_radiation: Rn TIFF.
 * @param  soil_heat: G TIFF.
 * @param  height_band: Band height.
 * @param  width_band: Band width.
 * @retval Candidate struct containing the cold pixel.
 */
Candidate select_cold_pixel(TIFF** ndvi, TIFF** surface_temperature, TIFF** net_radiation, TIFF** soil_heat, int height_band,
		int width_band, int threadNum) {

    //Timing
    std::chrono::steady_clock::time_point begin, end;
    std::chrono::duration< double, std::micro > time_span_us;

	//Auxiliary arrays
	double ndvi_line[width_band], surface_temperature_line[width_band];
	double net_radiation_line[width_band], soil_heat_line[width_band];
	double ho_line[width_band];
	int valid = 0;
	const int MAXC = 10000000;
	Candidate* pre_candidates;
	pre_candidates = (Candidate*) malloc(MAXC * sizeof(Candidate));
	if(pre_candidates == NULL) exit(1);

	//Device variables
	double *ndvi_dev, *ts_dev, *rn_dev, *soil_heat_dev, *ho_dev;
	int *valid_dev;
	Candidate* pre_candidates_dev;

	HANDLE_ERROR(cudaMalloc((void**) &pre_candidates_dev, MAXC * sizeof(Candidate)));
	HANDLE_ERROR(cudaMalloc((void**) &ndvi_dev, width_band * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &ts_dev, width_band * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &rn_dev, width_band * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &soil_heat_dev, width_band * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &ho_dev, width_band * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &valid_dev, sizeof(int)));

	//Contains the candidates with NDVI less than 0, which surface temperature is greater than 273.16
	//std::vector<Candidate> pre_candidates;
	begin = std::chrono::steady_clock::now();

	HANDLE_ERROR(cudaMemcpy(pre_candidates_dev, pre_candidates, MAXC*sizeof(Candidate), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(valid_dev, &valid, sizeof(int), cudaMemcpyHostToDevice));

	    //printf("PHASE 2 - PSC NDVI FILTER BEGIN, %d\n", int(time(NULL)));
	for (int line = 0; line < height_band; line++) {

		read_line_tiff(*net_radiation, net_radiation_line, line);
		read_line_tiff(*soil_heat, soil_heat_line, line);

		ho_function(net_radiation_line, soil_heat_line, width_band, ho_line);

		read_line_tiff(*ndvi, ndvi_line, line);
		read_line_tiff(*surface_temperature, surface_temperature_line, line);

		HANDLE_ERROR(cudaMemcpy(ndvi_dev, ndvi_line, width_band*sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(ts_dev, surface_temperature_line, width_band*sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(rn_dev, net_radiation_line, width_band*sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(soil_heat_dev, soil_heat_line, width_band*sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(ho_dev, ho_line, width_band*sizeof(double), cudaMemcpyHostToDevice));

		filterCold<<<(width_band + threadNum - 1) / threadNum, threadNum>>>(pre_candidates_dev, ndvi_dev, ts_dev, rn_dev, soil_heat_dev, ho_dev, valid_dev, line, width_band);

	}

	HANDLE_ERROR(cudaMemcpy(&valid, valid_dev, sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pre_candidates, pre_candidates_dev, MAXC * sizeof(Candidate), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(pre_candidates_dev));
	HANDLE_ERROR(cudaFree(ndvi_dev));
	HANDLE_ERROR(cudaFree(ts_dev));
	HANDLE_ERROR(cudaFree(rn_dev));
	HANDLE_ERROR(cudaFree(soil_heat_dev));
	HANDLE_ERROR(cudaFree(ho_dev));
	HANDLE_ERROR(cudaFree(valid_dev));

	end = std::chrono::steady_clock::now();
	time_span_us = std::chrono::duration_cast< std::chrono::duration<double, std::micro> >(end - begin);
	//printf("PHASE 2 - PSC NDVI FILTER DURATION, %.5f\n", time_span_us);
	
	if(valid <= 0) {
		std::cerr << "Pixel problem! - There are no precandidates";
		exit(15);
	}

	begin = std::chrono::steady_clock::now();
	//printf("PHASE 2 - PSC SORT BY TEMP BEGIN, %d\n", int(time(NULL)));
	//Sort the candidates by their temperatures and choose the surface temperature of the hot pixel
	std::sort(pre_candidates, pre_candidates + valid, compare_candidate_temperature);
	end = std::chrono::steady_clock::now();
	time_span_us = std::chrono::duration_cast< std::chrono::duration<double, std::micro> >(end - begin);
	//printf("PHASE 2 - PSC SORT BY TEMP DURATION, %.5f\n", time_span_us);
	int pos = floor(0.5 * valid);
	double surfaceTempCold = pre_candidates[pos].temperature;

	begin = std::chrono::steady_clock::now();
	    //printf("PHASE 2 - PSC HO MANIPULATION BEGIN, %d\n", int(time(NULL)));
	//Select only the ones with temperature equals the surface temperature of the Cold pixel
	std::vector<double> ho_candidates;
	Candidate lastHOCandidate;
	for (int i = 0; i < valid; i++) {
		if (essentiallyEqual(pre_candidates[i].temperature, surfaceTempCold)) {
			ho_candidates.push_back(pre_candidates[i].ho);
			lastHOCandidate = pre_candidates[i];
		}
	}

	free(pre_candidates);

	if (ho_candidates.size() == 1) {
		return lastHOCandidate;
	}

	//Select the limits of HOs
	sort(ho_candidates.begin(), ho_candidates.end());
	end = std::chrono::steady_clock::now();
	time_span_us = std::chrono::duration_cast< std::chrono::duration<double, std::micro> >(end - begin);
	//printf("PHASE 2 - PSC HO MANIPULATION DURATION, %.5f\n", time_span_us);

	begin = std::chrono::steady_clock::now();
	//printf("PHASE 2 - PSC SELECT FINAL CANDIDATES BEGIN, %d\n", int(time(NULL)));
	double HO_min = ho_candidates[floor(0.25 * ho_candidates.size())];
	double HO_max = ho_candidates[floor(0.75 * ho_candidates.size())];

	//Contains the final candidates which HO is in (HO_min, HO_max) and surface temperature is greater than 273.16
	std::vector<Candidate> final_candidates;

	for (int line = 0; line < height_band; line++) {

		read_line_tiff(*net_radiation, net_radiation_line, line);
		read_line_tiff(*soil_heat, soil_heat_line, line);

		ho_function(net_radiation_line, soil_heat_line, width_band, ho_line);

		read_line_tiff(*ndvi, ndvi_line, line);
		read_line_tiff(*surface_temperature, surface_temperature_line, line);

		for (int col = 0; col < width_band; col++) {
			if (definitelyGreaterThan(ho_line[col], HO_min) && definitelyLessThan(ho_line[col], HO_max)
					&& essentiallyEqual(surface_temperature_line[col], surfaceTempCold)) {
				final_candidates.push_back(
						Candidate(ndvi_line[col], surface_temperature_line[col], net_radiation_line[col], soil_heat_line[col], ho_line[col],
								line, col));
			}
		}

	}
	end = std::chrono::steady_clock::now();
	time_span_us = std::chrono::duration_cast< std::chrono::duration<double, std::micro> >(end - begin);
	//printf("PHASE 2 - PSC SELECT FINAL CANDIDATES DURATION, %.5f\n", time_span_us);
	
	if(final_candidates.size() <= 0) {
		std::cerr << "Pixel problem! - There are no final candidates";
		exit(15);
	}

	begin = std::chrono::steady_clock::now();
	//printf("PHASE 2 - PSC NN EXTRACT BEGIN, %d\n", int(time(NULL)));
	//Calculate the coefficient of variation, after the extract
	for (int i = 0; i < final_candidates.size(); i++) {
		final_candidates[i].extract_negative_neighbour(*ndvi);
	}
	end = std::chrono::steady_clock::now();
	time_span_us = std::chrono::duration_cast< std::chrono::duration<double, std::micro> >(end - begin);
	//printf("PHASE 2 - PSC NN EXTRACT DURATION, %.5f\n", time_span_us);

	begin = std::chrono::steady_clock::now();
	//printf("PHASE 2 - PSC FINAL BEGIN, %d\n", int(time(NULL)));
	//Choose as candidate the pixel with the minor CV
	Candidate choosen = final_candidates[0];
	for (int i = 1; i < final_candidates.size(); i++) {
		if (final_candidates[i].negative_neighbour > choosen.negative_neighbour)
			choosen = final_candidates[i];
	}
	end = std::chrono::steady_clock::now();
    time_span_us = std::chrono::duration_cast< std::chrono::duration<double, std::micro> >(end - begin);
	//printf("PHASE 2 - PSC FINAL DURATION, %.5f\n", time_span_us);
	//choosen.toString();
	return choosen;
}

/**
 * @brief  Computes the momentum roughness length (zom).
 * @param  A_ZOM: Correlation constant a.
 * @param  B_ZOM: Correlation constant b.
 * @param  ndvi_line[]: Array containing the specified line from the NDVI computation.
 * @param  width_band: Band width.
 * @param  zom_line[]: Auxiliary array for save the calculated value of zom for the line.
 */
void zom_fuction(double A_ZOM, double B_ZOM, double ndvi_line[], int width_band, double zom_line[]) {

	for (int col = 0; col < width_band; col++)
		zom_line[col] = exp(A_ZOM + B_ZOM * ndvi_line[col]);

}
;

/**
 * @brief  The friction velocity (u*) is computed.
 * @param  u200: Wind speed at 200 m.
 * @param  zom_line[]: Array containing the specified line from the zom computation.
 * @param  width_band: Band width.
 * @param  ustar_line[]: Auxiliary array for save the calculated value of ustar for the line.
 */
void ustar_fuction(double u200, double zom_line[], int width_band, double ustar_line[]) {

	for (int col = 0; col < width_band; col++)
		ustar_line[col] = (VON_KARMAN * u200) / log(200 / zom_line[col]);

}
;

/**
 * @brief  Computes the aerodynamic resistance (Rah).
 * @param  ustar_line[]: Array containing the specified line from the ustar computation.
 * @param  width_band: Band width.
 * @param  aerodynamic_resistance_line[]: Auxiliary array for save the calculated value of Rah for the line.
 */
void aerodynamic_resistance_fuction(double ustar_line[], int width_band, double aerodynamic_resistance_line[]) {

	for (int col = 0; col < width_band; col++)
		aerodynamic_resistance_line[col] = log(20) / (ustar_line[col] * VON_KARMAN);

}
;

/**
 * @brief  Computes Sensible Heat Flux (H).
 * @param  surface_temperature_line[]: Array containing the specified line from the TS computation.
 * @param  aerodynamic_resistance_line[]: Array containing the specified line from the Rah computation.
 * @param  net_radiation_line[]: Array containing the specified line from the Rn computation.
 * @param  soil_heat_line[]: Array containing the specified line from the G computation.
 * @param  sensible_heat_flux_line[]: Auxiliary array for save the calculated value of H computation for the line.
 * @param  a: Coefficient a.
 * @param  b: Coefficient b.
 * @param  width_band: Band width.
 */
void sensible_heat_flux_function(double surface_temperature_line[], double aerodynamic_resistance_line[], double net_radiation_line[],
		double soil_heat_line[], double sensible_heat_flux_line[], double a, double b, double width_band) {

	for (int col = 0; col < width_band; col++) {
		sensible_heat_flux_line[col] = RHO * SPECIFIC_HEAT_AIR * (a + b * (surface_temperature_line[col] - 273.15))
				/ aerodynamic_resistance_line[col];

		if (!std::isnan(sensible_heat_flux_line[col]) && sensible_heat_flux_line[col] > (net_radiation_line[col] - soil_heat_line[col])) {
			sensible_heat_flux_line[col] = net_radiation_line[col] - soil_heat_line[col];
		}

	}

}

/**
 * @brief  Computes Latent Heat Flux (LE).
 * @param  net_radiation_line[]: Array containing the specified line from the Rn computation.
 * @param  soil_heat_flux_line[]: Array containing the specified line from the G computation.
 * @param  sensible_heat_flux_line[]: Array containing the specified line from the H computation.
 * @param  width_band: Band width.
 * @param  latent_heat_flux[]: Auxiliary array for save the calculated value of LE for the line.
 */
void latent_heat_flux_function(double net_radiation_line[], double soil_heat_flux_line[], double sensible_heat_flux_line[], int width_band,
		double latent_heat_flux[]) {

	for (int col = 0; col < width_band; col++)
		latent_heat_flux[col] = net_radiation_line[col] - soil_heat_flux_line[col] - sensible_heat_flux_line[col];

}
;

/**
 * @brief  Calculates the Net Radiation for 24 hours (Rn24h).
 * @param  albedo_line[]: Array containing the specified line from the albedo computation.
 * @param  Ra24h: Extraterrestrial Radiation defined as solar short wave radiation in the absence of an atmosphere (Ra24h).
 * @param  Rs24h: Short wave radiation incident in 24 hours (Rs24h).
 * @param  width_band: Band width.
 * @param  net_radiation_24h_line[]: Auxiliary array for save the calculated value of Rn24h for the line.
 */
void net_radiation_24h_function(double albedo_line[], double Ra24h, double Rs24h, int width_band, double net_radiation_24h_line[]) {
	int FL = 110;

	for (int col = 0; col < width_band; col++)
		net_radiation_24h_line[col] = (1 - albedo_line[col]) * Rs24h - FL * Rs24h / Ra24h;

}
;

/**
 * @brief  The Reference ET Fraction (EF) is computed.
 * @param  latent_heat_flux_line[]: Array containing the specified line from the LE computation.
 * @param  net_radiation_line[]: Array containing the specified line from the Rn computation.
 * @param  soil_heat_line[]: Array containing the specified line from the G computation.
 * @param  width_band: Band width.
 * @param  evapotranspiration_fraction_line[]: Auxiliary array for save the calculated value of EF for the line.
 */
void evapotranspiration_fraction_fuction(double latent_heat_flux_line[], double net_radiation_line[], double soil_heat_line[],
		int width_band, double evapotranspiration_fraction_line[]) {

	for (int col = 0; col < width_band; col++)
		evapotranspiration_fraction_line[col] = latent_heat_flux_line[col] / (net_radiation_line[col] - soil_heat_line[col]);

}
;

/**
 * @brief  Computes Sensible Heat Flux for 24 hours (H24h).
 * @param  evapotranspiration_fraction_line[]: Array containing the specified line from the EF computation.
 * @param  net_radiation_24h_line[]: Array containing the specified line from the Rn24h computation.
 * @param  width_band: Band width.
 * @param  sensible_heat_flux_24h_line[]: Auxiliary array for save the calculated value of H24h for the line.
 */
void sensible_heat_flux_24h_fuction(double evapotranspiration_fraction_line[], double net_radiation_24h_line[], int width_band,
		double sensible_heat_flux_24h_line[]) {

	for (int col = 0; col < width_band; col++)
		sensible_heat_flux_24h_line[col] = (1 - evapotranspiration_fraction_line[col]) * net_radiation_24h_line[col];

}
;

/**
 * @brief  Calculates Latente Heat Flux for 24 hours (LE24h).
 * @param  evapotranspiration_fraction_line[]: Array containing the specified line from the EF computation.
 * @param  net_radiation_24h_line[]: Array containing the specified line from the Rn24h computation.
 * @param  width_band: Band width.
 * @param  latent_heat_flux_24h_line[]: Auxiliary array for save the calculated value of LE24h for the line.
 */
void latent_heat_flux_24h_function(double evapotranspiration_fraction_line[], double net_radiation_24h_line[], int width_band,
		double latent_heat_flux_24h_line[]) {

	for (int col = 0; col < width_band; col++)
		latent_heat_flux_24h_line[col] = evapotranspiration_fraction_line[col] * net_radiation_24h_line[col];

}
;

/**
 * @brief  Computes the Evapotranspiration for 24 hours (ET24h)
 * @param  latent_heat_flux_24h_line[]: Array containing the specified line from the LE24h computation.
 * @param  station: Station struct.
 * @param  width_band: Band width.
 * @param  evapotranspiration_24h_line[]: Auxiliary array for save the calculated value of ET24h for the line.
 */
void evapotranspiration_24h_function(double latent_heat_flux_24h_line[], Station station, int width_band,
		double evapotranspiration_24h_line[]) {

	for (int col = 0; col < width_band; col++)
		evapotranspiration_24h_line[col] = (latent_heat_flux_24h_line[col] * 86400)
				/ ((2.501 - 0.00236 * (station.v7_max + station.v7_min) / 2) * 1e+6);

}
;
