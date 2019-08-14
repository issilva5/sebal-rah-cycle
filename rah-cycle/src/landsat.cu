#include "landsat.h"
#include "rah_cycle.cuh"

/**
 * @brief  Empty constructor.
 */
Landsat::Landsat() {
}
;

/**
 * @brief  Constructor of the struct.
 * @param  tal_path: Path to tal TIFF.
 * @param  output_path: Output path where TIFF should be saved.
 */
Landsat::Landsat(std::string tal_path, std::string output_path) {
	this->tal_path = tal_path;
	this->output_path = output_path;

	//Initialize the path of products TIFF based on the output path.
	this->albedo_path = output_path + "/alb.tif";
	this->ndvi_path = output_path + "/NDVI.tif";
	this->evi_path = output_path + "/EVI.tif";
	this->lai_path = output_path + "/LAI.tif";
	this->soil_heat_path = output_path + "/G.tif";
	this->surface_temperature_path = output_path + "/TS.tif";
	this->net_radiation_path = output_path + "/Rn.tif";
	this->evapotranspiration_fraction_path = output_path + "/EF.tif";
	this->evapotranspiration_24h_path = output_path + "/ET24h.tif";
	this->zom_path = output_path + "/zom.tif";
	this->sensible_heat_flux_path = output_path + "/H.tif";
	this->ustar_tif0_path = output_path + "/ustar_tif0.tif";
	this->ustar_tif1_path = output_path + "/ustar_tif1.tif";
	this->aerodynamic_resistance_tif0_path = output_path + "/Rah_tif0.tif";
	this->aerodynamic_resistance_tif1_path = output_path + "/Rah_tif1.tif";
}
;

/**
 * @brief  Calculates the partials products (e. g. Albedo, NDVI, Rn, G) of the SEBAL execution.
 * @param  read_bands[]: Satellite images as TIFFs.
 * @param  mtl: MTL struct.
 * @param  station: Station struct.
 * @param  sensor: Sensor struct.
 */
void Landsat::process_partial_products(TIFF* read_bands[], MTL mtl, Station station, Sensor sensor) {
	uint32 height_band, width_band;

	TIFFGetField(read_bands[1], TIFFTAG_IMAGEWIDTH, &width_band);
	TIFFGetField(read_bands[1], TIFFTAG_IMAGELENGTH, &height_band);

	TIFF *tal = TIFFOpen(this->tal_path.c_str(), "rm");
	check_open_tiff(tal);

	double tal_line[width_band];

	TIFF *albedo, *ndvi, *evi, *lai, *soil_heat, *surface_temperature, *net_radiation;
	create_tiffs(&tal, &albedo, &ndvi, &evi, &lai, &soil_heat, &surface_temperature, &net_radiation);

	//Declare array with product information
	double albedo_line[width_band], ndvi_line[width_band];
	double evi_line[width_band], lai_line[width_band];
	double soil_heat_line[width_band], surface_temperature_line[width_band];
	double net_radiation_line[width_band];

	//Declare auxiliaries arrays
	double radiance_line[width_band][8];
	double reflectance_line[width_band][8];

	//Declare arrays of auxiliaries products
	double eo_emissivity_line[width_band], ea_emissivity_line[width_band], enb_emissivity_line[width_band];
	double large_wave_radiation_atmosphere_line[width_band], large_wave_radiation_surface_line[width_band];
	double short_wave_radiation_line[width_band];

	//Calculating the partial products for each line
	for (int line = 0; line < height_band; line++) {
		radiance_function(read_bands, mtl, sensor, width_band, line, radiance_line);
		reflectance_function(read_bands, mtl, sensor, radiance_line, width_band, line, reflectance_line);

		read_line_tiff(tal, tal_line, line);

		albedo_function(reflectance_line, sensor, tal_line, width_band, mtl.number_sensor, albedo_line);
		short_wave_radiation_function(tal_line, mtl, width_band, short_wave_radiation_line);
		ndvi_function(reflectance_line, width_band, ndvi_line);
		lai_function(reflectance_line, width_band, lai_line);
		evi_function(reflectance_line, width_band, evi_line);
		enb_emissivity_function(lai_line, ndvi_line, width_band, enb_emissivity_line);
		eo_emissivity_function(lai_line, ndvi_line, width_band, eo_emissivity_line);
		surface_temperature_function(radiance_line, enb_emissivity_line, mtl.number_sensor, width_band, surface_temperature_line);
		large_wave_radiation_surface_function(eo_emissivity_line, surface_temperature_line, width_band, large_wave_radiation_surface_line);
		ea_emissivity_function(tal_line, width_band, ea_emissivity_line);
		large_wave_radiation_atmosphere_function(ea_emissivity_line, width_band, station.temperature_image, large_wave_radiation_atmosphere_line);
		net_radiation_function(short_wave_radiation_line, large_wave_radiation_surface_line, large_wave_radiation_atmosphere_line, albedo_line,
				eo_emissivity_line, width_band, net_radiation_line);
		soil_heat_flux_function(ndvi_line, surface_temperature_line, albedo_line, net_radiation_line, width_band, soil_heat_line);

		save_tiffs(std::vector<double*> { albedo_line, ndvi_line, evi_line, lai_line, soil_heat_line, surface_temperature_line, net_radiation_line },
				std::vector<TIFF*> { albedo, ndvi, evi, lai, soil_heat, surface_temperature, net_radiation }, line);
	}

	//Closing the open TIFFs.
	TIFFClose(albedo);
	TIFFClose(ndvi);
	TIFFClose(evi);
	TIFFClose(lai);
	TIFFClose(soil_heat);
	TIFFClose(surface_temperature);
	TIFFClose(net_radiation);
	TIFFClose(tal);
}
;

/**
 * @brief  Process the final products (e. g. Evapotranspiration 24 hours) of the SEBAL execution.
 * @param  station: Station struct.
 * @param  mtl: MTL struct.
 */
void Landsat::process_final_products(Station station, MTL mtl) {
	TIFF *albedo, *ndvi, *soil_heat, *surface_temperature, *net_radiation;
	TIFF *evapotranspiration_fraction, *evapotranspiration_24h;

	open_tiffs(&albedo, &ndvi, &soil_heat, &surface_temperature, &net_radiation, &evapotranspiration_fraction, &evapotranspiration_24h);

	uint32 height_band, width_band;
	TIFFGetField(albedo, TIFFTAG_IMAGELENGTH, &height_band);
	TIFFGetField(albedo, TIFFTAG_IMAGEWIDTH, &width_band);

	// Selecting hot and cold pixels

	Candidate hot_pixel = select_hot_pixel(&ndvi, &surface_temperature, &net_radiation, &soil_heat, height_band, width_band);
	Candidate cold_pixel = select_cold_pixel(&ndvi, &surface_temperature, &net_radiation, &soil_heat, height_band, width_band);

	//Intermediaries products
	double sensible_heat_flux_line[width_band];
	double zom_line[width_band];
	double ustar_line[width_band];
	double aerodynamic_resistance_line[width_band];
	double latent_heat_flux_line[width_band];

	double ustar_station = (VON_KARMAN * station.v6) / (log(station.WIND_SPEED / station.SURFACE_ROUGHNESS));
	double u200 = (ustar_station / VON_KARMAN) * log(200 / station.SURFACE_ROUGHNESS);

	//Partial products
	double ndvi_line[width_band], surface_temperature_line[width_band];
	double soil_heat_line[width_band], net_radiation_line[width_band];
	double albedo_line[width_band];

	//Outhers products
	double net_radiation_24h_line[width_band];
	double evapotranspiration_fraction_line[width_band];
	double sensible_heat_flux_24h_line[width_band];
	double latent_heat_flux_24h_line[width_band];
	double evapotranspiration_24h_line[width_band];

	//Upscalling temporal
	double dr = (1 / mtl.distance_earth_sun) * (1 / mtl.distance_earth_sun);
	double sigma = 0.409 * sin(((2 * PI / 365) * mtl.julian_day) - 1.39);
	double phi = (PI / 180) * station.latitude;
	double omegas = acos(-tan(phi) * tan(sigma));
	double Ra24h = (((24 * 60 / PI) * GSC * dr) * (omegas * sin(phi) * sin(sigma) + cos(phi) * cos(sigma) * sin(omegas))) * (1000000 / 86400.0);

	//Short wave radiation incident in 24 hours (Rs24h)
	double Rs24h = station.INTERNALIZATION_FACTOR * sqrt(station.v7_max - station.v7_min) * Ra24h;

	//Auxiliary products TIFFs
	TIFF *zom, *ustar, *aerodynamic_resistance;
	zom = TIFFOpen(zom_path.c_str(), "w8m");
	setup(zom, albedo);

	ustar = TIFFOpen(ustar_tif0_path.c_str(), "w8m");
	setup(ustar, albedo);

	aerodynamic_resistance = TIFFOpen(aerodynamic_resistance_tif0_path.c_str(), "w8m");
	setup(aerodynamic_resistance, albedo);

	//Calculates initial values of zom, ustar and aerodynamic_resistance

	for (int line = 0; line < height_band; line++) {
		read_line_tiff(ndvi, ndvi_line, line);
		read_line_tiff(surface_temperature, surface_temperature_line, line);
		read_line_tiff(net_radiation, net_radiation_line, line);
		read_line_tiff(soil_heat, soil_heat_line, line);
		read_line_tiff(albedo, albedo_line, line);

		zom_fuction(station.A_ZOM, station.B_ZOM, ndvi_line, width_band, zom_line);
		ustar_fuction(u200, zom_line, width_band, ustar_line);
		aerodynamic_resistance_fuction(ustar_line, width_band, aerodynamic_resistance_line);

		save_tiffs(std::vector<double*> { zom_line, ustar_line, aerodynamic_resistance_line },
				std::vector<TIFF*> { zom, ustar, aerodynamic_resistance }, line);
	}

	//Initial zom, ustar and aerodynamic_resistance are calculated and saved.
	//Continuing the sebal calculation

	TIFFClose(ndvi);
	TIFFClose(zom);
	TIFFClose(ustar);
	TIFFClose(aerodynamic_resistance);

	aerodynamic_resistance = TIFFOpen(aerodynamic_resistance_tif0_path.c_str(), "rm");

	//Extract the hot pixel aerodynamic_resistance
	hot_pixel.aerodynamic_resistance.push_back(read_position_tiff(aerodynamic_resistance, hot_pixel.col, hot_pixel.line));
	double H_hot = hot_pixel.net_radiation - hot_pixel.soil_heat_flux;

	TIFFClose(aerodynamic_resistance);
	TIFF *ustar_tifR, *ustar_tifW, *aerodynamic_resistance_tifR, *aerodynamic_resistance_tifW;
	zom = TIFFOpen(zom_path.c_str(), "rm"); //It's not modified into the rah cycle

	//Auxiliaries arrays calculation
	double ustar_read_line[width_band], ustar_write_line[width_band];
	double aerodynamic_resistance_read_line[width_band], aerodynamic_resistance_write_line[width_band];

	/********** ALLOCATING VARIABLES IN DEVICE MEMORY BEGIN **********/

	//Auxiliaries arrays calculation to device
	double *devZom, *devTS;
	double *devUstarR, *devUstarW;
	double *devRahR, *devRahW;
	double *devA, *devB, *devU200;
	int *devSize;

	HANDLE_ERROR(cudaMalloc((void** ) &devA, sizeof(double)));

	HANDLE_ERROR(cudaMalloc((void** ) &devB, sizeof(double)));

	HANDLE_ERROR(cudaMalloc((void** ) &devU200, sizeof(double)));

	HANDLE_ERROR(cudaMalloc((void** ) &devSize, sizeof(int)));

	HANDLE_ERROR(cudaMalloc((void** ) &devZom, width_band * sizeof(double)));

	HANDLE_ERROR(cudaMalloc((void** ) &devTS, width_band * sizeof(double)));

	HANDLE_ERROR(cudaMalloc((void** ) &devUstarR, width_band * sizeof(double)));

	HANDLE_ERROR(cudaMalloc((void** ) &devUstarW, width_band * sizeof(double)));

	HANDLE_ERROR(cudaMalloc((void** ) &devRahR, width_band * sizeof(double)));

	HANDLE_ERROR(cudaMalloc((void** ) &devRahW, width_band * sizeof(double)));

	HANDLE_ERROR(cudaMemcpy(devSize, &width_band, sizeof(int), cudaMemcpyHostToDevice));

	/********** ALLOCATING VARIABLES IN DEVICE MEMORY BEGIN **********/

	//TODO PROFILING
	float timeline;
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	printf("loop, line, ms_time\n");
	int i = 0;
	bool Erro = true;
	double rah_hot0, rah_hot, dt_hot, a, b;
	//cudaProfilerStart();
	while (Erro) {

		rah_hot0 = hot_pixel.aerodynamic_resistance[i];

		if (i % 2) {
			//Since ustar is both write and read into the rah cycle, two TIFF will be needed
			ustar_tifR = TIFFOpen(ustar_tif1_path.c_str(), "rm");
			ustar_tifW = TIFFOpen(ustar_tif0_path.c_str(), "w8m");
			setup(ustar_tifW, albedo);

			//Since ustar is both write and read into the rah cycle, two TIFF will be needed
			aerodynamic_resistance_tifR = TIFFOpen(aerodynamic_resistance_tif1_path.c_str(), "rm");
			aerodynamic_resistance_tifW = TIFFOpen(aerodynamic_resistance_tif0_path.c_str(), "w8m");
			setup(aerodynamic_resistance_tifW, albedo);

		} else {
			//Since ustar is both write and read into the rah cycle, two TIFF will be needed
			ustar_tifR = TIFFOpen(ustar_tif0_path.c_str(), "rm");
			ustar_tifW = TIFFOpen(ustar_tif1_path.c_str(), "w8m");
			setup(ustar_tifW, albedo);

			//Since ustar is both write and read into the rah cycle, two TIFF will be needed
			aerodynamic_resistance_tifR = TIFFOpen(aerodynamic_resistance_tif0_path.c_str(), "rm");
			aerodynamic_resistance_tifW = TIFFOpen(aerodynamic_resistance_tif1_path.c_str(), "w8m");
			setup(aerodynamic_resistance_tifW, albedo);

		}

		dt_hot = H_hot * rah_hot0 / (RHO * SPECIFIC_HEAT_AIR);
		b = dt_hot / (hot_pixel.temperature - cold_pixel.temperature);
		a = -b * (cold_pixel.temperature - 273.15);

		/********** COPY VARIABLES FROM HOST TO DEVICE MEMORY BEGIN **********/
		//TODO use constant memory?
		HANDLE_ERROR(cudaMemcpy(devA, &a, sizeof(double), cudaMemcpyHostToDevice));

		HANDLE_ERROR(cudaMemcpy(devB, &b, sizeof(double), cudaMemcpyHostToDevice));

		HANDLE_ERROR(cudaMemcpy(devU200, &u200, sizeof(double), cudaMemcpyHostToDevice));

		/********** COPY  VARIABLES FROM HOST TO DEVICE MEMORY END **********/

		for (int line = 0; line < height_band; line++) {

			//Reading data needed
			read_line_tiff(surface_temperature, surface_temperature_line, line);
			read_line_tiff(zom, zom_line, line);
			read_line_tiff(ustar_tifR, ustar_read_line, line);
			read_line_tiff(aerodynamic_resistance_tifR, aerodynamic_resistance_read_line, line);

			/********** COPY HOST TO DEVICE MEMORY BEGIN **********/

			HANDLE_ERROR(cudaMemcpy(devTS, surface_temperature_line, width_band * sizeof(double), cudaMemcpyHostToDevice));

			HANDLE_ERROR(cudaMemcpy(devZom, zom_line, width_band * sizeof(double), cudaMemcpyHostToDevice));

			HANDLE_ERROR(cudaMemcpy(devUstarR, ustar_read_line, width_band * sizeof(double), cudaMemcpyHostToDevice));

			HANDLE_ERROR(cudaMemcpy(devRahR, aerodynamic_resistance_read_line, width_band * sizeof(double), cudaMemcpyHostToDevice));

			/********** COPY HOST TO DEVICE MEMORY END **********/

			/********** KERNEL BEGIN **********/

			cudaEventRecord(start, 0);
			correctionCycle<<<(width_band + 255) / 256, 256>>>(devTS, devZom, devUstarR, devUstarW, devRahR, devRahW, devA, devB, devU200, devSize);
			cudaDeviceSynchronize();

			cudaEventRecord(end, 0);
			cudaEventElapsedTime(&timeline, start, end);
			cudaEventSynchronize(end);

			printf("%d, %d, %.3f\n", i, line, timeline);
//			acummulated += timeline;

			/********** KERNEL END **********/

			/********** COPY DEVICE TO HOST MEMORY BEGIN **********/

			HANDLE_ERROR(cudaMemcpy(ustar_write_line, devUstarW, width_band * sizeof(double), cudaMemcpyDeviceToHost));

			HANDLE_ERROR(cudaMemcpy(aerodynamic_resistance_write_line, devRahW, width_band * sizeof(double), cudaMemcpyDeviceToHost));

			/********** COPY DEVICE TO HOST MEMORY END **********/

			if (line == hot_pixel.line) {
				rah_hot = aerodynamic_resistance_write_line[hot_pixel.col];
				hot_pixel.aerodynamic_resistance.push_back(rah_hot);
			}

			//Saving new ustar e rah
			save_tiffs(std::vector<double*> { ustar_write_line, aerodynamic_resistance_write_line }, std::vector<TIFF*> { ustar_tifW,
					aerodynamic_resistance_tifW }, line);

		}

		TIFFClose(ustar_tifR);
		TIFFClose(ustar_tifW);
		TIFFClose(aerodynamic_resistance_tifR);
		TIFFClose(aerodynamic_resistance_tifW);

		std::cout << "rah_hot0: " << rah_hot0 << " " << "rah_hot: " << rah_hot << std::endl;
		std::cout << fabs(1 - rah_hot0 / rah_hot) << std::endl;

		Erro = (fabs(1 - rah_hot0 / rah_hot) >= 0.05);
		i++;
		printf("%d\n", i);

	}

	TIFFClose(zom);

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
	//cudaProfilerStop();

	if (i % 2) {

		//printf("Rah_after is aerodynamic_resistance_tif1_path\n");
		aerodynamic_resistance_tifR = TIFFOpen(aerodynamic_resistance_tif1_path.c_str(), "rm");

	} else {

		//printf("Rah_after is aerodynamic_resistance_tif0_path\n");
		aerodynamic_resistance_tifR = TIFFOpen(aerodynamic_resistance_tif0_path.c_str(), "rm");

	}

	//End of Rah correction

	dt_hot = H_hot * rah_hot / (RHO * SPECIFIC_HEAT_AIR);
	b = dt_hot / (hot_pixel.temperature - cold_pixel.temperature);
	a = -b * (cold_pixel.temperature - 273.15);

	//Continuing to the final products

	for (int line = 0; line < height_band; line++) {
		read_line_tiff(aerodynamic_resistance_tifR, aerodynamic_resistance_line, line);
		read_line_tiff(surface_temperature, surface_temperature_line, line);
		read_line_tiff(net_radiation, net_radiation_line, line);
		read_line_tiff(soil_heat, soil_heat_line, line);
		read_line_tiff(albedo, albedo_line, line);

		sensible_heat_flux_function(surface_temperature_line, aerodynamic_resistance_line, net_radiation_line, soil_heat_line,
				sensible_heat_flux_line, a, b, width_band);
		latent_heat_flux_function(net_radiation_line, soil_heat_line, sensible_heat_flux_line, width_band, latent_heat_flux_line);
		net_radiation_24h_function(albedo_line, Ra24h, Rs24h, width_band, net_radiation_24h_line);
		evapotranspiration_fraction_fuction(latent_heat_flux_line, net_radiation_line, soil_heat_line, width_band, evapotranspiration_fraction_line);
		sensible_heat_flux_24h_fuction(evapotranspiration_fraction_line, net_radiation_24h_line, width_band, sensible_heat_flux_24h_line);
		latent_heat_flux_24h_function(evapotranspiration_fraction_line, net_radiation_24h_line, width_band, latent_heat_flux_24h_line);
		evapotranspiration_24h_function(latent_heat_flux_24h_line, station, width_band, evapotranspiration_24h_line);

		save_tiffs(std::vector<double*> { evapotranspiration_fraction_line, evapotranspiration_24h_line }, std::vector<TIFF*> {
				evapotranspiration_fraction, evapotranspiration_24h }, line);

	}

	TIFFClose(surface_temperature);
	TIFFClose(aerodynamic_resistance_tifR);
	TIFFClose(albedo);
	TIFFClose(soil_heat);
	TIFFClose(net_radiation);
	TIFFClose(evapotranspiration_fraction);
	TIFFClose(evapotranspiration_24h);

}
;

/**
 * @brief  Initializes TIFFs of the partial execution products as writable. Doing their setup based upon the tal TIFF characteristics.
 * @param  **tal: Tal TIFF.
 * @param  **albedo: Albedo TIFF.
 * @param  **ndvi: NDVI TIFF.
 * @param  **evi: EVI TIFF.
 * @param  **lai: LAI TIFF.
 * @param  **soil_heat: Soil heat flux TIFF.
 * @param  **surface_temperature: Surface temperature TIFF.
 * @param  **net_radiation: Net radiation TIFF.
 */
void Landsat::create_tiffs(TIFF **tal, TIFF **albedo, TIFF **ndvi, TIFF **evi, TIFF **lai, TIFF **soil_heat, TIFF **surface_temperature,
		TIFF **net_radiation) {
	*albedo = TIFFOpen(albedo_path.c_str(), "w8m");
	setup(*albedo, *tal);

	*ndvi = TIFFOpen(ndvi_path.c_str(), "w8m");
	setup(*ndvi, *tal);

	*evi = TIFFOpen(evi_path.c_str(), "w8m");
	setup(*evi, *tal);

	*lai = TIFFOpen(lai_path.c_str(), "w8m");
	setup(*lai, *tal);

	*soil_heat = TIFFOpen(soil_heat_path.c_str(), "w8m");
	setup(*soil_heat, *tal);

	*surface_temperature = TIFFOpen(surface_temperature_path.c_str(), "w8m");
	setup(*surface_temperature, *tal);

	*net_radiation = TIFFOpen(net_radiation_path.c_str(), "w8m");
	setup(*net_radiation, *tal);
}
;

/**
 * @brief  Open the partial products TIFF as readble TIFFs and create the final products TIFF (evapotranspiration_fraction and evapotranspiration_24h) as writable. For use them at the final phase.
 * @param  **albedo: Albedo TIFF.
 * @param  **ndvi: NDVI TIFF.
 * @param  **soil_heat: Soil heat flux TIFF.
 * @param  **surface_temperature: Surface temperature TIFF.
 * @param  **net_radiation: Net radiation TIFF.
 * @param  **evapotranspiration_fraction: Evapotranspiration fraction TIFF.
 * @param  **evapotranspiration_24h: Evapotranspiration 24 hours TIFF.
 */
void Landsat::open_tiffs(TIFF **albedo, TIFF **ndvi, TIFF **soil_heat, TIFF **surface_temperature, TIFF **net_radiation,
		TIFF **evapotranspiration_fraction, TIFF **evapotranspiration_24h) {

	*albedo = TIFFOpen(albedo_path.c_str(), "rm");
	*ndvi = TIFFOpen(ndvi_path.c_str(), "rm");
	*soil_heat = TIFFOpen(soil_heat_path.c_str(), "rm");
	*surface_temperature = TIFFOpen(surface_temperature_path.c_str(), "rm");
	*net_radiation = TIFFOpen(net_radiation_path.c_str(), "rm");

	*evapotranspiration_fraction = TIFFOpen(evapotranspiration_fraction_path.c_str(), "w8m");
	setup(*evapotranspiration_fraction, *albedo);

	*evapotranspiration_24h = TIFFOpen(evapotranspiration_24h_path.c_str(), "w8m");
	setup(*evapotranspiration_24h, *albedo);

}
