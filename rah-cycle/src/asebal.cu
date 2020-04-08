#include "asebal.h"

/**
 * @brief   Calculates the four quartiles of an input TIFF.
 * @param   target: The input TIFF.
 * @param   vQuartile: The output array, with a size of four.
 * @param   height_band: Band height.
 * @param   width_band: Band width.
 */
void quartile(TIFF* target, double* vQuartile, int height_band, int width_band){

    const int SIZE = height_band * width_band;
    double target_line[width_band];
    double* target_values = (double*) malloc(sizeof(double) * SIZE);
    if(target_values == NULL) exit(15);
    int pos = 0;

    for(int line = 0; line < height_band; line++){
        read_line_tiff(target, target_line, line);

        for(int col = 0; col < width_band; col++){

            if(!std::isnan(target_line[col]) && !std::isinf(target_line[col])) {

                target_values[pos] = target_line[col];
                pos++;

            }

        }

    }

    std::sort(target_values, target_values + pos);

    //First quartile
    vQuartile[0] = target_values[int(floor(0.25 * pos))];

    //Second quartile
    vQuartile[1] = target_values[int(floor(0.5 * pos))];

    //Third quartile
    vQuartile[2] = target_values[int(floor(0.75 * pos))];

    //Fourth quartile
    vQuartile[3] = target_values[pos-1];

    free(target_values);

}

/**
 * @brief  Computes the HO.
 * @param  net_radiation_line[]: Array containing the specified line from the Rn computation.
 * @param  soil_heat_flux[]: Array containing the specified line from the G computation.
 * @param  width_band: Band width.
 * @param  ho_line[]: Auxiliary array for save the calculated value of HO for the line.
 */
void hoFunction(double net_radiation_line[], double soil_heat_flux[], int width_band, double ho_line[]){

    for(int col = 0; col < width_band; col++)
        ho_line[col] = net_radiation_line[col] - soil_heat_flux[col];

};

/**
 * @brief  Select the hot pixel.
 * @param  ndvi: NDVI TIFF.
 * @param  surface_temperature: TS TIFF.
 * @param  albedo: Albedo TIFF.
 * @param  net_radiation: Rn TIFF.
 * @param  soil_heat: G TIFF.
 * @param  height_band: Band height.
 * @param  width_band: Band width.
 * @retval Candidate struct containing the hot pixel.
 */
Candidate getHotPixel(TIFF** ndvi, TIFF** surface_temperature, TIFF** albedo, TIFF** net_radiation, TIFF** soil_heat, int height_band, int width_band, int threadNum){

    double ndvi_line[width_band], surface_temperature_line[width_band];
    double net_radiation_line[width_band], soil_heat_line[width_band];
    double ho_line[width_band], albedo_line[width_band];

    double* ndviQuartile = (double*) malloc(sizeof(double) * 4);
    double* tsQuartile = (double*) malloc(sizeof(double) * 4);
    double* albedoQuartile = (double*) malloc(sizeof(double) * 4);

    int valid = 0;
    const int MAXC = 10000000;

    quartile(*ndvi, ndviQuartile, height_band, width_band);
    quartile(*surface_temperature, tsQuartile, height_band, width_band);
    quartile(*albedo, albedoQuartile, height_band, width_band);

    //Creating first pixel group
    Candidate* candidatesGroupI;
    candidatesGroupI = (Candidate*) malloc(MAXC * sizeof(Candidate));
    if(candidatesGroupI == NULL) exit(15);

    //Creating device arrays
    double *ndvi_dev, *ts_dev, *albedo_dev, *rn_dev, *soil_dev, *ho_dev;
    double *ndviQ_dev, *tsQ_dev, *albQ_dev;
    int *valid_dev;
    Candidate *candidates_dev;

    HANDLE_ERROR(cudaMalloc((void**) &candidates_dev, MAXC * sizeof(Candidate)));
	HANDLE_ERROR(cudaMalloc((void**) &ndvi_dev, width_band * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &ts_dev, width_band * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &albedo_dev, width_band * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &rn_dev, width_band * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &soil_dev, width_band * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &ho_dev, width_band * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &ndviQ_dev, 4 * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &tsQ_dev, 4 * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &albQ_dev, 4 * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**) &valid_dev, sizeof(int)));

	HANDLE_ERROR(cudaMemcpy(candidates_dev, candidatesGroupI, MAXC*sizeof(Candidate), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(ndviQ_dev, ndviQuartile, 4*sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(tsQ_dev, tsQuartile, 4*sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(albQ_dev, albedoQuartile, 4*sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(valid_dev, &valid, sizeof(int), cudaMemcpyHostToDevice));


    for(int line = 0; line < height_band; line ++){
        read_line_tiff(*net_radiation, net_radiation_line, line);
        read_line_tiff(*soil_heat, soil_heat_line, line);

        hoFunction(net_radiation_line, soil_heat_line, width_band, ho_line);

        read_line_tiff(*ndvi, ndvi_line, line);
        read_line_tiff(*surface_temperature, surface_temperature_line, line);
        read_line_tiff(*albedo, albedo_line, line);

        HANDLE_ERROR(cudaMemcpy(ndvi_dev, ndvi_line, width_band*sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(ts_dev, surface_temperature_line, width_band*sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(albedo_dev, albedo_line, width_band*sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(rn_dev, net_radiation_line, width_band*sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(soil_dev, soil_heat_line, width_band*sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(ho_dev, ho_line, width_band*sizeof(double), cudaMemcpyHostToDevice));

		asebalFilterHot<<<(width_band + threadNum - 1) / threadNum, threadNum>>>(candidates_dev, ndvi_dev, ts_dev, albedo_dev, rn_dev, soil_dev, ho_dev, valid_dev, albQ_dev, ndviQ_dev, tsQ_dev, line, width_band);

    }

    HANDLE_ERROR(cudaMemcpy(&valid, valid_dev, sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(candidatesGroupI, candidates_dev, MAXC * sizeof(Candidate), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(candidates_dev));
	HANDLE_ERROR(cudaFree(ndvi_dev));
	HANDLE_ERROR(cudaFree(ts_dev));
	HANDLE_ERROR(cudaFree(albedo_dev));
	HANDLE_ERROR(cudaFree(ndviQ_dev));
	HANDLE_ERROR(cudaFree(tsQ_dev));
	HANDLE_ERROR(cudaFree(albQ_dev));
	HANDLE_ERROR(cudaFree(rn_dev));
	HANDLE_ERROR(cudaFree(soil_dev));
	HANDLE_ERROR(cudaFree(ho_dev));
	HANDLE_ERROR(cudaFree(valid_dev));

	if(valid <= 0) {
		std::cerr << "Pixel problem! - There are no precandidates";
		exit(15);
	}

    //Creating second pixel group, all values lower than the 3rd quartile are excluded
    std::sort(candidatesGroupI, candidatesGroupI + valid, compare_candidate_temperature);
    unsigned int pos = int(floor(valid * 0.75));
    std::vector<Candidate> candidatesGroupII(candidatesGroupI + pos, candidatesGroupI + valid);

    if(candidatesGroupII.size() <= 0) {
		std::cerr << "Pixel problem! - There are no final candidates";
		exit(15);
	}

    pos = int(floor(candidatesGroupII.size() * 0.5));
    Candidate hotPixel = candidatesGroupII[pos];

    free(ndviQuartile);
    free(tsQuartile);
    free(albedoQuartile);

    //hotPixel.toString();

    return hotPixel;
}

/**
 * @brief  Select the cold pixel.
 * @param  ndvi: NDVI TIFF.
 * @param  surface_temperature: TS TIFF.
 * @param  albedo: Albedo TIFF.
 * @param  net_radiation: Rn TIFF.
 * @param  soil_heat: G TIFF.
 * @param  height_band: Band height.
 * @param  width_band: Band width.
 * @retval Candidate struct containing the cold pixel.
 */
Candidate getColdPixel(TIFF** ndvi, TIFF** surface_temperature, TIFF** albedo, TIFF** net_radiation, TIFF** soil_heat, int height_band, int width_band, int threadNum) {

	 double ndvi_line[width_band], surface_temperature_line[width_band];
	    double net_radiation_line[width_band], soil_heat_line[width_band];
	    double ho_line[width_band], albedo_line[width_band];

	    double* ndviQuartile = (double*) malloc(sizeof(double) * 4);
	    double* tsQuartile = (double*) malloc(sizeof(double) * 4);
	    double* albedoQuartile = (double*) malloc(sizeof(double) * 4);

	    int valid = 0;
	    const int MAXC = 10000000;

	    quartile(*ndvi, ndviQuartile, height_band, width_band);
	    quartile(*surface_temperature, tsQuartile, height_band, width_band);
	    quartile(*albedo, albedoQuartile, height_band, width_band);

	    //Creating first pixel group
	    Candidate* candidatesGroupI;
	    candidatesGroupI = (Candidate*) malloc(MAXC * sizeof(Candidate));
	    if(candidatesGroupI == NULL) exit(15);

	    //Creating device arrays
	    double *ndvi_dev, *ts_dev, *albedo_dev, *rn_dev, *soil_dev, *ho_dev;
	    double *ndviQ_dev, *tsQ_dev, *albQ_dev;
	    int *valid_dev;
	    Candidate *candidates_dev;

	    HANDLE_ERROR(cudaMalloc((void**) &candidates_dev, MAXC * sizeof(Candidate)));
		HANDLE_ERROR(cudaMalloc((void**) &ndvi_dev, width_band * sizeof(double)));
		HANDLE_ERROR(cudaMalloc((void**) &ts_dev, width_band * sizeof(double)));
		HANDLE_ERROR(cudaMalloc((void**) &albedo_dev, width_band * sizeof(double)));
		HANDLE_ERROR(cudaMalloc((void**) &rn_dev, width_band * sizeof(double)));
		HANDLE_ERROR(cudaMalloc((void**) &soil_dev, width_band * sizeof(double)));
		HANDLE_ERROR(cudaMalloc((void**) &ho_dev, width_band * sizeof(double)));
		HANDLE_ERROR(cudaMalloc((void**) &ndviQ_dev, 4 * sizeof(double)));
		HANDLE_ERROR(cudaMalloc((void**) &tsQ_dev, 4 * sizeof(double)));
		HANDLE_ERROR(cudaMalloc((void**) &albQ_dev, 4 * sizeof(double)));
		HANDLE_ERROR(cudaMalloc((void**) &valid_dev, sizeof(int)));

		HANDLE_ERROR(cudaMemcpy(candidates_dev, candidatesGroupI, MAXC*sizeof(Candidate), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(ndviQ_dev, ndviQuartile, 4*sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(tsQ_dev, tsQuartile, 4*sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(albQ_dev, albedoQuartile, 4*sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(valid_dev, &valid, sizeof(int), cudaMemcpyHostToDevice));


	    for(int line = 0; line < height_band; line ++){
	        read_line_tiff(*net_radiation, net_radiation_line, line);
	        read_line_tiff(*soil_heat, soil_heat_line, line);

	        hoFunction(net_radiation_line, soil_heat_line, width_band, ho_line);

	        read_line_tiff(*ndvi, ndvi_line, line);
	        read_line_tiff(*surface_temperature, surface_temperature_line, line);
	        read_line_tiff(*albedo, albedo_line, line);

	        HANDLE_ERROR(cudaMemcpy(ndvi_dev, ndvi_line, width_band*sizeof(double), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(ts_dev, surface_temperature_line, width_band*sizeof(double), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(albedo_dev, albedo_line, width_band*sizeof(double), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(rn_dev, net_radiation_line, width_band*sizeof(double), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(soil_dev, soil_heat_line, width_band*sizeof(double), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(ho_dev, ho_line, width_band*sizeof(double), cudaMemcpyHostToDevice));

			asebalFilterCold<<<(width_band + threadNum - 1) / threadNum, threadNum>>>(candidates_dev, ndvi_dev, ts_dev, albedo_dev, rn_dev, soil_dev, ho_dev, valid_dev, albQ_dev, ndviQ_dev, tsQ_dev, line, width_band);

	    }

	    HANDLE_ERROR(cudaMemcpy(&valid, valid_dev, sizeof(int), cudaMemcpyDeviceToHost));
	    HANDLE_ERROR(cudaMemcpy(candidatesGroupI, candidates_dev, MAXC * sizeof(Candidate), cudaMemcpyDeviceToHost));

		HANDLE_ERROR(cudaFree(candidates_dev));
		HANDLE_ERROR(cudaFree(ndvi_dev));
		HANDLE_ERROR(cudaFree(ts_dev));
		HANDLE_ERROR(cudaFree(albedo_dev));
		HANDLE_ERROR(cudaFree(ndviQ_dev));
		HANDLE_ERROR(cudaFree(tsQ_dev));
		HANDLE_ERROR(cudaFree(albQ_dev));
		HANDLE_ERROR(cudaFree(rn_dev));
		HANDLE_ERROR(cudaFree(soil_dev));
		HANDLE_ERROR(cudaFree(ho_dev));
		HANDLE_ERROR(cudaFree(valid_dev));

		if(valid <= 0) {
			std::cerr << "Pixel problem! - There are no precandidates";
			exit(15);
		}

	    //Creating second pixel group, all values lower than the 3rd quartile are excluded
	    std::sort(candidatesGroupI, candidatesGroupI + valid, compare_candidate_temperature);
	    unsigned int pos = int(floor(valid * 0.25));
	    std::vector<Candidate> candidatesGroupII(candidatesGroupI, candidatesGroupI + pos);

	    if(candidatesGroupII.size() <= 0) {
			std::cerr << "Pixel problem! - There are no final candidates";
			exit(15);
		}

	    pos = int(floor(candidatesGroupII.size() * 0.5));
	    Candidate coldPixel = candidatesGroupII[pos];

	    free(ndviQuartile);
	    free(tsQuartile);
	    free(albedoQuartile);

	    //coldPixel.toString();

	    return coldPixel;
}
