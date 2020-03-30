#include "esaSebal.h"

/**
 * @brief   Tests the land cover homogeneity of a pixel.
 * @note    A the land cover of a pixel is homogeneous if every neighbour pixel inside a 7x7 window is also agricultural field.
 * @param   landCover: Land Cover TIFF.
 * @param   mask: Output binary TIFF, where pixels with 1 means that is a homogeneous pixel and 0 means the otherwise.
 */
void testLandCoverHomogeneity(TIFF* landCover, TIFF* mask, int threadNum){

	uint32 height_band, width_band;
	TIFFGetField(landCover, TIFFTAG_IMAGELENGTH, &height_band);
	TIFFGetField(landCover, TIFFTAG_IMAGEWIDTH, &width_band);

	double* buffer = (double *) malloc(7 * width_band * sizeof(double));
	int* output_line = (int*) malloc(width_band * sizeof(int));

	double* buffer_dev;
	cudaMalloc((void**) &buffer_dev, 7 * width_band * sizeof(double*));

	int* output_dev;
	cudaMalloc((void**) &output_dev, width_band * sizeof(int*));

	int relation[7] = {-1, -1, -1, -1, -1, -1, -1};

	for(int line = 0; line < height_band; line++) {

		for(int i = -3; i < 4; i++) {

			if(line + i >= 0 && line + i < height_band){

				if(relation[(line + i) % 7] != (line + i)) {
					read_line_tiff(landCover, buffer + ((line + i) % 7) * width_band, line + i);
					relation[(line + i) % 7] = line + i;
				}

			}

		}

		cudaMemcpy(buffer_dev, buffer, 7 * width_band * sizeof(double), cudaMemcpyHostToDevice);

		landCoverHomogeneityKernel<<< (width_band + threadNum - 1) / threadNum , threadNum>>>(buffer_dev, output_dev, line, width_band, height_band);

		cudaMemcpy(output_line, output_dev, width_band * sizeof(int), cudaMemcpyDeviceToHost);

		write_line_tiff(mask, output_line, line);

	}

	free(buffer);
	free(output_line);
	cudaFree(buffer_dev);
	cudaFree(output_dev);

}

/**
 * @brief   Tests the ndvi, surface_temperature and albedo homogeneity of a pixel.
 * @note    A pixel is homogeneous in these criteria if inside a 7x7 window the coefficient of variation of the albedo and ndvi is less or equal than 25%
 *          and the surface temperature has a standard deviation less or equal than 1.5 K.
 * @param   ndvi: NDVI TIFF.
 * @param   surface_temperature: TS TIFF.
 * @param   albedo: Albedo TIFF.
 * @param   maskLC: A binary TIFF conteining the data of the land cover homogeneity.
 * @param   output: A binary TIFF, where pixels with 1 means that is a homogeneous pixel in land cover, ndvi, surface temperature and albedo, and 0 means otherwise.
 */
void testHomogeneity(TIFF* ndvi, TIFF* surface_temperature, TIFF* albedo, TIFF* maskLC, TIFF* output, int threadNum){

    uint32 height_band, width_band;
    TIFFGetField(ndvi, TIFFTAG_IMAGELENGTH, &height_band);
    TIFFGetField(ndvi, TIFFTAG_IMAGEWIDTH, &width_band);

    double *bufferTS = (double *) malloc(7 * width_band * sizeof(double));
    double *bufferNDVI = (double *) malloc(7 * width_band * sizeof(double));
    double *bufferAlb = (double *) malloc(7 * width_band * sizeof(double));
    int *bufferLandCover = (int *) malloc(width_band * sizeof(int));
    int *output_line = (int *) malloc(width_band * sizeof(int));

    double* bufferNDVI_dev;
    double* bufferTS_dev;
    double* bufferAlb_dev;
    int* bufferLand_dev;
    int* output_dev;
    cudaMalloc((void**) &bufferNDVI_dev, 7 * width_band * sizeof(double*));
    cudaMalloc((void**) &bufferTS_dev, 7 * width_band * sizeof(double*));
    cudaMalloc((void**) &bufferAlb_dev, 7 * width_band * sizeof(double*));
    cudaMalloc((void**) &bufferLand_dev, width_band * sizeof(int*));
    cudaMalloc((void**) &output_dev, width_band * sizeof(int*));

    int relation[7] = {-1, -1, -1, -1, -1, -1, -1};

    for(int line = 0; line < height_band; line++) {

    	read_line_tiff(maskLC, bufferLandCover, line);

    	for(int i = -3; i < 4; i++) {

			if(line + i >= 0 && line + i < height_band){

				if(relation[(line + i) % 7] != (line + i)) {
					read_line_tiff(ndvi, bufferNDVI + ((line + i) % 7) * width_band, line + i);
					read_line_tiff(surface_temperature, bufferTS + ((line + i) % 7) * width_band, line + i);
					read_line_tiff(albedo, bufferAlb + ((line + i) % 7) * width_band, line + i);
					relation[(line + i) % 7] = line + i;
				}

			}

		}

    	cudaMemcpy(bufferNDVI_dev, bufferNDVI, 7 * width_band * sizeof(double), cudaMemcpyHostToDevice);
    	cudaMemcpy(bufferTS_dev, bufferTS, 7 * width_band * sizeof(double), cudaMemcpyHostToDevice);
    	cudaMemcpy(bufferAlb_dev, bufferAlb, 7 * width_band * sizeof(double), cudaMemcpyHostToDevice);
    	cudaMemcpy(bufferLand_dev, bufferLandCover, width_band * sizeof(int), cudaMemcpyHostToDevice);

    	testHomogeneityKernel<<<(width_band + threadNum - 1) / threadNum , threadNum>>>(bufferNDVI_dev, bufferTS_dev, bufferAlb_dev, bufferLand_dev, output_dev, line, width_band, height_band);

    	cudaMemcpy(output_line, output_dev, width_band * sizeof(int), cudaMemcpyDeviceToHost);

        write_line_tiff(output, output_line, line);

    }

    free(bufferAlb);
    free(bufferNDVI);
    free(bufferTS);
    free(bufferLandCover);
    free(output_line);
    cudaFree(bufferNDVI_dev);
    cudaFree(bufferTS_dev);
    cudaFree(bufferAlb_dev);
    cudaFree(bufferLand_dev);
    cudaFree(output_dev);

}

/**
 * @brief   Removes from the binary TIFF gave as input, groups of pixel with value 1, that have less pixels than a specified value.
 * @param   input: A binary TIFF to be processed.
 * @param   output: A binary TIFF.
 * @param   groupSize: The inferior limit of the group of pixels size.
 */
void testMorphological(TIFF* input, TIFF* output, int groupSize){

    // Read the entire TIFF to the memory
    // Create an same size matrix to serve as output

    uint32 height_band, width_band;
    TIFFGetField(input, TIFFTAG_IMAGELENGTH, &height_band);
    TIFFGetField(input, TIFFTAG_IMAGEWIDTH, &width_band);

    int** inputM = (int **) malloc(height_band * sizeof(int *));
    int** outputM = (int **) malloc(height_band * sizeof(int *));

    for(int i = 0; i < height_band; i++){

        inputM[i] = (int *) malloc(width_band * sizeof(int));
        outputM[i] = (int *) malloc(width_band * sizeof(int));

    }

    for(int i = 0; i < height_band; i++){

        read_line_tiff(input, inputM[i], i);
        read_line_tiff(input, outputM[i], i);

    }

    // Apply the routine

    std::queue< std::pair<int, int> > fila;
    std::set< std::pair<int, int> > cont;

    for(int line = 0; line < height_band; line++) {

        for(int col = 0; col < width_band; col++) {

            if(inputM[line][col] == 1) {

                fila.push({line, col});
                cont.insert({line, col});
                inputM[line][col] = -1;

                while (!fila.empty()) {

                    int i = fila.front().first;
                    int j = fila.front().second;
                    fila.pop();

                    if(j + 1 < width_band){

                        if(inputM[i][j+1] == 1) {
                            fila.push({i, j+1});
                            cont.insert({i, j+1});
                            inputM[i][j+1] = -1;
                        }

                        if(i + 1 < height_band && inputM[i+1][j+1] == 1){
                            fila.push({i+1, j+1});
                            cont.insert({i+1, j+1});
                            inputM[i+1][j+1] = -1;
                        }

                        if(i > 0 && inputM[i-1][j+1] == 1){
                            fila.push({i-1, j+1});
                            cont.insert({i-1, j+1});
                            inputM[i-1][j+1] = -1;
                        }

                    }

                    if(j > 0){

                        if(inputM[i][j-1] == 1){
                            fila.push({i, j-1});
                            cont.insert({i, j-1});
                            inputM[i][j-1] = -1;
                        }

                        if(i + 1 < height_band && inputM[i+1][j-1] == 1){
                            fila.push({i+1, j-1});
                            cont.insert({i+1, j-1});
                            inputM[i+1][j-1] = -1;
                        }

                        if(i > 0 && inputM[i-1][j-1] == 1){
                            fila.push({i-1, j-1});
                            cont.insert({i-1, j-1});
                            inputM[i-1][j-1] = -1;
                        }

                    }

                    if(i + 1 < height_band && inputM[i+1][j] == 1){
                        fila.push({i+1, j});
                        cont.insert({i+1, j});
                        inputM[i+1][j] = -1;
                    }

                    if(i > 0 && inputM[i-1][j] == 1){
                        fila.push({i-1, j});
                        cont.insert({i-1, j});
                        inputM[i-1][j] = -1;
                    }

                }

                int group = cont.size();

                for(auto elem : cont) {

                    outputM[elem.first][elem.second] = (group >= groupSize);

                }

                cont.clear();

            } else if (inputM[line][col] == 0) {

                outputM[line][col] = 0;

            }

        }

    }

    // Write output TIFF

    for(int i = 0; i < height_band; i++){

        write_line_tiff(output, outputM[i], i);

    }

    for(int i = 0; i < height_band; i++){

        free(inputM[i]);
        free(outputM[i]);

    }

    free(inputM);
    free(outputM);

}

/**
 * @brief  Computes the HO.
 * @param  net_radiation_line[]: Array containing the specified line from the Rn computation.
 * @param  soil_heat_flux[]: Array containing the specified line from the G computation.
 * @param  width_band: Band width.
 * @param  ho_line[]: Auxiliary array for save the calculated value of HO for the line.
 */
void hoCalc(double net_radiation_line[], double soil_heat_flux[], int width_band, double ho_line[]){

    for(int col = 0; col < width_band; col++)
        ho_line[col] = net_radiation_line[col] - soil_heat_flux[col];

};

/**
 * @brief   Select a pair of pixels, which will be used as hot and cold pixels of the SEBAL Algorithm.
 * @note    For further information, check https://www.sciencedirect.com/science/article/pii/S0034425717302018.
 * @param   ndvi: NDVI TIFF.
 * @param   surface_temperature: TS TIFF.
 * @param   albedo: Albedo TIFF.
 * @param   net_radiation: Rn TIFF.
 * @param   soil_heat: G TIFF.
 * @param   landCover: Land Cover TIFF.
 * @param   height_band: Band height.
 * @param   width_band: Band width.
 * @param   output_path: A path where will be written auxiliary TIFFs generated in the process.
 * @retval  A pair struct of pixel, where the first one is the hot pixel selected, and the second is the cold one.
 */
std::pair<Candidate, Candidate> esaPixelSelect(TIFF** ndvi, TIFF** surface_temperature, TIFF** albedo, TIFF** net_radiation, TIFF** soil_heat, TIFF** landCover, int height_band, int width_band, std::string output_path, int threadNum){

    //Testing land cover homogeneity
    TIFF* outputLC = TIFFOpen((output_path + "/outLC.tif").c_str(), "w8m");
    setup(outputLC, width_band, height_band, 32, 2);
    testLandCoverHomogeneity(*landCover, outputLC, threadNum);
    TIFFClose(outputLC);

    //Testing the other params homogeneity
    outputLC = TIFFOpen((output_path + "/outLC.tif").c_str(), "r");
    TIFF* outputH = TIFFOpen((output_path + "/outH.tif").c_str(), "w8m");
    setup(outputH, width_band, height_band, 32, 2);
    testHomogeneity(*ndvi, *surface_temperature, *albedo, outputLC, outputH, threadNum);
    TIFFClose(outputLC);
    TIFFClose(outputH);

    //Morphological test
    outputH = TIFFOpen((output_path + "/outH.tif").c_str(), "r");
    TIFF* outputAll = TIFFOpen((output_path + "/outAll.tif").c_str(), "w8m");
    setup(outputAll, width_band, height_band, 32, 2);
    testMorphological(outputH, outputAll, 50);
    TIFFClose(outputH);
    TIFFClose(outputAll);

    //Getting the candidates
    outputAll = TIFFOpen((output_path + "/outAll.tif").c_str(), "r");
    int all_condition[width_band];
    std::vector<Candidate> listTS;

    //Auxiliary arrays
    double ndvi_line[width_band], surface_temperature_line[width_band];
    double net_radiation_line[width_band], soil_heat_line[width_band];
    double ho_line[width_band];

    //Creating candidates array for TS and then for NDVI as a copy
    for(int line = 0; line < height_band; line++){

        read_line_tiff(outputAll, all_condition, line);

        read_line_tiff(*net_radiation, net_radiation_line, line);
        read_line_tiff(*soil_heat, soil_heat_line, line);

        hoCalc(net_radiation_line, soil_heat_line, width_band, ho_line);

        read_line_tiff(*ndvi, ndvi_line, line);
        read_line_tiff(*surface_temperature, surface_temperature_line, line);

        for(int col = 0; col < width_band; col++) {

            if(all_condition[col] && !isnan(ndvi_line[col])){

                listTS.push_back(Candidate(ndvi_line[col],
                                           surface_temperature_line[col],
                                           net_radiation_line[col],
                                           soil_heat_line[col],
                                           ho_line[col],
                                           line, col));

            }

        }

    }

    if(listTS.size() <= 0) {
		std::cerr << "Pixel problem! - There are no precandidates";
		exit(15);
	}

    std::vector<Candidate> listNDVI (listTS);

    std::sort(listTS.begin(), listTS.end(), compare_candidate_temperature);
    std::sort(listNDVI.begin(), listNDVI.end(), compare_candidate_ndvi);

    double ts_min = listTS[0].temperature, ts_max = listTS[listTS.size() - 1].temperature;
    double ndvi_min = listNDVI[0].ndvi, ndvi_max = listNDVI[listNDVI.size() - 1].ndvi;
    int binTS = int(ceil((ts_max - ts_min)/0.25)); //0.25 is TS bin size
    int binNDVI = int(ceil((ndvi_max - ndvi_min)/0.01)); //0.01 is ndvi bin size

    std::vector<Candidate> histTS[binTS], final_histTS;

    for(Candidate c : listTS) {

        int pos = int(ceil((c.temperature - ts_min)/0.25));
        histTS[pos > 0 ? pos-1 : 0].push_back(c);

    }

    for(int i = 0; i < binTS; i++) {

        if(histTS[i].size() > 50) {

            for(Candidate c : histTS[i])
                final_histTS.push_back(c);

        }

    }

    if(final_histTS.size() <= 0) {
		std::cerr << "Pixel problem! - There are no final TS candidates";
		exit(15);
	}

    std::vector<Candidate> histNDVI[binNDVI], final_histNDVI;
    for(Candidate c : listNDVI) {

        int pos = int(ceil((c.ndvi - ndvi_min)/0.01));
        histNDVI[pos > 0 ? pos-1 : 0].push_back(c);

    }

    for(int i = 0; i < binNDVI; i++) {

        if(histNDVI[i].size() > 50) {

            for(Candidate c : histNDVI[i])
                final_histNDVI.push_back(c);

        }

    }

    if(final_histNDVI.size() <= 0) {
		std::cerr << "Pixel problem! - There are no final NDVI candidates";
		exit(15);
	}

    // Select cold pixel
    int pixel_count = 0, n1 = 1, n2 = 1, ts_pos, ndvi_pos, beginTs = 0, beginNDVI = final_histNDVI.size() - 1;
    std::vector<Candidate> coldPixels;
    while (pixel_count < 10 && !(n2 == 10 && n1 == 10)) {

        ts_pos = int(floor(n1/100.0 * final_histTS.size()));
        ndvi_pos = int(floor((100 - n2)/100.0 * final_histNDVI.size()));

        for(int i = beginTs; i <= ts_pos && pixel_count < 10; i++) {

            for(int j = beginNDVI; j >= ndvi_pos && pixel_count < 10; j--) {

                if(equals(final_histTS[i], final_histNDVI[j])){

                    coldPixels.push_back(final_histTS[i]);
                    pixel_count++;

                }

            }

        }

	    beginTs = ts_pos;
	    beginNDVI = ndvi_pos;

        if(n2 < 10) n2++;
        else if(n1 < 10){
            n1++;
            beginNDVI = final_histNDVI.size() - 1;
        }

    }

    if(coldPixels.size() <= 0) {
		std::cerr << "Pixel problem! - There are no cold candidates";
		exit(15);
	}

    //Select hot pixel
    pixel_count = 0, n1 = 1, n2 = 1;
    std::vector<Candidate> hotPixels;
    beginTs = final_histTS.size() - 1, beginNDVI = 0;
    while (pixel_count < 10 && !(n2 == 10 && n1 == 10)) {

        ts_pos = int(floor((100 - n1)/100.0 * final_histTS.size()));
        ndvi_pos = int(floor(n2/100.0 * final_histNDVI.size()));

        for(int i = beginNDVI; i <= ndvi_pos && pixel_count < 10; i++) {

            for(int j = beginTs; j >= ts_pos && pixel_count < 10; j--) {

                if(equals(final_histTS[j], final_histNDVI[i])){

                    hotPixels.push_back(final_histTS[j]);
                    pixel_count++;

                }

            }

        }

	    beginTs = ts_pos;
	    beginNDVI = ndvi_pos;

        if(n2 < 10) n2++;
        else if(n1 < 10){
            n1++;
            beginTs = final_histTS.size() - 1;
        }

    }

    if(hotPixels.size() <= 0) {
		std::cerr << "Pixel problem! - There are no hot candidates";
		exit(15);
	}

    std::sort(coldPixels.begin(), coldPixels.end(), compare_candidate_ndvi);
    std::sort(hotPixels.begin(), hotPixels.end(), compare_candidate_temperature);

    return {hotPixels[hotPixels.size() - 1], coldPixels[coldPixels.size() - 1]};

}
