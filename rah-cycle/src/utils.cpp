#include "utils.h"

/**
 * @brief  Configures a TIFF based on a second TIFF.
 * @param  new_tif: TIFF to be configured.
 * @param  base_tif: TIFF used to provide the configurations.
 */
void setup(TIFF* new_tif, TIFF* base_tif){
    uint32 image_width, image_length;

    TIFFGetField(base_tif, TIFFTAG_IMAGEWIDTH,      &image_width);
    TIFFGetField(base_tif, TIFFTAG_IMAGELENGTH,     &image_length);

    TIFFSetField(new_tif, TIFFTAG_IMAGEWIDTH     , image_width);
    TIFFSetField(new_tif, TIFFTAG_IMAGELENGTH    , image_length);
    TIFFSetField(new_tif, TIFFTAG_BITSPERSAMPLE  , 64);
    TIFFSetField(new_tif, TIFFTAG_SAMPLEFORMAT   , 3);
    TIFFSetField(new_tif, TIFFTAG_COMPRESSION    , 1);
    TIFFSetField(new_tif, TIFFTAG_PHOTOMETRIC    , 1);
    TIFFSetField(new_tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(new_tif, TIFFTAG_ROWSPERSTRIP   , 1);
    TIFFSetField(new_tif, TIFFTAG_RESOLUTIONUNIT , 1);
    TIFFSetField(new_tif, TIFFTAG_XRESOLUTION    , 1);
    TIFFSetField(new_tif, TIFFTAG_YRESOLUTION    , 1);
    TIFFSetField(new_tif, TIFFTAG_PLANARCONFIG   , PLANARCONFIG_CONTIG);
};

/**
 * @brief  Verifies if a TIFF was open correctly.
 * @param  tif: TIFF to be verified
 * @throws Throw an error with exit code 1 if the TIFF isn't open.
 */
void check_open_tiff(TIFF* tif){
    if(!tif){
        std::cerr << "Open tiff problem" << std::endl;
        exit(1);
    }
};

/**
 * @brief  Reads the values of a line in a TIFF saving them into an array.
 * @param  tif: TIFF who line should be read.
 * @param  tif_line[]: Array where the data will be saved.
 * @param  line: Number of the line to be read.
 * @throws Throw an error with exit code 3 if the read couldn't be done.
 */
void read_line_tiff(TIFF* tif, double tif_line[], int line){
    if(TIFFReadScanline(tif, tif_line, line) < 0){
    	std::cerr << "Read problem" << std::endl;
        exit(3);
    }
};

/**
 * @brief  Reads the values of a line in a TIFF saving them into an array.
 * @param  tif: TIFF who line should be read.
 * @param  tif_line: image data ref
 * @param  line: Number of the line to be read.
 * @throws Throw an error with exit code 3 if the read couldn't be done.
 */
void read_line_tiff(TIFF* tif, tdata_t tif_line, int line){
    if(TIFFReadScanline(tif, tif_line, line) < 0){
    	std::cerr << "Read problem" << std::endl;
        exit(3);
    }
};

/**
 * @brief  Reads the value contained in a specific position of a TIFF.
 * @param  tif: TIFF who value should be read.
 * @param  col: Number of the column to be read.
 * @param  line: Number of the line to be read.
 * @throws Throw an error with exit code 3 if the read couldn't be done.
 */
double read_position_tiff(TIFF* tif, int col, int line){
    uint32 width_band;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width_band);

    double tif_line[width_band];

    read_line_tiff(tif, tif_line, line);

    return tif_line[col];
};

/**
 * @brief  Writes values from an array to a specific line in a TIFF.
 * @param  tif: TIFF who line should be written.
 * @param  tif_line[]: Array containing the values to be written.
 * @param  line: Number of the line to be read.
 * @throws Throw an error with exit code 4 if the write couldn't be done.
 */
void write_line_tiff(TIFF* tif, double tif_line[], int line){

    if (TIFFWriteScanline(tif, tif_line, line) < 0){
    	std::cerr << "Write problem!" << std::endl;
        exit(4);
    }

};

/**
 * @brief  Closes open TIFFs.
 * @param  tiffs[]: Array containing opened tiffs to be closed.
 * @param  quant_tiffs: Length of the array or number of tiffs.
 */
void close_tiffs(TIFF* tiffs[], int quant_tiffs){
    for(int i = 1; i < quant_tiffs; i++)
        TIFFClose(tiffs[i]);
};

/**
 * @brief  Writes values from an array to a specific line in a TIFF. Doing this for each respective array and TIFF at the vectors parameters passed.
 * @note:  The positions both vectors should be corresponding arrays and TIFFs.
 * @param  products_line: Vector containing the arrays of a line to be written on a respective TIFF.
 * @param  products: Vector containing the respective TIFF for each array.
 * @param  line: Number of the line that should be written.
 */
void save_tiffs(std::vector<double*> products_line, std::vector<TIFF*> products, int line){

    for (unsigned i = 0; i < products.size(); i++)
        write_line_tiff(products[i], products_line[i], line);

};