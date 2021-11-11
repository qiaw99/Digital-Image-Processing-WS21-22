//============================================================================
// Name        : Dip2.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip2.h"

namespace dip2 {


/**
 * @brief Convolution in spatial domain.
 * @details Performs spatial convolution of image and filter kernel.
 * @params src Input image
 * @params kernel Filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel)
{
   Mat_<float> res = src.clone();
    int k = kernel.cols / 2;
    Mat_<float> padding(src.rows + k*2, src.cols + k*2, src.depth());
    copyMakeBorder(src, padding, k, k, k, k, BORDER_REPLICATE);

    flip(kernel, kernel, -1);
    for (int y = k; y < padding.rows-k; y++){
        for (int x = k; x < padding.cols-k; x++){
            float conv = 0;
            for (int j = 0; j < kernel.rows; j++){
                for (int i = 0; i < kernel.cols; i++){
                    conv += padding.at<float>(y - (kernel.rows/2) + j, x - (kernel.cols/2) + i)*kernel.at<float>(j, i);
                }
            }
            res.at<float>(y-k, x-k) = conv;
        }
    }
    return res;
}

/**
 * @brief Moving average filter (aka box filter)
 * @note: you might want to use Dip2::spatialConvolution(...) within this function
 * @param src Input image
 * @param kSize Window size used by local average
 * @returns Filtered image
 */
cv::Mat_<float> averageFilter(const cv::Mat_<float>& src, int kSize)
{
   Mat weight = Mat::ones(kSize, kSize, CV_64FC1);
    Mat input = src.clone();

    // Apply convolution on input
    Mat res = spatialConvolution(input, weight);

    // Average the result
    res /= kSize * kSize;

    return res;
}

/**
 * @brief Median filter
 * @param src Input image
 * @param kSize Window size used by median operation
 * @returns Filtered image
 */
cv::Mat_<float> medianFilter(const cv::Mat_<float>& src, int kSize)
{
   // TO DO !!
   return src.clone();
}

/**
 * @brief Bilateral filer
 * @param src Input image
 * @param kSize Size of the kernel
 * @param sigma_spatial Standard-deviation of the spatial kernel
 * @param sigma_radiometric Standard-deviation of the radiometric kernel
 * @returns Filtered image
 */
cv::Mat_<float> bilateralFilter(const cv::Mat_<float>& src, int kSize, float sigma_spatial, float sigma_radiometric)
{
    // TO DO !!
    return src.clone();
}

/**
 * @brief Non-local means filter
 * @note: This one is optional!
 * @param src Input image
 * @param searchSize Size of search region
 * @param sigma Optional parameter for weighting function
 * @returns Filtered image
 */
cv::Mat_<float> nlmFilter(const cv::Mat_<float>& src, int searchSize, double sigma)
{
    return src.clone();
}



/**
 * @brief Chooses the right algorithm for the given noise type
 * @note: Figure out what kind of noise NOISE_TYPE_1 and NOISE_TYPE_2 are and select the respective "right" algorithms.
 */
NoiseReductionAlgorithm chooseBestAlgorithm(NoiseType noiseType)
{
    // TO DO !!
    return (NoiseReductionAlgorithm) -1;
}



cv::Mat_<float> denoiseImage(const cv::Mat_<float> &src, NoiseType noiseType, dip2::NoiseReductionAlgorithm noiseReductionAlgorithm)
{
    // TO DO !!

    // for each combination find reasonable filter parameters

    switch (noiseReductionAlgorithm) {
        case dip2::NR_MOVING_AVERAGE_FILTER:
            switch (noiseType) {
                case NOISE_TYPE_1:
    		    return dip2::averageFilter(src, 5);
		case NOISE_TYPE_2:
		    return dip2::averageFilter(src, 3);
		default:
		    throw std::runtime_error("Unhandled noise type!");
		}
	case dip2::NR_MEDIAN_FILTER:
		switch (noiseType) {
		case NOISE_TYPE_1:
		    return dip2::medianFilter(src, 5);
		case NOISE_TYPE_2:
		    return dip2::medianFilter(src, 5);
                default:
                    throw std::runtime_error("Unhandled noise type!");
            }
        case dip2::NR_BILATERAL_FILTER:
            switch (noiseType) {
                case NOISE_TYPE_1:
                    return dip2::bilateralFilter(src, 1, 1.0f, 1.0f);
                case NOISE_TYPE_2:
                    return dip2::bilateralFilter(src, 1, 1.0f, 1.0f);
                default:
                    throw std::runtime_error("Unhandled noise type!");
            }
        default:
            throw std::runtime_error("Unhandled filter type!");
    }
}





// Helpers, don't mind these

const char *noiseTypeNames[NUM_NOISE_TYPES] = {
    "NOISE_TYPE_1",
    "NOISE_TYPE_2",
};

const char *noiseReductionAlgorithmNames[NUM_FILTERS] = {
    "NR_MOVING_AVERAGE_FILTER",
    "NR_MEDIAN_FILTER",
    "NR_BILATERAL_FILTER",
};


}
