//============================================================================
// Name        : Dip2.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip2.h"
#include <algorithm>
#include <math.h>
#include<time.h>
#include <thread>
#include <chrono>
using namespace cv;
using namespace std;

namespace dip2 {


/**
 * @brief Convolution in spatial domain.
 * @details Performs spatial convolution of image and filter kernel.
 * @params src Input image
 * @params kernel Filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel){

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
    cout << "average filter starting" <<endl;
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
    cout << "median filter starting" <<endl;
    Mat_<float> padding = src.clone();
    Mat_<float> res = src.clone();

    int rows = res.rows;
    int cols = res.cols;

    // Add padding
    copyMakeBorder(padding, padding, kSize/2, kSize/2, kSize/2, kSize/2, BORDER_REPLICATE);

    int anchor = kSize / 2;

    Mat_<float> temp;
    clock_t startTime, endTime;
    startTime = clock();

    // Loop through the whole image
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            vector<float> vec;
            for(int p = 0; p < kSize; p++){
                for(int q = 0; q < kSize; q++){
                    vec.push_back(padding.at<float>(i+p, j+q));
                }
            }
            sort(vec.begin(), vec.end());
            res.at<float>(i, j) = vec[kSize * kSize / 2];
        }
    }
    endTime = clock();
    cout << "\nProcessed time for median: " << endTime - startTime << "ms" << endl;

    return res;

}

/**
 * @brief Get gaussian distribution given x and covaraince
 * @returns gaussian values
 */
float gaussian(float x, float sigma) {
    return exp(-(pow(x, 2))/(2 * pow(sigma, 2)));

}

/**
 * @brief compute distance between points (x,y) and (i,j) in L2 norm
 * @returns distances
 */
float distance(int x, int y, int i, int j) {
    return sqrt(pow(x - i, 2) + pow(y - j, 2));
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

    Mat_<float> res = src.clone();
    Mat_<float> padding = src.clone();

    int rows = res.rows;
    int cols = res.cols;

    // Add padding
    copyMakeBorder(padding, padding, kSize/2, kSize/2, kSize/2, kSize/2, BORDER_REPLICATE);

    int anchor = kSize / 2;

    Mat_<float> temp;
    float part1 = 0;
    float part2 = 0;
    // Loop through the whole image
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            float gi = 0;
            float gs = 0;
            float filtered = 0;
            float part = 0;

            for(int p = 0; p < kSize; p++){
                for(int q = 0; q < kSize; q++){
                    float f_prime = padding.at<float>(i+p, j+q);
                    float fx = src.at<float>(i,j);
                    gi = gaussian(distance(i, j, p, q), sigma_radiometric);
                    gs = gaussian(f_prime-fx, sigma_spatial);
                    part += gi * gs;
                    filtered += gi * gs * f_prime;
                }
            }
            res.at<float>(i,j) = filtered / part;
        }
    }

    return res;
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
    // TODO
    return src.clone();
}



/**
 * @brief Chooses the right algorithm for the given noise type
 * @note: Figure out what kind of noise NOISE_TYPE_1 and NOISE_TYPE_2 are and select the respective "right" algorithms.
 */
NoiseReductionAlgorithm chooseBestAlgorithm(NoiseType noiseType)
{
    switch (noiseType) {
        // Shot noise
        case NOISE_TYPE_1:
            return NR_MEDIAN_FILTER;
        // Gaussian noise
        case NOISE_TYPE_2:
            return NR_MOVING_AVERAGE_FILTER;
        default:
            break;
	}
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
                    return dip2::averageFilter(src, 5);
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
                    return dip2::bilateralFilter(src, 3, 100.0f, 1.0f);
                case NOISE_TYPE_2:
                    return dip2::bilateralFilter(src, 3, 100.0f, 1.0f);
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

