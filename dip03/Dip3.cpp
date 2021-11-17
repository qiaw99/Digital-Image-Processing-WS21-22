//============================================================================
// Name    : Dip3.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip3.h"

#include <stdexcept>
#include <math.h>
//#include <vector>

using namespace std;
using namespace cv;

namespace dip3 {

const char * const filterModeNames[NUM_FILTER_MODES] = {
    "FM_SPATIAL_CONVOLUTION",
    "FM_FREQUENCY_CONVOLUTION",
    "FM_SEPERABLE_FILTER",
    "FM_INTEGRAL_IMAGE",
};

/**
 * @brief Generates 1D gaussian filter kernel of given size
 * @param kSize Kernel size (used to calculate standard deviation)
 * @returns The generated filter kernel
 */
cv::Mat_<float> createGaussianKernel1D(int kSize){

    // According to exercise sheet
    float sigma = 1 / (float) kSize;
    float mean = (kSize - 1) / 2;

    Mat_<float> kernel = Mat::zeros(Size(kSize, 1), CV_64FC1);
    float sum = 0.0;
    float inter = 0.0;

    for(int i = 0; i < kSize; i++){
        inter = 1 / (2 * CV_PI * sigma) * exp(-pow(i - mean, 2) / (2 * pow(sigma, 2)));
        kernel.at<float>(0, i) = inter;
        sum += inter;
    }

    kernel /= sum;

    return kernel.clone();
}

/**
 * @brief Generates 2D gaussian filter kernel of given size
 * @param kSize Kernel size (used to calculate standard deviation)
 * @returns The generated filter kernel
 */
cv::Mat_<float> createGaussianKernel2D(int kSize){

    float sigma = 1 / (float) kSize;
    float mean = (kSize - 1) / 2;
    Mat_<float> kernel = Mat::zeros(Size(kSize, kSize), CV_64FC1);

    float sum = 0.0;
    float inter = 0.0;

    for(int i = 0; i < kSize; i++){
        for(int j = 0; j < kSize; j++){
            inter = 1 / (2 * CV_PI * sigma * sigma) * exp(-pow(i - mean, 2) / (2 * pow(sigma, 2)) - pow(j - mean, 2) / (2 * pow(sigma, 2)));
            kernel.at<float>(i, j) = inter;
            sum += inter;
        }
    }

    kernel /= sum;

    return kernel.clone();
}

/**
 * @brief Performes a circular shift in (dx,dy) direction
 * @param in Input matrix
 * @param dx Shift in x-direction
 * @param dy Shift in y-direction
 * @returns Circular shifted matrix
 */
cv::Mat_<float> circShift(const cv::Mat_<float>& in, int dx, int dy){

	Mat_<float> res = Mat::zeros(in.size(), CV_64FC1);
	Mat_<float> src = in.clone();
	int cols = src.cols;
	int rows = src.rows;
    int x, y = 0;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if ((x = (i + dx) % cols) < 0) {
                x = cols + (i + dx) % cols;
            }

			if ((y = (j + dy) % rows) <  0) {
				y = rows + (j + dy) % rows;
			}

			res.at<float>(x, y) = src.at<float>(i, j);
		}
	}

	return res.clone();
}


/**
 * @brief Performes convolution by multiplication in frequency domain
 * @param in Input image
 * @param kernel Filter kernel
 * @returns Output image
 */
cv::Mat_<float> frequencyConvolution(const cv::Mat_<float>& in, const cv::Mat_<float>& kernel){

    Mat_<float> res = in.clone();
    Mat_<float> resized = Mat::zeros(in.size(), CV_64FC1);

    // Put kernel at origin
	kernel.copyTo(resized(Rect(0, 0, kernel.cols, kernel.rows)));

	Mat result = Mat::zeros(in.rows, in.cols, CV_32FC1);
	Mat input_dft, kernel_dft;

	resized = circShift(resized, -1, -1);
	dft(res, res, 0);
	dft(resized, resized, 0);

	mulSpectrums(res, resized, res, 0);
	dft(res, res, DFT_INVERSE + DFT_SCALE);
	return res;
}


/**
 * @brief  Performs UnSharp Masking to enhance fine image structures
 * @param in The input image
 * @param filterMode How convolution for smoothing operation is done
 * @param size Size of used smoothing kernel
 * @param thresh Minimal intensity difference to perform operation
 * @param scale Scaling of edge enhancement
 * @returns Enhanced image
 */
cv::Mat_<float> usm(const cv::Mat_<float>& in, FilterMode filterMode, int size, float thresh, float scale)
{
    Mat_<float> src = in.clone();

    // l0
    Mat_<float> original = in.clone();

    // l1
    Mat_<float> smoothed = smoothImage(src, size, filterMode);

    Mat_<float> l2 = smoothed - original;
    Mat_<float> absL2 = cv::abs(l2.clone());

    for(int i = 0; i < absL2.rows; i++){
        for(int j = 0; j < absL2.cols; j++){
            if(absL2.at<float>(i ,j) > thresh){
                original.at<float>(i, j) += scale * l2.at<float>(i, j);
            }
        }
    }

    return original.clone();
}


/**
 * @brief Convolution in spatial domain
 * @param src Input image
 * @param kernel Filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel)
{
    // Reused from 2nd. exercise
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
 * @brief Convolution in spatial domain by seperable filters
 * @param src Input image
 * @param size Size of filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> separableFilter(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel){

    Mat_<float> res = spatialConvolution(src, kernel);
    transpose(res, res);
    res = spatialConvolution(res, kernel);
    transpose(res, res);

    return res.clone();

}


/**
 * @brief Convolution in spatial domain by integral images
 * @param src Input image
 * @param size Size of filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> satFilter(const cv::Mat_<float>& src, int size){

   // optional

   return src;

}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

/**
 * @brief Performs a smoothing operation but allows the algorithm to be chosen
 * @param in Input image
 * @param size Size of filter kernel
 * @param type How is smoothing performed?
 * @returns Smoothed image
 */
cv::Mat_<float> smoothImage(const cv::Mat_<float>& in, int size, FilterMode filterMode)
{
    switch(filterMode) {
        case FM_SPATIAL_CONVOLUTION: return spatialConvolution(in, createGaussianKernel2D(size));	// 2D spatial convolution
        case FM_FREQUENCY_CONVOLUTION: return frequencyConvolution(in, createGaussianKernel2D(size));	// 2D convolution via multiplication in frequency domain
        case FM_SEPERABLE_FILTER: return separableFilter(in, createGaussianKernel1D(size));	// seperable filter
        case FM_INTEGRAL_IMAGE: return satFilter(in, size);		// integral image
        default:
            throw std::runtime_error("Unhandled filter type!");
    }
}



}

