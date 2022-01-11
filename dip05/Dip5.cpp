//============================================================================
// Name        : Dip5.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip5.h"
#include <math.h>
using namespace cv;
using namespace std;


namespace dip5 {


/**
* @brief Generates gaussian filter kernel of given size
* @param kSize Kernel size (used to calculate standard deviation)
* @returns The generated filter kernel
*/
cv::Mat_<float> createGaussianKernel1D(float sigma)
{
    unsigned kSize = getOddKernelSizeForSigma(sigma);
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
cv::Mat_<float> separableFilter(const cv::Mat_<float>& src, const cv::Mat_<float>& kernelX, const cv::Mat_<float>& kernelY)
{
    Mat_<float> res = spatialConvolution(src, kernelX);
    transpose(res, res);
    res = spatialConvolution(res, kernelY);
    transpose(res, res);

    return res.clone();
}


/**
 * @brief Creates kernel representing fst derivative of a Gaussian kernel (1-dimensional)
 * @param sigma standard deviation of the Gaussian kernel
 * @returns the calculated kernel
 */
cv::Mat_<float> createFstDevKernel1D(float sigma)
{
    unsigned kSize = getOddKernelSizeForSigma(sigma);
    Mat_<float> gaussianKernel = createGaussianKernel1D(sigma);

    float temp = 0.0;
    float inter = 0.0;
    for(int i = 0; i < kSize; i++){
        temp = gaussianKernel.at<float>(0, i);
        inter = -temp / (2 * CV_PI * pow(sigma, 4)) * exp(-pow(temp, 2) / (2 * pow(sigma, 2)));
        gaussianKernel.at<float>(0, i) = inter;
    }

    return gaussianKernel;
}


/**
 * @brief Calculates the directional gradients through convolution
 * @param img The input image
 * @param sigmaGrad The standard deviation of the Gaussian kernel for the directional gradients
 * @param gradX Matrix through which to return the x component of the directional gradients
 * @param gradY Matrix through which to return the y component of the directional gradients
 */
void calculateDirectionalGradients(const cv::Mat_<float>& img, float sigmaGrad,
                            cv::Mat_<float>& gradX, cv::Mat_<float>& gradY)
{
    gradX.create(img.rows, img.cols);
    gradY.create(img.rows, img.cols);
    Mat_<float> gaussianKernel = createGaussianKernel1D(sigmaGrad);
    Mat_<float> gaussianKernelDerivative = createFstDevKernel1D(sigmaGrad);

    gradX = separableFilter(img, gaussianKernelDerivative, gaussianKernel);
    gradY = separableFilter(img, gaussianKernel, gaussianKernelDerivative);
}

/**
 * @brief Calculates the structure tensors (per pixel)
 * @param gradX The x component of the directional gradients
 * @param gradY The y component of the directional gradients
 * @param sigmaNeighborhood The standard deviation of the Gaussian kernel for computing the "neighborhood summation".
 * @param A00 Matrix through which to return the A_{0,0} elements of the structure tensor of each pixel.
 * @param A01 Matrix through which to return the A_{0,1} elements of the structure tensor of each pixel.
 * @param A11 Matrix through which to return the A_{1,1} elements of the structure tensor of each pixel.
 */
void calculateStructureTensor(const cv::Mat_<float>& gradX, const cv::Mat_<float>& gradY, float sigmaNeighborhood,
                            cv::Mat_<float>& A00, cv::Mat_<float>& A01, cv::Mat_<float>& A11)
{
    A00.create(gradX.rows, gradX.cols);
    A01.create(gradX.rows, gradX.cols);
    A11.create(gradX.rows, gradX.cols);

    cv::Mat_<float> gaussianKernel = createGaussianKernel1D(sigmaNeighborhood);
    Mat_<float> gxgx, gxgy, gygy;
    gxgx = gradX.mul(gradX);
    gxgy = gradX.mul(gradY);
    gygy = gradY.mul(gradY);
    A00 = separableFilter(gxgx, gaussianKernel, gaussianKernel);
    A01 = separableFilter(gxgy, gaussianKernel, gaussianKernel);
    A11 = separableFilter(gygy, gaussianKernel, gaussianKernel);
}

/**
 * @brief Calculates the feature point weight and isotropy from the structure tensors.
 * @param A00 The A_{0,0} elements of the structure tensor of each pixel.
 * @param A01 The A_{0,1} elements of the structure tensor of each pixel.
 * @param A11 The A_{1,1} elements of the structure tensor of each pixel.
 * @param weight Matrix through which to return the weights of each pixel.
 * @param isotropy Matrix through which to return the isotropy of each pixel.
 */
void calculateFoerstnerWeightIsotropy(const cv::Mat_<float>& A00, const cv::Mat_<float>& A01, const cv::Mat_<float>& A11,
                                    cv::Mat_<float>& weight, cv::Mat_<float>& isotropy)
{
    weight.create(A00.rows, A00.cols);
    isotropy.create(A00.rows, A00.cols);

    Mat_<float> trace = A00 + A11;
    Mat_<float> det = A00.mul(A11) - A01.mul(A01);
    for(int i = 0; i < A00.rows; i++){
        for(int j = 0; j < A00.cols; j++){
            weight.at<float>(i, j) = det.at<float>(i, j) / max(trace.at<float>(i, j), 1e-8f);
            isotropy.at<float>(i, j) = 4 * det.at<float>(i, j) / pow(max(trace.at<float>(i, j), 1e-8f), 2);
        }
    }
}


/**
 * @brief Finds Foerstner interest points in an image and returns their location.
 * @param img The greyscale input image
 * @param sigmaGrad The standard deviation of the Gaussian kernel for the directional gradients
 * @param sigmaNeighborhood The standard deviation of the Gaussian kernel for computing the "neighborhood summation" of the structure tensor.
 * @param fractionalMinWeight Threshold on the weight as a fraction of the mean of all locally maximal weights.
 * @param minIsotropy Threshold on the isotropy of interest points.
 * @returns List of interest point locations.
 */
std::vector<cv::Vec2i> getFoerstnerInterestPoints(const cv::Mat_<float>& img, float sigmaGrad, float sigmaNeighborhood, float fractionalMinWeight, float minIsotropy)
{
    Mat_<float> gradX, gradY, A00, A11, A01, weight, isotropy;
    vector<cv::Vec2i> res;

    calculateDirectionalGradients(img, sigmaGrad, gradX, gradY);
    calculateStructureTensor(gradX, gradY, sigmaNeighborhood, A00, A01, A11);
    calculateFoerstnerWeightIsotropy(A00, A01, A11, weight, isotropy);
    for(int i = 0; i < weight.rows; i++){
        for(int j = 0; j < weight.cols; j++){
            if(weight.at<float>(i, j) > fractionalMinWeight){
                if(isotropy.at<float>(i, j) > minIsotropy){
                    if(isLocalMaximum(weight, j, i)){
                        res.push_back(Vec2i(j, i));
                    }
                }
            }
        }
    }

    return res;
}



/* *****************************
  GIVEN FUNCTIONS
***************************** */


// Use this to compute kernel sizes so that the unit tests can simply hard checks for correctness.
unsigned getOddKernelSizeForSigma(float sigma)
{
    unsigned kSize = (unsigned) std::ceil(5.0f * sigma) | 1;
    if (kSize < 3) kSize = 3;
    return kSize;
}

bool isLocalMaximum(const cv::Mat_<float>& weight, int x, int y)
{
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++) {
            int x_ = std::min(std::max(x+j, 0), weight.cols-1);
            int y_ = std::min(std::max(y+i, 0), weight.rows-1);
            if (weight(y_, x_) > weight(y, x))
                return false;
        }
    return true;
}

}
