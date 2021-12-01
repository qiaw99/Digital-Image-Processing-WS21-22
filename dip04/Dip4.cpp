//============================================================================
// Name        : Dip4.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip4.h"

using namespace std;
using namespace cv;

namespace dip4 {

using namespace std::complex_literals;

/*

===== std::complex cheat sheet =====

Initialization:

std::complex<float> a(1.0f, 2.0f);
std::complex<float> a = 1.0f + 2.0if;

Common Operations:

std::complex<float> a, b, c;

a = b + c;
a = b - c;
a = b * c;
a = b / c;

std::sin, std::cos, std::tan, std::sqrt, std::pow, std::exp, .... all work as expected

Access & Specific Operations:

std::complex<float> a = ...;

float real = a.real();
float imag = a.imag();
float phase = std::arg(a);
float magnitude = std::abs(a);
float squared_magnitude = std::norm(a);

std::complex<float> complex_conjugate_a = std::conj(a);

*/



/**
 * @brief Computes the complex valued forward DFT of a real valued input
 * @param input real valued input
 * @return Complex valued output, each pixel storing real and imaginary parts
 */
cv::Mat_<std::complex<float>> DFTReal2Complex(const cv::Mat_<float>& input)
{
    Mat_<float> img = input.clone();
    Mat_<float> img2;

    // Convert to 2 channels
    img.convertTo(img2, CV_32FC2);
    Mat_<complex<float>> img_dft;

    dft(img2, img_dft, DFT_COMPLEX_OUTPUT);
    return img_dft;
}


/**
 * @brief Computes the real valued inverse DFT of a complex valued input
 * @param input Complex valued input, each pixel storing real and imaginary parts
 * @return Real valued output
 */
cv::Mat_<float> IDFTComplex2Real(const cv::Mat_<std::complex<float>>& input)
{
    Mat_<complex<float>> in = input.clone();
    Mat res = Mat::zeros(in.size(), in.type());

    idft(in, res, DFT_INVERSE | DFT_REAL_OUTPUT + DFT_SCALE);
    return res.clone();
}

/**
 * @brief Performes a circular shift in (dx,dy) direction
 * @param in Input matrix
 * @param dx Shift in x-direction
 * @param dy Shift in y-direction
 * @return Circular shifted matrix
*/
cv::Mat_<float> circShift(const cv::Mat_<float>& in, int dx, int dy)
{
    Mat dst(in.size(), CV_32FC1);
	int x, y;

	if (dx == 0 && dy == 0)
		return in;
	if (dx >= 0)
		x = dx % in.cols;
	else
		x = dx % in.cols + in.cols;
	if (dy >= 0)
		y = dy % in.rows;
	else
		y = dy % in.rows + in.rows;

	Mat tmp0, tmp1, tmp2, tmp3;
	Mat part0(in, Rect(0, 0, in.cols - x, in.rows - y));
	Mat part1(in, Rect(in.cols - x, 0, x, in.rows - y));
	Mat part2(in, Rect(0, in.rows - y, in.cols - x, y));
	Mat part3(in, Rect(in.cols - x, in.rows - y, x, y));

	part0.copyTo(tmp0);
	part1.copyTo(tmp1);
	part2.copyTo(tmp2);
	part3.copyTo(tmp3);

	tmp0.copyTo(in(Rect(x, y, in.cols - x, in.rows - y)));
	tmp1.copyTo(in(Rect(0, y, x, in.rows - y)));
	tmp2.copyTo(in(Rect(x, 0, in.cols - x, y)));
	tmp3.copyTo(in(Rect(0, 0, x, y)));

	return in;
}


/**
 * @brief Computes the thresholded inverse filter
 * @param input Blur filter in frequency domain (complex valued)
 * @param eps Factor to compute the threshold (relative to the max amplitude)
 * @return The inverse filter in frequency domain (complex valued)
 */
cv::Mat_<std::complex<float>> computeInverseFilter(const cv::Mat_<std::complex<float>>& input, const float eps)
{

    Mat_<complex<float>> src = input.clone();
    complex<float> one = 1;
    float maximum = abs(src.at<std::complex<float>>(0, 0));
    float temp;
    Mat res = Mat::zeros(src.size(), src.type());

    // Find max amplitude
    for(int i = 0; i < input.rows; i++){
        for(int j = 0; j < input.cols; j++){
            temp = abs(src.at<complex<float>>(i, j));
            if(temp > maximum){
                maximum = temp;
            }
        }
    }

    maximum *= eps;

    for(int i = 0; i < input.rows; i++){
        for(int j = 0; j < input.cols; j++){
            temp = abs(src.at<complex<float>>(i, j));
            if(temp >= maximum){
                res.at<complex<float>>(i, j) = one / src.at<complex<float>>(i, j);
            } else {
                res.at<complex<float>>(i, j) = one / maximum;
            }
        }
    }

    return res.clone();

}


/**
 * @brief Applies a filter (in frequency domain)
 * @param input Image in frequency domain (complex valued)
 * @param filter Filter in frequency domain (complex valued), same size as input
 * @return The filtered image, complex valued, in frequency domain
 */
cv::Mat_<std::complex<float>> applyFilter(const cv::Mat_<std::complex<float>>& input, const cv::Mat_<std::complex<float>>& filter)
{
    Mat_<complex<float>> res = Mat::zeros(input.size(), input.type());
    mulSpectrums(input.clone(), filter.clone(), res, DFT_COMPLEX_OUTPUT);
    return res.clone();
}


/**
 * @brief Function applies the inverse filter to restorate a degraded image
 * @param degraded Degraded input image
 * @param filter Filter which caused degradation
 * @param eps Factor to compute the threshold (relative to the max amplitude)
 * @return Restorated output image
 */
cv::Mat_<float> inverseFilter(const cv::Mat_<float>& degraded, const cv::Mat_<float>& filter, const float eps)
{
    Mat_<float> in = degraded.clone();
    Mat_<float> resized = Mat::zeros(in.size(), in.type());

    // Put kernel at origin
	filter.copyTo(resized(Rect(0, 0, filter.cols, filter.rows)));
	resized = circShift(resized, -filter.cols/2, -filter.rows/2);
	in = DFTReal2Complex(in);
	resized = DFTReal2Complex(resized);

    Mat_<complex<float>> inversed = computeInverseFilter(resized, eps);
    Mat_<complex<float>> img = applyFilter(in, inversed);

    Mat_<float> res = IDFTComplex2Real(img);

    return res;
}


/**
 * @brief Computes the Wiener filter
 * @param input Blur filter in frequency domain (complex valued)
 * @param snr Signal to noise ratio
 * @return The wiener filter in frequency domain (complex valued)
 */
cv::Mat_<std::complex<float>> computeWienerFilter(const cv::Mat_<std::complex<float>>& input, const float snr)
{
    Mat_<complex<float>> in = input.clone();
    Mat_<complex<float>> conju = Mat::zeros(in.size(), in.type());

    for(int i = 0; i < in.rows; i++){
        for(int j = 0; j < in.cols; j++){
            complex<float> temp_complex = conj(in.at<complex<float>>(i, j));
            float temp = pow(abs(in.at<complex<float>>(i, j)), 2);
            conju.at<complex<float>>(i, j) = temp_complex / complex<float>(temp + 1 / pow(snr,2));
        }
    }

    return conju.clone();
}

/**
 * @brief Function applies the wiener filter to restore a degraded image
 * @param degraded Degraded input image
 * @param filter Filter which caused degradation
 * @param snr Signal to noise ratio of the input image
 * @return Restored output image
 */
cv::Mat_<float> wienerFilter(const cv::Mat_<float>& degraded, const cv::Mat_<float>& filter, float snr)
{

    Mat_<float> in = degraded.clone();
    Mat_<float> resized = Mat::zeros(in.size(), in.type());

    // Put kernel at origin
	filter.copyTo(resized(Rect(0, 0, filter.cols, filter.rows)));
	resized = circShift(resized, -filter.cols/2, -filter.rows/2);

	Mat_<complex<float>> complex_resized = DFTReal2Complex(resized);
	Mat_<complex<float>> complex_in = DFTReal2Complex(in);

	Mat_<complex<float>> wienerFilter = computeWienerFilter(complex_resized, snr);
    Mat_<complex<float>> filteredImage = applyFilter(complex_in, wienerFilter);
    Mat_<float> res = IDFTComplex2Real(filteredImage);

    return res;
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

/**
 * function degrades the given image with gaussian blur and additive gaussian noise
 * @param img Input image
 * @param degradedImg Degraded output image
 * @param filterDev Standard deviation of kernel for gaussian blur
 * @param snr Signal to noise ratio for additive gaussian noise
 * @return The used gaussian kernel
 */
cv::Mat_<float> degradeImage(const cv::Mat_<float>& img, cv::Mat_<float>& degradedImg, float filterDev, float snr)
{

    int kSize = round(filterDev*3)*2 - 1;

    cv::Mat gaussKernel = cv::getGaussianKernel(kSize, filterDev, CV_32FC1);
    gaussKernel = gaussKernel * gaussKernel.t();

    cv::Mat imgs = img.clone();
    cv::dft( imgs, imgs, img.rows);
    cv::Mat kernels = cv::Mat::zeros( img.rows, img.cols, CV_32FC1);
    int dx, dy; dx = dy = (kSize-1)/2.;
    for(int i=0; i<kSize; i++)
        for(int j=0; j<kSize; j++)
            kernels.at<float>((i - dy + img.rows) % img.rows,(j - dx + img.cols) % img.cols) = gaussKernel.at<float>(i,j);
	cv::dft( kernels, kernels );
	cv::mulSpectrums( imgs, kernels, imgs, 0 );
	cv::dft( imgs, degradedImg,  cv::DFT_INVERSE + cv::DFT_SCALE, img.rows );

    cv::Mat mean, stddev;
    cv::meanStdDev(img, mean, stddev);

    cv::Mat noise = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
    cv::randn(noise, 0, stddev.at<double>(0)/snr);
    degradedImg = degradedImg + noise;
    cv::threshold(degradedImg, degradedImg, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(degradedImg, degradedImg, 0, 0, cv::THRESH_TOZERO);

    return gaussKernel;
}


}
