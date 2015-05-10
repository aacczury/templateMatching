#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <omp.h>

using namespace std;
using namespace cv;

/// Global Variables
Mat img; Mat templ; Mat result;
Mat resizeImg, resizeTempl; // resize + rgba
Mat splitImg[4], splitTempl[4]; // resize + r,g,b,a
Mat rgbImg, rgbTempl; // resize + rgb
char* image_window = "Source Image";
char* result_window = "Result window";

int match_method;
int max_Trackbar = 5;

/// Function Headers
void MatchingMethod(int, void*);

void testTM(){
	/// Source image to display
	Mat img_display;
	resizeImg.copyTo(img_display);

	/// Create the result matrix
	int result_cols = resizeImg.cols - resizeTempl.cols + 1;
	int result_rows = resizeImg.rows - resizeTempl.rows + 1;

	result.create(result_rows, result_cols, CV_32FC1);

	Mat _T = rgbTempl - Mat(rgbTempl.rows, rgbTempl.cols, CV_32FC3, mean(rgbTempl));
	int i, j, x, y;
	for (j = 0; j < result_rows; ++j){
		for (i = 0; i < result_cols; ++i){
			Mat I(rgbImg, Rect(i, j, rgbTempl.cols, rgbTempl.rows));
			Mat _I = I - Mat(rgbTempl.rows, rgbTempl.cols, CV_32FC3, mean(I));
			((float *)result.data)[j * result_cols + i] = (float)norm(sum(_I.mul(_T)));
			//((float *)result.data)[j * result_cols + i] = norm(sum(_I.mul(_T)));
			//result.at<Vec4f>(j, i) = sum(_I.mul(_T)); (slower than access address)
			I.release();
			_I.release();
		}
		printf("%d\n", j);
	}
	cout << "finish" << endl;
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
	resizeTempl.copyTo(img_display(Rect(maxLoc.x, maxLoc.y, resizeTempl.cols, resizeTempl.rows)));

	imshow(image_window, result);
	imshow(result_window, img_display);
	img_display.convertTo(img_display, CV_8UC4, 255.0);
	imwrite("out.png", img_display);
	return;
}


/** @function main */
int main(int argc, char** argv)
{
	double scale = 0.2;
	/// Load image and template (contain alpha channel)
	img = imread("input.png", CV_LOAD_IMAGE_UNCHANGED);
	templ = imread("match.png", CV_LOAD_IMAGE_UNCHANGED);
	img.convertTo(img, CV_32FC4, 1.0 / 255.0);
	templ.convertTo(templ, CV_32FC4, 1.0 / 255.0);

	resize(img, resizeImg, Size(img.cols * scale, img.rows * scale));
	resize(templ, resizeTempl, Size(templ.cols * scale, templ.rows * scale));

	split(resizeImg, splitImg);
	merge(splitImg, 3, rgbImg);
	split(resizeTempl, splitTempl);
	merge(splitTempl, 3, rgbTempl);

	/// Create windows
	namedWindow(image_window, CV_WINDOW_AUTOSIZE);
	namedWindow(result_window, CV_WINDOW_AUTOSIZE);

	testTM();

	/// Create Trackbar
	//char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
	//createTrackbar(trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod);

	//MatchingMethod(0, 0);

	waitKey(0);
	return 0;
}

/**
* @function MatchingMethod
* @brief Trackbar callback
*/
void MatchingMethod(int, void*)
{
	/// Source image to display
	Mat img_display;
	resizeImg.copyTo(img_display);

	/// Create the result matrix
	int result_cols = resizeImg.cols - resizeTempl.cols + 1;
	int result_rows = resizeImg.rows - resizeTempl.rows + 1;

	result.create(result_rows, result_cols, CV_32FC1);

	/// Do the Matching and Normalize
	matchTemplate(resizeImg, resizeTempl, result, match_method);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}

	/// Show me what you got
	rectangle(img_display, matchLoc, Point(matchLoc.x + resizeTempl.cols, matchLoc.y + resizeTempl.rows), Scalar::all(0), 2, 8, 0);
	rectangle(result, matchLoc, Point(matchLoc.x + resizeTempl.cols, matchLoc.y + resizeTempl.rows), Scalar::all(0), 2, 8, 0);

	imshow(image_window, img_display);
	imwrite("out.png", img_display);
	imshow(result_window, result);

	return;
}