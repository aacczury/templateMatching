#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

class Parallel_process : public ParallelLoopBody{
private:
	Mat input;
	Mat &output;
	Mat templ;
	Mat m;

public:
	Parallel_process(Mat inputImg, Mat &outputImg, Mat templImg, Mat Mask)
		:input(inputImg), output(outputImg), templ(templImg), m(Mask){}

	virtual void operator()(const Range &range) const{
		for (int j = range.start; j < range.end; ++j){
			for (int i = 0; i < output.cols; ++i){
				Mat tmpI(input, Rect(i, j, templ.cols, templ.rows));
				Mat I;
				tmpI.copyTo(I, m);
				tmpI.release();
				((float *)output.data)[j * output.cols + i] = (float)norm(sum((I - templ).mul(I - templ)));
				I.release();
			}
			printf("%d\n", j);
		}
	}
};

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

	parallel_for_(Range(0, result.rows), Parallel_process(rgbImg, result, rgbTempl, splitTempl[3]));
	cout << "finish" << endl;
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
	resizeTempl.copyTo(img_display(Rect(minLoc.x, minLoc.y, resizeTempl.cols, resizeTempl.rows)), splitTempl[3]);

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

	splitTempl[3].convertTo(splitTempl[3], CV_8U, 255.0); // alpha channel is mask would be CV_8U

	/// Create windows
	namedWindow(image_window, CV_WINDOW_AUTOSIZE);
	namedWindow(result_window, CV_WINDOW_AUTOSIZE);
		
	testTM();

	waitKey(0);
	return 0;
}