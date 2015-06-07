#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;

typedef struct tmpMatchDiff{
	char *filename;
	Vec2i pos;
	float diff;
}matchDiff;

void VectorAdd(Mat inputImg, Mat &resultImg, Mat templImg, Mat Mask){
	//int i = threadIdx.x;
	int result_cols = inputImg.cols - templImg.cols + 1;
	int result_rows = inputImg.rows - templImg.rows + 1;


	for (int j = 0; j < result_rows; ++j){
		for (int i = 0; i < result_cols; ++i){
			Mat tmpI(inputImg, Rect(i, j, templImg.cols, templImg.rows)), I;
			tmpI.copyTo(I, Mask); tmpI.release();
			((float *)resultImg.data)[j * result_cols + i] = (float)norm(sum((I-templImg).mul(I-templImg)));
			I.release();
		}
	}
}

void templateMatching(Mat inputImg, Mat &resultImg, Mat templImg, Mat Mask, vector<matchDiff> &allDiff, char *fileWithDir){
	/// Create the result matrix
	int result_cols = inputImg.cols - templImg.cols + 1;
	int result_rows = inputImg.rows - templImg.rows + 1;

	resultImg.create(result_rows, result_cols, CV_32FC1);

	gpu::GpuMat d_inputImg(inputImg), d_templImg(templImg), d_Mask(Mask), d_resultImg(resultImg);
	for (int j = 0; j < result_rows; ++j){
		for (int i = 0; i < result_cols; ++i){
			gpu::GpuMat tmpI(d_inputImg, Rect(i, j, templImg.cols, templImg.rows)), d_I;
			tmpI.copyTo(d_I, d_Mask); tmpI.release();
			gpu::GpuMat d_disMat;
			gpu::subtract(d_I, d_templImg, d_disMat);
			gpu::multiply(d_disMat, d_disMat, d_disMat);
			(float)gpu::norm(d_disMat);
			d_I.release();
		}
		//printf("%d\n", j);
	}

	for (int j = 0; j < resultImg.rows; ++j)
		for (int i = 0; i < resultImg.cols; ++i){
			matchDiff pDiff;
			pDiff.filename = fileWithDir;
			pDiff.pos = Vec2i(i, j); // (col, row) (x, y)
			pDiff.diff = ((float *)resultImg.data)[j * resultImg.cols + i];
			allDiff.push_back(pDiff);
		}
	//cout << "finish" << endl;
	normalize(resultImg, resultImg, 0, 1, NORM_MINMAX, -1, Mat());
	resultImg.convertTo(resultImg, CV_8UC1, 255.0);

	return;
}

int main(int argc, char* argv[])
{
	printf("Device count: %i\n", cv::gpu::getCudaEnabledDeviceCount());
	cv::gpu::DeviceInfo dev_info(0);
	printf("Device compatible: %d\n", dev_info.isCompatible());

	try
	{
		/*Mat src_host = imread("input.png", CV_LOAD_IMAGE_GRAYSCALE);
		gpu::GpuMat dst, src(src_host);

		gpu::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);

		Mat result_host(dst);
		imshow("Result", result_host);*/


		Mat img, outputImg;
		double scale = 0.2;
		vector<matchDiff> allDiff;
		img = imread("input.png", CV_LOAD_IMAGE_UNCHANGED);

		Mat templ;
		Mat resizeImg, resizeTempl; // resize + rgba
		Mat splitImg[4], splitTempl[4]; // resize + r,g,b,a
		Mat rgbImg, rgbTempl; // resize + rgb
		char *inputDir = "match", *outputDir = "result";
		allDiff.clear();

		resize(img, resizeImg, Size(img.cols * scale, img.rows * scale));
		split(resizeImg, splitImg);
		merge(splitImg, 3, rgbImg);
		rgbImg.convertTo(rgbImg, CV_32FC4, 1.0 / 255.0);
		templ = imread("match.png", CV_LOAD_IMAGE_UNCHANGED);
		resize(templ, resizeTempl, Size(templ.cols * scale, templ.rows * scale));
		split(resizeTempl, splitTempl); // alpha channel is mask would be CV_8U
		merge(splitTempl, 3, rgbTempl);
		rgbTempl.convertTo(rgbTempl, CV_32FC4, 1.0 / 255.0);

		Mat resultImg;

		templateMatching(rgbImg, resultImg, rgbTempl, splitTempl[3], allDiff, "match.png");
		//runTemplateMatching(img, allDiff, scale);
		//outputDiff(allDiff);
		//outputByTxt(img, outputImg, scale);

		namedWindow("output", CV_WINDOW_AUTOSIZE);
		imshow("output", resultImg);
		imwrite("outputImg.png", resultImg);
	}
	catch (const cv::Exception& ex)
	{
		cout << "Error: " << ex.what() << endl;
	}

	cv::waitKey(0);
	return 0;
}

/*
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

using namespace System;
using namespace System::IO;
using namespace System::Collections;

typedef struct tmpMatchDiff{
	char *filename;
	Vec2i pos;
	float diff;
}matchDiff;
bool compareByDiff(const matchDiff &a, const matchDiff &b){
	return a.diff < b.diff;
}

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
				Mat tmpI(input, Rect(i, j, templ.cols, templ.rows)), I;
				tmpI.copyTo(I, m); tmpI.release();
				((float *)output.data)[j * output.cols + i] = (float)norm(sum((I - templ).mul(I - templ)));
				I.release();
			}
			//printf("%d\n", j);
		}
	}
};

void templateMatching(Mat inputImg, Mat &resultImg, Mat templImg, Mat Mask, vector<matchDiff> &allDiff, char *fileWithDir){
	/// Create the result matrix
	int result_cols = inputImg.cols - templImg.cols + 1;
	int result_rows = inputImg.rows - templImg.rows + 1;

	resultImg.create(result_rows, result_cols, CV_32FC1);

	parallel_for_(Range(0, resultImg.rows), Parallel_process(inputImg, resultImg, templImg, Mask));
	for (int j = 0; j < resultImg.rows; ++ j)
		for (int i = 0; i < resultImg.cols; ++i){
			matchDiff pDiff;
			pDiff.filename = fileWithDir;
			pDiff.pos = Vec2i(i, j); // (col, row) (x, y)
			pDiff.diff = ((float *)resultImg.data)[j * resultImg.cols + i];
			allDiff.push_back(pDiff);
		}
	//cout << "finish" << endl;
	normalize(resultImg, resultImg, 0, 1, NORM_MINMAX, -1, Mat());
	resultImg.convertTo(resultImg, CV_8UC1, 255.0);
	
	return;
}

void runTemplateMatching(Mat img, vector<matchDiff> &allDiff, double scale){
	Mat templ;
	Mat resizeImg, resizeTempl; // resize + rgba
	Mat splitImg[4], splitTempl[4]; // resize + r,g,b,a
	Mat rgbImg, rgbTempl; // resize + rgb
	char *inputDir = "match", *outputDir = "result";
	allDiff.clear();

	resize(img, resizeImg, Size(img.cols * scale, img.rows * scale));
	split(resizeImg, splitImg);
	merge(splitImg, 3, rgbImg);
	rgbImg.convertTo(rgbImg, CV_32FC4, 1.0 / 255.0);

	System::String^ folder = gcnew System::String(inputDir);
	array<System::String ^>^ file = Directory::GetFiles(folder);
	for (int i = 0; i < file->Length; ++i){
		char *fileWithDir = (char*)System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(file[i]).ToPointer();
		char *fileName = (char*)System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(file[i]->Split('\\')[1]).ToPointer();
		printf("%s\n", fileName);
		
		templ = imread(fileWithDir, CV_LOAD_IMAGE_UNCHANGED);
		resize(templ, resizeTempl, Size(templ.cols * scale, templ.rows * scale));
		split(resizeTempl, splitTempl); // alpha channel is mask would be CV_8U
		merge(splitTempl, 3, rgbTempl);
		rgbTempl.convertTo(rgbTempl, CV_32FC4, 1.0 / 255.0);

		Mat resultImg;
		templateMatching(rgbImg, resultImg, rgbTempl, splitTempl[3], allDiff, fileWithDir);
	}
	sort(allDiff.begin(), allDiff.end(), compareByDiff);
}

void outputByTxt(Mat img, Mat &outputImg, double scale){
	img.copyTo(outputImg);
	Hashtable ^fileNameHash = gcnew Hashtable();

	FILE *pFile;
	pFile = fopen("outAllDiff.txt", "r");
	char str[1000];
	int count = 0;
	vector<matchDiff> outputMatch; outputMatch.clear();
	while (fgets(str, 1000, pFile)){
		char *filename = (char *)malloc(100 * sizeof(char));
		int x, y;
		char *pch = strtok(str, ",");
		strcpy(filename, pch);
		pch = strtok(NULL, ",");
		x = atoi(pch);
		pch = strtok(NULL, ",");
		y = atoi(pch);

		if (!fileNameHash->ContainsKey(gcnew System::String(filename))){
			//printf("%s\t(%d,%d)\n", filename, x, y);
			count++;
			fileNameHash->Add(gcnew System::String(filename), 1);
			matchDiff pDiff; pDiff.filename = filename; pDiff.pos = Vec2i(x, y);
			outputMatch.push_back(pDiff);
		}
		if (count >= 100)
			break;
	}

	Mat templ, oSplitTempl[4];
	for (int i = outputMatch.size() - 1; i >= 0; --i){
		templ = imread(outputMatch[i].filename, CV_LOAD_IMAGE_UNCHANGED);
		split(templ, oSplitTempl);
		templ.copyTo(outputImg(Rect(outputMatch[i].pos[0] / scale, outputMatch[i].pos[1] / scale, templ.cols, templ.rows)), oSplitTempl[3]);
	}
}

void outputDiff(vector<matchDiff> allDiff){
	FILE *pFile;
	pFile = fopen("outAllDiff.txt", "w");
	for (int i = 0; i < allDiff.size(); ++i)
		fprintf(pFile, "%s,%d,%d,%f\n", allDiff[i].filename, allDiff[i].pos[0], allDiff[i].pos[1], allDiff[i].diff);
	fclose(pFile);
}

// @function main 
int main(int argc, char** argv)
{
	Mat img, outputImg;
	double scale = 0.2;
	vector<matchDiff> allDiff;

	img = imread("input.png", CV_LOAD_IMAGE_UNCHANGED);

	runTemplateMatching(img, allDiff, scale);
	outputDiff(allDiff);
	outputByTxt(img, outputImg, scale);

	namedWindow("output", CV_WINDOW_AUTOSIZE);
	imshow("output", outputImg);
	imwrite("outputImg.png", outputImg);

	waitKey(0);
	return 0;
}
*/