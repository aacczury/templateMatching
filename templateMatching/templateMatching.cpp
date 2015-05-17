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

/** @function main */
int main(int argc, char** argv)
{
	Mat img, templ;
	Mat oSplitTempl[4]; // r,g,b,a
	Mat resizeImg, resizeTempl; // resize + rgba
	Mat splitImg[4], splitTempl[4]; // resize + r,g,b,a
	Mat rgbImg, rgbTempl; // resize + rgb
	double scale = 0.2;
	char *inputDir = "match", *outputDir = "result";

	img = imread("input.png", CV_LOAD_IMAGE_UNCHANGED);
	resize(img, resizeImg, Size(img.cols * scale, img.rows * scale));
	split(resizeImg, splitImg);
	merge(splitImg, 3, rgbImg);
	rgbImg.convertTo(rgbImg, CV_32FC4, 1.0 / 255.0);

	//System::String^ folder = gcnew System::String(inputDir);
	//array<System::String ^>^ file = Directory::GetFiles(folder);
	//vector<matchDiff> allDiff; allDiff.clear();
	//for (int i = 0; i < file->Length; ++i){
	//	char *fileWithDir = (char*)System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(file[i]).ToPointer();
	//	char *fileName = (char*)System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(file[i]->Split('\\')[1]).ToPointer();
	//	
	//	printf("%s\n", fileName);
	//	
	//	templ = imread(fileWithDir, CV_LOAD_IMAGE_UNCHANGED);
	//	split(templ, oSplitTempl);
	//	resize(templ, resizeTempl, Size(templ.cols * scale, templ.rows * scale));
	//	split(resizeTempl, splitTempl); // alpha channel is mask would be CV_8U
	//	merge(splitTempl, 3, rgbTempl);
	//	rgbTempl.convertTo(rgbTempl, CV_32FC4, 1.0 / 255.0);

	//	Mat resultImg;
	//	templateMatching(rgbImg, resultImg, rgbTempl, splitTempl[3], allDiff, fileWithDir);

	//	/// Source image to display
	//	/*Mat outputImg;
	//	img.copyTo(outputImg);

	//	double minVal, maxVal;
	//	Point minLoc, maxLoc;
	//	minMaxLoc(resultImg, &minVal, &maxVal, &minLoc, &maxLoc);
	//	templ.copyTo(outputImg(Rect(minLoc.x / scale, minLoc.y / scale, templ.cols, templ.rows)), oSplitTempl[3]);
	//	
	//	char resultFile[50], outputFile[50];
	//	sprintf(resultFile, "%s/result_%s", outputDir, fileName);
	//	sprintf(outputFile, "%s/output_%s", outputDir, fileName);
	//	imwrite(resultFile, resultImg);
	//	imwrite(outputFile, outputImg);*/
	//}

	//Mat outputImg;
	//img.copyTo(outputImg);
	//sort(allDiff.begin(), allDiff.end(), compareByDiff);
	//for (int i = 0; i < 100; ++i){
	//	templ = imread(allDiff[i].filename, CV_LOAD_IMAGE_UNCHANGED);
	//	split(templ, oSplitTempl);
	//	templ.copyTo(outputImg(Rect(allDiff[i].pos[0] / scale, allDiff[i].pos[1] / scale, templ.cols, templ.rows)), oSplitTempl[3]);
	//	//printf("%s\t(%d, %d): %f\n", allDiff[i].filename, allDiff[i].pos[0], allDiff[i].pos[1], allDiff[i].diff);
	//}

	Mat outputImg;
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

	for (int i = outputMatch.size() - 1; i >= 0; --i){
		templ = imread(outputMatch[i].filename, CV_LOAD_IMAGE_UNCHANGED);
		split(templ, oSplitTempl);
		templ.copyTo(outputImg(Rect(outputMatch[i].pos[0] / scale, outputMatch[i].pos[1] / scale, templ.cols, templ.rows)), oSplitTempl[3]);
	}
	

	namedWindow("output", CV_WINDOW_AUTOSIZE);
	imshow("output", outputImg);
	imwrite("outputImg.png", outputImg);

	//FILE *pFile;
	//pFile = fopen("outAllDiff.txt", "w");
	//for (int i = 0; i < allDiff.size(); ++i)
	//	fprintf(pFile, "%s,%d,%d,%f\n", allDiff[i].filename, allDiff[i].pos[0], allDiff[i].pos[1], allDiff[i].diff);
	//fclose(pFile);

	waitKey(0);
	return 0;
}