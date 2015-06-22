#include <Windows.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "matchTemplate.h"

using namespace cv;
using namespace std;
using namespace System;
using namespace System::Collections;
using namespace System::Collections::Generic;

ref class matchDiff{
public:
	std::string *filename;
	short x, y; // int 會超過記憶體限制
	float diff, scale;
};


int compareByDiff(matchDiff ^a, matchDiff ^b){
	return a->diff != b->diff ? a->diff < b->diff : a->scale > b->scale;
}
void runTemplateMatching(Mat img, List<matchDiff ^>^ allDiff, double scale){
	Mat templ;
	Mat resizeImg, resizeTempl; // resize + rgba
	Mat splitImg[4], splitTempl[4]; // resize + r,g,b,a
	Mat rgbImg, rgbTempl; // resize + rgb
	char *inputDir = "match", *outputDir = "result";
	allDiff = gcnew List<matchDiff ^>();

	resize(img, resizeImg, Size(img.cols * scale, img.rows * scale));
	resizeImg.convertTo(resizeImg, CV_32FC4, 1.0 / 255.0);
	split(resizeImg, splitImg);
	merge(splitImg, 3, rgbImg);
	cvtColor(rgbImg, rgbImg, CV_RGB2HSV);
	GpuMat d_rgbImg(rgbImg);

	char systemStr[100];
	sprintf(systemStr, "dir /B %s > matchDir.txt", inputDir);
	system(systemStr);
	FILE *file = fopen("matchDir.txt", "r");
	char fileName[100];
	while (fgets(fileName, 100, file)){
		strtok(fileName, "\n");
		char *fileWithDir = (char *)malloc(40 * sizeof(char));
		sprintf(fileWithDir, "%s\\%s", inputDir, fileName);
		printf("%s\n", fileName);

		templ = imread(fileWithDir, CV_LOAD_IMAGE_UNCHANGED);

		for (double s = 0.08; s <= scale ; s += 0.04){
			resize(templ, resizeTempl, Size(templ.cols * s, templ.rows * s));
			resizeTempl.convertTo(resizeTempl, CV_32FC4, 1.0 / 255.0);
			split(resizeTempl, splitTempl);
			merge(splitTempl, 3, rgbTempl);
			cvtColor(rgbTempl, rgbTempl, CV_RGB2HSV);

			GpuMat d_resultImg, d_rgbTempl(rgbTempl), d_mask(splitTempl[3]);
			matchTemplate_SQDIFF_32F(d_rgbImg, d_rgbTempl, d_resultImg, d_mask);

			Mat resultImg;
			d_resultImg.download(resultImg);

			for (int x = 0; x < resultImg.cols; ++x){
				for (int y = 0; y < resultImg.rows; ++y){
					matchDiff ^pDiff = gcnew matchDiff;
					pDiff->filename = new std::string(fileWithDir);
					pDiff->x = (short)(x / scale);
					pDiff->y = (short)(y / scale);
					pDiff->diff = ((float *)resultImg.data)[y * resultImg.cols + x];
					pDiff->scale = s / scale;
					allDiff->Add(pDiff);
				}
			}
		}
	}
	fclose(file);
	system("rm matchDir.txt");
	printf("Matching complete !!\n");
}
/*
void outputDiff(vector<matchDiff> &allDiff, char *filename){
	FILE *pFile;
	pFile = fopen(filename, "w");
	for (int i = 0; i < allDiff.size(); ++i)
		fprintf(pFile, "%s,%d,%d,%f,%f\n", allDiff[i].filename, allDiff[i].x, allDiff[i].y, allDiff[i].scale, allDiff[i].diff);
	fclose(pFile);
	printf("write all diff file\n");
	allDiff.clear();
}
*/
bool checkout(Mat &check, int x, int y, int dis){
	for (int i = x - dis; i <= x + dis; ++i)
		for (int j = y - dis; j <= y + dis; ++j)
		if (((uchar *)check.data)[j * check.cols + i])
			return true;
	return false;
}
void outputByObj(Mat img, Mat &outputImg, double scale, List<matchDiff ^>^ allDiff){
	img.copyTo(outputImg);
	/*map<string, int>fileNameHash;

	vector<matchDiff> outputMatch; outputMatch.clear();
	for(int i = 0; i < allDiff.size() - 1 && outputMatch.size() < 100; ++ i){
		string fileStr(allDiff[i].filename);
		if (fileNameHash.find(fileStr) == fileNameHash.end()){
			fileNameHash[fileStr] = 1;
			matchDiff pDiff = allDiff[i];
			outputMatch.push_back(pDiff);
		}
	}

	Mat templ, mask[4];
	for (int i = outputMatch.size() - 1; i >= 0; --i){
		templ = imread(outputMatch[i].filename, CV_LOAD_IMAGE_UNCHANGED);
		resize(templ, templ, Size(templ.cols * outputMatch[i].scale, templ.rows * outputMatch[i].scale));
		split(templ, mask);
		templ.copyTo(outputImg(Rect(outputMatch[i].x, outputMatch[i].y, templ.cols, templ.rows)), mask[3]);
	}*/

	Mat templ, mask[4];
	Mat check(img.rows, img.cols, CV_8UC1);
	for (int i = 49999; i >= 0; --i){
		if (checkout(check, allDiff[i]->x, allDiff[i]->y, 100))
			continue;
		((uchar *)check.data)[(allDiff[i]->y + 2) * check.cols + allDiff[i]->x + 2] = 1;
		templ = imread(*allDiff[i]->filename, CV_LOAD_IMAGE_UNCHANGED);
		resize(templ, templ, Size(templ.cols * allDiff[i]->scale, templ.rows * allDiff[i]->scale));
		split(templ, mask);
		templ.copyTo(outputImg(Rect(allDiff[i]->x, allDiff[i]->y, templ.cols, templ.rows)), mask[3]);
	}
}
/*
void outputByTxt(Mat img, Mat &outputImg, double scale, char *filename){
	img.copyTo(outputImg);
	map<string, int>fileNameHash;

	FILE *pFile;
	pFile = fopen(filename, "r");
	char str[200];
	vector<matchDiff> outputMatch; outputMatch.clear();
	while (fgets(str, 200, pFile)){
		char *filename = (char *)malloc(40 * sizeof(char));
		short x, y;
		double t_scale;
		char *pch = strtok(str, ","); strcpy(filename, pch);
		pch = strtok(NULL, ","); x = atoi(pch);
		pch = strtok(NULL, ","); y = atoi(pch);
		pch = strtok(NULL, ","); t_scale = atof(pch);

		string fileStr(filename);
		if (fileNameHash.find(fileStr) == fileNameHash.end()){
			fileNameHash[fileStr] = 1;
			matchDiff pDiff; pDiff.filename = *filename; pDiff.x = x; pDiff.y = y; pDiff.scale = t_scale;
			outputMatch.push_back(pDiff);
		}
		if (outputMatch.size() >= 100)
			break;
	}
	fclose(pFile);

	Mat templ, mask[4];
	for (int i = outputMatch.size() - 1; i >= 0; --i){
		templ = imread(outputMatch[i].filename, CV_LOAD_IMAGE_UNCHANGED);
		resize(templ, templ, Size(templ.cols * outputMatch[i].scale, templ.rows * outputMatch[i].scale));
		split(templ, mask);
		templ.copyTo(outputImg(Rect(outputMatch[i].x, outputMatch[i].y, templ.cols, templ.rows)), mask[3]);
	}
}
*/
int main(int argc, char* argv[])
{

	// test gpu support
	printf("Device count: %i\n", getCudaEnabledDeviceCount());
	DeviceInfo dev_info(0);
	printf("Device compatible: %d\n", dev_info.isCompatible());

	Mat img, outputImg;
	double scale = 0.2;
	//vector<matchDiff> allDiff;
	List<matchDiff ^>^ allDiff;

	img = imread("input.png", CV_LOAD_IMAGE_UNCHANGED);


	LARGE_INTEGER startTime, endTime, fre;
	QueryPerformanceFrequency(&fre); // 取得CPU頻率
	QueryPerformanceCounter(&startTime); // 取得開機到現在經過幾個CPU Cycle
	
	runTemplateMatching(img, allDiff, scale);

	QueryPerformanceCounter(&endTime); //取得開機到程式執行完成經過幾個CPU Cycle
	printf("Elapsed Time: %lf\n", ((double)endTime.QuadPart - (double)startTime.QuadPart) / fre.QuadPart);

	allDiff->Sort(gcnew Comparison<matchDiff ^>(compareByDiff));
	printf("Sort by diff finish !! size: %d\n", allDiff->Count);

	outputByObj(img, outputImg, scale, allDiff);

	//outputDiff(allDiff, "outAllDiff.txt");
	//outputByTxt(img, outputImg, scale, "outAllDiff.txt");

	namedWindow("output", CV_WINDOW_AUTOSIZE);
	imshow("output", outputImg);
	imwrite("outputImg.png", outputImg);

	waitKey(0);
	return 0;
}