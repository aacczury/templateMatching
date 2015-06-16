#include <Windows.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/stream_accessor.hpp>
#include <opencv2/gpu/device/vec_traits.hpp>

#include <cuda_runtime.h>



using namespace cv;
using namespace cv::gpu;
using namespace std;


__device__ __forceinline__ float sum(float v) { return v; }
__device__ __forceinline__ float sum(float2 v) { return v.x + v.y; }
__device__ __forceinline__ float sum(float3 v) { return v.x + v.y + v.z; }
__device__ __forceinline__ float sum(float4 v) { return v.x + v.y + v.z + v.w; }

__device__ __forceinline__ float add(float a, float b) { return a + b; }
__device__ __forceinline__ float2 add(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ float3 add(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float4 add(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }

__device__ __forceinline__ float mul(float a, float b) { return a * b; }
__device__ __forceinline__ float2 mul(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
__device__ __forceinline__ float3 mul(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ __forceinline__ float4 mul(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }

__device__ __forceinline__ float sub(float a, float b) { return a - b; }
__device__ __forceinline__ float2 sub(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ float3 sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float4 sub(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }

template <typename T, int cn>
__global__ void matchTemplateNaiveKernel_SQDIFF(int w, int h, const PtrStepb image, const PtrStepb templ, PtrStepSzf result, PtrStepSzf mask){
	typedef typename device::TypeVec<T, cn>::vec_type Type;
	typedef typename device::TypeVec<float, cn>::vec_type Typef;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < result.cols && y < result.rows){
		Typef res = device::VecTraits<Typef>::all(0);
		Typef delta;

		for (int i = 0; i < h; ++i){
			const Type* image_ptr = (const Type*)image.ptr(y + i);
			const Type* templ_ptr = (const Type*)templ.ptr(i);
			float *mask_ptr = (float *)mask.ptr(i);
			for (int j = 0; j < w; ++j){
				if (mask_ptr[j] > 0.5){
					delta = sub(image_ptr[x + j], templ_ptr[j]);
					res = add(res, mul(delta, delta));
				}
			}
		}

		result.ptr(y)[x] = sum(res);
	}
}

template <typename T, int cn>
void matchTemplateNaive_SQDIFF(const PtrStepSzb image, const PtrStepSzb templ, PtrStepSzf result, PtrStepSzf mask){
	const dim3 threads(32, 8);
	const dim3 grid(divUp(result.cols, threads.x), divUp(result.rows, threads.y));

	matchTemplateNaiveKernel_SQDIFF<T, cn> << <grid, threads >> >(templ.cols, templ.rows, image, templ, result, mask);
	cudaSafeCall(cudaGetLastError());

	cudaSafeCall(cudaDeviceSynchronize());
}

void matchTemplateNaive_SQDIFF_32F(const PtrStepSzb image, const PtrStepSzb templ, PtrStepSzf result, int cn, PtrStepSzf mask){
	typedef void(*caller_t)(const PtrStepSzb image, const PtrStepSzb templ, PtrStepSzf result, PtrStepSzf mask);

	static const caller_t callers[] = {
		0, matchTemplateNaive_SQDIFF<float, 1>, matchTemplateNaive_SQDIFF<float, 2>, matchTemplateNaive_SQDIFF<float, 3>, matchTemplateNaive_SQDIFF<float, 4>
	};

	callers[cn](image, templ, result, mask);
}

void matchTemplate_SQDIFF_32F(const GpuMat& image, const GpuMat& templ, GpuMat& result, GpuMat& mask){
	result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32F);
	matchTemplateNaive_SQDIFF_32F(image, templ, result, image.channels(), mask);
}

typedef struct tmpMatchDiff{
	char *filename;
	Vec2i pos;
	float diff;
}matchDiff;


bool compareByDiff(const matchDiff &a, const matchDiff &b){
	return a.diff < b.diff;
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
	GpuMat d_rgbImg(rgbImg);

	char systemStr[100];
	sprintf(systemStr, "dir /B %s > matchDir.txt", inputDir);
	system(systemStr);
	FILE *file = fopen("matchDir.txt", "r");
	char fileName[100];
	while (fgets(fileName, 100, file)){
		strtok(fileName, "\n");
		char *fileWithDir = (char *)malloc(100 * sizeof(char));
		sprintf(fileWithDir, "%s\\%s", inputDir, fileName);
		printf("%s\n", fileName);

		templ = imread(fileWithDir, CV_LOAD_IMAGE_UNCHANGED);
		resize(templ, resizeTempl, Size(templ.cols * scale, templ.rows * scale));
		resizeTempl.convertTo(resizeTempl, CV_32FC4, 1.0 / 255.0);
		split(resizeTempl, splitTempl);
		merge(splitTempl, 3, rgbTempl);

		GpuMat d_resultImg, d_rgbTempl(rgbTempl), d_mask(splitTempl[3]);
		matchTemplate_SQDIFF_32F(d_rgbImg, d_rgbTempl, d_resultImg, d_mask);

		Mat resultImg;
		d_resultImg.download(resultImg);

		for (int x = 0; x < resultImg.cols; ++x){
			for (int y = 0; y < resultImg.rows; ++y){
				matchDiff pDiff;
				pDiff.filename = fileWithDir;
				pDiff.pos = Vec2i(x, y); // (col, row) (x, y)
				pDiff.diff = ((float *)resultImg.data)[y * resultImg.cols + x];
				allDiff.push_back(pDiff);
			}
		}
	}
	fclose(file);
	system("rm matchDir.txt");
	sort(allDiff.begin(), allDiff.end(), compareByDiff);
}

void outputDiff(vector<matchDiff> allDiff){
	FILE *pFile;
	pFile = fopen("outAllDiff.txt", "w");
	for (int i = 0; i < allDiff.size(); ++i)
		fprintf(pFile, "%s,%d,%d,%f\n", allDiff[i].filename, allDiff[i].pos[0], allDiff[i].pos[1], allDiff[i].diff);
	fclose(pFile);
}

void outputByTxt(Mat img, Mat &outputImg, double scale){
	img.copyTo(outputImg);
	map<string, int>fileNameHash;

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

		string fileStr(filename);
		if (fileNameHash.find(fileStr) == fileNameHash.end()){
			//cout << fileStr << endl;
			count++;
			fileNameHash[fileStr] = 1;
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

int main(int argc, char* argv[])
{

	// test gpu support
	printf("Device count: %i\n", getCudaEnabledDeviceCount());
	DeviceInfo dev_info(0);
	printf("Device compatible: %d\n", dev_info.isCompatible());

	Mat img, outputImg;
	double scale = 0.2;
	vector<matchDiff> allDiff;

	img = imread("input.png", CV_LOAD_IMAGE_UNCHANGED);


	LARGE_INTEGER startTime, endTime, fre;
	QueryPerformanceFrequency(&fre); // 取得CPU頻率
	QueryPerformanceCounter(&startTime); // 取得開機到現在經過幾個CPU Cycle
	
	runTemplateMatching(img, allDiff, scale);

	QueryPerformanceCounter(&endTime); //取得開機到程式執行完成經過幾個CPU Cycle
	printf("Elapsed Time: %lf\n", ((double)endTime.QuadPart - (double)startTime.QuadPart) / fre.QuadPart);

	outputDiff(allDiff);
	outputByTxt(img, outputImg, scale);

	namedWindow("output", CV_WINDOW_AUTOSIZE);
	imshow("output", outputImg);
	imwrite("outputImg.png", outputImg);

	waitKey(0);
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