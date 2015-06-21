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
	short x, y; // int 會超過記憶體限制
	float diff, scale;
}matchDiff;


bool compareByDiff(const matchDiff &a, const matchDiff &b){
	return a.diff != b.diff ? a.diff < b.diff : a.scale > b.scale;
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
		char *fileWithDir = (char *)malloc(40 * sizeof(char));
		sprintf(fileWithDir, "%s\\%s", inputDir, fileName);
		printf("%s\n", fileName);

		templ = imread(fileWithDir, CV_LOAD_IMAGE_UNCHANGED);

		for (double s = 0.1; s <= scale ; s += 0.05){
			resize(templ, resizeTempl, Size(templ.cols * s, templ.rows * s));
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
					pDiff.x = (short)(x / scale);
					pDiff.y = (short)(y / scale);
					pDiff.diff = ((float *)resultImg.data)[y * resultImg.cols + x];
					pDiff.scale = s / scale;
					allDiff.push_back(pDiff);
				}
			}
		}
	}
	fclose(file);
	system("rm matchDir.txt");
	printf("Matching complete !!\n");
}

void outputDiff(vector<matchDiff> &allDiff, char *filename){
	FILE *pFile;
	pFile = fopen(filename, "w");
	for (int i = 0; i < allDiff.size(); ++i)
		fprintf(pFile, "%s,%d,%d,%f,%f\n", allDiff[i].filename, allDiff[i].x, allDiff[i].y, allDiff[i].scale, allDiff[i].diff);
	fclose(pFile);
	printf("write all diff file\n");
	allDiff.clear();
}

void outputByObj(Mat img, Mat &outputImg, double scale, vector<matchDiff> &allDiff){
	img.copyTo(outputImg);
	map<string, int>fileNameHash;

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
	}
}

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
			matchDiff pDiff; pDiff.filename = filename; pDiff.x = x; pDiff.y = y; pDiff.scale = t_scale;
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

	sort(allDiff.begin(), allDiff.end(), compareByDiff);
	printf("Sort by diff finish !! size: %d\n", allDiff.size());

	outputByObj(img, outputImg, scale, allDiff);

	//outputDiff(allDiff, "outAllDiff.txt");
	//outputByTxt(img, outputImg, scale, "outAllDiff.txt");

	namedWindow("output", CV_WINDOW_AUTOSIZE);
	imshow("output", outputImg);
	imwrite("outputImg.png", outputImg);

	waitKey(0);
	return 0;
}