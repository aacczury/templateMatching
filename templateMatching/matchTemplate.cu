#include "matchTemplate.h"

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
