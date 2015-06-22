#pragma once

typedef unsigned int uint;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/stream_accessor.hpp>
#include <opencv2/gpu/device/vec_traits.hpp>

#include <cuda_runtime.h>

using namespace cv;
using namespace cv::gpu;

void matchTemplate_SQDIFF_32F(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&);