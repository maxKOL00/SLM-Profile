#pragma once
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"
#include <vector>



class cudaCore {
	public:
		__host__ cudaCore::cudaCore(size_t nx, size_t ny, int activen, double trap_spacing, int dim);
		__host__ cudaCore::~cudaCore();
		__host__ void execfft();
		__host__ void execifft();
		__host__ void swapPhase();
		__host__ void swapAmp();
		__host__ void setDim(int Dim);
		__host__ void weightImage();
		__host__ void weightImage2();
		__host__ void savePhaseN();
		__host__ void swapPhaseN();
		__host__ double gaussian(int i, int j);
		__host__ cufftDoubleComplex* get_image_plane();
		__host__ cufftDoubleComplex* get_diff_plane();
		__host__ void setPaddedSize(int size);
		__host__ void setActiveSize(int size);
		__host__ void setTrapSpacing(double spacing);
		__host__ void saveDiffPlane();
		__host__ void saveImageBitmap();
		__host__ void createArrays();
		__host__ void saveTarget();
		__host__ void savePhasePlot();
	private:
		size_t NX;
		size_t NY;
		int activeN;
		int mid, dim;
		int norm, st;
		double trap_spacing;
		std::string GPU_name;
		double *phaseN; // can't use vector or other stl function in cuda
		cufftDoubleReal* input_intensity;
		cufftDoubleReal* target;
		cufftHandle plan;
		cufftDoubleComplex* DiffPlane;
		cufftDoubleComplex* ImagePlane;
		cudaError_t err;
};