#pragma once

//#include "Control.h"
#include "cudaCore.cuh"


class Hologram
{

	public:
		Hologram::Hologram(size_t nx, size_t ny, int activen, double spacing, int Dim);
		Hologram::Hologram();
		void setNX(size_t nx);
		void setNY(size_t ny);
		void setActive(int active);
		void setSpace(double space);
		void setDim(int Dim);
		void gerchbergPhaseLoop(int itterations);
		void populateArrays();
		void gerchbergAmpLoop(int itterations);
		void saveImageAsBitmap();
		void saveDiffPlaneAsBitmap();
	private:
		cudaCore Core;
};

