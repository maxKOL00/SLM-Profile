//#include "pch.h"
#include "Hologram.h"
#include "Control.h"


Hologram::Hologram(size_t nx, size_t ny, int activen, double spacing, int Dim) :
	Core(nx, ny, activen, spacing, Dim) {

}

Hologram::Hologram() : Core(8192, 8192, 1024, 20, 10) {

}

void Hologram::setNX(size_t nx) { Core.setDim(nx); }
void Hologram::setNY(size_t ny) { Core.setPaddedSize(ny); }
void Hologram::setActive(int active) { Core.setActiveSize(active); }
void Hologram::setSpace(double space) { Core.setTrapSpacing(space); }
void Hologram::setDim(int Dim) { Core.setDim(Dim); }
void Hologram::populateArrays() { Core.createArrays(); }

void Hologram::gerchbergPhaseLoop(int itterations) {
	for (int i = 0; i < itterations; i++) {
		Core.execfft();
		Core.weightImage();
		Core.swapAmp();
		Core.execifft();
		Core.swapPhase();
	}
	Core.execfft();
	Core.savePhaseN();//store final phase (in image plane) for amp loop
}

void Hologram::gerchbergAmpLoop(int itterations) {
	for (int i = 0; i < itterations; i++) {
		Core.execfft();
		Core.weightImage();
		Core.swapPhaseN();//image plane is now target image plug phaseN
		Core.execifft();
		Core.swapPhase(); //update the new phase pattern
	}
}

void Hologram::saveImageAsBitmap() {
	Core.saveImageBitmap();
}

void Hologram::saveDiffPlaneAsBitmap() {
	Core.saveDiffPlane();
}