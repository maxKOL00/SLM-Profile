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
		Core.saveImageBitmap();
		Core.execfft();
		Core.savePhasePlot();
		Core.saveImageBitmap();
		Core.saveTarget();
		Core.weightImage();
		Core.swapAmp();
		Core.execifft();
		Core.swapPhase();
		Core.saveDiffPlane();
		Core.savePhasePlot();
	}
	Core.execfft();//don't do this and you should have good image plane!!!!!!!!!!!!!why?
	Core.saveImageBitmap();
	Core.savePhaseN();//store final phase (in image plane) for amp loop
}

void Hologram::gerchbergAmpLoop(int itterations) {
	for (int i = 0; i < itterations; i++) {
		Core.execfft();
		Core.weightImage2();
		Core.swapPhaseN();//image plane is now target image plug phaseN
		Core.execifft();
		Core.swapPhase(); //update the new phase pattern
	}
	Core.execfft();
}

void Hologram::saveImageAsBitmap() {
	Core.saveImageBitmap();
}

void Hologram::saveDiffPlaneAsBitmap() {
	Core.saveDiffPlane();
}